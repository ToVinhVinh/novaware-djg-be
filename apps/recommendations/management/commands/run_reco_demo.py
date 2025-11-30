from __future__ import annotations

from dataclasses import asdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser
from django.utils import timezone

from apps.products.models import Category, Product
from apps.recommendations.cbf.models import engine as cbf_engine, recommend_cbf
from apps.recommendations.gnn.models import engine as gnn_engine, recommend_gnn
from apps.recommendations.hybrid.models import engine as hybrid_engine, recommend_hybrid
from apps.recommendations.common.exceptions import ModelNotTrainedError
from apps.recommendations.common.filters import CandidateFilter
from apps.users.models import User, UserInteraction

class Command(BaseCommand):
    help = "Train one selected recommendation model, run recommendations for a demo user and product, and log results to a .txt file."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--model",
            choices=["gnn", "cbf", "hybrid"],
            default="cbf",
            help="Which model to train before running the demo.",
        )
        parser.add_argument(
            "--outfile",
            default="recommendation_demo_results.txt",
            help="Output file name (relative to BASE_DIR).",
        )
        parser.add_argument(
            "--top-k-personal",
            type=int,
            default=5,
            help="Top K personalized items.",
        )
        parser.add_argument(
            "--top-k-outfit",
            type=int,
            default=4,
            help="Top K items per category for outfit.",
        )

    def handle(self, *args, **options) -> str | None:
        model_to_train: str = options["model"]
        outfile: str = options["outfile"]
        top_k_personal: int = options["top_k_personal"]
        top_k_outfit: int = options["top_k_outfit"]

        user, product = self._ensure_demo_data()

        self.stdout.write(self.style.NOTICE(f"Demo user: {user.email} (id={user.id})"))
        self.stdout.write(self.style.NOTICE(f"Current product: {product.name} (id={product.id})"))

        self.stdout.write(self.style.WARNING(f"Training model: {model_to_train} ..."))
        if model_to_train == "cbf":
            cbf_engine.train(force_retrain=True)
        elif model_to_train == "gnn":
            gnn_engine.train(force_retrain=True)
        elif model_to_train == "hybrid":
            hybrid_engine.train(force_retrain=True)

        request_params = {
            "user_id": str(user.id),
            "current_product_id": str(product.id),
            "top_k_personal": top_k_personal,
            "top_k_outfit": top_k_outfit,
        }

        results: dict[str, Any] = {}
        for name, fn in {
            "gnn": recommend_gnn,
            "cbf": recommend_cbf,
            "hybrid": recommend_hybrid,
        }.items():
            try:
                payload = fn(
                    user_id=user.id,
                    current_product_id=product.id,
                    top_k_personal=top_k_personal,
                    top_k_outfit=top_k_outfit,
                    request_params=request_params,
                )
                results[name] = {"status": "ok", "payload": payload}
            except ModelNotTrainedError as exc:
                results[name] = {"status": "error", "detail": str(exc)}
            except Exception as exc:
                results[name] = {"status": "error", "detail": f"{type(exc).__name__}: {exc}"}

        out_path = Path(settings.BASE_DIR) / outfile
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            f.write(f"Recommendation Demo Results ({timezone.now().isoformat()})\n")
            f.write(f"Trained model: {model_to_train}\n")
            f.write(f"User: {user.id} - {user.email}\n")
            f.write(f"Current product: {product.id} - {product.name}\n\n")
            for model_name, obj in results.items():
                f.write(f"=== Model: {model_name} ===\n")
                if obj.get("status") == "ok":
                    payload = obj.get("payload") or {}
                    f.write("Personalized:\n")
                    for item in payload.get("personalized", []):
                        f.write(f"- {item.get('product_id')} | {item.get('name')} | score={item.get('score')} | reason={item.get('reason')}\n")
                    f.write("\nOutfit:\n")
                    outfit = payload.get("outfit", {})
                    for cat, items in outfit.items():
                        f.write(f"* {cat}:\n")
                        for it in items:
                            f.write(f"  - {it.get('product_id')} | {it.get('name')} | score={it.get('score')}\n")
                    f.write(f"\nOutfit complete score: {payload.get('outfit_complete_score')}\n")
                else:
                    f.write(f"ERROR: {obj.get('detail')}\n")
                f.write("\n")

        self.stdout.write(self.style.SUCCESS(f"Wrote results to: {out_path}"))
        return None

    def _ensure_demo_data(self) -> tuple[User, Product]:
        cat_tops, _ = Category.objects.get_or_create(name="Tops")
        cat_bottoms, _ = Category.objects.get_or_create(name="Bottoms")
        cat_shoes, _ = Category.objects.get_or_create(name="Shoes")
        cat_accessories, _ = Category.objects.get_or_create(name="Accessories")

        user, _ = User.objects.get_or_create(
            email="demo_user@example.com",
            defaults={
                "password": "demo123!@
                "gender": User.Gender.FEMALE,
                "age": 25,
                "username": "demo_user",
            },
        )

        def ensure_product(name: str, slug: str, gender: str, age: str, cat_type: str, category: Category) -> Product:
            obj, _ = Product.objects.get_or_create(
                slug=slug,
                defaults={
                    "user": user,
                    "category": category,
                    "name": name,
                    "description": "Demo product",
                    "images": [],
                    "rating": 0,
                    "num_reviews": 0,
                    "price": Decimal("99.99"),
                    "sale": Decimal("0.00"),
                    "count_in_stock": 10,
                    "size": {},
                    "outfit_tags": ["streetwear", "casual"],
                    "style_tags": ["streetwear", "casual"],
                    "feature_vector": [],
                    "gender": gender,
                    "age_group": age,
                    "category_type": cat_type,
                },
            )
            return obj

        current = ensure_product(
            name="Demo Female Top",
            slug="demo-female-top",
            gender=Product.GenderChoices.FEMALE,
            age=Product.AgeGroupChoices.ADULT,
            cat_type=Product.CategoryTypeChoices.TOPS,
            category=cat_tops,
        )
        history_top = ensure_product(
            name="History Female Top",
            slug="history-female-top",
            gender=Product.GenderChoices.FEMALE,
            age=Product.AgeGroupChoices.ADULT,
            cat_type=Product.CategoryTypeChoices.TOPS,
            category=cat_tops,
        )
        candidate_bottom = ensure_product(
            name="Candidate Female Bottom",
            slug="candidate-female-bottom",
            gender=Product.GenderChoices.FEMALE,
            age=Product.AgeGroupChoices.ADULT,
            cat_type=Product.CategoryTypeChoices.BOTTOMS,
            category=cat_bottoms,
        )
        candidate_shoes = ensure_product(
            name="Candidate Unisex Shoes",
            slug="candidate-unisex-shoes",
            gender=Product.GenderChoices.UNISEX,
            age=Product.AgeGroupChoices.ADULT,
            cat_type=Product.CategoryTypeChoices.SHOES,
            category=cat_shoes,
        )
        candidate_accessories = ensure_product(
            name="Candidate Female Accessory",
            slug="candidate-female-accessory",
            gender=Product.GenderChoices.FEMALE,
            age=Product.AgeGroupChoices.ADULT,
            cat_type=Product.CategoryTypeChoices.ACCESSORIES,
            category=cat_accessories,
        )

        if not UserInteraction.objects.filter(user=user, product=history_top).exists():
            UserInteraction.objects.create(
                user=user,
                product=history_top,
                interaction_type=UserInteraction.InteractionType.PURCHASE,
            )

        CandidateFilter.build_context(
            user_id=user.id,
            current_product_id=current.id,
            top_k_personal=5,
            top_k_outfit=4,
        )

        return user, current


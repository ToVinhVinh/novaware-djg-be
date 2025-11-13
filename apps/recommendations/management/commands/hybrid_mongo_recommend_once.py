from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from apps.recommendations.hybrid.mongo_engine import recommend_hybrid_mongo


class Command(BaseCommand):
    help = "Run a single Mongo-native hybrid recommendation (ObjectId inputs)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--user-id", required=True, help="Mongo ObjectId of the user")
        parser.add_argument("--product-id", required=True, help="Mongo ObjectId of the current product")
        parser.add_argument("--top-k-personal", type=int, default=5)
        parser.add_argument("--top-k-outfit", type=int, default=4)
        parser.add_argument("--alpha", type=float, default=0.6, help="Blend weight between graph and content scores")
        parser.add_argument("--outfile", default="hybrid_mongo_result.json")

    def handle(self, *args, **options) -> str | None:
        user_id = options["user_id"]
        product_id = options["product_id"]
        top_k_personal: int = options["top_k_personal"]
        top_k_outfit: int = options["top_k_outfit"]
        alpha: float = options["alpha"]
        outfile: str = options["outfile"]

        payload: dict[str, Any] = recommend_hybrid_mongo(
            user_id=user_id,
            current_product_id=product_id,
            top_k_personal=top_k_personal,
            top_k_outfit=top_k_outfit,
            alpha=alpha,
        )

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.stdout.write(text)

        out_path = Path(settings.BASE_DIR) / outfile
        out_path.write_text(text, encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Wrote result to: {out_path}"))
        return None



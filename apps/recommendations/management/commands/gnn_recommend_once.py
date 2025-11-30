from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from django.conf import settings
from django.core.management.base import BaseCommand, CommandParser

from apps.recommendations.gnn.models import recommend_gnn

class Command(BaseCommand):
    help = "Run a single GNN recommendation for given user_id and current_product_id (accepts SQL int or Mongo ObjectId)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--user-id", required=True, help="User ID (SQL int or Mongo ObjectId)")
        parser.add_argument("--product-id", required=True, help="Current product ID (SQL int or Mongo ObjectId)")
        parser.add_argument("--top-k-personal", type=int, default=5)
        parser.add_argument("--top-k-outfit", type=int, default=4)
        parser.add_argument(
            "--outfile",
            default="gnn_recommend_once.json",
            help="Output file name (relative to BASE_DIR).",
        )

    def handle(self, *args, **options) -> str | None:
        user_id = options["user_id"]
        product_id = options["product_id"]
        top_k_personal: int = options["top_k_personal"]
        top_k_outfit: int = options["top_k_outfit"]
        outfile: str = options["outfile"]

        request_params = {
            "user_id": str(user_id),
            "current_product_id": str(product_id),
            "top_k_personal": top_k_personal,
            "top_k_outfit": top_k_outfit,
        }

        payload: dict[str, Any] = recommend_gnn(
            user_id=user_id,
            current_product_id=product_id,
            top_k_personal=top_k_personal,
            top_k_outfit=top_k_outfit,
            request_params=request_params,
        )

        text = json.dumps(payload, ensure_ascii=False, indent=2)
        self.stdout.write(text)

        out_path = Path(settings.BASE_DIR) / outfile
        out_path.write_text(text, encoding="utf-8")
        self.stdout.write(self.style.SUCCESS(f"Wrote result to: {out_path}"))
        return None


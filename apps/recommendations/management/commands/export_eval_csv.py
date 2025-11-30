from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Optional

from django.core.management.base import BaseCommand

from config.mongodb import connect_mongodb

from apps.users.mongo_models import User as MongoUser
from apps.users.mongo_models import UserInteraction as MongoInteraction
from apps.products.mongo_models import Product as MongoProduct

try:
    from bson import ObjectId
except Exception:
    ObjectId = None

def _to_str(x: Any) -> str:
    try:
        if ObjectId is not None and isinstance(x, ObjectId):
            return str(x)
    except Exception:
        pass
    if isinstance(x, (list, tuple)):
        return ";".join(_to_str(i) for i in x)
    if x is None:
        return ""
    return str(x)

def _dt_iso(x: Any) -> str:
    if isinstance(x, datetime):
        try:
            return x.isoformat()
        except Exception:
            pass
    if isinstance(x, str):
        for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(x, fmt).isoformat()
            except Exception:
                continue
        try:
            return datetime.fromisoformat(x).isoformat()
        except Exception:
            return x
    return _to_str(x)

def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None

def export_users(out_dir: str) -> str:
    path = os.path.join(out_dir, "users.csv")
    headers = [
        "id",
        "email",
        "username",
        "name",
        "gender",
        "age",
        "preferences_styles",
        "interaction_history_count",
        "interaction_history",
        "created_at",
        "updated_at",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for u in MongoUser.objects:
            prefs = getattr(u, "preferences", {}) or {}
            styles = []
            if isinstance(prefs, dict):
                styles = prefs.get("styles") or prefs.get("style_tags") or []

            interaction_history = getattr(u, "interaction_history", []) or []
            interaction_history_str = _to_str(interaction_history)

            w.writerow([
                _to_str(getattr(u, "id", None)),
                _to_str(getattr(u, "email", None)),
                _to_str(getattr(u, "username", None)),
                _to_str(getattr(u, "name", None)),
                _to_str(getattr(u, "gender", None)),
                _to_str(getattr(u, "age", None)),
                _to_str(styles),
                str(len(interaction_history)),
                interaction_history_str,
                _dt_iso(getattr(u, "created_at", getattr(u, "createdAt", None))),
                _dt_iso(getattr(u, "updated_at", getattr(u, "updatedAt", None))),
            ])
    return path

def export_products(out_dir: str) -> str:
    path = os.path.join(out_dir, "products.csv")
    headers = [
        "id",
        "gender",
        "age_group",
        "category_type",
        "masterCategory",
        "subCategory",
        "articleType",
        "baseColour",
        "season",
        "usage",
        "productDisplayName",
        "style_tags",
        "outfit_tags",
        "colors",
        "name",
        "slug",
        "price",
        "rating",
        "created_at",
        "updated_at",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for p in MongoProduct.objects:
            colors = getattr(p, "colors", []) or []

            row = [
                _to_str(getattr(p, "id", None)),
                _to_str(getattr(p, "gender", None)),
                _to_str(getattr(p, "age_group", None)),
                _to_str(getattr(p, "category_type", None)),
                _to_str(getattr(p, "masterCategory", None)),
                _to_str(getattr(p, "subCategory", None)),
                _to_str(getattr(p, "articleType", None)),
                _to_str(getattr(p, "baseColour", None)),
                _to_str(getattr(p, "season", None)),
                _to_str(getattr(p, "usage", None)),
                _to_str(getattr(p, "productDisplayName", None)),
                _to_str(getattr(p, "style_tags", [])),
                _to_str(getattr(p, "outfit_tags", [])),
                _to_str(colors),
                _to_str(getattr(p, "name", None)),
                _to_str(getattr(p, "slug", None)),
                _to_str(getattr(p, "price", None)),
                _to_str(getattr(p, "rating", None)),
                _dt_iso(getattr(p, "created_at", None)),
                _dt_iso(getattr(p, "updated_at", None)),
            ]
            w.writerow(row)
    return path

def export_interactions(out_dir: str, start: Optional[str] = None, end: Optional[str] = None) -> str:
    path = os.path.join(out_dir, "interactions.csv")
    headers = ["user_id", "product_id", "interaction_type", "rating", "timestamp"]
    q = {}
    dt_start = _parse_dt(start)
    dt_end = _parse_dt(end)
    if dt_start or dt_end:
        ts_q = {}
        if dt_start:
            ts_q["$gte"] = dt_start
        if dt_end:
            ts_q["$lte"] = dt_end
        q["timestamp"] = ts_q
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for it in MongoInteraction.objects(__raw__=q).order_by("-timestamp"):
            w.writerow([
                _to_str(getattr(it, "user_id", None)),
                _to_str(getattr(it, "product_id", None)),
                _to_str(getattr(it, "interaction_type", None)),
                _to_str(getattr(it, "rating", None)),
                _dt_iso(getattr(it, "timestamp", None)),
            ])
    return path

def export_pairs(out_dir: str, max_pairs: int = 10000) -> str:
    path = os.path.join(out_dir, "pairs.csv")
    headers = ["user_id", "current_product_id", "timestamp"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for it in MongoInteraction.objects.only("user_id", "product_id", "timestamp").order_by("-timestamp").limit(max_pairs):
            w.writerow([
                _to_str(getattr(it, "user_id", None)),
                _to_str(getattr(it, "product_id", None)),
                _dt_iso(getattr(it, "timestamp", None)),
            ])
    return path

class Command(BaseCommand):
    help = "Export evaluation-related data from MongoDB collections to CSV files."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--out", default="./exports", help="Output directory for CSV files")
        parser.add_argument("--start", default=None, help="Start ISO timestamp filter for interactions (optional)")
        parser.add_argument("--end", default=None, help="End ISO timestamp filter for interactions (optional)")
        parser.add_argument("--max-pairs", type=int, default=10000, help="Max rows for pairs.csv")

    def handle(self, *args, **options):
        out_dir: str = options["out"]
        start: Optional[str] = options.get("start")
        end: Optional[str] = options.get("end")
        max_pairs: int = options.get("max_pairs", 10000)

        os.makedirs(out_dir, exist_ok=True)

        connect_mongodb()

        users_csv = export_users(out_dir)
        prods_csv = export_products(out_dir)
        inter_csv = export_interactions(out_dir, start, end)
        pairs_csv = export_pairs(out_dir, max_pairs)

        self.stdout.write(self.style.SUCCESS("Export completed:"))
        self.stdout.write(f"- {users_csv}")
        self.stdout.write(f"- {prods_csv}")
        self.stdout.write(f"- {inter_csv}")
        self.stdout.write(f"- {pairs_csv}")


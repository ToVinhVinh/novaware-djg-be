"""
Export users from MongoDB to CSV file.

This script exports users data to CSV format matching the recommendation system format:
id,name,email,age,gender,interaction_history

Usage:
    python manage.py export_users_csv --out ./exports/users.csv
    python manage.py export_users_csv --out ./exports/users.csv --format-interactions
"""

from __future__ import annotations

import csv
import os
from datetime import datetime
from typing import Any, Optional

from django.core.management.base import BaseCommand

from novaware.mongodb import connect_mongodb
from apps.users.mongo_models import User as MongoUser

try:
    from bson import ObjectId
except ImportError:
    ObjectId = None


def _to_str(x: Any) -> str:
    """Convert value to string, handling ObjectId, lists, and None."""
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


def format_interaction_history(interactions: list) -> str:
    """
    Format interaction history to string format matching CSV format.
    
    Format: "{'product_id': 10866, 'interaction_type': 'purchase', 'timestamp': datetime.datetime(...), ...};{...}"
    
    Handles different interaction formats:
    - Direct dict: {'product_id': 10866, 'interaction_type': 'purchase', ...}
    - With ObjectId: {'_id': ObjectId('...'), 'productId': ObjectId('...'), ...}
    """
    if not interactions:
        return ""
    
    formatted_interactions = []
    for interaction in interactions:
        if not isinstance(interaction, dict):
            # Try to convert if it's a string representation
            if isinstance(interaction, str):
                try:
                    import ast
                    interaction = ast.literal_eval(interaction)
                except:
                    continue
            else:
                continue
        
        # Build interaction dict string
        parts = []
        for key, value in interaction.items():
            if value is None:
                continue
            
            # Format value based on type
            if isinstance(value, datetime):
                # Format: datetime.datetime(2025, 11, 17, 11, 10, 42, 24000)
                # Note: microseconds might be in different format
                microsecond = getattr(value, 'microsecond', 0)
                value_str = f"datetime.datetime({value.year}, {value.month}, {value.day}, {value.hour}, {value.minute}, {value.second}, {microsecond})"
            elif ObjectId is not None and isinstance(value, ObjectId):
                value_str = f"ObjectId('{str(value)}')"
            elif isinstance(value, str):
                # Escape single quotes in string
                escaped_value = value.replace("'", "\\'")
                value_str = f"'{escaped_value}'"
            elif isinstance(value, (int, float)):
                value_str = str(value)
            elif isinstance(value, bool):
                value_str = str(value)
            else:
                # For other types, convert to string and escape
                escaped_value = str(value).replace("'", "\\'")
                value_str = f"'{escaped_value}'"
            
            parts.append(f"'{key}': {value_str}")
        
        if parts:
            formatted_interactions.append("{" + ", ".join(parts) + "}")
    
    return ";".join(formatted_interactions)


def export_users_to_csv(output_path: str, format_interactions: bool = True) -> None:
    """
    Export users from MongoDB to CSV file.
    
    Args:
        output_path: Path to output CSV file
        format_interactions: If True, format interaction_history as string with datetime/ObjectId
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # CSV headers matching the expected format
    headers = ["id", "name", "email", "age", "gender", "interaction_history"]
    
    # Connect to MongoDB
    connect_mongodb()
    
    # Count total users
    total_users = MongoUser.objects.count()
    print(f"Found {total_users} users in MongoDB")
    
    # Write CSV file
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        
        processed = 0
        for user in MongoUser.objects:
            # Get user data
            user_id = _to_str(getattr(user, "id", None))
            name = _to_str(getattr(user, "name", None))
            email = _to_str(getattr(user, "email", None))
            age = _to_str(getattr(user, "age", None))
            gender = _to_str(getattr(user, "gender", None))
            
            # Get interaction_history
            interaction_history = getattr(user, "interaction_history", []) or []
            
            # Handle different formats of interaction_history
            if isinstance(interaction_history, str):
                # Already a string, use as is
                interaction_history_str = interaction_history
            elif format_interactions and interaction_history:
                # Format interaction_history with datetime/ObjectId
                interaction_history_str = format_interaction_history(interaction_history)
            else:
                # Simple string representation
                interaction_history_str = _to_str(interaction_history)
            
            # Write row
            writer.writerow([
                user_id,
                name,
                email,
                age,
                gender,
                interaction_history_str
            ])
            
            processed += 1
            if processed % 100 == 0:
                print(f"Processed {processed}/{total_users} users...")
    
    print(f"Successfully exported {processed} users to {output_path}")


class Command(BaseCommand):
    help = "Export users from MongoDB to CSV file"

    def add_arguments(self, parser):
        parser.add_argument(
            "--out",
            type=str,
            default="./exports/users.csv",
            help="Output CSV file path (default: ./exports/users.csv)"
        )
        parser.add_argument(
            "--format-interactions",
            action="store_true",
            default=True,
            help="Format interaction_history with datetime and ObjectId (default: True)"
        )
        parser.add_argument(
            "--no-format-interactions",
            action="store_false",
            dest="format_interactions",
            help="Don't format interaction_history (simple string representation)"
        )

    def handle(self, *args, **options):
        output_path = options["out"]
        format_interactions = options.get("format_interactions", True)
        
        self.stdout.write(f"Exporting users to {output_path}...")
        
        try:
            export_users_to_csv(output_path, format_interactions)
            self.stdout.write(
                self.style.SUCCESS(f"Successfully exported users to {output_path}")
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Error exporting users: {e}")
            )
            raise


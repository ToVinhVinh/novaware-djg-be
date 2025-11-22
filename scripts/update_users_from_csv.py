"""
Script to update users data in MongoDB from CSV file.
This script reads users.csv and updates the corresponding users in MongoDB.
"""

import os
import sys
import django
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'novaware.settings')
django.setup()

import csv
import logging
from datetime import datetime
from bson import ObjectId
import mongoengine
from django.conf import settings

from apps.users.mongo_models import User as MongoUser
from novaware.mongodb import connect_mongodb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_interaction_history(history_str):
    """Parse interaction_history string from CSV into list of dicts."""
    if not history_str or history_str.strip() == '':
        return []
    
    interactions = []
    # Split by semicolon
    parts = history_str.split(';')
    
    for part in parts:
        if not part.strip():
            continue
        
        try:
            # Parse the dict-like string
            # Format: {'product_id': 10866, 'interaction_type': 'purchase', ...}
            # or {'_id': ObjectId(...), 'productId': ObjectId(...), ...}
            
            # Remove outer braces and parse
            part = part.strip()
            if part.startswith('{') and part.endswith('}'):
                part = part[1:-1]
            
            # Parse key-value pairs
            interaction = {}
            pairs = part.split(',')
            
            for pair in pairs:
                pair = pair.strip()
                if ':' not in pair:
                    continue
                
                key, value = pair.split(':', 1)
                key = key.strip().strip("'\"")
                value = value.strip().strip("'\"")
                
                # Handle different value types
                if value.startswith('ObjectId(') and value.endswith(')'):
                    # Extract ObjectId string
                    oid_str = value[9:-1].strip().strip("'\"")
                    interaction[key] = ObjectId(oid_str)
                elif value.startswith('datetime.datetime('):
                    # Parse datetime
                    # Format: datetime.datetime(2025, 11, 17, 11, 10, 42, 24000)
                    try:
                        dt_str = value[18:-1]  # Remove 'datetime.datetime(' and ')'
                        parts = [int(x.strip()) for x in dt_str.split(',')]
                        interaction[key] = datetime(*parts)
                    except:
                        interaction[key] = datetime.utcnow()
                elif value.isdigit():
                    interaction[key] = int(value)
                elif value.replace('.', '', 1).isdigit():
                    interaction[key] = float(value)
                else:
                    interaction[key] = value
            
            # Normalize field names
            if 'productId' in interaction:
                interaction['product_id'] = interaction.pop('productId')
            if 'interactionType' in interaction:
                interaction['interaction_type'] = interaction.pop('interactionType')
            
            interactions.append(interaction)
        except Exception as e:
            logger.warning(f"Failed to parse interaction: {part[:50]}... Error: {e}")
            continue
    
    return interactions


def update_users_from_csv(csv_path):
    """Update users in MongoDB from CSV file."""
    csv_path = Path(csv_path)
    
    if not csv_path.exists():
        logger.error(f"CSV file not found: {csv_path}")
        return
    
    # Connect to MongoDB using project's connection method
    connect_mongodb()
    
    logger.info(f"Reading users from {csv_path}")
    
    updated_count = 0
    created_count = 0
    error_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            try:
                user_id = row.get('id', '').strip()
                if not user_id:
                    continue
                
                # Try to get user by ID
                try:
                    user_obj_id = ObjectId(user_id) if len(user_id) == 24 else user_id
                except:
                    user_obj_id = user_id
                
                user = MongoUser.objects(id=user_obj_id).first()
                
                if not user:
                    # Create new user
                    user = MongoUser(id=user_obj_id)
                    created_count += 1
                    logger.info(f"Creating new user: {user_id}")
                else:
                    updated_count += 1
                    logger.info(f"Updating user: {user_id}")
                
                # Update basic fields
                if row.get('name'):
                    user.name = row['name']
                if row.get('email'):
                    user.email = row['email']
                if row.get('age'):
                    try:
                        user.age = int(row['age'])
                    except:
                        pass
                if row.get('gender'):
                    gender = row['gender'].lower()
                    if gender in ['male', 'female', 'other']:
                        user.gender = gender
                
                # Update interaction_history
                if row.get('interaction_history'):
                    interaction_history = parse_interaction_history(row['interaction_history'])
                    if interaction_history:
                        # Merge with existing interactions (avoid duplicates)
                        existing_ids = set()
                        if user.interaction_history:
                            for existing in user.interaction_history:
                                # Create unique ID from product_id and timestamp
                                prod_id = str(existing.get('product_id', ''))
                                ts = str(existing.get('timestamp', ''))
                                existing_ids.add(f"{prod_id}_{ts}")
                        
                        # Add new interactions
                        for interaction in interaction_history:
                            prod_id = str(interaction.get('product_id', ''))
                            ts = str(interaction.get('timestamp', ''))
                            unique_id = f"{prod_id}_{ts}"
                            
                            if unique_id not in existing_ids:
                                user.interaction_history.append(interaction)
                                existing_ids.add(unique_id)
                        
                        logger.info(f"Updated interaction_history for user {user_id}: {len(interaction_history)} interactions")
                
                # Save user
                user.save()
                logger.info(f"Successfully saved user {user_id}")
                
            except Exception as e:
                error_count += 1
                logger.error(f"Error processing user {row.get('id', 'unknown')}: {e}")
                continue
    
    logger.info(f"\n=== Summary ===")
    logger.info(f"Created: {created_count}")
    logger.info(f"Updated: {updated_count}")
    logger.info(f"Errors: {error_count}")
    logger.info(f"Total processed: {created_count + updated_count + error_count}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Update users in MongoDB from CSV file')
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to users.csv file',
        default='exports/users.csv',
        nargs='?'
    )
    
    args = parser.parse_args()
    
    update_users_from_csv(args.csv_path)


"""Data loading utilities for Streamlit recommendation system."""

import ast
import csv
import datetime
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def parse_interaction_history(history_str: str) -> List[Dict]:
    """Parse interaction history string to list of dictionaries."""
    if not history_str or (isinstance(history_str, float) and pd.isna(history_str)):
        return []
    
    interactions = []
    # Split by semicolon
    parts = str(history_str).split(';')
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        try:
            # Try to parse as dictionary string
            # Handle both dict format and string format
            if part.startswith('{') and part.endswith('}'):
                # Replace datetime.datetime(...) with string
                part = re.sub(
                    r"datetime\.datetime\((\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)(?:,\s*(\d+))?\)",
                    r'"\1-\2-\3 \4:\5:\6"',
                    part
                )
                # Replace ObjectId(...) with string
                part = re.sub(r"ObjectId\('([^']+)'\)", r'"\1"', part)
                # Replace single quotes with double quotes for JSON
                part = part.replace("'", '"')
                
                try:
                    interaction = ast.literal_eval(part)
                    interactions.append(interaction)
                except:
                    # Try JSON parsing
                    try:
                        import json
                        interaction = json.loads(part)
                        interactions.append(interaction)
                    except:
                        continue
        except Exception as e:
            continue
    
    return interactions


def load_users_csv(file_path: str) -> pd.DataFrame:
    """Load users CSV file."""
    df = pd.read_csv(file_path)
    
    # Parse interaction_history
    if 'interaction_history' in df.columns:
        df['parsed_interactions'] = df['interaction_history'].apply(parse_interaction_history)
    
    return df


def load_products_csv(file_path: str) -> pd.DataFrame:
    """Load products CSV file."""
    df = pd.read_csv(file_path)
    return df


def load_interactions_csv(file_path: str) -> pd.DataFrame:
    """Load interactions CSV file."""
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime if needed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    return df


def prepare_data_for_models(
    users_df: pd.DataFrame,
    products_df: pd.DataFrame,
    interactions_df: pd.DataFrame
) -> Tuple[Dict, Dict, pd.DataFrame]:
    """
    Prepare data for recommendation models.
    
    Returns:
        Tuple of (user_dict, product_dict, interactions_df)
    """
    # Create user mapping
    user_dict = {}
    for _, row in users_df.iterrows():
        user_id = str(row['id'])
        user_dict[user_id] = {
            'id': user_id,
            'name': row.get('name', ''),
            'email': row.get('email', ''),
            'age': row.get('age', None),
            'gender': row.get('gender', ''),
            'interactions': row.get('parsed_interactions', [])
        }
    
    # Create product mapping
    product_dict = {}
    for _, row in products_df.iterrows():
        product_id = str(row['id'])
        product_dict[product_id] = {
            'id': product_id,
            'gender': row.get('gender', ''),
            'masterCategory': row.get('masterCategory', ''),
            'subCategory': row.get('subCategory', ''),
            'articleType': row.get('articleType', ''),
            'baseColour': row.get('baseColour', ''),
            'season': row.get('season', ''),
            'year': row.get('year', None),
            'usage': row.get('usage', ''),
            'productDisplayName': row.get('productDisplayName', '')
        }
    
    # Ensure interactions have proper types
    interactions_df['user_id'] = interactions_df['user_id'].astype(str)
    interactions_df['product_id'] = interactions_df['product_id'].astype(str)
    
    return user_dict, product_dict, interactions_df


def get_user_interactions(user_id: str, interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Get all interactions for a specific user."""
    return interactions_df[interactions_df['user_id'] == user_id].copy()


def get_product_interactions(product_id: str, interactions_df: pd.DataFrame) -> pd.DataFrame:
    """Get all interactions for a specific product."""
    return interactions_df[interactions_df['product_id'] == product_id].copy()


def filter_products_by_gender_age(
    products_df: pd.DataFrame,
    user_gender: str,
    user_age: int = None
) -> pd.DataFrame:
    """Filter products based on user gender and age."""
    filtered = products_df.copy()
    
    # Gender filtering
    if user_gender:
        user_gender_lower = user_gender.lower()
        # Allow matching gender or Unisex
        gender_mask = (
            (filtered['gender'].str.lower() == user_gender_lower) |
            (filtered['gender'].str.lower() == 'unisex')
        )
        filtered = filtered[gender_mask]
    
    # Age-based category filtering (e.g., Boys/Girls for age <= 12)
    if user_age is not None:
        if user_age <= 12:
            # For children, prefer Boys/Girls categories
            age_mask = (
                (filtered['gender'].str.lower().isin(['boys', 'girls', 'unisex'])) |
                (filtered['gender'].str.lower() == user_gender.lower())
            )
            filtered = filtered[age_mask]
    
    return filtered


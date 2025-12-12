from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict

import pandas as pd
from django.conf import settings
from rest_framework import permissions, status
from rest_framework.response import Response
from rest_framework.views import APIView

from apps.products.mongo_models import Product as MongoProduct
from apps.products.mongo_serializers import ProductSerializer as MongoProductSerializer

from apps.recommendations.common.exceptions import ModelNotTrainedError
from apps.utils.hybrid_utils import combine_hybrid_scores
from apps.utils.cbf_utils import get_allowed_genders
from apps.utils.user_profile import INTERACTION_WEIGHTS

from .serializers import HybridRecommendationSerializer

logger = logging.getLogger(__name__)

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
EXPORTS_DIR = BASE_DIR / "apps" / "exports"
ARTIFACTS_DIR = BASE_DIR / "artifacts"


def load_products_data() -> Optional[pd.DataFrame]:
    """Load products dataset from exports directory."""
    csv_path = EXPORTS_DIR / 'products.csv'
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
            df = df.set_index('id')
        return df
    except Exception as e:
        logger.error(f"Error loading products data: {e}")
        return None


def load_users_data() -> Optional[pd.DataFrame]:
    """Load users dataset from exports directory."""
    csv_path = EXPORTS_DIR / 'users.csv'
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'id' in df.columns:
            df['id'] = df['id'].astype(str)
            df = df.set_index('id')
        return df
    except Exception as e:
        logger.error(f"Error loading users data: {e}")
        return None


def load_interactions_data() -> Optional[pd.DataFrame]:
    """Load interactions dataset from exports directory."""
    csv_path = EXPORTS_DIR / 'interactions.csv'
    if not csv_path.exists():
        return None
    try:
        df = pd.read_csv(csv_path)
        if 'user_id' in df.columns:
            df['user_id'] = df['user_id'].astype(str)
        if 'product_id' in df.columns:
            df['product_id'] = df['product_id'].astype(str)
        return df
    except Exception as e:
        logger.error(f"Error loading interactions data: {e}")
        return None


def get_user_record(user_id: str, users_df: pd.DataFrame):
    """Get user record from DataFrame."""
    if users_df is None or user_id is None:
        return None
    try:
        if user_id in users_df.index:
            return users_df.loc[user_id]
        return users_df.loc[users_df.index.astype(str) == str(user_id)].iloc[0]
    except Exception:
        return None


def get_product_record(product_id: str, products_df: pd.DataFrame):
    """Get product record from DataFrame."""
    if products_df is None or product_id is None:
        return None
    product_key = str(product_id)
    try:
        if products_df.index.name is not None or not isinstance(products_df.index, pd.RangeIndex):
            if product_key in products_df.index.astype(str):
                return products_df.loc[product_key]
        if 'id' in products_df.columns:
            match = products_df[products_df['id'].astype(str) == product_key]
            if not match.empty:
                return match.iloc[0]
    except Exception:
        return None
    return None


def load_cached_predictions() -> Dict:
    """Load cached predictions from artifacts directory (giống Streamlit)."""
    predictions = {}
    
    # Load CBF predictions
    cbf_path = ARTIFACTS_DIR / "streamlit_cbf_predictions.pkl"
    if cbf_path.exists():
        try:
            with open(cbf_path, 'rb') as f:
                predictions['cbf'] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load CBF predictions: {e}")
    
    # Load GNN predictions (ưu tiên gnn_predictions, fallback gnn_training)
    gnn_path = ARTIFACTS_DIR / "streamlit_gnn_predictions.pkl"
    gnn_training_path = ARTIFACTS_DIR / "streamlit_gnn_training.pkl"
    
    if gnn_path.exists():
        try:
            with open(gnn_path, 'rb') as f:
                predictions['gnn'] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load GNN predictions: {e}")
    elif gnn_training_path.exists():
        # Fallback to gnn_training (giống Streamlit)
        try:
            with open(gnn_training_path, 'rb') as f:
                predictions['gnn'] = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load GNN training: {e}")
    
    return predictions


def ensure_hybrid_predictions(alpha: float, candidate_pool: int = 200) -> Optional[Dict]:
    """
    Ensure hybrid predictions are available.
    Recompute when alpha changes or cached predictions missing.
    """
    cached = load_cached_predictions()
    cbf_predictions = cached.get('cbf')
    gnn_predictions = cached.get('gnn')
    
    if cbf_predictions and gnn_predictions:
        combined = combine_hybrid_scores(
            cbf_predictions,
            gnn_predictions,
            alpha=alpha,
            top_k=max(candidate_pool, 50)
        )
        return combined
    
    if cbf_predictions and not gnn_predictions:
        fallback = {
            'predictions': cbf_predictions.get('predictions', {}),
            'rankings': cbf_predictions.get('rankings', {}),
            'alpha': alpha,
            'stats': {'note': 'Fallback to CBF scores (GNN predictions missing)'}
        }
        return fallback
    
    return None


def build_user_interaction_preferences(
    user_id: str,
    interactions_df: pd.DataFrame,
    products_df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Derive normalized preference weights from user interaction history.
    Returns dict with article, usage, gender preference maps in [0,1].
    """
    preference_maps = {
        'articleType': defaultdict(float),
        'usage': defaultdict(float),
        'gender': defaultdict(float)
    }
    
    if (
        interactions_df is None
        or products_df is None
        or interactions_df.empty
        or user_id is None
    ):
        return {k: {} for k in preference_maps}
    
    user_history = interactions_df[interactions_df['user_id'] == str(user_id)]
    if user_history.empty:
        return {k: {} for k in preference_maps}
    
    for _, row in user_history.iterrows():
        product_id = str(row.get('product_id'))
        interaction_type = row.get('interaction_type', '').lower()
        weight = INTERACTION_WEIGHTS.get(interaction_type, 1.0)
        product_row = get_product_record(product_id, products_df)
        if product_row is None:
            continue
        
        article = str(product_row.get('articleType', '')).strip()
        usage = str(product_row.get('usage', '')).strip()
        gender = str(product_row.get('gender', '')).strip()
        
        if article:
            preference_maps['articleType'][article] += weight
        if usage:
            preference_maps['usage'][usage] += weight
        if gender:
            preference_maps['gender'][gender] += weight
    
    normalized = {}
    for key, counter in preference_maps.items():
        if not counter:
            normalized[key] = {}
            continue
        max_val = max(counter.values())
        if max_val == 0:
            normalized[key] = {k: 0.0 for k in counter}
        else:
            normalized[key] = {k: v / max_val for k, v in counter.items()}
    
    return normalized


def build_personalized_candidates(
    user_id: str,
    payload_product_id: str,
    hybrid_predictions: Dict,
    products_df: pd.DataFrame,
    users_df: pd.DataFrame,
    interactions_df: pd.DataFrame,
    top_k: int = 10,
    usage_bonus: float = 0.08,
    gender_primary_bonus: float = 0.06,
    gender_secondary_bonus: float = 0.03,
    interaction_weight: float = 0.05,
    usage_pref_weight: float = 0.04
) -> List[Dict]:
    """Compute prioritized personalized recommendations."""
    if (
        hybrid_predictions is None
        or products_df is None
        or payload_product_id is None
    ):
        return []
    
    payload_row = get_product_record(payload_product_id, products_df)
    if payload_row is None:
        return []
    
    payload_article = str(payload_row.get('articleType', '')).strip()
    payload_usage = str(payload_row.get('usage', '')).strip()
    payload_gender = str(payload_row.get('gender', '')).strip()
    payload_gender_lower = payload_gender.lower()
    
    user_record = get_user_record(user_id, users_df)
    user_age = None
    if user_record is not None:
        try:
            user_age = int(user_record.get('age')) if pd.notna(user_record.get('age')) else None
        except (ValueError, TypeError):
            user_age = None
    user_gender = user_record.get('gender') if user_record is not None else None
    
    allowed_genders = get_allowed_genders(user_age, user_gender)
    preference_maps = build_user_interaction_preferences(
        user_id,
        interactions_df,
        products_df
    )
    
    # Robustly fetch user scores regardless of user_id key type (str/int)
    predictions_by_user = hybrid_predictions.get('predictions', {}) or {}
    user_scores = None
    user_key_str = str(user_id)
    if user_key_str in predictions_by_user:
        user_scores = predictions_by_user[user_key_str]
    else:
        for key, val in predictions_by_user.items():
            if str(key) == user_key_str:
                user_scores = val
                break
    
    if not user_scores:
        return []
    
    prioritized = []
    seen_product_ids = set()  # Track để tránh duplicate
    
    for product_id, base_score in sorted(
        user_scores.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        product_id_str = str(product_id)
        
        # Skip payload product
        if product_id_str == str(payload_product_id):
            continue
        
        # Skip nếu đã thêm product này rồi (tránh duplicate)
        if product_id_str in seen_product_ids:
            continue
        
        product_row = get_product_record(product_id, products_df)
        if product_row is None:
            continue
        
        article_type = str(product_row.get('articleType', '')).strip()
        if not article_type or article_type != payload_article:
            continue  # strict articleType requirement
        
        product_usage = str(product_row.get('usage', '')).strip()
        product_gender = str(product_row.get('gender', '')).strip() or 'Unspecified'
        product_gender_lower = product_gender.lower()
        payload_gender_match = False
        payload_unisex_fallback = False

        if payload_gender:
            if product_gender_lower == payload_gender_lower:
                payload_gender_match = True
            elif product_gender_lower == 'unisex':
                payload_gender_match = True
                payload_unisex_fallback = True
            else:
                continue  # Skip products outside payload gender scope
        
        score = float(base_score)
        reasons = []
        
        if payload_usage and product_usage and product_usage == payload_usage:
            score += usage_bonus
            reasons.append("Ưu tiên do cùng usage")
        
        if payload_gender:
            if payload_gender_match and not payload_unisex_fallback:
                score += gender_primary_bonus
                reasons.append("Phù hợp gender với sản phẩm đang xem")
            elif payload_unisex_fallback:
                score += gender_secondary_bonus
                reasons.append("Unisex phù hợp với sản phẩm đang xem")
        else:
            if product_gender in allowed_genders:
                score += gender_primary_bonus
                reasons.append("Phù hợp giới tính/độ tuổi")
            elif product_gender_lower == 'unisex' and (user_age or 0) >= 13:
                score += gender_secondary_bonus
                reasons.append("Unisex phù hợp (>=13)")
            else:
                score -= 0.01
        
        article_pref = preference_maps.get('articleType', {}).get(article_type, 0.0)
        if article_pref > 0:
            score += interaction_weight * article_pref
            reasons.append("Trọng số lịch sử articleType")
        
        usage_pref = preference_maps.get('usage', {}).get(product_usage, 0.0)
        if usage_pref > 0:
            score += usage_pref_weight * usage_pref
            reasons.append("Trọng số lịch sử usage")
        
        prioritized.append({
            'product_id': product_id_str,
            'score': score,
            'base_score': base_score,
            'usage_match': product_usage == payload_usage and bool(payload_usage),
            'gender_match': payload_gender_match if payload_gender else (product_gender in allowed_genders),
            'reasons': reasons,
            'product_row': product_row
        })
        
        seen_product_ids.add(product_id_str)  # Mark as seen
    
    # Sort by score (desc), then by product_id (asc) for deterministic ordering
    prioritized.sort(key=lambda x: (-x['score'], x['product_id']))
    return prioritized[:top_k]


def build_outfit_suggestions(
    user_id: str,
    payload_product_id: str,
    personalized_items: List[Dict],
    products_df: pd.DataFrame,
    hybrid_predictions: Dict,
    user_age: Optional[int],
    user_gender: Optional[str],
    max_outfits: int = 3
) -> List[Dict]:
    """
    Create outfits based on Item-Item complement relationships.
    Uses complement dictionary to find compatible items instead of usage-based filtering.
    """
    if (
        products_df is None
        or personalized_items is None
        or hybrid_predictions is None
    ):
        return []
    
    payload_row = get_product_record(payload_product_id, products_df)
    if payload_row is None:
        return []
    
    # Item-Item complement dictionary
    complement = {
        'Trousers': [['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Formal Shoes'], ['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Casual Shoes'], ['Tshirts', 'Shirts', 'Jackets', 'Sweaters', 'Sweatshirts', 'Sports Shoes']],
        'Tshirts': [['Watches', 'Jeans', 'Casual Shoes'], ['Watches', 'Jeans', 'Flip Flops']],
        'Shirts': [['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Formal Shoes', 'Watches'], ['Shorts', 'Formal Shoes', 'Watches'], ['Trousers', 'Casual Shoes', 'Watches'], ['Jeans', 'Casual Shoes', 'Watches'], ['Shorts', 'Casual Shoes', 'Watches'], ['Trousers', 'Formal Shoes', 'Belts'], ['Jeans', 'Formal Shoes', 'Belts'], ['Shorts', 'Formal Shoes', 'Belts'], ['Trousers', 'Casual Shoes', 'Belts'], ['Jeans', 'Casual Shoes', 'Belts'], ['Shorts', 'Casual Shoes', 'Belts'], ['Trousers', 'Formal Shoes', 'Watches', 'Belts'], ['Jeans', 'Formal Shoes', 'Watches', 'Belts'], ['Trousers', 'Casual Shoes', 'Watches', 'Belts'], ['Jeans', 'Casual Shoes', 'Watches', 'Belts']],
        'Dresses': [['Watches', 'Casual Shoes'], ['Watches', 'Flats'], ['Watches', 'Flip Flops']],
        'Tops': [['Trousers', 'Casual Shoes'], ['Jeans', 'Casual Shoes'], ['Shorts', 'Casual Shoes'], ['Skirts', 'Casual Shoes'], ['Capris', 'Casual Shoes'], ['Trousers', 'Sports Shoes'], ['Jeans', 'Sports Shoes'], ['Shorts', 'Sports Shoes'], ['Skirts', 'Sports Shoes'], ['Capris', 'Sports Shoes']],
        'Shorts': [['Tshirts', 'Sweatshirts', 'Sports Shoes'], ['Tops', 'Sweatshirts', 'Sports Shoes'], ['Tshirts', 'Sweatshirts', 'Casual Shoes'], ['Tops', 'Sweatshirts', 'Casual Shoes'], ['Tshirts', 'Sweatshirts', 'Flip Flops'], ['Tops', 'Sweatshirts', 'Flip Flops'], ['Watches', 'Tshirts', 'Sports Shoes']],
        'Skirts': [['Tshirts', 'Tunics', 'Jackets', 'Heels'], ['Tops', 'Tunics', 'Jackets', 'Heels'], ['Tshirts', 'Tunics', 'Jackets', 'Flats'], ['Tops', 'Tunics', 'Jackets', 'Flats'], ['Tshirts', 'Tunics', 'Jackets', 'Casual Shoes'], ['Tops', 'Tunics', 'Jackets', 'Casual Shoes'], ['Watches', 'Tshirts', 'Casual Shoes'], ['Watches', 'Tshirts', 'Flats'], ['Watches', 'Tshirts', 'Flip Flops']],
        'Jeans': [['Tshirts', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Casual Shoes'], ['Tops', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Casual Shoes'], ['Tshirts', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Sports Shoes'], ['Tops', 'Shirts', 'Sweaters', 'Sweatshirts', 'Jackets', 'Sports Shoes'], ['Watches', 'Tshirts', 'Flip Flops'], ['Watches', 'Shirts', 'Casual Shoes']],
        'Formal Shoes': ['Watches', 'Shirts', 'Trousers'],
        'Casual Shoes': [['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
        'Sports Shoes': [['Tshirts', 'Shorts'], ['Tops', 'Shorts'], ['Tshirts', 'Track Pants'], ['Tops', 'Track Pants'], ['Tshirts', 'Capris'], ['Tops', 'Capris'], ['Watches', 'Tshirts', 'Shorts'], ['Watches', 'Tshirts', 'Track Pants']],
        'Heels': [['Watches', 'Tshirts', 'Skirts'], ['Watches', 'Dresses']],
        'Flats': [['Watches', 'Tshirts', 'Skirts', 'Dresses'], ['Watches', 'Tshirts', 'Shorts', 'Dresses']],
        'Sandals': [['Tshirts', 'Shorts'], ['Tops', 'Shorts'], ['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
        'Flip Flops': [['Watches', 'Tshirts', 'Jeans'], ['Watches', 'Shirts', 'Jeans']],
        'Handbags': [['Tshirts', 'Skirts', 'Casual Shoes'], ['Tshirts', 'Skirts', 'Flats'], ['Tshirts', 'Skirts', 'Flip Flops'], ['Dresses', 'Flip Flops'], ['Dresses', 'Flats']],
        'Jackets': [['Trousers', 'Tshirts', 'Dresses', 'Shirts'], ['Jeans', 'Tshirts', 'Dresses', 'Shirts'], ['Trousers', 'Tops', 'Dresses', 'Shirts'], ['Jeans', 'Tops', 'Dresses', 'Shirts']],
        'Sweaters': [['Trousers', 'Dresses'], ['Jeans', 'Dresses']],
        'Sweatshirts': [['Trousers', 'Casual Shoes', 'Watches'], ['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Sports Shoes', 'Watches'], ['Jeans', 'Casual Shoes', 'Watches'], ['Shorts', 'Sports Shoes', 'Watches'], ['Shorts', 'Flip Flops', 'Watches'], ['Track Pants', 'Sports Shoes', 'Watches'], ['Trousers', 'Casual Shoes', 'Caps'], ['Trousers', 'Formal Shoes', 'Caps'], ['Jeans', 'Sports Shoes', 'Caps'], ['Jeans', 'Casual Shoes', 'Caps'], ['Shorts', 'Sports Shoes', 'Caps'], ['Shorts', 'Flip Flops', 'Caps'], ['Track Pants', 'Sports Shoes', 'Caps'], ['Trousers', 'Casual Shoes', 'Backpacks'], ['Jeans', 'Sports Shoes', 'Backpacks'], ['Shorts', 'Sports Shoes', 'Backpacks'], ['Track Pants', 'Sports Shoes', 'Backpacks'], ['Trousers', 'Casual Shoes', 'Watches', 'Caps'], ['Trousers', 'Formal Shoes', 'Watches'], ['Jeans', 'Sports Shoes', 'Watches', 'Caps'], ['Shorts', 'Sports Shoes', 'Watches', 'Caps'], ['Track Pants', 'Sports Shoes', 'Watches', 'Caps']],
        'Backpacks': [['Tshirts', 'Jeans', 'Flip Flops'], ['Shirts', 'Jeans', 'Casual Shoes']],
        'Belts': [['Tshirts', 'Jeans', 'Flip Flops'], ['Shirts', 'Jeans', 'Casual Shoes']],
        'Capris': [['Caps', 'Jackets', 'Sports Shoes'], ['Caps', 'Tshirts', 'Sports Shoes']],
        'Caps': [['Tshirts', 'Shorts', 'Sports Shoes'], ['Tshirts', 'Track Pants', 'Sports Shoes']]
    }

    target_gender = str(payload_row.get('gender', '')).strip()
    
    allowed_genders_for_user = get_allowed_genders(user_age, user_gender)

    def gender_allowed(gender_value: str) -> bool:
        gender_clean = str(gender_value).strip()
        if not target_gender:
            return True
        if not gender_clean:
            return False
        gender_lower = gender_clean.lower()
        target_lower = target_gender.lower()
        if gender_lower == target_lower:
            return True
        return gender_lower == 'unisex'

    # Map articleType directly to complement keys (using exact articleType from CSV)
    def map_to_complement_key(row) -> Optional[str]:
        """Map product articleType to complement dictionary key (using exact articleType from CSV)."""
        article_type = str(row.get('articleType', '')).strip()
        
        # Direct mapping: use articleType as-is if it exists in complement dictionary
        if article_type in complement:
            return article_type
        
        # Normalize and map common variations
        article_lower = article_type.lower()
        
        # Map variations to standard articleType keys
        if article_lower in ['t-shirt', 't shirt', 'tshirt']:
            return 'Tshirts'
        if article_lower in ['dress']:
            return 'Dresses'
        if article_lower in ['formal shoe', 'formal']:
            return 'Formal Shoes'
        if article_lower in ['casual shoe', 'casual']:
            return 'Casual Shoes'
        if article_lower in ['sports shoe', 'sport shoe']:
            return 'Sports Shoes'
        if article_lower in ['flip flop', 'flipflop']:
            return 'Flip Flops'
        if article_lower in ['sandal']:
            return 'Sandals'
        if article_lower in ['heel']:
            return 'Heels'
        if article_lower in ['flat']:
            return 'Flats'
        if article_lower in ['handbag', 'bag']:
            return 'Handbags'
        if article_lower in ['sweater']:
            return 'Sweaters'
        if article_lower in ['sweatshirt']:
            return 'Sweatshirts'
        if article_lower in ['jacket']:
            return 'Jackets'
        if article_lower in ['short']:
            return 'Shorts'
        if article_lower in ['skirt']:
            return 'Skirts'
        if article_lower in ['jean']:
            return 'Jeans'
        if article_lower in ['trouser', 'pant']:
            return 'Trousers'
        if article_lower in ['shirt']:
            return 'Shirts'
        if article_lower in ['top']:
            return 'Tops'
        if article_lower in ['track pant', 'trackpant']:
            return 'Track Pants'
        if article_lower in ['capri']:
            return 'Capris'
        if article_lower in ['tunic']:
            return 'Tunics'
        if article_lower in ['backpack']:
            return 'Backpacks'
        if article_lower in ['belt']:
            return 'Belts'
        if article_lower in ['cap', 'hat']:
            return 'Caps'
        if article_lower in ['watch', 'watches']:
            return 'Watches'
        if article_lower in ['shoe', 'shoes']:
            return 'Casual Shoes'
        
        return None

    payload_complement_key = map_to_complement_key(payload_row)
    if payload_complement_key is None:
        # Fallback: try to infer from subCategory
        payload_sub = str(payload_row.get('subCategory', '')).strip().lower()
        payload_article = str(payload_row.get('articleType', '')).strip().lower()
        
        if payload_sub == 'bottomwear':
            if 'trouser' in payload_article or 'pant' in payload_article:
                payload_complement_key = 'Trousers'
            elif 'jean' in payload_article:
                payload_complement_key = 'Jeans'
            elif 'short' in payload_article:
                payload_complement_key = 'Shorts'
            elif 'skirt' in payload_article:
                payload_complement_key = 'Skirts'
            else:
                payload_complement_key = 'Trousers'
        elif payload_sub == 'topwear':
            if 'tshirt' in payload_article or 't-shirt' in payload_article:
                payload_complement_key = 'Tshirts'
            elif 'shirt' in payload_article:
                payload_complement_key = 'Shirts'
            elif 'top' in payload_article:
                payload_complement_key = 'Tops'
            elif 'sweater' in payload_article:
                payload_complement_key = 'Sweaters'
            elif 'sweatshirt' in payload_article:
                payload_complement_key = 'Sweatshirts'
            elif 'jacket' in payload_article:
                payload_complement_key = 'Jackets'
            else:
                payload_complement_key = 'Tshirts'
        elif payload_sub == 'dress':
            payload_complement_key = 'Dresses'
        elif payload_sub in ['shoes', 'sandal', 'flip flops']:
            if 'formal' in payload_article:
                payload_complement_key = 'Formal Shoes'
            elif 'casual' in payload_article:
                payload_complement_key = 'Casual Shoes'
            elif 'sport' in payload_article:
                payload_complement_key = 'Sports Shoes'
            elif 'heel' in payload_article:
                payload_complement_key = 'Heels'
            elif 'flat' in payload_article:
                payload_complement_key = 'Flats'
            elif 'sandal' in payload_article:
                payload_complement_key = 'Sandals'
            elif 'flip' in payload_article:
                payload_complement_key = 'Flip Flops'
            else:
                payload_complement_key = 'Casual Shoes'
        elif payload_sub == 'bags':
            if 'backpack' in payload_article:
                payload_complement_key = 'Backpacks'
            else:
                payload_complement_key = 'Handbags'
        elif payload_sub in ['accessories', 'wallets', 'belts']:
            if 'belt' in payload_article:
                payload_complement_key = 'Belts'
            elif 'cap' in payload_article or 'hat' in payload_article:
                payload_complement_key = 'Caps'
            elif 'watch' in payload_article:
                payload_complement_key = 'Watches'
            else:
                payload_complement_key = 'Tshirts'
        else:
            # Default fallback
            payload_complement_key = 'Tshirts'

    # Get compatible item types for payload
    # Handle both old format (flat list) and new format (list of lists)
    complement_value = complement.get(payload_complement_key, [])
    if complement_value and isinstance(complement_value[0], list):
        # New format: list of lists - flatten and get unique types
        compatible_types = list(set([item for sublist in complement_value for item in sublist]))
        complement_rules = complement_value  # Store rules for outfit building
    else:
        # Old format: flat list
        compatible_types = complement_value if complement_value else []
        complement_rules = [compatible_types] if compatible_types else []  # Treat as single rule

    # Filter products by gender compatibility
    gender_filtered = products_df.copy()
    if 'gender' in gender_filtered.columns and target_gender:
        gender_filtered = gender_filtered[gender_filtered['gender'].apply(gender_allowed)]
    if gender_filtered.empty:
        gender_filtered = products_df.copy()

    user_gender_filtered = products_df.copy()
    if 'gender' in user_gender_filtered.columns and allowed_genders_for_user:
        allowed_set = {str(g).strip().lower() for g in allowed_genders_for_user + ["Unisex"]}
        user_gender_filtered = user_gender_filtered[
            user_gender_filtered['gender'].astype(str).str.strip().str.lower().isin(allowed_set)
        ]
    if user_gender_filtered.empty:
        user_gender_filtered = products_df.copy()

    unisex_filtered = products_df.copy()
    if 'gender' in unisex_filtered.columns:
        unisex_filtered = unisex_filtered[
            unisex_filtered['gender'].astype(str).str.strip().str.lower() == 'unisex'
        ]
    if unisex_filtered.empty:
        unisex_filtered = products_df.copy()
    
    score_lookup = {
        item['product_id']: item['score']
        for item in personalized_items
    }
    predictions_by_user = hybrid_predictions.get('predictions', {}) or {}
    user_scores = None
    user_key_str = str(user_id)
    if user_key_str in predictions_by_user:
        user_scores = predictions_by_user[user_key_str]
    else:
        for key, val in predictions_by_user.items():
            if str(key) == user_key_str:
                user_scores = val
                break
    if user_scores is None:
        user_scores = {}
    
    def get_product_score(pid: str) -> float:
        """Robust lookup product score from score_lookup or user_scores."""
        if pid in score_lookup:
            return score_lookup[pid]
        pid_str = str(pid)
        if pid_str in user_scores:
            return user_scores[pid_str]
        try:
            pid_int = int(pid)
            if pid_int in user_scores:
                return user_scores[pid_int]
        except (ValueError, TypeError):
            pass
        for key, val in user_scores.items():
            if str(key) == pid_str:
                return val
        return 0.0

    def is_compatible_with_payload(product_row) -> bool:
        """Check if product is compatible with payload based on complement rules."""
        product_complement_key = map_to_complement_key(product_row)
        if product_complement_key is None:
            return False
        
        # Check if product's complement key is in compatible_types
        return product_complement_key in compatible_types

    def get_products_by_complement_type(complement_type: str, df: pd.DataFrame) -> pd.DataFrame:
        """Get products that match a complement type (using map_to_complement_key logic)."""
        # Use the same mapping logic to find products
        matching_products = []
        
        for idx, row in df.iterrows():
            product_complement_key = map_to_complement_key(row)
            if product_complement_key == complement_type:
                matching_products.append(idx)
        
        if matching_products:
            return df.loc[matching_products]
        
        # Fallback: try direct match
        exact_match = df[df['articleType'].astype(str).str.strip() == complement_type]
        if not exact_match.empty:
            return exact_match
        
        # Fallback: case-insensitive match
        article_lower = complement_type.lower()
        mask = df['articleType'].astype(str).str.lower().str.strip() == article_lower
        
        return df[mask]

    # Build candidate pools for each compatible type
    def build_candidate_pool(complement_type: str, df: pd.DataFrame) -> List[str]:
        """Build sorted candidate list for a complement type."""
        type_df = get_products_by_complement_type(complement_type, df)
        if type_df.empty:
            return []
        
        ids = type_df.index.astype(str)
        scores = [get_product_score(pid) for pid in ids]
        ordered = sorted(zip(ids, scores), key=lambda x: (-x[1], x[0]))
        return [pid for pid, _ in ordered]

    # Build candidate pools with different filtering strategies
    candidates_gender = {}
    candidates_user_gender = {}
    candidates_unisex = {}
    candidates_any = {}

    for comp_type in compatible_types:
        candidates_gender[comp_type] = build_candidate_pool(comp_type, gender_filtered)
        candidates_user_gender[comp_type] = build_candidate_pool(comp_type, user_gender_filtered)
        candidates_unisex[comp_type] = build_candidate_pool(comp_type, unisex_filtered)
        candidates_any[comp_type] = build_candidate_pool(comp_type, products_df)

    # Also include Shoes and Bag as they're common complements
    if 'Shoes' not in compatible_types:
        compatible_types.append('Shoes')
        candidates_gender['Shoes'] = build_candidate_pool('Shoes', gender_filtered)
        candidates_user_gender['Shoes'] = build_candidate_pool('Shoes', user_gender_filtered)
        candidates_unisex['Shoes'] = build_candidate_pool('Shoes', unisex_filtered)
        candidates_any['Shoes'] = build_candidate_pool('Shoes', products_df)

    # Handbags are already included in complement dictionary for Dresses
    # No need for separate handling

    outfits = []
    category_offsets = defaultdict(int)

    def pick_candidate(comp_type: str, used: set) -> Optional[str]:
        """Pick a candidate product for a complement type."""
        is_payload_unisex = str(target_gender).strip().lower() == 'unisex'
        
        if is_payload_unisex:
            pools = [
                ('gender', candidates_gender.get(comp_type, [])),
            ]
        else:
            pools = [
                ('gender', candidates_gender.get(comp_type, [])),
                ('user_gender', candidates_user_gender.get(comp_type, [])),
                ('unisex', candidates_unisex.get(comp_type, [])),
                ('any', candidates_any.get(comp_type, [])),
            ]
        
        for pool_key, pool in pools:
            if not pool:
                continue
            offset_key = f"{comp_type}:{pool_key}"
            start = category_offsets[offset_key]
            for shift in range(len(pool)):
                idx = (start + shift) % len(pool)
                pid = pool[idx]
                if pid in used or pid == str(payload_product_id):
                    continue
                # Verify compatibility
                product_row = get_product_record(pid, products_df)
                if product_row is not None and is_compatible_with_payload(product_row):
                    category_offsets[offset_key] = idx + 1
                    return pid
        return None

    # Build outfits using complement relationships
    for outfit_idx in range(max_outfits):
        used = {str(payload_product_id)}
        ordered_products = [str(payload_product_id)]
        
        # Try to add compatible items
        for comp_type in compatible_types[:4]:  # Limit to top 4 compatible types
            if len(ordered_products) >= 5:  # Limit outfit size
                break
            candidate = pick_candidate(comp_type, used)
            if candidate:
                used.add(candidate)
                ordered_products.append(candidate)

        # Calculate outfit score based on complement compatibility
        base_score = sum(get_product_score(pid) for pid in ordered_products)
        
        # Bonus for complement compatibility
        complement_bonus = 0.0
        for pid in ordered_products[1:]:  # Skip payload
            product_row = get_product_record(pid, products_df)
            if product_row is not None and is_compatible_with_payload(product_row):
                complement_bonus += 0.1
        
        final_score = base_score + complement_bonus
        
        if len(ordered_products) > 1:  # At least payload + 1 item
            outfits.append({
                'products': ordered_products,
                'score': final_score
            })
    
    return outfits


class RecommendHybridView(APIView):
    serializer_class = HybridRecommendationSerializer
    permission_classes = [permissions.AllowAny]
    authentication_classes = []
    
    def post(self, request, *args, **kwargs):
        serializer = self.serializer_class(data=request.data)
        serializer.is_valid(raise_exception=True)
        
        user_id = serializer.validated_data["user_id"]
        current_product_id = serializer.validated_data["current_product_id"]
        alpha = serializer.validated_data.get("alpha", 0.5)
        top_k_personalized = serializer.validated_data.get("top_k_personalized", 6)
        top_k_outfit = serializer.validated_data.get("top_k_outfit", 3)
        
        # Load data
        products_df = load_products_data()
        users_df = load_users_data()
        interactions_df = load_interactions_data()
        
        if products_df is None or users_df is None:
            return Response(
                {
                    "detail": "Không tìm thấy dữ liệu `products.csv` hoặc `users.csv`. Vui lòng chạy bước xuất dữ liệu trước.",
                    "error": "missing_data"
                },
                status=status.HTTP_400_BAD_REQUEST,
            )
        
        # Ensure hybrid predictions
        candidate_pool = max(int(top_k_personalized * 3), 100)
        hybrid_data = ensure_hybrid_predictions(alpha, candidate_pool)
        
        if hybrid_data is None:
            return Response(
                {
                    "detail": "Không tìm thấy dữ liệu hybrid predictions. Vui lòng chạy các bước Training trước.",
                    "error": "missing_predictions"
                },
                status=status.HTTP_409_CONFLICT,
            )
        
        # Get user info
        user_record = get_user_record(user_id, users_df)
        user_age = None
        if user_record is not None and pd.notna(user_record.get('age')):
            try:
                user_age = int(user_record.get('age'))
            except (ValueError, TypeError):
                user_age = None
        user_gender = user_record.get('gender') if user_record is not None else None
        
        # Build personalized candidates
        personalized_items = build_personalized_candidates(
            user_id=user_id,
            payload_product_id=current_product_id,
            hybrid_predictions=hybrid_data,
            products_df=products_df,
            users_df=users_df,
            interactions_df=interactions_df,
            top_k=int(top_k_personalized)
        )
        
        if not personalized_items:
            preds = hybrid_data.get("predictions", {}) or {}
            has_hybrid_for_user = any(str(k) == str(user_id) for k in preds.keys())
            if not has_hybrid_for_user:
                return Response(
                    {
                        "detail": "Không có bất kỳ điểm Hybrid nào cho user này (chưa được train hoặc đã bị lọc ở bước trước). Vui lòng kiểm tra lại dữ liệu train hoặc chọn user khác.",
                        "error": "no_predictions_for_user"
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
            else:
                return Response(
                    {
                        "detail": "Không tìm thấy sản phẩm nào thỏa **articleType = articleType của sản phẩm đầu vào** trong Top candidate Hybrid. Vui lòng thử sản phẩm khác hoặc nới lỏng điều kiện.",
                        "error": "no_matching_products"
                    },
                    status=status.HTTP_404_NOT_FOUND,
                )
        
        personalized_products = []
        for idx, item in enumerate(personalized_items, start=1):
            row = item["product_row"]
            # Convert pandas Series to dict
            if hasattr(row, "to_dict"):
                product_dict = row.to_dict()
            else:
                product_dict = dict(row)

            personalized_products.append(
                {
                    "rank": idx,
                    "product_id": item["product_id"],
                    "name": row.get("productDisplayName", "N/A"),
                    "articleType": row.get("articleType", "N/A"),
                    "usage": row.get("usage", "N/A"),
                    "gender": row.get("gender", "N/A"),
                    "hybrid_score": round(item["base_score"], 4),
                    "priority_score": round(item["score"], 4),
                    "highlights": " • ".join(item["reasons"]) if item["reasons"] else "-",
                    # Tạm thời giữ product_dict, sẽ được thay thế bằng dữ liệu Mongo (nếu có)
                    "product": product_dict,
                }
            )
        
        # Build outfit suggestions
        outfits = build_outfit_suggestions(
            user_id=user_id,
            payload_product_id=current_product_id,
            personalized_items=personalized_items,
            products_df=products_df,
            hybrid_predictions=hybrid_data,
            user_age=user_age,
            user_gender=user_gender,
            max_outfits=int(top_k_outfit)
        )
        
        formatted_outfits = []
        for idx, outfit in enumerate(outfits, start=1):
            outfit_products = []
            for pid in outfit["products"]:
                product_row = get_product_record(pid, products_df)
                if product_row is not None:
                    # Convert pandas Series to dict
                    if hasattr(product_row, "to_dict"):
                        product_dict = product_row.to_dict()
                    else:
                        product_dict = dict(product_row)
                    outfit_products.append(
                        {
                            "product_id": pid,
                            # Tạm dùng product_dict, sẽ được thay bằng dữ liệu Mongo nếu available
                            "product": product_dict,
                        }
                    )
            formatted_outfits.append(
                {
                    "outfit_number": idx,
                    "score": round(outfit["score"], 4),
                    "products": outfit_products,
                }
            )

        # Enrich tất cả product bằng dữ liệu đầy đủ từ MongoDB
        try:
            # Thu thập toàn bộ product_id cần query
            all_product_ids: set[int] = set()
            for p in personalized_products:
                try:
                    all_product_ids.add(int(p["product_id"]))
                except (TypeError, ValueError):
                    continue
            for outfit in formatted_outfits:
                for p in outfit.get("products", []):
                    try:
                        all_product_ids.add(int(p["product_id"]))
                    except (TypeError, ValueError):
                        continue

            mongo_products_map: dict[str, dict] = {}
            if all_product_ids:
                try:
                    mongo_qs = MongoProduct.objects(id__in=list(all_product_ids))
                    serializer = MongoProductSerializer(mongo_qs, many=True)
                    for prod in serializer.data:
                        mongo_products_map[str(prod.get("id"))] = prod
                except Exception:
                    mongo_products_map = {}

            # Thay thế field "product" bằng dữ liệu Mongo nếu tìm thấy
            if mongo_products_map:
                for p in personalized_products:
                    pid_str = str(p.get("product_id"))
                    full_prod = mongo_products_map.get(pid_str)
                    if full_prod:
                        p["product"] = full_prod

                for outfit in formatted_outfits:
                    for p in outfit.get("products", []):
                        pid_str = str(p.get("product_id"))
                        full_prod = mongo_products_map.get(pid_str)
                        if full_prod:
                            p["product"] = full_prod
        except Exception:
            # Nếu có lỗi khi enrich, vẫn trả về dữ liệu gốc từ CSV
            pass
        
        allowed_genders = get_allowed_genders(user_age, user_gender)
        
        response = {
            "personalized_products": personalized_products,
            "outfits": formatted_outfits,
            "metadata": {
                "user_id": user_id,
                "current_product_id": current_product_id,
                "alpha": alpha,
                "top_k_personalized": top_k_personalized,
                "top_k_outfit": top_k_outfit,
                "allowed_genders": allowed_genders,
                "user_age": user_age,
                "user_gender": user_gender
            }
        }
        
        return Response(response, status=status.HTTP_200_OK)


"""
Hybrid recommendation engine with late fusion.
Combines LightGCN (CF) and Sentence-BERT (Content) with weighted sum (default: 0.7 CF + 0.3 Content).
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from bson import ObjectId

from apps.products.mongo_models import Product as MongoProduct
from apps.users.mongo_models import User as MongoUser
from apps.recommendations.utils import (
    filter_by_age_gender,
    get_outfit_categories,
    generate_english_reason,
    map_subcategory_to_tag,
)

# Import the two engines
from apps.recommendations.gnn.engine_lightgcn import get_engine as get_gnn_engine
from apps.recommendations.cbf.engine_sbert_faiss import get_engine as get_cbf_engine

logger = logging.getLogger(__name__)


class HybridRecommendationEngine:
    """
    Hybrid recommendation engine combining GNN (LightGCN) and CBF (Sentence-BERT + FAISS).
    Uses late fusion with weighted sum.
    """
    
    def __init__(self, model_dir: str = "models"):
        """
        Initialize the engine.
        
        Args:
            model_dir: Directory to save/load models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.gnn_engine = get_gnn_engine()
        self.cbf_engine = get_cbf_engine()
        
        self.is_trained = False
    
    def train(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Train both GNN and CBF models.
        
        Args:
            force_retrain: Force retraining even if models exist
            
        Returns:
            Training metrics and info
        """
        logger.info("Training Hybrid model (GNN + CBF)...")
        
        # Train GNN model
        logger.info("Training GNN component...")
        gnn_result = self.gnn_engine.train(force_retrain=force_retrain)
        
        # Train CBF model
        logger.info("Training CBF component...")
        cbf_result = self.cbf_engine.train(force_retrain=force_retrain)
        
        self.is_trained = True
        
        return {
            "status": "success",
            "message": "Hybrid model trained successfully",
            "gnn_result": gnn_result,
            "cbf_result": cbf_result,
            "trained_at": datetime.now().isoformat(),
        }
    
    def recommend(
        self,
        user_id: str,
        current_product_id: str,
        top_k_personal: int = 5,
        top_k_outfit: int = 4,
        alpha: float = 0.7,  # Weight for GNN (CF), 1-alpha for CBF (Content)
    ) -> Dict[str, Any]:
        """
        Generate hybrid recommendations.
        
        Args:
            user_id: User ID (MongoDB ObjectId as string)
            current_product_id: Current product ID
            top_k_personal: Number of personalized recommendations
            top_k_outfit: Number of outfit recommendations per category
            alpha: Weight for GNN component (default 0.7), CBF gets (1-alpha)
            
        Returns:
            Recommendation results with personalized and outfit sections
        """
        if not self.is_trained:
            # Try to load models
            try:
                self.gnn_engine.load_model()
                self.cbf_engine.load_model()
                self.is_trained = True
            except FileNotFoundError:
                raise ValueError("Models not trained. Please train the models first.")
        
        logger.info(f"Generating hybrid recommendations with alpha={alpha}")
        
        # Load user and current product
        user = MongoUser.objects(id=ObjectId(user_id)).first()
        if not user:
            raise ValueError(f"User {user_id} not found")
        
        current_product = MongoProduct.objects(id=ObjectId(current_product_id)).first()
        if not current_product:
            raise ValueError(f"Product {current_product_id} not found")
        
        # Get recommendations from both engines
        # We'll get more candidates and then re-rank
        k_candidates = top_k_personal * 3
        
        try:
            gnn_results = self.gnn_engine.recommend(
                user_id=user_id,
                current_product_id=current_product_id,
                top_k_personal=k_candidates,
                top_k_outfit=top_k_outfit,
            )
        except Exception as e:
            logger.warning(f"GNN recommendation failed: {e}, using CBF only")
            gnn_results = {"personalized": [], "outfit": {}}
        
        try:
            cbf_results = self.cbf_engine.recommend(
                user_id=user_id,
                current_product_id=current_product_id,
                top_k_personal=k_candidates,
                top_k_outfit=top_k_outfit,
            )
        except Exception as e:
            logger.warning(f"CBF recommendation failed: {e}, using GNN only")
            cbf_results = {"personalized": [], "outfit": {}}
        
        # Combine scores using late fusion
        combined_scores: Dict[str, float] = {}
        product_map: Dict[str, MongoProduct] = {}
        
        # Add GNN scores
        for item in gnn_results.get("personalized", []):
            product_id = item["product"]["id"]
            combined_scores[product_id] = alpha * item["score"]
            # Load product if not in map
            if product_id not in product_map:
                product = MongoProduct.objects(id=ObjectId(product_id)).first()
                if product:
                    product_map[product_id] = product
        
        # Add CBF scores
        for item in cbf_results.get("personalized", []):
            product_id = item["product"]["id"]
            if product_id in combined_scores:
                combined_scores[product_id] += (1 - alpha) * item["score"]
            else:
                combined_scores[product_id] = (1 - alpha) * item["score"]
            # Load product if not in map
            if product_id not in product_map:
                product = MongoProduct.objects(id=ObjectId(product_id)).first()
                if product:
                    product_map[product_id] = product
        
        # Sort by combined score
        sorted_products = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Filter by age and gender
        exclude_ids = {current_product_id}
        candidate_products = [
            product_map[product_id]
            for product_id, _ in sorted_products
            if product_id in product_map
        ]
        
        filtered_products = filter_by_age_gender(candidate_products, user, exclude_ids)
        
        # Generate personalized recommendations
        personalized = []
        for product in filtered_products[:top_k_personal]:
            product_id = str(product.id)
            score = combined_scores.get(product_id, 0.0)
            
            # Generate reason mentioning both components
            reason = generate_english_reason(
                product=product,
                user=user,
                reason_type="personalized",
                interaction_history=getattr(user, 'interaction_history', []),
            )
            reason += f" (Hybrid: {alpha:.0%} CF + {1-alpha:.0%} Content)"
            
            personalized.append({
                "product": self._serialize_product(product),
                "score": float(score),
                "reason": reason,
            })
        
        # Generate outfit recommendations
        # Use GNN outfit results as base, enhance with CBF scores
        outfit = {}
        
        current_tag = map_subcategory_to_tag(
            current_product.subCategory,
            current_product.articleType
        )
        
        outfit_categories = get_outfit_categories(current_tag or "tops", user.gender)
        
        # Combine outfit results from both engines
        gnn_outfit = gnn_results.get("outfit", {})
        cbf_outfit = cbf_results.get("outfit", {})
        
        for category in outfit_categories:
            # Get candidates from both engines
            candidates = []
            
            if category in gnn_outfit and gnn_outfit[category]:
                gnn_item = gnn_outfit[category]
                product_id = gnn_item["product"]["id"]
                product = MongoProduct.objects(id=ObjectId(product_id)).first()
                if product:
                    score = alpha * gnn_item["score"]
                    candidates.append((product, score))
            
            if category in cbf_outfit and cbf_outfit[category]:
                cbf_item = cbf_outfit[category]
                product_id = cbf_item["product"]["id"]
                product = MongoProduct.objects(id=ObjectId(product_id)).first()
                if product:
                    # Check if already added
                    existing = next((p for p, _ in candidates if str(p.id) == product_id), None)
                    if existing:
                        # Update score
                        for i, (p, s) in enumerate(candidates):
                            if str(p.id) == product_id:
                                candidates[i] = (p, s + (1 - alpha) * cbf_item["score"])
                    else:
                        score = (1 - alpha) * cbf_item["score"]
                        candidates.append((product, score))
            
            # Pick best candidate
            if candidates:
                candidates.sort(key=lambda x: x[1], reverse=True)
                product, score = candidates[0]
                
                reason = generate_english_reason(
                    product=product,
                    user=user,
                    reason_type="outfit",
                    current_product=current_product,
                )
                reason += f" (Hybrid fusion)"
                
                outfit[category] = {
                    "product": self._serialize_product(product),
                    "score": float(score),
                    "reason": reason,
                }
        
        # Generate overall reasons
        reasons = {
            "personalized": [item["reason"] for item in personalized],
            "outfit": [f"Perfect combination with {current_product.articleType or 'current product'} (Hybrid: {alpha:.0%} GNN + {1-alpha:.0%} CBF)"]
        }
        
        return {
            "personalized": personalized,
            "outfit": outfit,
            "reasons": reasons,
            "fusion_weights": {
                "gnn_cf": alpha,
                "cbf_content": 1 - alpha,
            },
        }
    
    def _serialize_product(self, product: MongoProduct) -> Dict[str, Any]:
        """Serialize product to dictionary."""
        return {
            "id": str(product.id),
            "name": product.productDisplayName or "",
            "gender": product.gender or "",
            "masterCategory": product.masterCategory or "",
            "subCategory": product.subCategory or "",
            "articleType": product.articleType or "",
            "baseColour": product.baseColour or "",
            "season": product.season or "",
            "usage": product.usage or "",
            "images": product.images or [],
        }


# Global engine instance
_engine: Optional[HybridRecommendationEngine] = None


def get_engine() -> HybridRecommendationEngine:
    """Get or create the global engine instance."""
    global _engine
    if _engine is None:
        _engine = HybridRecommendationEngine()
    return _engine


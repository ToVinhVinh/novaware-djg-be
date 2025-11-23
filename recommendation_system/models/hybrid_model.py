"""
Hybrid Model
Kết hợp GNN và Content-Based Filtering
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import time


class HybridRecommender:
    """Hybrid model kết hợp GNN và Content-Based"""
    
    def __init__(
        self,
        gnn_model,
        content_based_model,
        alpha: float = 0.5
    ):
        """
        Args:
            gnn_model: GNN recommender model
            content_based_model: Content-based recommender model
            alpha: Trọng số cho GNN (1-alpha cho Content-Based)
                  alpha = 0.5 nghĩa là cân bằng giữa 2 models
        """
        self.gnn_model = gnn_model
        self.content_based_model = content_based_model
        self.alpha = alpha
        
        self.training_time = 0
        
    def normalize_scores(self, scores: List[Tuple[int, float]]) -> Dict[int, float]:
        """
        Normalize scores về range [0, 1]
        
        Args:
            scores: List of (product_idx, score)
        
        Returns:
            Dictionary {product_idx: normalized_score}
        """
        if len(scores) == 0:
            return {}
        
        score_dict = {idx: score for idx, score in scores}
        scores_array = np.array(list(score_dict.values()))
        
        # Min-max normalization
        min_score = scores_array.min()
        max_score = scores_array.max()
        
        if max_score - min_score > 0:
            normalized = {
                idx: (score - min_score) / (max_score - min_score)
                for idx, score in score_dict.items()
            }
        else:
            normalized = {idx: 0.5 for idx in score_dict.keys()}
        
        return normalized
    
    def combine_scores(
        self,
        gnn_scores: List[Tuple[int, float]],
        cb_scores: List[Tuple[int, float]]
    ) -> List[Tuple[int, float]]:
        """
        Kết hợp scores từ 2 models
        
        Args:
            gnn_scores: Scores từ GNN model
            cb_scores: Scores từ Content-Based model
        
        Returns:
            Combined scores
        """
        # Normalize scores
        gnn_norm = self.normalize_scores(gnn_scores)
        cb_norm = self.normalize_scores(cb_scores)
        
        # Combine scores
        all_products = set(gnn_norm.keys()) | set(cb_norm.keys())
        
        combined = {}
        for prod_idx in all_products:
            gnn_score = gnn_norm.get(prod_idx, 0)
            cb_score = cb_norm.get(prod_idx, 0)
            
            # Weighted combination
            combined[prod_idx] = self.alpha * gnn_score + (1 - self.alpha) * cb_score
        
        # Sort by combined score
        sorted_scores = sorted(
            combined.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_scores
    
    def train(self):
        """
        Train hybrid model
        Note: GNN và Content-Based đã được train riêng
        """
        print("\n" + "="*80)
        print("TRAINING HYBRID MODEL")
        print("="*80)
        
        start_time = time.time()
        
        # Hybrid model không cần train riêng
        # Chỉ cần kết hợp predictions từ 2 models
        
        print(f"[Hybrid] Using alpha = {self.alpha}")
        print(f"  GNN weight: {self.alpha}")
        print(f"  Content-Based weight: {1 - self.alpha}")
        
        self.training_time = (
            self.gnn_model.training_time + 
            self.content_based_model.training_time
        )
        
        print(f"\n[Hybrid] Total training time: {self.training_time:.2f}s")
        print(f"  GNN: {self.gnn_model.training_time:.2f}s")
        print(f"  Content-Based: {self.content_based_model.training_time:.2f}s")
        print("="*80)
        
    def recommend_personalized(
        self,
        user_info: Dict,
        user_idx: int,
        user_history: pd.DataFrame,
        payload_product_idx: int,
        top_k: int = 10
    ) -> Tuple[List[Tuple[int, float]], float]:
        """
        Gợi ý sản phẩm theo tiêu chí Personalized
        Sử dụng Late Fusion thông minh:
        - Lấy candidates từ GNN và CB KHÔNG filter strict sớm
        - Combine với dynamic weight (ưu tiên GNN)
        - Chỉ apply strict filter ở cuối
        
        Args:
            user_info: Thông tin user
            user_idx: User index
            user_history: Lịch sử tương tác
            payload_product_idx: Index của sản phẩm payload
            top_k: Số lượng recommendations
        
        Returns:
            (List of (product_idx, score), inference_time)
        """
        start_time = time.time()
        
        # Lấy thông tin sản phẩm payload để filter sau
        payload_product = self.content_based_model.products_df[
            self.content_based_model.products_df['product_idx'] == payload_product_idx
        ].iloc[0]
        
        target_article_type = payload_product['articleType']
        gender_filter = [user_info['product_gender'], 'Unisex']
        
        # ========== BƯỚC 1: Lấy candidates từ GNN (KHÔNG filter strict) ==========
        # Sử dụng get_user_recommendations trực tiếp để tránh filter sớm
        gnn_candidates = self.gnn_model.get_user_recommendations(
            user_idx=user_idx,
            top_k=top_k * 5,  # Lấy nhiều candidates để có đủ sau khi filter
            exclude_interacted=True
        )
        
        # ========== BƯỚC 2: Lấy candidates từ Content-Based (KHÔNG filter strict sớm) ==========
        # Sử dụng get_similar_products với non-strict filters
        non_strict_filters = {
            'baseColour': payload_product['baseColour'],  # Boost only
            'usage': payload_product['usage']             # Boost only
        }
        
        cb_candidates = self.content_based_model.get_similar_products(
            product_idx=payload_product_idx,
            top_k=top_k * 5,  # Lấy nhiều candidates
            filters=non_strict_filters  # Chỉ boost, không filter strict
        )
        
        # ========== BƯỚC 3: Late Fusion với Dynamic Weight ==========
        # Dynamic weight: Ưu tiên GNN cao hơn vì nó tốt hơn CB
        # GNN weight = 0.8, CB weight = 0.2 (có thể điều chỉnh)
        gnn_weight = 0.8
        cb_weight = 0.2
        
        # Normalize scores từng model
        gnn_norm = self.normalize_scores(gnn_candidates)
        cb_norm = self.normalize_scores(cb_candidates)
        
        # Combine tất cả candidates
        all_products = set(gnn_norm.keys()) | set(cb_norm.keys())
        
        combined_scores = {}
        for prod_idx in all_products:
            gnn_score = gnn_norm.get(prod_idx, 0)
            cb_score = cb_norm.get(prod_idx, 0)
            
            # Weighted combination với dynamic weight
            combined_scores[prod_idx] = gnn_weight * gnn_score + cb_weight * cb_score
        
        # ========== BƯỚC 4: Apply strict filters (CHỈ Ở CUỐI) ==========
        # Tạo mapping product_idx -> product info chỉ cho candidates (tối ưu performance)
        products_df = self.content_based_model.products_df
        product_info_map = {}
        # Chỉ tạo map cho các products có trong combined_scores
        candidate_indices = set(combined_scores.keys())
        for _, row in products_df.iterrows():
            prod_idx = int(row['product_idx'])
            if prod_idx in candidate_indices:
                product_info_map[prod_idx] = {
                    'gender': row['gender'],
                    'articleType': row['articleType']
                }
        
        filtered_results = []
        filtered_set = set()  # Để check nhanh hơn
        
        # Sort combined scores một lần
        sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        for prod_idx, score in sorted_combined:
            # Lấy thông tin sản phẩm từ map
            if prod_idx not in product_info_map:
                continue
            
            product_info = product_info_map[prod_idx]
            
            # Strict filter 1: Gender
            if product_info['gender'] not in gender_filter:
                continue
            
            # Strict filter 2: ArticleType
            if product_info['articleType'] != target_article_type:
                continue
            
            filtered_results.append((prod_idx, score))
            filtered_set.add(prod_idx)
            
            # Đủ top_k thì dừng
            if len(filtered_results) >= top_k:
                break
        
        # ========== BƯỚC 5: Fallback nếu không đủ candidates ==========
        if len(filtered_results) < top_k:
            # Fallback: Chỉ filter gender, không filter articleType
            for prod_idx, score in sorted_combined:
                if prod_idx in filtered_set:
                    continue
                
                if prod_idx not in product_info_map:
                    continue
                
                product_info = product_info_map[prod_idx]
                
                if product_info['gender'] in gender_filter:
                    filtered_results.append((prod_idx, score))
                    filtered_set.add(prod_idx)
                    if len(filtered_results) >= top_k:
                        break
        
        # Nếu vẫn không đủ, lấy top từ combined (không filter)
        if len(filtered_results) < top_k:
            for prod_idx, score in sorted_combined:
                if prod_idx not in filtered_set:
                    filtered_results.append((prod_idx, score))
                    filtered_set.add(prod_idx)
                    if len(filtered_results) >= top_k:
                        break
        
        inference_time = time.time() - start_time
        
        return filtered_results[:top_k], inference_time
    
    def recommend_outfit(
        self,
        user_info: Dict,
        payload_product_idx: int,
        user_history: pd.DataFrame = None
    ) -> Tuple[Dict[str, List[Tuple[int, float]]], float]:
        """
        Gợi ý outfit hoàn chỉnh
        
        Args:
            user_info: Thông tin user
            payload_product_idx: Index của sản phẩm payload
            user_history: Lịch sử tương tác
        
        Returns:
            (Dictionary chứa các items trong outfit, inference_time)
        """
        # Sử dụng Content-Based cho outfit recommendation
        # vì nó có logic phù hợp hơn cho việc match categories
        return self.content_based_model.recommend_outfit(
            user_info=user_info,
            payload_product_idx=payload_product_idx,
            user_history=user_history
        )
    
    def get_user_recommendations(
        self,
        user_idx: int,
        top_k: int = 20,
        exclude_interacted: bool = True,
        user_history: pd.DataFrame = None
    ) -> List[Tuple[int, float]]:
        """
        Gợi ý sản phẩm cho user (cho mục đích evaluation)
        
        Args:
            user_idx: Index của user
            top_k: Số lượng gợi ý
            exclude_interacted: Có loại bỏ sản phẩm đã tương tác không
            user_history: Lịch sử tương tác của user (cho Content-Based)
            
        Returns:
            List of (product_idx, score)
        """
        # Get recommendations từ GNN
        gnn_recs = []
        if hasattr(self.gnn_model, 'get_user_recommendations'):
            gnn_recs = self.gnn_model.get_user_recommendations(
                user_idx=user_idx,
                top_k=top_k * 2,
                exclude_interacted=exclude_interacted
            )
            
        # Get recommendations từ Content-Based
        cb_recs = []
        if hasattr(self.content_based_model, 'get_user_recommendations'):
            if user_history is not None:
                cb_recs = self.content_based_model.get_user_recommendations(
                    user_idx=user_idx,
                    user_history=user_history,
                    top_k=top_k * 2,
                    exclude_interacted=exclude_interacted
                )
        
        # Combine scores
        combined_scores = self.combine_scores(gnn_recs, cb_recs)
        
        return combined_scores[:top_k]
    
    def get_model_info(self) -> Dict:
        """Lấy thông tin về model"""
        return {
            'model_name': 'Hybrid (GNN + Content-Based)',
            'alpha': self.alpha,
            'gnn_weight': self.alpha,
            'cb_weight': 1 - self.alpha,
            'training_time': self.training_time,
            'gnn_info': self.gnn_model.get_model_info(),
            'cb_info': self.content_based_model.get_model_info()
        }


if __name__ == "__main__":
    print("Testing Hybrid Recommender...")

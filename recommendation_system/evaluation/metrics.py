"""
Evaluation Metrics Module
Tính toán các metrics để đánh giá models
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import time


class RecommendationEvaluator:
    """Class để evaluate recommendation models"""
    
    def __init__(self, test_interactions: pd.DataFrame, products_df: pd.DataFrame, train_interactions: pd.DataFrame = None):
        """
        Args:
            test_interactions: DataFrame chứa test interactions
            products_df: DataFrame chứa thông tin products
            train_interactions: DataFrame chứa train interactions (cần cho Content-Based)
        """
        self.test_interactions = test_interactions
        self.products_df = products_df
        self.train_interactions = train_interactions
        
        self.ground_truth = self._build_ground_truth()
        
    def _build_ground_truth(self) -> Dict[int, Set[int]]:
        """Xây dựng ground truth từ test interactions"""
        ground_truth = defaultdict(set)
        
        interaction_counts = self.test_interactions['interaction_type'].value_counts()
        print("\nTest Interaction Types:")
        print(interaction_counts)
        
        if self.test_interactions['user_idx'].isna().any():
            print("WARNING: Found NaN values in user_idx!")
        if self.test_interactions['product_idx'].isna().any():
            print("WARNING: Found NaN values in product_idx!")
        
        for _, row in self.test_interactions.iterrows():
            if pd.isna(row['user_idx']) or pd.isna(row['product_idx']):
                continue
                
            user_idx = int(row['user_idx'])
            product_idx = int(row['product_idx'])
            
            if row['interaction_type'] in ['purchase', 'cart', 'like']:
                ground_truth[user_idx].add(product_idx)
        
        print(f"\nGround Truth Statistics:")
        print(f"  Users with relevant interactions: {len(ground_truth)}")
        total_items = sum(len(items) for items in ground_truth.values())
        print(f"  Total relevant items: {total_items}")
        if len(ground_truth) > 0:
            print(f"  Avg items per user: {total_items / len(ground_truth):.2f}")
            # Show sample
            sample_user = list(ground_truth.keys())[0]
            sample_items = list(ground_truth[sample_user])[:5]
            print(f"  Sample user {sample_user} has items: {sample_items}")
        else:
            print("  WARNING: Ground truth is empty! Check test_interactions data.")
        
        return dict(ground_truth)
    
    def recall_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính Recall@K
        
        Recall@K = (Số sản phẩm relevant được recommend) / (Tổng số sản phẩm relevant)
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            Recall@K score
        """
        recalls = []
        
        for user_idx, rec_products in recommendations.items():
            if user_idx not in self.ground_truth:
                continue
            
            # Lấy top K
            top_k = rec_products[:k] if rec_products else []
            
            # Ground truth
            relevant = self.ground_truth[user_idx]
            
            if len(relevant) == 0:
                continue
            
            # Ensure types are consistent (convert to int)
            try:
                top_k_set = {int(x) for x in top_k if x is not None and not pd.isna(x)}
                relevant_set = {int(x) for x in relevant if x is not None and not pd.isna(x)}
            except (ValueError, TypeError) as e:
                print(f"Warning: Type conversion error for user {user_idx}: {e}")
                print(f"  top_k types: {[type(x).__name__ for x in top_k[:3]]}")
                print(f"  relevant types: {[type(x).__name__ for x in list(relevant)[:3]]}")
                continue
            
            # Tính recall
            hits = len(top_k_set & relevant_set)
            recall = hits / len(relevant_set) if len(relevant_set) > 0 else 0.0
            
            recalls.append(recall)
        
        return np.mean(recalls) if recalls else 0.0
    
    def precision_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính Precision@K
        
        Precision@K = (Số sản phẩm relevant được recommend) / K
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            Precision@K score
        """
        precisions = []
        
        for user_idx, rec_products in recommendations.items():
            if user_idx not in self.ground_truth:
                continue
            
            # Lấy top K
            top_k = rec_products[:k] if rec_products else []
            
            # Ground truth
            relevant = self.ground_truth[user_idx]
            
            if len(relevant) == 0:
                continue
            
            # Ensure types are consistent
            try:
                top_k_set = {int(x) for x in top_k if x is not None and not pd.isna(x)}
                relevant_set = {int(x) for x in relevant if x is not None and not pd.isna(x)}
            except (ValueError, TypeError) as e:
                continue
            
            # Tính precision
            hits = len(top_k_set & relevant_set)
            precision = hits / k if k > 0 else 0.0
            
            precisions.append(precision)
        
        return np.mean(precisions) if precisions else 0.0
    
    def ndcg_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính NDCG@K (Normalized Discounted Cumulative Gain)
        
        NDCG@K đo ranking quality, càng cao càng tốt
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            NDCG@K score
        """
        ndcgs = []
        
        for user_idx, rec_products in recommendations.items():
            if user_idx not in self.ground_truth:
                continue
            
            # Lấy top K
            top_k = rec_products[:k] if rec_products else []
            
            # Ground truth
            relevant = self.ground_truth[user_idx]
            
            if len(relevant) == 0:
                continue
            
            # Ensure types are consistent
            try:
                relevant_set = {int(x) for x in relevant if x is not None and not pd.isna(x)}
                top_k_ints = [int(x) for x in top_k if x is not None and not pd.isna(x)]
            except (ValueError, TypeError) as e:
                continue
            
            # DCG (Discounted Cumulative Gain)
            dcg = 0
            for i, prod_idx in enumerate(top_k_ints):
                if prod_idx in relevant_set:
                    # rel = 1 nếu relevant, 0 nếu không
                    dcg += 1 / np.log2(i + 2)  # i+2 vì index từ 0
            
            # IDCG (Ideal DCG)
            idcg = 0
            for i in range(min(len(relevant), k)):
                idcg += 1 / np.log2(i + 2)
            
            # NDCG
            ndcg = dcg / idcg if idcg > 0 else 0
            ndcgs.append(ndcg)
        
        return np.mean(ndcgs) if ndcgs else 0.0
    
    def hit_rate_at_k(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính Hit Rate@K
        
        Hit Rate@K = (Số users có ít nhất 1 hit) / (Tổng số users)
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            Hit Rate@K score
        """
        hits = 0
        total = 0
        
        for user_idx, rec_products in recommendations.items():
            if user_idx not in self.ground_truth:
                continue
            
            # Lấy top K
            top_k = rec_products[:k] if rec_products else []
            
            # Ground truth
            relevant = self.ground_truth[user_idx]
            
            if len(relevant) == 0:
                continue
            
            # Ensure types are consistent
            try:
                top_k_set = {int(x) for x in top_k if x is not None and not pd.isna(x)}
                relevant_set = {int(x) for x in relevant if x is not None and not pd.isna(x)}
            except (ValueError, TypeError) as e:
                continue
            
            # Check hit
            if len(top_k_set & relevant_set) > 0:
                hits += 1
            
            total += 1
        
        return hits / total if total > 0 else 0.0
    
    def mrr(self, recommendations: Dict[int, List[int]]) -> float:
        """
        Tính MRR (Mean Reciprocal Rank)
        
        MRR = Average(1 / rank_of_first_relevant_item)
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
        
        Returns:
            MRR score
        """
        reciprocal_ranks = []
        
        for user_idx, rec_products in recommendations.items():
            if user_idx not in self.ground_truth:
                continue
            
            # Ground truth
            relevant = self.ground_truth[user_idx]
            
            if len(relevant) == 0:
                continue
            
            # Ensure types are consistent
            try:
                relevant_set = {int(x) for x in relevant if x is not None and not pd.isna(x)}
                rec_products_ints = [int(x) for x in rec_products if x is not None and not pd.isna(x)]
            except (ValueError, TypeError) as e:
                continue
            
            # Tìm rank của first relevant item
            found = False
            for i, prod_idx in enumerate(rec_products_ints):
                if prod_idx in relevant_set:
                    reciprocal_ranks.append(1 / (i + 1))
                    found = True
                    break
            
            if not found:
                # Không tìm thấy relevant item
                reciprocal_ranks.append(0)
        
        return np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0
    
    def coverage(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính Coverage
        
        Coverage = (Số sản phẩm unique được recommend) / (Tổng số sản phẩm)
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            Coverage score
        """
        recommended_products = set()
        
        for rec_products in recommendations.values():
            recommended_products.update(rec_products[:k])
        
        total_products = len(self.products_df)
        
        return len(recommended_products) / total_products
    
    def diversity(
        self,
        recommendations: Dict[int, List[int]],
        k: int = 10
    ) -> float:
        """
        Tính Diversity
        
        Diversity = Average pairwise distance giữa các sản phẩm được recommend
        
        Args:
            recommendations: Dictionary {user_idx: [product_idx_1, product_idx_2, ...]}
            k: Top K
        
        Returns:
            Diversity score
        """
        diversities = []
        
        for rec_products in recommendations.values():
            top_k = rec_products[:k]
            
            if len(top_k) < 2:
                continue
            
            # Lấy categories của các sản phẩm
            categories = []
            for prod_idx in top_k:
                product = self.products_df[
                    self.products_df['product_idx'] == prod_idx
                ]
                if len(product) > 0:
                    cat = product.iloc[0]['articleType']
                    categories.append(cat)
            
            # Tính diversity = số lượng unique categories / k
            diversity = len(set(categories)) / len(categories) if categories else 0
            diversities.append(diversity)
        
        return np.mean(diversities) if diversities else 0.0
    
    def evaluate_model(
        self,
        model,
        model_name: str,
        users_df: pd.DataFrame,
        k_values: List[int] = [10, 20]
    ) -> Dict:
        """
        Evaluate model với tất cả metrics
        
        Args:
            model: Model cần evaluate
            model_name: Tên model
            users_df: DataFrame chứa thông tin users
            k_values: List các giá trị K để evaluate
        
        Returns:
            Dictionary chứa tất cả metrics
        """
        print(f"\n{'='*80}")
        print(f"EVALUATING {model_name}")
        print(f"{'='*80}")
        
        # Generate recommendations cho tất cả users
        print(f"\n[{model_name}] Generating recommendations...")
        
        recommendations = {}
        inference_times = []
        
        for user_idx in self.ground_truth.keys():
            # Lấy user info
            user = users_df[users_df['user_idx'] == user_idx]
            if len(user) == 0:
                continue
            
            user_info = user.iloc[0].to_dict()
            
            try:
                # Generate recommendations
                start_time = time.time()
                
                if hasattr(model, 'get_user_recommendations'):
                    # Chuẩn bị arguments
                    kwargs = {
                        'user_idx': user_idx,
                        'top_k': max(k_values),
                        'exclude_interacted': True
                    }
                    
                    # Nếu model cần user_history (như Content-Based)
                    # Kiểm tra signature hoặc cứ truyền vào nếu có train_interactions
                    if self.train_interactions is not None:
                        user_history = self.train_interactions[
                            self.train_interactions['user_idx'] == user_idx
                        ]
                        kwargs['user_history'] = user_history
                    
                    # Gọi method với kwargs
                    try:
                        recs = model.get_user_recommendations(**kwargs)
                    except TypeError:
                        # Nếu model không nhận user_history (như GNN cũ), bỏ nó ra
                        if 'user_history' in kwargs:
                            del kwargs['user_history']
                        recs = model.get_user_recommendations(**kwargs)
                        
                    # Extract product indices from recommendations
                    if recs is None or len(recs) == 0:
                        # Debug: Check why recommendations are empty
                        if self.train_interactions is not None:
                            user_train_count = len(self.train_interactions[
                                self.train_interactions['user_idx'] == user_idx
                            ])
                            if user_train_count == 0:
                                print(f"Warning: User {user_idx} has no training interactions (cold start)")
                            else:
                                print(f"Warning: Empty recommendations for user {user_idx} (has {user_train_count} train interactions)")
                        else:
                            print(f"Warning: Empty recommendations for user {user_idx}")
                        rec_products = []
                    else:
                        # Handle both tuple format (prod_idx, score) and list format
                        rec_products = []
                        for item in recs:
                            try:
                                if isinstance(item, tuple):
                                    prod_idx = item[0]
                                elif isinstance(item, (int, np.integer)):
                                    prod_idx = item
                                elif isinstance(item, dict) and 'product_idx' in item:
                                    prod_idx = item['product_idx']
                                else:
                                    continue
                                # Ensure product_idx is int and valid
                                prod_idx_int = int(prod_idx)
                                if prod_idx_int >= 0:  # Valid index
                                    rec_products.append(prod_idx_int)
                            except (ValueError, TypeError, IndexError) as e:
                                print(f"Warning: Error extracting product_idx from item {item}: {e}")
                                continue
                else:
                    # Không có method này, skip
                    continue
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                recommendations[user_idx] = rec_products
                
            except Exception as e:
                print(f"Error generating recommendations for user {user_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"Generated recommendations for {len(recommendations)} users")
        
        # Debug: Check recommendation statistics
        total_recs = sum(len(recs) for recs in recommendations.values())
        avg_recs = total_recs / len(recommendations) if recommendations else 0
        empty_recs = sum(1 for recs in recommendations.values() if len(recs) == 0)
        
        print(f"\n[{model_name}] Recommendation Statistics:")
        print(f"  Total users with recommendations: {len(recommendations)}")
        print(f"  Total recommendations: {total_recs}")
        print(f"  Average recommendations per user: {avg_recs:.2f}")
        print(f"  Users with empty recommendations: {empty_recs}")
        
        # Debug: Check overlap with ground truth
        if recommendations:
            # Check multiple users for better debugging
            sample_users = list(recommendations.keys())[:3]
            for sample_user in sample_users:
                sample_recs = recommendations[sample_user][:10] if recommendations[sample_user] else []
                sample_gt = list(self.ground_truth.get(sample_user, []))[:10]
                overlap = set(sample_recs) & set(sample_gt)
                print(f"\n[{model_name}] Sample (user {sample_user}):")
                print(f"  Recommendations ({len(sample_recs)}): {sample_recs}")
                print(f"  Ground truth ({len(sample_gt)}): {sample_gt}")
                print(f"  Overlap ({len(overlap)}): {overlap}")
                
                # Check type consistency
                if sample_recs and sample_gt:
                    rec_types = [type(x).__name__ for x in sample_recs[:3]]
                    gt_types = [type(x).__name__ for x in sample_gt[:3]]
                    print(f"  Types - Recs: {rec_types}, GT: {gt_types}")
        
        # Additional validation: Check if recommendations are empty for all users
        if not recommendations or all(len(recs) == 0 for recs in recommendations.values()):
            print(f"\n⚠️  WARNING: All recommendations are empty for {model_name}!")
            print(f"   This will result in all metrics being 0.")
            print(f"   Ground truth users: {len(self.ground_truth)}")
            print(f"   Users with recommendations: {len(recommendations)}")
            
            # Debug: Check if users in ground truth have training data
            if self.train_interactions is not None:
                users_with_train = set(self.train_interactions['user_idx'].unique())
                users_in_gt = set(self.ground_truth.keys())
                users_without_train = users_in_gt - users_with_train
                print(f"   Users in ground truth: {len(users_in_gt)}")
                print(f"   Users with training data: {len(users_with_train)}")
                print(f"   Users in GT without training: {len(users_without_train)}")
                if len(users_without_train) > 0:
                    print(f"   Sample users without training: {list(users_without_train)[:5]}")
        
        # Tính metrics
        results = {
            'model_name': model_name,
            'training_time': model.training_time if hasattr(model, 'training_time') else 0,
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
        }
        
        for k in k_values:
            print(f"\n[{model_name}] Computing metrics @{k}...")
            
            results[f'recall@{k}'] = self.recall_at_k(recommendations, k)
            results[f'precision@{k}'] = self.precision_at_k(recommendations, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(recommendations, k)
            results[f'hit_rate@{k}'] = self.hit_rate_at_k(recommendations, k)
            results[f'coverage@{k}'] = self.coverage(recommendations, k)
            results[f'diversity@{k}'] = self.diversity(recommendations, k)
        
        results['mrr'] = self.mrr(recommendations)
        
        # Print results
        print(f"\n{'='*80}")
        print(f"RESULTS FOR {model_name}")
        print(f"{'='*80}")
        
        for metric, value in results.items():
            if metric != 'model_name':
                print(f"{metric}: {value:.4f}")
        
        print(f"{'='*80}\n")
        
        # Tự động kiểm tra metrics có giá trị = 0
        self._check_zero_metrics(results, model_name)
        
        return results
    
    def _check_zero_metrics(self, results: Dict, model_name: str):
        """
        Kiểm tra và cảnh báo nếu có metrics = 0
        """
        metric_columns = [k for k in results.keys() 
                         if k not in ['model_name', 'training_time', 'avg_inference_time']]
        
        zero_metrics = []
        for col in metric_columns:
            value = results.get(col, 0)
            if value == 0.0 or (isinstance(value, (int, float)) and value == 0):
                zero_metrics.append(col)
        
        if zero_metrics:
            print(f"\n⚠️  CẢNH BÁO: {model_name} có {len(zero_metrics)} metrics = 0:")
            for metric in zero_metrics:
                print(f"   - {metric}")
        else:
            print(f"\n✅ {model_name}: Tất cả metrics đều khác 0!")


if __name__ == "__main__":
    print("Testing Recommendation Evaluator...")

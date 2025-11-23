"""
Content-Based Filtering Model
Gợi ý sản phẩm dựa trên content similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
import time


class ContentBasedRecommender:
    """Content-Based Filtering Model"""
    
    def __init__(self, products_df: pd.DataFrame):
        """
        Args:
            products_df: DataFrame chứa thông tin sản phẩm
        """
        self.products_df = products_df.copy()
        self.product_features = None
        self.similarity_matrix = None
        self.tfidf_vectorizer = None
        
        # Training metrics
        self.training_time = 0
        
    def create_product_features(self):
        """Tạo feature vector cho mỗi sản phẩm"""
        print("\n[Content-Based] Creating product features...")
        
        # Kết hợp các features thành text
        feature_cols = ['gender', 'masterCategory', 'subCategory', 
                       'articleType', 'baseColour', 'usage']
        
        def create_feature_text(row):
            """Tạo text từ các features"""
            features = []
            for col in feature_cols:
                if pd.notna(row[col]):
                    # Lặp lại feature quan trọng để tăng trọng số
                    if col == 'articleType':
                        features.extend([str(row[col])] * 3)  # articleType quan trọng nhất
                    elif col == 'subCategory':
                        features.extend([str(row[col])] * 2)
                    else:
                        features.append(str(row[col]))
            return ' '.join(features)
        
        self.products_df['feature_text'] = self.products_df.apply(
            create_feature_text, axis=1
        )
        
        print(f"Created feature text for {len(self.products_df)} products")
        
    def build_similarity_matrix(self):
        """Xây dựng ma trận similarity giữa các sản phẩm"""
        print("\n[Content-Based] Building similarity matrix...")
        
        # TF-IDF vectorization
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            self.products_df['feature_text']
        )
        
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Compute cosine similarity
        self.similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
        print(f"Average similarity: {self.similarity_matrix.mean():.4f}")
        
    def train(self):
        """Train model"""
        print("\n" + "="*80)
        print("TRAINING CONTENT-BASED FILTERING MODEL")
        print("="*80)
        
        start_time = time.time()
        
        # Bước 1: Tạo features
        self.create_product_features()
        
        # Bước 2: Build similarity matrix
        self.build_similarity_matrix()
        
        self.training_time = time.time() - start_time
        
        print(f"\n[Content-Based] Training completed in {self.training_time:.2f}s")
        print("="*80)
        
    def get_similar_products(
        self, 
        product_idx: int, 
        top_k: int = 20,
        filters: Dict = None
    ) -> List[Tuple[int, float]]:
        """
        Tìm sản phẩm tương tự
        
        Args:
            product_idx: Index của sản phẩm
            top_k: Số lượng sản phẩm tương tự
            filters: Dictionary chứa các điều kiện lọc
                - gender: List giới tính phù hợp
                - articleType: articleType phải giống (strict)
                - baseColour: Màu sắc ưu tiên (non-strict)
                - usage: Mục đích sử dụng ưu tiên (non-strict)
        
        Returns:
            List of (product_idx, similarity_score)
        """
        if product_idx >= len(self.similarity_matrix):
            return []
        
        # Lấy similarity scores
        similarities = self.similarity_matrix[product_idx]
        
        # Tạo DataFrame để filter
        candidates = pd.DataFrame({
            'product_idx': range(len(similarities)),
            'similarity': similarities
        })
        
        # Merge với product info
        candidates = candidates.merge(
            self.products_df[['product_idx', 'gender', 'articleType', 
                            'baseColour', 'usage']],
            on='product_idx',
            how='left'
        )
        
        # Loại bỏ chính sản phẩm đó
        candidates = candidates[candidates['product_idx'] != product_idx]
        
        # Apply filters
        if filters:
            # Filter 1: Gender (nếu có)
            if 'gender' in filters and filters['gender']:
                gender_list = filters['gender'] if isinstance(filters['gender'], list) else [filters['gender']]
                # Thêm Unisex vào danh sách
                if 'Unisex' not in gender_list:
                    gender_list.append('Unisex')
                candidates = candidates[candidates['gender'].isin(gender_list)]
            
            # Filter 2: ArticleType (STRICT - phải giống)
            if 'articleType' in filters and filters['articleType']:
                source_article_type = filters['articleType']
                candidates = candidates[candidates['articleType'] == source_article_type]
            
            # Filter 3: BaseColour (NON-STRICT - boost score nếu giống)
            if 'baseColour' in filters and filters['baseColour']:
                source_colour = filters['baseColour']
                # Tăng similarity score nếu màu giống
                candidates.loc[candidates['baseColour'] == source_colour, 'similarity'] *= 1.2
            
            # Filter 4: Usage (NON-STRICT - boost score nếu giống)
            if 'usage' in filters and filters['usage']:
                source_usage = filters['usage']
                # Tăng similarity score nếu usage giống
                candidates.loc[candidates['usage'] == source_usage, 'similarity'] *= 1.1
        
        # Sort by similarity
        candidates = candidates.sort_values('similarity', ascending=False)
        
        # Return top K
        results = list(zip(
            candidates['product_idx'].values[:top_k],
            candidates['similarity'].values[:top_k]
        ))
        
        return results
    
    def recommend_personalized(
        self,
        user_info: Dict,
        user_history: pd.DataFrame,
        payload_product_idx: int,
        top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Gợi ý sản phẩm theo tiêu chí Personalized
        
        Args:
            user_info: Thông tin user (age, gender, product_gender)
            user_history: Lịch sử tương tác của user
            payload_product_idx: Index của sản phẩm payload
            top_k: Số lượng sản phẩm gợi ý
        
        Returns:
            List of (product_idx, score)
        """
        start_time = time.time()
        
        # Lấy thông tin sản phẩm payload
        payload_product = self.products_df[
            self.products_df['product_idx'] == payload_product_idx
        ].iloc[0]
        
        # BƯỚC 1: Candidate Generation - KHÔNG áp dụng strict filters
        # Chỉ áp dụng non-strict filters (boosts) để có nhiều candidates
        non_strict_filters = {
            'baseColour': payload_product['baseColour'],    # NON-STRICT (boost only)
            'usage': payload_product['usage']               # NON-STRICT (boost only)
        }
        
        # Lấy pool lớn candidates (top 50) mà không filter strict
        similar_products = self.get_similar_products(
            payload_product_idx,
            top_k=50,  # Lấy top 50 để có đủ candidates
            filters=non_strict_filters  # Chỉ boost, không filter strict
        )
        
        # BƯỚC 2: Apply strict filters trên top 50 candidates
        gender_filter = [user_info['product_gender'], 'Unisex']
        target_article_type = payload_product['articleType']
        
        filtered_products = []
        for prod_idx, score in similar_products:
            product = self.products_df[
                self.products_df['product_idx'] == prod_idx
            ].iloc[0]
            
            # Apply strict filters
            # Gender filter
            if product['gender'] not in gender_filter:
                continue
            
            # ArticleType filter (strict)
            if product['articleType'] != target_article_type:
                continue
            
            filtered_products.append((prod_idx, score))
        
        # Nếu sau strict filter không còn candidates, fallback về candidates không filter strict
        if len(filtered_products) == 0:
            # Fallback: chỉ filter gender, không filter articleType
            for prod_idx, score in similar_products:
                product = self.products_df[
                    self.products_df['product_idx'] == prod_idx
                ].iloc[0]
                
                if product['gender'] in gender_filter:
                    filtered_products.append((prod_idx, score))
        
        # Nếu vẫn không có, lấy tất cả candidates (trường hợp user mới, không có lịch sử)
        if len(filtered_products) == 0:
            filtered_products = similar_products[:top_k * 2]
        
        # BƯỚC 3: Boost score dựa trên interaction history
        if len(user_history) > 0:
            # Lấy các sản phẩm user đã tương tác
            interacted_products = set(user_history['product_idx'].values)
            
            # Lấy baseColour và usage từ history
            history_colours = user_history.merge(
                self.products_df[['product_idx', 'baseColour', 'usage']],
                on='product_idx',
                how='left'
            )
            
            # Đếm màu và usage phổ biến
            colour_counts = history_colours['baseColour'].value_counts()
            usage_counts = history_colours['usage'].value_counts()
            
            # Boost score
            boosted_results = []
            for prod_idx, score in filtered_products:
                # Không recommend sản phẩm đã tương tác
                if prod_idx in interacted_products:
                    continue
                
                product = self.products_df[
                    self.products_df['product_idx'] == prod_idx
                ].iloc[0]
                
                # Boost nếu màu phổ biến trong history
                if product['baseColour'] in colour_counts:
                    score *= (1 + 0.1 * colour_counts[product['baseColour']])
                
                # Boost nếu usage phổ biến trong history
                if product['usage'] in usage_counts:
                    score *= (1 + 0.1 * usage_counts[product['usage']])
                
                boosted_results.append((prod_idx, score))
            
            filtered_products = boosted_results
        
        # BƯỚC 4: Sort và lấy top K
        final_results = sorted(filtered_products, key=lambda x: x[1], reverse=True)[:top_k]
        
        inference_time = time.time() - start_time
        
        return final_results, inference_time
    
    def recommend_outfit(
        self,
        user_info: Dict,
        payload_product_idx: int,
        user_history: pd.DataFrame = None
    ) -> Dict[str, List[Tuple[int, float]]]:
        """
        Gợi ý outfit hoàn chỉnh
        
        Args:
            user_info: Thông tin user
            payload_product_idx: Index của sản phẩm payload
            user_history: Lịch sử tương tác (optional)
        
        Returns:
            Dictionary chứa các items trong outfit
        """
        start_time = time.time()
        
        # Lấy thông tin sản phẩm payload
        payload_product = self.products_df[
            self.products_df['product_idx'] == payload_product_idx
        ].iloc[0]
        
        outfit = {
            'payload': payload_product_idx,
            'accessories': [],
            'topwear': [],
            'bottomwear': [],
            'dress': [],
            'footwear': []
        }
        
        # Gender filter
        gender_filter = [user_info['product_gender'], 'Unisex']
        is_female = user_info['gender'] == 'female'
        
        # Base filters cho outfit matching
        base_filters = {
            'gender': gender_filter,
            'baseColour': payload_product['baseColour'],
            'usage': payload_product['usage']
        }
        
        # 1. Accessories (1 trong Bags, Belts, Headwear, Watches)
        for sub_cat in ['Bags', 'Belts', 'Headwear', 'Watches']:
            candidates = self.products_df[
                (self.products_df['masterCategory'] == 'Accessories') &
                (self.products_df['subCategory'] == sub_cat) &
                (self.products_df['gender'].isin(gender_filter))
            ]
            
            if len(candidates) > 0:
                # Lấy sản phẩm tương tự nhất
                for _, product in candidates.head(3).iterrows():
                    outfit['accessories'].append((product['product_idx'], 0.9))
        
        # 2. Topwear (BẮT BUỘC)
        topwear_candidates = self.products_df[
            (self.products_df['masterCategory'] == 'Apparel') &
            (self.products_df['subCategory'] == 'Topwear') &
            (self.products_df['gender'].isin(gender_filter))
        ]
        
        if len(topwear_candidates) > 0:
            for _, product in topwear_candidates.head(5).iterrows():
                outfit['topwear'].append((product['product_idx'], 0.95))
        
        # 3. Bottomwear (BẮT BUỘC)
        bottomwear_candidates = self.products_df[
            (self.products_df['masterCategory'] == 'Apparel') &
            (self.products_df['subCategory'] == 'Bottomwear') &
            (self.products_df['gender'].isin(gender_filter))
        ]
        
        if len(bottomwear_candidates) > 0:
            for _, product in bottomwear_candidates.head(5).iterrows():
                outfit['bottomwear'].append((product['product_idx'], 0.95))
        
        # 4. Dress (TÙY CHỌN - chỉ nếu user là nữ)
        if is_female:
            dress_candidates = self.products_df[
                (self.products_df['masterCategory'] == 'Apparel') &
                (self.products_df['subCategory'] == 'Dress') &
                (self.products_df['gender'].isin(gender_filter))
            ]
            
            if len(dress_candidates) > 0:
                for _, product in dress_candidates.head(3).iterrows():
                    outfit['dress'].append((product['product_idx'], 0.85))
        
        # 4.5. Innerwear (TÙY CHỌN)
        innerwear_candidates = self.products_df[
            (self.products_df['masterCategory'] == 'Apparel') &
            (self.products_df['subCategory'] == 'Innerwear') &
            (self.products_df['gender'].isin(gender_filter))
        ]
        
        if len(innerwear_candidates) > 0:
            outfit['innerwear'] = []
            for _, product in innerwear_candidates.head(3).iterrows():
                outfit['innerwear'].append((product['product_idx'], 0.80))
        
        # 5. Footwear (1 trong Shoes, Sandal, Flip Flops)
        for sub_cat in ['Shoes', 'Sandal', 'Flip Flops']:
            footwear_candidates = self.products_df[
                (self.products_df['masterCategory'] == 'Footwear') &
                (self.products_df['subCategory'] == sub_cat) &
                (self.products_df['gender'].isin(gender_filter))
            ]
            
            if len(footwear_candidates) > 0:
                for _, product in footwear_candidates.head(3).iterrows():
                    outfit['footwear'].append((product['product_idx'], 0.9))
        
        inference_time = time.time() - start_time
        
        return outfit, inference_time
    
    def get_user_recommendations(
        self,
        user_idx: int,
        user_history: pd.DataFrame = None,
        top_k: int = 20,
        exclude_interacted: bool = True
    ) -> List[Tuple[int, float]]:
        """
        Gợi ý sản phẩm cho user dựa trên lịch sử tương tác (cho mục đích evaluation)
        
        Args:
            user_idx: Index của user
            user_history: Lịch sử tương tác của user (bắt buộc)
            top_k: Số lượng gợi ý
            exclude_interacted: Có loại bỏ sản phẩm đã tương tác không
            
        Returns:
            List of (product_idx, score)
        """
        if user_history is None or len(user_history) == 0:
            # Cold start: Recommend popular products based on average similarity
            # Use average similarity across all products as a proxy for popularity
            if self.similarity_matrix is None or self.similarity_matrix.shape[0] == 0:
                return []
            
            # Calculate average similarity score for each product (popularity proxy)
            avg_similarities = self.similarity_matrix.mean(axis=0)
            
            # Sort by average similarity
            scores = list(enumerate(avg_similarities))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)
            
            # Return top K - map row index to product_idx
            # similarity_matrix row i corresponds to products_df.iloc[i]
            results = []
            for row_idx, score in scores[:top_k * 2]:  # Get more to filter
                # Check if row_idx is valid
                if 0 <= row_idx < len(self.products_df):
                    product = self.products_df.iloc[row_idx]
                    # Use the actual product_idx from products_df
                    actual_product_idx = int(product['product_idx'])
                    results.append((actual_product_idx, float(score)))
                    if len(results) >= top_k:
                        break
            
            return results[:top_k]
            
        # Lấy danh sách sản phẩm đã tương tác
        interacted_indices = user_history['product_idx'].values
        
        # Tính User Profile Vector (Mean của các sản phẩm đã tương tác)
        # Cần map product_idx sang index trong similarity_matrix
        # products_df có product_idx được encode, cần map sang index trong DataFrame
        
        # IMPORTANT: similarity_matrix[i][j] corresponds to products_df.iloc[i] and products_df.iloc[j]
        # But product_idx in interactions may not match row index
        # We need to map product_idx -> row index in products_df
        
        # Create mapping: product_idx -> row index in products_df
        product_idx_to_row = {}
        for row_idx, row in self.products_df.iterrows():
            product_idx_to_row[int(row['product_idx'])] = row_idx
        
        # Find valid row indices for interacted products
        valid_row_indices = []
        for prod_idx in interacted_indices:
            prod_idx_int = int(prod_idx)
            if prod_idx_int in product_idx_to_row:
                row_idx = product_idx_to_row[prod_idx_int]
                # Ensure row_idx is within bounds of similarity_matrix
                if 0 <= row_idx < self.similarity_matrix.shape[0]:
                    valid_row_indices.append(row_idx)
        
        if not valid_row_indices:
            # Fallback to cold start
            return self.get_user_recommendations(user_idx, None, top_k, exclude_interacted)
            
        # Cộng dồn similarity vector của các sản phẩm đã tương tác
        user_sim_vector = np.zeros(self.similarity_matrix.shape[0])
        
        for row_idx in valid_row_indices:
            user_sim_vector += self.similarity_matrix[row_idx]
            
        # Normalize
        user_sim_vector /= len(valid_row_indices)
        
        # Sort scores by row index (which corresponds to position in products_df)
        scores = list(enumerate(user_sim_vector))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        # Filter and map back to product_idx
        final_recs = []
        interacted_set = {int(x) for x in interacted_indices}
        
        for row_idx, score in scores:
            if row_idx >= len(self.products_df):
                continue
                
            # Get actual product_idx from products_df at this row
            product = self.products_df.iloc[row_idx]
            actual_product_idx = int(product['product_idx'])
            
            if exclude_interacted and actual_product_idx in interacted_set:
                continue
                
            final_recs.append((actual_product_idx, float(score)))
            
            if len(final_recs) >= top_k:
                break
                
        return final_recs

    def get_model_info(self) -> Dict:
        """Lấy thông tin về model"""
        return {
            'model_name': 'Content-Based Filtering',
            'n_products': len(self.products_df),
            'feature_dim': self.similarity_matrix.shape[1] if self.similarity_matrix is not None else 0,
            'training_time': self.training_time,
            'avg_similarity': float(self.similarity_matrix.mean()) if self.similarity_matrix is not None else 0
        }


if __name__ == "__main__":
    # Test
    print("Testing Content-Based Recommender...")

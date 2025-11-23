"""
Data Preprocessing Module
Xử lý và chuẩn bị dữ liệu cho các models
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    """Class xử lý dữ liệu cho hệ thống gợi ý"""
    
    def __init__(self, users_path: str, products_path: str, interactions_path: str):
        """
        Args:
            users_path: Đường dẫn đến file users.csv
            products_path: Đường dẫn đến file products.csv
            interactions_path: Đường dẫn đến file interactions.csv
        """
        self.users_path = users_path
        self.products_path = products_path
        self.interactions_path = interactions_path
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.product_encoder = LabelEncoder()
        
        # Data
        self.users_df = None
        self.products_df = None
        self.interactions_df = None
        
        # Processed data
        self.train_interactions = None
        self.test_interactions = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load dữ liệu từ các file CSV"""
        print("Loading data...")
        
        # Load users
        self.users_df = pd.read_csv(self.users_path)
        print(f"Loaded {len(self.users_df)} users")
        
        # Load products
        self.products_df = pd.read_csv(self.products_path)
        print(f"Loaded {len(self.products_df)} products")
        
        # Load interactions
        self.interactions_df = pd.read_csv(self.interactions_path)
        print(f"Loaded {len(self.interactions_df)} interactions")
        
        return self.users_df, self.products_df, self.interactions_df
    
    def clean_data(self):
        """Làm sạch dữ liệu"""
        print("\nCleaning data...")
        
        # 1. Clean users
        # Remove admin user without age/gender
        self.users_df = self.users_df.dropna(subset=['age', 'gender'])
        print(f"Users after removing missing age/gender: {len(self.users_df)}")
        
        # 2. Clean interactions
        # Convert product_id to int, remove invalid ones
        # Lưu ý: product_id trong interactions có thể là string, cần convert
        self.interactions_df['product_id'] = pd.to_numeric(
            self.interactions_df['product_id'], errors='coerce'
        )
        self.interactions_df = self.interactions_df.dropna(subset=['product_id'])
        self.interactions_df['product_id'] = self.interactions_df['product_id'].astype(int)
        
        # Keep only interactions with valid products
        valid_products = set(self.products_df['id'].values)
        valid_users = set(self.users_df['id'].values)
        
        # Filter: chỉ giữ interactions có product_id hợp lệ
        before_filter = len(self.interactions_df)
        self.interactions_df = self.interactions_df[
            self.interactions_df['product_id'].isin(valid_products)
        ]
        
        # Tạo users mới cho các user_id không có trong users.csv
        interactions_user_ids = set(self.interactions_df['user_id'].unique())
        missing_user_ids = interactions_user_ids - valid_users
        
        if len(missing_user_ids) > 0:
            print(f"Creating {len(missing_user_ids)} new users for interactions...")
            # Tạo DataFrame cho users mới với thông tin mặc định
            new_users_data = []
            for user_id in missing_user_ids:
                new_users_data.append({
                    'id': user_id,
                    'name': f'User_{user_id[:8]}',
                    'age': 25,  # Default age
                    'gender': 'male',  # Default gender
                    'email': f'{user_id}@example.com'
                })
            
            new_users_df = pd.DataFrame(new_users_data)
            self.users_df = pd.concat([self.users_df, new_users_df], ignore_index=True)
            print(f"Total users after adding new ones: {len(self.users_df)}")
        
        # Bây giờ filter lại để đảm bảo user_id hợp lệ
        valid_users = set(self.users_df['id'].values)
        self.interactions_df = self.interactions_df[
            self.interactions_df['user_id'].isin(valid_users)
        ]
        
        after_filter = len(self.interactions_df)
        print(f"Interactions after cleaning: {after_filter} (removed {before_filter - after_filter})")
        
        # Nếu quá ít interactions sau khi clean, cảnh báo
        if after_filter < 100:
            print(f"WARNING: Only {after_filter} interactions after cleaning!")
            print(f"   Valid users: {len(valid_users)}, Valid products: {len(valid_products)}")
            print(f"   Users in interactions: {self.interactions_df['user_id'].nunique()}")
            print(f"   Products in interactions: {self.interactions_df['product_id'].nunique()}")
        
        # 3. Convert timestamp to datetime (handle mixed formats)
        self.interactions_df['timestamp'] = pd.to_datetime(
            self.interactions_df['timestamp'], 
            format='mixed',
            errors='coerce'
        )
        # Remove rows with invalid timestamps
        self.interactions_df = self.interactions_df.dropna(subset=['timestamp'])
        
    def create_user_features(self):
        """Tạo features cho users"""
        print("\nCreating user features...")
        
        # Age group
        self.users_df['age_group'] = pd.cut(
            self.users_df['age'], 
            bins=[0, 18, 25, 35, 50, 100],
            labels=['<18', '18-25', '26-35', '36-50', '50+']
        )
        
        # Gender mapping for product matching
        def map_gender_to_product(row):
            """Map user gender và age sang product gender"""
            if row['gender'] == 'male':
                return 'Boys' if row['age'] < 18 else 'Men'
            else:
                return 'Girls' if row['age'] < 18 else 'Women'
        
        self.users_df['product_gender'] = self.users_df.apply(map_gender_to_product, axis=1)
        
        print(f"Created age_group and product_gender features")
        
    def create_product_features(self):
        """Tạo features cho products"""
        print("\nCreating product features...")
        
        # Encode categorical features
        categorical_cols = ['gender', 'masterCategory', 'subCategory', 
                          'articleType', 'baseColour', 'season', 'usage']
        
        for col in categorical_cols:
            if col in self.products_df.columns:
                # Fill missing values
                self.products_df[col] = self.products_df[col].fillna('Unknown')
        
        print(f"Processed {len(categorical_cols)} categorical features")
        
    def create_interaction_weights(self):
        """Tạo trọng số cho các loại interaction"""
        print("\nCreating interaction weights...")
        
        # Định nghĩa trọng số
        interaction_weights = {
            'purchase': 1.0,
            'cart': 0.7,
            'like': 0.5,
            'view': 0.3
        }
        
        self.interactions_df['weight'] = self.interactions_df['interaction_type'].map(
            interaction_weights
        )
        
        print("Interaction weights:")
        for itype, weight in interaction_weights.items():
            count = len(self.interactions_df[self.interactions_df['interaction_type'] == itype])
            print(f"  {itype}: {weight} (count: {count})")
    
    def encode_ids(self):
        """Encode user_id và product_id thành indices"""
        print("\nEncoding IDs...")
        
        # Encode users
        self.users_df['user_idx'] = self.user_encoder.fit_transform(self.users_df['id'])
        
        # Encode products
        self.products_df['product_idx'] = self.product_encoder.fit_transform(
            self.products_df['id']
        )
        
        # Map to interactions
        user_id_to_idx = dict(zip(self.users_df['id'], self.users_df['user_idx']))
        product_id_to_idx = dict(zip(self.products_df['id'], self.products_df['product_idx']))
        
        self.interactions_df['user_idx'] = self.interactions_df['user_id'].map(user_id_to_idx)
        self.interactions_df['product_idx'] = self.interactions_df['product_id'].map(
            product_id_to_idx
        )
        
        print(f"Encoded {len(user_id_to_idx)} users and {len(product_id_to_idx)} products")
        
    def create_train_test_split(self, test_size=0.2, random_state=42):
        """Chia dữ liệu thành train và test sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        
        # Sort by timestamp
        self.interactions_df = self.interactions_df.sort_values('timestamp')
        
        # Split by time (temporal split)
        split_idx = int(len(self.interactions_df) * (1 - test_size))
        self.train_interactions = self.interactions_df.iloc[:split_idx].copy()
        self.test_interactions = self.interactions_df.iloc[split_idx:].copy()
        
        print(f"Train interactions: {len(self.train_interactions)}")
        print(f"Test interactions: {len(self.test_interactions)}")
        
        # Statistics
        print("\nTrain set statistics:")
        print(f"  Unique users: {self.train_interactions['user_idx'].nunique()}")
        print(f"  Unique products: {self.train_interactions['product_idx'].nunique()}")
        print(f"  Sparsity: {len(self.train_interactions) / (self.users_df['user_idx'].nunique() * self.products_df['product_idx'].nunique()) * 100:.2f}%")
        
        print("\nTest set statistics:")
        print(f"  Unique users: {self.test_interactions['user_idx'].nunique()}")
        print(f"  Unique products: {self.test_interactions['product_idx'].nunique()}")
        
    def get_user_interaction_history(self, user_idx: int) -> pd.DataFrame:
        """Lấy lịch sử tương tác của user"""
        return self.train_interactions[
            self.train_interactions['user_idx'] == user_idx
        ].sort_values('timestamp', ascending=False)
    
    def get_product_info(self, product_idx: int) -> Dict:
        """Lấy thông tin sản phẩm"""
        product = self.products_df[self.products_df['product_idx'] == product_idx]
        if len(product) == 0:
            return None
        return product.iloc[0].to_dict()
    
    def get_user_info(self, user_idx: int) -> Dict:
        """Lấy thông tin user"""
        user = self.users_df[self.users_df['user_idx'] == user_idx]
        if len(user) == 0:
            return None
        return user.iloc[0].to_dict()
    
    def preprocess_all(self):
        """Chạy toàn bộ pipeline preprocessing"""
        print("="*80)
        print("STARTING DATA PREPROCESSING PIPELINE")
        print("="*80)
        
        # Step 1: Load data
        self.load_data()
        
        # Step 2: Clean data
        self.clean_data()
        
        # Step 3: Create features
        self.create_user_features()
        self.create_product_features()
        
        # Step 4: Create interaction weights
        self.create_interaction_weights()
        
        # Step 5: Encode IDs
        self.encode_ids()
        
        # Step 6: Train/test split
        self.create_train_test_split()
        
        print("\n" + "="*80)
        print("PREPROCESSING COMPLETED")
        print("="*80)
        
        return self
    
    def get_statistics(self) -> Dict:
        """Lấy thống kê về dữ liệu"""
        stats = {
            'n_users': len(self.users_df),
            'n_products': len(self.products_df),
            'n_interactions': len(self.interactions_df),
            'n_train': len(self.train_interactions),
            'n_test': len(self.test_interactions),
            'sparsity': len(self.interactions_df) / (len(self.users_df) * len(self.products_df)),
            'interaction_types': self.interactions_df['interaction_type'].value_counts().to_dict(),
            'gender_distribution': self.users_df['gender'].value_counts().to_dict(),
            'age_distribution': self.users_df['age_group'].value_counts().to_dict(),
            'product_categories': self.products_df['masterCategory'].value_counts().to_dict()
        }
        return stats


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DataPreprocessor(
        users_path="../exports/users.csv",
        products_path="../exports/products.csv",
        interactions_path="../exports/interactions.csv"
    )
    
    preprocessor.preprocess_all()
    
    # Print statistics
    stats = preprocessor.get_statistics()
    print("\n" + "="*80)
    print("DATA STATISTICS")
    print("="*80)
    for key, value in stats.items():
        print(f"{key}: {value}")

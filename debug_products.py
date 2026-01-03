import pandas as pd
import os
import sys

# Define mapping function to simulate the app logic exactly
def map_to_complement_key(article_type):
    article_type = str(article_type).strip()
    article_lower = article_type.lower()
    
    if article_lower in ['t-shirt', 't shirt', 'tshirt']:
        return 'Tshirts'
    if article_lower in ['shirt']:
        return 'Shirts'
    if article_lower in ['top']:
        return 'Tops'
    return None

try:
    csv_path = os.path.join(os.getcwd(), 'apps', 'exports', 'products.csv')
    print(f"Reading {csv_path}")
    if not os.path.exists(csv_path):
        print("File not found!")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Total products: {len(df)}")
    
    # Filter for Girls
    girls_df = df[df['gender'].str.strip().str.lower() == 'girls']
    print(f"Girls products: {len(girls_df)}")
    
    # Check articleTypes
    article_counts = girls_df['articleType'].value_counts()
    print("\nArticle Types for Girls:")
    print(article_counts)
    
    # Check mapping
    print("\nMapped Keys for Girls products:")
    mapped_counts = {}
    for atype in girls_df['articleType'].unique():
        key = map_to_complement_key(atype)
        if key:
            mapped_counts[key] = mapped_counts.get(key, 0) + len(girls_df[girls_df['articleType'] == atype])
            
    print(mapped_counts)
    
except Exception as e:
    print(f"Error: {e}")

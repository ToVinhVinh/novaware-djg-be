"""
Direct test of outfit building logic without API call
"""
import os
import sys
import django

# Setup Django
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
django.setup()

# Now import after Django setup
from apps.recommendations.hybrid.views import build_outfit_suggestions
from apps.utils.cbf_utils import get_allowed_genders
import pandas as pd

print("=" * 60)
print("TESTING OUTFIT BUILDING LOGIC")
print("=" * 60)

# Load data
products_df = pd.read_csv('apps/exports/products.csv')
if 'id' in products_df.columns:
    products_df = products_df.set_index('id')

# Test parameters
user_id = "690bf40623150d4eec246874"
payload_product_id = "10003"
user_age = 21
user_gender = "male"

# Mock data
personalized_items = []
hybrid_predictions = {'predictions': {user_id: {}}}

print(f"\nTest case:")
print(f"  User ID: {user_id}")
print(f"  Product ID: {payload_product_id}")
print(f"  User age: {user_age}, gender: {user_gender}")

# Get payload product info
payload_row = products_df.loc[int(payload_product_id)]
print(f"\nPayload product:")
print(f"  ArticleType: {payload_row['articleType']}")
print(f"  Gender: {payload_row['gender']}")

try:
    # Build outfits
    outfits = build_outfit_suggestions(
        user_id=user_id,
        payload_product_id=payload_product_id,
        personalized_items=personalized_items,
        products_df=products_df,
        hybrid_predictions=hybrid_predictions,
        user_age=user_age,
        user_gender=user_gender,
        max_outfits=1
    )
    
    print(f"\n✅ Generated {len(outfits)} outfit(s)")
    
    if outfits:
        outfit = outfits[0]
        print(f"\nOutfit items ({len(outfit['products'])} total):")
        for i, pid in enumerate(outfit['products'], 1):
            product = products_df.loc[int(pid)]
            print(f"  {i}. {product['articleType']} (ID: {pid}, Gender: {product['gender']})")
        
        print(f"\nExpected structure:")
        print(f"  1. Tshirts (payload)")
        print(f"  2. Watches")
        print(f"  3. Jeans/Trousers/Shorts/Skirts (bottom)")
        print(f"  4. Casual Shoes/Flip Flops/etc (shoes)")
    else:
        print("\n❌ No outfits generated!")
        
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

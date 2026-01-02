import requests
import json

try:
    r = requests.post(
        'http://localhost:8000/api/v1/hybrid/recommend/',
        json={
            'user_id': '690bf40623150d4eec246874',
            'current_product_id': '10003',
            'top_k_personalized': 6,
            'top_k_outfit': 1
        },
        timeout=30
    )
    
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        outfit = data['outfits'][0]
        
        print(f"\nOutfit has {len(outfit['products'])} items:")
        for i, p in enumerate(outfit['products'], 1):
            article = p['product']['articleType']
            pid = p['product_id']
            print(f"  {i}. {article} (ID: {pid})")
        
        print(f"\nExpected: [Tshirts, Watches, Jeans/Trousers/Shorts/Skirts, Casual Shoes/Flip Flops/etc]")
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"Exception: {e}")

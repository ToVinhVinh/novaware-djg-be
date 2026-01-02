import requests
import json

url = 'http://localhost:8000/api/v1/hybrid/recommend/'
payload = {
    'user_id': '690bf40623150d4eec246874',
    'current_product_id': '10003',
    'top_k_personalized': 6,
    'top_k_outfit': 1
}

try:
    r = requests.post(url, json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    
    if r.status_code == 200:
        data = r.json()
        
        print("\n=== OUTFIT RESULT ===")
        if 'outfits' in data and len(data['outfits']) > 0:
            outfit = data['outfits'][0]
            print(f"Total items: {len(outfit['products'])}")
            print("\nItems:")
            for i, p in enumerate(outfit['products'], 1):
                article = p['product']['articleType']
                pid = p['product_id']
                gender = p['product'].get('gender', 'N/A')
                print(f"  {i}. {article} (ID: {pid}, Gender: {gender})")
            
            print(f"\nExpected: 4 items (Tshirts + Watches + Bottom + Shoes)")
            
            # Check if outfit matches complement rules
            articles = [p['product']['articleType'] for p in outfit['products']]
            print(f"\nArticle types: {articles}")
            
            # Save full response
            with open('outfit_test_result.json', 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print("\n✅ Full response saved to outfit_test_result.json")
        else:
            print("No outfits in response")
    else:
        print(f"Error: {r.text}")
        
except Exception as e:
    print(f"Exception: {e}")
    import traceback
    traceback.print_exc()

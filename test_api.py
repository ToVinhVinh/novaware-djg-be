import urllib.request
import json
import sys

url = "http://localhost:8000/api/v1/hybrid/recommend/"
payload = {
    "user_id": "690bf40623150d4eec246874",
    "current_product_id": 10604,
    "top_k_personal": 5,
    "top_k_outfit": 10
}

try:
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, headers={
        'Content-Type': 'application/json',
        'User-Agent': 'Mozilla/5.0'
    })
    
    with urllib.request.urlopen(req) as response:
        status_code = response.getcode()
        print(f"Status Code: {status_code}")
        
        response_body = response.read().decode('utf-8')
        result = json.loads(response_body)
        
        outfits = result.get('outfits', [])
        print(f"Number of outfits returned: {len(outfits)}")
        
        for i, outfit in enumerate(outfits):
            products = outfit.get('products', [])
            print(f"Outfit {i+1} has {len(products)} products:")
            for p in products:
                prod_details = p.get('product', {})
                name = prod_details.get('productDisplayName', 'N/A')
                article = prod_details.get('articleType', 'N/A')
                gender = prod_details.get('gender', 'N/A')
                pid = p.get('product_id')
                print(f"  - [{pid}] {article} ({gender}): {name}")

except urllib.error.HTTPError as e:
    print(f"HTTP Error: {e.code} - {e.reason}")
    print(e.read().decode('utf-8'))
except Exception as e:
    print(f"Error: {e}")

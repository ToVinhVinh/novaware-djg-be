import requests
import json

payload = {
    "user_id": "690bf0f2d0c3753df0ecbe08",
    "current_product_id": "68fd618251f9562e9cca76e8",
    "top_k_personal": 10,
    "top_k_outfit": 5,
    "alpha": 0.5
}

try:
    response = requests.post('http://localhost:8000/api/hybrid/recommend/', json=payload)
    response.raise_for_status()
    result = response.json()
    print(json.dumps(result, indent=2))
    
    # Check structure
    print("\n=== Structure Check ===")
    print(f"Has 'personalized': {'personalized' in result}")
    if 'personalized' in result and result['personalized']:
        print(f"Personalized count: {len(result['personalized'])}")
    
    print(f"\nHas 'outfit': {'outfit' in result}")
    if 'outfit' in result:
        outfit = result['outfit']
        print(f"Outfit categories: {list(outfit.keys())}")
        expected_categories = ["accessories", "bottoms", "shoes", "tops"]
        missing = [cat for cat in expected_categories if cat not in outfit]
        if missing:
            print(f"  MISSING categories: {missing}")
            print("  ERROR: Not all required categories present!")
        else:
            print(f"  OK: All expected categories present")
        
        for category in expected_categories:
            if category in outfit:
                item = outfit[category]
                print(f"  {category}: OK (score: {item.get('score', 'N/A')})")
            else:
                print(f"  {category}: MISSING")
    
    print(f"\nHas 'outfit_complete_score': {'outfit_complete_score' in result}")
    print(f"Outfit complete score: {result.get('outfit_complete_score', 'N/A')}")
    
except requests.exceptions.ConnectionError:
    print("Error: Could not connect to server. Is it running on http://localhost:8000?")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

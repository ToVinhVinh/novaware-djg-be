"""
Automated script to run GNN training with all products
"""
import os
import sys
import django
import requests
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'novaware.settings')
django.setup()

BASE_URL = "http://localhost:8000"
MODEL = "gnn"

print("=" * 70)
print("AUTOMATED GNN TRAINING WITH ALL PRODUCTS")
print("=" * 70)

# Step 1: Start training
print(f"\n🚀 Starting {MODEL.upper()} training...")
url = f"{BASE_URL}/api/v1/{MODEL}/train"

payload = {
    "force_retrain": True,
    "sync": False  # Run async to monitor progress
}

try:
    print(f"   URL: {url}")
    print(f"   Payload: {json.dumps(payload, indent=2)}")
    
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    print(f"\n✅ Response:")
    print(json.dumps(data, indent=2, ensure_ascii=False))
    
    if "task_id" in data:
        task_id = data["task_id"]
        print(f"\n📋 Task ID: {task_id}")
        print(f"\n⏳ Waiting for training to complete...")
        print(f"   (Checking every 3 seconds)\n")
        
        # Step 2: Polling status
        start_time = time.time()
        max_wait = 600  # 10 minutes
        check_interval = 3
        
        while True:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                print(f"\n⏰ Timeout after {max_wait} seconds")
                break
            
            # Check status
            status_response = requests.post(url, json={"task_id": task_id}, timeout=10)
            status_data = status_response.json()
            
            status = status_data.get("status", "unknown")
            progress = status_data.get("progress", 0)
            message = status_data.get("message", "")
            
            print(f"[{elapsed:.1f}s] Status: {status.upper()} | Progress: {progress}% | {message}")
            
            if status == "success":
                print(f"\n✅ Training completed after {elapsed:.1f} seconds!")
                print("\n📊 Results:")
                print(json.dumps(status_data, indent=2, ensure_ascii=False))
                
                # Check matrix_data
                if "matrix_data" in status_data:
                    matrix = status_data["matrix_data"]
                    print(f"\n📈 Matrix Data:")
                    print(f"   - Shape: {matrix.get('shape')}")
                    print(f"   - Sparsity: {matrix.get('sparsity')}")
                
                break
            elif status == "failure":
                print(f"\n❌ Training failed!")
                print(f"   Error: {status_data.get('error', 'Unknown error')}")
                break
            elif status == "not_found":
                print(f"\n❌ Task not found!")
                break
            
            time.sleep(check_interval)
    else:
        print("\n⚠️  No task_id in response")
        print("   (Training may have run in sync mode)")

except requests.exceptions.RequestException as e:
    print(f"\n❌ Error calling API: {e}")
    if hasattr(e, 'response') and e.response is not None:
        print(f"   Response: {e.response.text}")
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)


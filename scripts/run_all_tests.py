"""Script chạy tất cả các tests và checks."""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Thêm thư mục gốc vào Python path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import django
from django.conf import settings

# Setup Django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
django.setup()

import subprocess


def run_script(script_name, description):
    """Chạy một script và hiển thị kết quả."""
    print("\n" + "=" * 60)
    print(f"🚀 {description}")
    print("=" * 60)
    
    script_path = Path(__file__).parent / script_name
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Lỗi khi chạy script: {e}")
        return False


def main():
    """Hàm chính."""
    print("\n" + "=" * 60)
    print("🚀 CHẠY TẤT CẢ TESTS VÀ CHECKS")
    print("=" * 60)
    
    scripts = [
        ("test_mongodb_connection.py", "Kiểm tra MongoDB Connection và Indexes"),
        ("test_endpoints.py", "Test các Endpoints API"),
        ("monitor_performance.py", "Monitor Performance"),
    ]
    
    results = {}
    
    for script_name, description in scripts:
        success = run_script(script_name, description)
        results[script_name] = success
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TỔNG KẾT")
    print("=" * 60)
    
    for script_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {script_name}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✅ Tất cả tests đã pass!")
        sys.exit(0)
    else:
        print("\n⚠️  Một số tests đã fail. Vui lòng kiểm tra lại.")
        sys.exit(1)


if __name__ == "__main__":
    main()


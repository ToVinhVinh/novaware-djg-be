"""Script test các endpoints API."""

from __future__ import annotations

import json
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

import requests
from django.contrib.auth import get_user_model
from rest_framework.test import APIClient

from apps.users.mongo_models import User


class EndpointTester:
    """Class để test các endpoints."""
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.client = APIClient()
        self.token = None
        self.user = None
        self.admin_token = None
        self.admin_user = None
    
    def print_section(self, title):
        """In tiêu đề section."""
        print("\n" + "=" * 60)
        print(f"🔍 {title}")
        print("=" * 60)
    
    def print_test(self, test_name, status, message=""):
        """In kết quả test."""
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {test_name}")
        if message:
            print(f"   {message}")
    
    def create_test_user(self, email="test@example.com", password="testpass123", is_admin=False):
        """Tạo user test."""
        try:
            user = User.objects(email=email).first()
            if user:
                user.delete()
            
            user = User(
                email=email,
                name="Test User",
                is_admin=is_admin,
                is_active=True,
            )
            user.set_password(password)
            user.save()
            return user
        except Exception as e:
            print(f"❌ Lỗi tạo user: {e}")
            return None
    
    def login(self, email="test@example.com", password="testpass123"):
        """Đăng nhập và lấy token."""
        url = f"{self.base_url}/api/v1/auth/login/"
        data = {"email": email, "password": password}
        
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                token_data = response.json()
                self.token = token_data.get("access")
                self.client.credentials(HTTP_AUTHORIZATION=f"Bearer {self.token}")
                return True
            return False
        except Exception as e:
            print(f"❌ Lỗi login: {e}")
            return False
    
    def test_auth_endpoints(self):
        """Test authentication endpoints."""
        self.print_section("TEST AUTHENTICATION ENDPOINTS")
        
        # Test register
        url = f"{self.base_url}/api/v1/auth/register/"
        data = {
            "email": "newuser@example.com",
            "password": "newpass123",
            "name": "New User",
        }
        try:
            response = requests.post(url, json=data)
            self.print_test(
                "Register",
                response.status_code in [200, 201],
                f"Status: {response.status_code}"
            )
        except Exception as e:
            self.print_test("Register", False, str(e))
        
        # Test login
        self.user = self.create_test_user()
        success = self.login()
        self.print_test("Login", success)
        
        # Test admin login
        self.admin_user = self.create_test_user(
            email="admin@example.com",
            password="adminpass123",
            is_admin=True
        )
        url = f"{self.base_url}/api/v1/auth/login/"
        data = {"email": "admin@example.com", "password": "adminpass123"}
        try:
            response = requests.post(url, json=data)
            if response.status_code == 200:
                token_data = response.json()
                self.admin_token = token_data.get("access")
                self.print_test("Admin Login", True)
            else:
                self.print_test("Admin Login", False, f"Status: {response.status_code}")
        except Exception as e:
            self.print_test("Admin Login", False, str(e))
    
    def test_user_endpoints(self):
        """Test user endpoints."""
        self.print_section("TEST USER ENDPOINTS")
        
        if not self.token:
            self.print_test("User Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test get profile
        url = f"{self.base_url}/api/v1/users/profile/"
        response = self.client.get(url)
        self.print_test(
            "Get Profile",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        
        # Test update profile
        url = f"{self.base_url}/api/v1/users/profile/"
        data = {"name": "Updated Name"}
        response = self.client.patch(url, data, format="json")
        self.print_test(
            "Update Profile",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        
        # Test list users (admin only)
        url = f"{self.base_url}/api/v1/users/"
        response = self.client.get(url)
        self.print_test(
            "List Users",
            response.status_code in [200, 403],
            f"Status: {response.status_code}"
        )
    
    def test_product_endpoints(self):
        """Test product endpoints."""
        self.print_section("TEST PRODUCT ENDPOINTS")
        
        if not self.token:
            self.print_test("Product Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test list categories
        url = f"{self.base_url}/api/v1/products/categories/"
        response = self.client.get(url)
        self.print_test(
            "List Categories",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        
        # Test list products
        url = f"{self.base_url}/api/v1/products/"
        response = self.client.get(url)
        self.print_test(
            "List Products",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
        
        # Test search products
        url = f"{self.base_url}/api/v1/products/?search=test"
        response = self.client.get(url)
        self.print_test(
            "Search Products",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    
    def test_brand_endpoints(self):
        """Test brand endpoints."""
        self.print_section("TEST BRAND ENDPOINTS")
        
        if not self.token:
            self.print_test("Brand Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test list brands
        url = f"{self.base_url}/api/v1/brands/"
        response = self.client.get(url)
        self.print_test(
            "List Brands",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    
    def test_order_endpoints(self):
        """Test order endpoints."""
        self.print_section("TEST ORDER ENDPOINTS")
        
        if not self.token:
            self.print_test("Order Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test list orders
        url = f"{self.base_url}/api/v1/orders/"
        response = self.client.get(url)
        self.print_test(
            "List Orders",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    
    def test_chat_endpoints(self):
        """Test chat endpoints."""
        self.print_section("TEST CHAT ENDPOINTS")
        
        if not self.token:
            self.print_test("Chat Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test list chat threads
        url = f"{self.base_url}/api/v1/chat/"
        response = self.client.get(url)
        self.print_test(
            "List Chat Threads",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    
    def test_recommendation_endpoints(self):
        """Test recommendation endpoints."""
        self.print_section("TEST RECOMMENDATION ENDPOINTS")
        
        if not self.token:
            self.print_test("Recommendation Endpoints", False, "Chưa đăng nhập")
            return
        
        # Test list outfits
        url = f"{self.base_url}/api/v1/recommendations/outfits/"
        response = self.client.get(url)
        self.print_test(
            "List Outfits",
            response.status_code == 200,
            f"Status: {response.status_code}"
        )
    
    def cleanup(self):
        """Dọn dữ liệu test."""
        try:
            if self.user:
                self.user.delete()
            if self.admin_user:
                self.admin_user.delete()
            # Xóa test users
            User.objects(email__in=["newuser@example.com", "test@example.com", "admin@example.com"]).delete()
        except Exception:
            pass
    
    def run_all_tests(self):
        """Chạy tất cả tests."""
        print("\n" + "=" * 60)
        print("🚀 BẮT ĐẦU TEST ENDPOINTS")
        print("=" * 60)
        
        try:
            self.test_auth_endpoints()
            self.test_user_endpoints()
            self.test_product_endpoints()
            self.test_brand_endpoints()
            self.test_order_endpoints()
            self.test_chat_endpoints()
            self.test_recommendation_endpoints()
        finally:
            self.cleanup()
        
        print("\n" + "=" * 60)
        print("✅ HOÀN TẤT TEST ENDPOINTS")
        print("=" * 60)


def main():
    """Hàm chính."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test các endpoints API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL của API (default: http://localhost:8000)"
    )
    
    args = parser.parse_args()
    
    tester = EndpointTester(base_url=args.base_url)
    tester.run_all_tests()


if __name__ == "__main__":
    main()


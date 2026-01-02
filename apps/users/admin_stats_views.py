from __future__ import annotations

from datetime import datetime, timedelta
from bson import ObjectId
from rest_framework import permissions, status, viewsets
from rest_framework.decorators import action
from rest_framework.response import Response

from apps.utils import api_error, api_success
from apps.users.mongo_models import User
from apps.products.mongo_models import Product
from apps.orders.mongo_models import Order


class AdminStatsViewSet(viewsets.ViewSet):
    """
    Admin Statistics API for Dashboard Analytics
    """
    permission_classes = [permissions.AllowAny]
    authentication_classes = []

    @action(detail=False, methods=["get"], url_path="overview")
    def overview(self, request):
        """
        GET /api/v1/admin-stats/overview
        
        Tổng quan thống kê tổng thể:
        - Tổng số users, products, orders
        - Doanh thu tổng, doanh thu tháng này
        - Tăng trưởng so với tháng trước
        """
        # Total counts
        total_users = User.objects.count()
        total_products = Product.objects.count()
        total_orders = Order.objects.count()
        
        # Revenue calculations
        all_paid_orders = Order.objects.filter(is_paid=True)
        total_revenue = sum(order.total_price for order in all_paid_orders if hasattr(order, 'total_price'))
        
        # This month's data
        now = datetime.utcnow()
        first_day_this_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        this_month_orders = Order.objects.filter(
            is_paid=True,
            paid_at__gte=first_day_this_month
        )
        this_month_revenue = sum(order.total_price for order in this_month_orders if hasattr(order, 'total_price'))
        this_month_order_count = this_month_orders.count()
        
        # Last month's data for comparison
        first_day_last_month = (first_day_this_month - timedelta(days=1)).replace(day=1)
        last_month_orders = Order.objects.filter(
            is_paid=True,
            paid_at__gte=first_day_last_month,
            paid_at__lt=first_day_this_month
        )
        last_month_revenue = sum(order.total_price for order in last_month_orders if hasattr(order, 'total_price'))
        last_month_order_count = last_month_orders.count()
        
        # Calculate growth rates
        revenue_growth = 0
        if last_month_revenue > 0:
            revenue_growth = ((this_month_revenue - last_month_revenue) / last_month_revenue) * 100
        
        order_growth = 0
        if last_month_order_count > 0:
            order_growth = ((this_month_order_count - last_month_order_count) / last_month_order_count) * 100
        
        # New users this month
        new_users_this_month = User.objects.filter(created_at__gte=first_day_this_month).count()
        new_users_last_month = User.objects.filter(
            created_at__gte=first_day_last_month,
            created_at__lt=first_day_this_month
        ).count()
        
        user_growth = 0
        if new_users_last_month > 0:
            user_growth = ((new_users_this_month - new_users_last_month) / new_users_last_month) * 100
        
        return api_success(
            "Overview statistics retrieved successfully",
            {
                "totals": {
                    "users": total_users,
                    "products": total_products,
                    "orders": total_orders,
                    "revenue": round(total_revenue, 2)
                },
                "this_month": {
                    "revenue": round(this_month_revenue, 2),
                    "orders": this_month_order_count,
                    "new_users": new_users_this_month
                },
                "growth": {
                    "revenue_percent": round(revenue_growth, 2),
                    "orders_percent": round(order_growth, 2),
                    "users_percent": round(user_growth, 2)
                }
            }
        )

    @action(detail=False, methods=["get"], url_path="revenue-chart")
    def revenue_chart(self, request):
        """
        GET /api/v1/admin-stats/revenue-chart?period=7d|30d|90d|1y
        
        Dữ liệu biểu đồ doanh thu theo thời gian
        - period: 7d (7 ngày), 30d (30 ngày), 90d (90 ngày), 1y (1 năm)
        
        Response format phù hợp cho Line Chart hoặc Area Chart
        """
        period = request.query_params.get("period", "30d")
        
        # Calculate date range
        now = datetime.utcnow()
        if period == "7d":
            start_date = now - timedelta(days=7)
            group_by = "day"
        elif period == "90d":
            start_date = now - timedelta(days=90)
            group_by = "day"
        elif period == "1y":
            start_date = now - timedelta(days=365)
            group_by = "month"
        else:  # default 30d
            start_date = now - timedelta(days=30)
            group_by = "day"
        
        # Get orders in period - use created_at if paid_at is not available
        orders = Order.objects.filter(
            is_paid=True,
            created_at__gte=start_date
        ).order_by("created_at")
        
        # Group data
        revenue_data = {}
        order_count_data = {}
        
        for order in orders:
            # Use paid_at if available, otherwise use created_at
            date_field = order.paid_at if order.paid_at else order.created_at
            if not date_field:
                continue
                
            if group_by == "day":
                key = date_field.strftime("%Y-%m-%d")
            else:  # month
                key = date_field.strftime("%Y-%m")
            
            if key not in revenue_data:
                revenue_data[key] = 0
                order_count_data[key] = 0
            
            revenue_data[key] += float(order.total_price) if order.total_price else 0
            order_count_data[key] += 1
        
        # Format for chart
        labels = sorted(revenue_data.keys())
        revenues = [round(revenue_data[label], 2) for label in labels]
        order_counts = [order_count_data[label] for label in labels]
        
        return api_success(
            "Revenue chart data retrieved successfully",
            {
                "period": period,
                "group_by": group_by,
                "labels": labels,
                "datasets": [
                    {
                        "name": "Revenue",
                        "data": revenues
                    },
                    {
                        "name": "Orders",
                        "data": order_counts
                    }
                ]
            }
        )

    @action(detail=False, methods=["get"], url_path="top-products")
    def top_products(self, request):
        """
        GET /api/v1/admin-stats/top-products?limit=10&sort_by=revenue|quantity
        
        Top sản phẩm bán chạy
        - limit: Số lượng sản phẩm (default: 10)
        - sort_by: revenue (doanh thu) hoặc quantity (số lượng bán)
        
        Response format phù hợp cho Bar Chart hoặc Table
        """
        limit = int(request.query_params.get("limit", 10))
        sort_by = request.query_params.get("sort_by", "revenue")
        
        # Aggregate product sales from orders
        product_stats = {}
        
        paid_orders = Order.objects.filter(is_paid=True)
        for order in paid_orders:
            if not order.items:
                continue
                
            for item in order.items:
                # OrderItem is an EmbeddedDocument, access fields directly
                product_id = str(item.product_id) if hasattr(item, 'product_id') else None
                if not product_id:
                    continue
                
                if product_id not in product_stats:
                    product_stats[product_id] = {
                        'product_id': product_id,
                        'quantity': 0,
                        'revenue': 0,
                        'name': item.name if hasattr(item, 'name') else 'Unknown Product'
                    }
                
                # Use 'qty' and 'price_sale' from OrderItem model
                quantity = item.qty if hasattr(item, 'qty') else 0
                price = float(item.price_sale) if hasattr(item, 'price_sale') else 0
                
                product_stats[product_id]['quantity'] += quantity
                product_stats[product_id]['revenue'] += quantity * price
        
        # Sort and limit
        products_list = list(product_stats.values())
        if sort_by == "quantity":
            products_list.sort(key=lambda x: x['quantity'], reverse=True)
        else:  # revenue
            products_list.sort(key=lambda x: x['revenue'], reverse=True)
        
        top_products = products_list[:limit]
        
        # Round revenue
        for product in top_products:
            product['revenue'] = round(product['revenue'], 2)
        
        return api_success(
            "Top products retrieved successfully",
            {
                "sort_by": sort_by,
                "limit": limit,
                "products": top_products
            }
        )

    @action(detail=False, methods=["get"], url_path="user-demographics")
    def user_demographics(self, request):
        """
        GET /api/v1/admin-stats/user-demographics
        
        Thống kê nhân khẩu học người dùng:
        - Phân bố theo giới tính
        - Phân bố theo độ tuổi
        
        Response format phù hợp cho Pie Chart hoặc Donut Chart
        """
        # Gender distribution
        male_count = User.objects.filter(gender__iexact="male").count()
        female_count = User.objects.filter(gender__iexact="female").count()
        other_count = User.objects.filter(gender__exists=True, gender__nin=["male", "Male", "female", "Female"]).count()
        no_gender_count = User.objects.filter(gender__exists=False).count() + User.objects.filter(gender=None).count()
        
        # Age distribution
        users_with_age = User.objects.filter(age__exists=True, age__ne=None)
        age_groups = {
            "Under 18": 0,
            "18-24": 0,
            "25-34": 0,
            "35-44": 0,
            "45-54": 0,
            "55+": 0
        }
        
        for user in users_with_age:
            age = user.age
            if age < 18:
                age_groups["Under 18"] += 1
            elif age <= 24:
                age_groups["18-24"] += 1
            elif age <= 34:
                age_groups["25-34"] += 1
            elif age <= 44:
                age_groups["35-44"] += 1
            elif age <= 54:
                age_groups["45-54"] += 1
            else:
                age_groups["55+"] += 1
        
        return api_success(
            "User demographics retrieved successfully",
            {
                "gender": {
                    "labels": ["Male", "Female", "Other", "Not Specified"],
                    "data": [male_count, female_count, other_count, no_gender_count]
                },
                "age_groups": {
                    "labels": list(age_groups.keys()),
                    "data": list(age_groups.values())
                }
            }
        )

    @action(detail=False, methods=["get"], url_path="order-status")
    def order_status(self, request):
        """
        GET /api/v1/admin-stats/order-status
        
        Thống kê trạng thái đơn hàng
        
        Response format phù hợp cho Pie Chart hoặc Donut Chart
        """
        paid_count = Order.objects.filter(is_paid=True).count()
        unpaid_count = Order.objects.filter(is_paid=False).count()
        delivered_count = Order.objects.filter(is_delivered=True).count()
        cancelled_count = Order.objects.filter(is_cancelled=True).count()
        
        # Processing = not delivered and not cancelled
        processing_count = Order.objects.filter(
            is_delivered=False,
            is_cancelled=False
        ).count()
        
        return api_success(
            "Order status statistics retrieved successfully",
            {
                "payment_status": {
                    "labels": ["Paid", "Unpaid"],
                    "data": [paid_count, unpaid_count]
                },
                "fulfillment_status": {
                    "labels": ["Delivered", "Processing", "Cancelled"],
                    "data": [delivered_count, processing_count, cancelled_count]
                }
            }
        )

    @action(detail=False, methods=["get"], url_path="recent-orders")
    def recent_orders(self, request):
        """
        GET /api/v1/admin-stats/recent-orders?limit=10
        
        Danh sách đơn hàng gần đây
        
        Response format phù hợp cho Table
        """
        limit = int(request.query_params.get("limit", 10))
        
        from apps.orders.mongo_serializers import OrderSerializer
        
        recent_orders = Order.objects.all().order_by("-created_at").limit(limit)
        serializer = OrderSerializer(recent_orders, many=True)
        
        return api_success(
            "Recent orders retrieved successfully",
            {
                "limit": limit,
                "orders": serializer.data
            }
        )

    @action(detail=False, methods=["get"], url_path="product-categories")
    def product_categories(self, request):
        """
        GET /api/v1/admin-stats/product-categories
        
        Phân bố sản phẩm theo danh mục
        
        Response format phù hợp cho Pie Chart hoặc Bar Chart
        """
        # Group by masterCategory
        pipeline = [
            {
                "$group": {
                    "_id": "$masterCategory",
                    "count": {"$sum": 1}
                }
            },
            {
                "$sort": {"count": -1}
            }
        ]
        
        category_stats = list(Product.objects.aggregate(*pipeline))
        
        labels = [item['_id'] or 'Unknown' for item in category_stats]
        data = [item['count'] for item in category_stats]
        
        return api_success(
            "Product category distribution retrieved successfully",
            {
                "labels": labels,
                "data": data
            }
        )

    @action(detail=False, methods=["get"], url_path="sales-by-gender")
    def sales_by_gender(self, request):
        """
        GET /api/v1/admin-stats/sales-by-gender
        
        Doanh thu theo giới tính sản phẩm
        
        Response format phù hợp cho Pie Chart hoặc Bar Chart
        """
        gender_revenue = {}
        
        paid_orders = Order.objects.filter(is_paid=True)
        for order in paid_orders:
            if not order.items:
                continue
            
            for item in order.items:
                # OrderItem is an EmbeddedDocument, access fields directly
                product_id = item.product_id if hasattr(item, 'product_id') else None
                if not product_id:
                    continue
                
                try:
                    # Product.id is IntField, not ObjectId
                    product = Product.objects.get(id=int(product_id))
                    gender = product.gender if product.gender else "Unisex"
                    
                    if gender not in gender_revenue:
                        gender_revenue[gender] = 0
                    
                    # Use 'qty' and 'price_sale' from OrderItem model
                    quantity = item.qty if hasattr(item, 'qty') else 0
                    price = float(item.price_sale) if hasattr(item, 'price_sale') else 0
                    gender_revenue[gender] += quantity * price
                except Exception as e:
                    # Skip if product not found
                    continue
        
        labels = list(gender_revenue.keys())
        data = [round(gender_revenue[label], 2) for label in labels]
        
        return api_success(
            "Sales by gender retrieved successfully",
            {
                "labels": labels,
                "data": data
            }
        )

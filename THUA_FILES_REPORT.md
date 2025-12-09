#### ❌ **CÓ THỂ XÓA**:
- `apps/recommendations/admin.py` - Không liên quan
- `apps/recommendations/apps.py` - Config app
- `apps/recommendations/migrations/` - Migration files
- `apps/recommendations/models.py` - Django models (không được sử dụng)
- `apps/recommendations/mongo_models.py` - Có thể thừa nếu không được sử dụng
- `apps/recommendations/mongo_serializers.py` - Có thể thừa nếu không được sử dụng
- `apps/recommendations/mongo_services.py` - Cần kiểm tra xem có được sử dụng không
- `apps/recommendations/mongo_views.py` - **THỪA** - API endpoints không liên quan đến hybrid/recommend
- `apps/recommendations/serializers.py` - Django serializers (không được sử dụng)
- `apps/recommendations/services.py` - Cần kiểm tra xem có được sử dụng không
- `apps/recommendations/tasks.py` - Celery tasks (có thể thừa)
- `apps/recommendations/tests/` - **THỪA** - Test files không cần thiết cho production
- `apps/recommendations/urls.py` - **THỪA** - API endpoints không liên quan đến hybrid/recommend
- `apps/recommendations/views.py` - Django views (không được sử dụng)
- `apps/recommendations/management/commands/` - **THỪA** - Management commands không liên quan đến app_recommendation.py hoặc hybrid/recommend

---
"""
Django management command để kiểm tra dữ liệu training
Sử dụng: python manage.py check_training_data
"""

from django.core.management.base import BaseCommand
from apps.users.models import UserInteraction as SqlInteraction
from apps.users.models import User as SqlUser
from apps.products.models import Product as SqlProduct


class Command(BaseCommand):
    help = 'Kiểm tra dữ liệu training trong database (SQL và MongoDB)'

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write(self.style.SUCCESS("KIỂM TRA DỮ LIỆU TRAINING"))
        self.stdout.write("=" * 60)

        # Kiểm tra SQL Database
        self.stdout.write("\n📊 SQL DATABASE:")
        self.stdout.write("-" * 60)

        sql_interactions = SqlInteraction.objects.all()
        sql_users = SqlUser.objects.all()
        sql_products = SqlProduct.objects.all()

        self.stdout.write(f"✅ Users: {sql_users.count()}")
        self.stdout.write(f"✅ Products: {sql_products.count()}")
        self.stdout.write(f"✅ Interactions: {sql_interactions.count()}")

        if sql_interactions.exists():
            self.stdout.write("\n📝 Sample Interactions (SQL):")
            for i, interaction in enumerate(sql_interactions[:5]):
                self.stdout.write(
                    f"  - User: {interaction.user_id}, Product: {interaction.product_id}, "
                    f"Type: {interaction.interaction_type}"
                )

        # Kiểm tra MongoDB
        self.stdout.write("\n📊 MONGODB:")
        self.stdout.write("-" * 60)

        try:
            from apps.users.mongo_models import UserInteraction as MongoInteraction
            from apps.users.mongo_models import User as MongoUser
            from apps.products.mongo_models import Product as MongoProduct

            mongo_interactions = MongoInteraction.objects.all()
            mongo_users = MongoUser.objects.all()
            mongo_products = MongoProduct.objects.all()

            self.stdout.write(f"✅ Users: {mongo_users.count()}")
            self.stdout.write(f"✅ Products: {mongo_products.count()}")
            self.stdout.write(f"✅ Interactions: {mongo_interactions.count()}")

            if mongo_interactions.count() > 0:
                self.stdout.write("\n📝 Sample Interactions (MongoDB):")
                for i, interaction in enumerate(mongo_interactions[:5]):
                    self.stdout.write(
                        f"  - User: {interaction.user_id}, Product: {interaction.product_id}, "
                        f"Type: {interaction.interaction_type}"
                    )

            # Kiểm tra chi tiết interactions
            if mongo_interactions.count() > 0:
                self.stdout.write("\n📈 Interaction Statistics (MongoDB):")
                interaction_types = {}
                for interaction in mongo_interactions:
                    itype = interaction.interaction_type
                    interaction_types[itype] = interaction_types.get(itype, 0) + 1

                for itype, count in interaction_types.items():
                    self.stdout.write(f"  - {itype}: {count}")

                # Kiểm tra unique users và products
                unique_users = set()
                unique_products = set()
                for interaction in mongo_interactions:
                    if interaction.user_id:
                        unique_users.add(str(interaction.user_id))
                    if interaction.product_id:
                        unique_products.add(str(interaction.product_id))

                self.stdout.write(f"\n  - Unique Users: {len(unique_users)}")
                self.stdout.write(f"  - Unique Products: {len(unique_products)}")

        except Exception as e:
            self.stdout.write(
                self.style.WARNING(f"❌ Error connecting to MongoDB: {e}")
            )
            self.stdout.write("   (MongoDB may not be configured or has no data)")

        # Tổng kết
        self.stdout.write("\n" + "=" * 60)
        self.stdout.write("TỔNG KẾT:")
        self.stdout.write("=" * 60)

        total_interactions = sql_interactions.count()
        try:
            total_interactions += mongo_interactions.count()
        except:
            pass

        if total_interactions == 0:
            self.stdout.write(
                self.style.WARNING("⚠️  KHÔNG CÓ INTERACTIONS TRONG DATABASE!")
            )
            self.stdout.write("\n💡 Để training hoạt động, bạn cần:")
            self.stdout.write("   1. Thêm dữ liệu interactions vào MongoDB hoặc SQL database")
            self.stdout.write("   2. Đảm bảo có ít nhất một số users và products")
            self.stdout.write("   3. Tạo interactions giữa users và products")
        else:
            self.stdout.write(
                self.style.SUCCESS(f"✅ Tìm thấy {total_interactions} interactions")
            )
            self.stdout.write("   Training sẽ hoạt động với dữ liệu này")

        self.stdout.write("\n" + "=" * 60)


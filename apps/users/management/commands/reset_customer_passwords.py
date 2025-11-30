from django.core.management.base import BaseCommand

from apps.users.mongo_models import User

class Command(BaseCommand):
    help = (
        'Cập nhật mật khẩu "Customer123!" cho tất cả User có isAdmin == False'
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Chỉ hiển thị số lượng user sẽ được cập nhật mà không ghi vào DB",
        )

    def handle(self, *args, **options):
        target_users = User.objects(is_admin=False)
        count = target_users.count()

        if count == 0:
            self.stdout.write(self.style.WARNING("Không tìm thấy user nào cần cập nhật."))
            return

        if options["dry_run"]:
            self.stdout.write(
                self.style.SUCCESS(
                    f"Dry run: sẽ có {count} user không phải admin được cập nhật."
                )
            )
            return

        for user in target_users:
            user.set_password("Customer123!")
            user.save()

        self.stdout.write(
            self.style.SUCCESS(
                f"Đã cập nhật mật khẩu cho {count} user không phải admin."
            )
        )


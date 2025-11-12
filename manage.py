#!/usr/bin/env python
"""Entrypoint cho các lệnh quản trị Django."""

import os
import sys


def main() -> None:
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "novaware.settings")
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Không thể import Django. Đảm bảo rằng môi trường ảo và requirements đã được cài đặt."
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()


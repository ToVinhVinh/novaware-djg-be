from __future__ import annotations

import csv
import logging
import os
from pathlib import Path

from django.core.management.base import BaseCommand
from django.conf import settings

logger = logging.getLogger(__name__)

class Command(BaseCommand):
    help = "Sync products from CSV file to MongoDB (only specified fields)"

    def add_arguments(self, parser):
        parser.add_argument(
            '--csv-file',
            type=str,
            default='exports/products.csv',
            help='Path to the CSV file (default: exports/products.csv)',
        )
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Run without actually saving to MongoDB (for testing)',
        )

    def handle(self, *args, **options):
        csv_file_path = options['csv_file']
        dry_run = options['dry_run']

        if not os.path.isabs(csv_file_path):
            base_dir = Path(settings.BASE_DIR) if isinstance(settings.BASE_DIR, str) else settings.BASE_DIR
            csv_file_path = str(base_dir / csv_file_path)

        self.stdout.write("=" * 80)
        self.stdout.write("Syncing Products from CSV to MongoDB")
        self.stdout.write("=" * 80)
        self.stdout.write(f"CSV File: {csv_file_path}")
        self.stdout.write(f"Dry Run: {dry_run}")
        self.stdout.write("=" * 80)

        if not os.path.exists(csv_file_path):
            self.stdout.write(self.style.ERROR(f"Error: CSV file not found at {csv_file_path}"))
            return

        try:
            from config.mongodb import connect_mongodb
            connect_mongodb()
            from apps.products.mongo_models import Product
        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error connecting to MongoDB: {e}"))
            return

        fields_to_sync = [
            'id',
            'gender',
            'masterCategory',
            'subCategory',
            'articleType',
            'baseColour',
            'season',
            'year',
            'usage',
            'productDisplayName'
        ]

        stats = {
            'total': 0,
            'created': 0,
            'updated': 0,
            'errors': 0,
            'skipped': 0
        }

        try:
            with open(csv_file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)

                for row_num, row in enumerate(reader, start=2):
                    stats['total'] += 1

                    try:
                        product_id = row.get('id', '').strip()
                        if not product_id:
                            self.stdout.write(
                                self.style.WARNING(f"Row {row_num}: Skipping - missing ID")
                            )
                            stats['skipped'] += 1
                            continue

                        try:
                            product_id = int(product_id)
                        except ValueError:
                            self.stdout.write(
                                self.style.WARNING(f"Row {row_num}: Skipping - invalid ID: {product_id}")
                            )
                            stats['skipped'] += 1
                            continue

                        product_data = {}
                        for field in fields_to_sync:
                            value = row.get(field, '').strip()

                            if field == 'year':
                                if value:
                                    try:
                                        product_data[field] = int(value)
                                    except ValueError:
                                        self.stdout.write(
                                            self.style.WARNING(
                                                f"Row {row_num}: Invalid year '{value}', skipping year field"
                                            )
                                        )
                                        product_data[field] = None
                                else:
                                    product_data[field] = None
                            else:
                                product_data[field] = value if value else None

                        existing_product = None
                        try:
                            existing_product = Product.objects(id=product_id).first()
                        except Exception as e:
                            self.stdout.write(
                                self.style.WARNING(f"Row {row_num}: Error checking existing product: {e}")
                            )

                        if existing_product:
                            if not dry_run:
                                for field, value in product_data.items():
                                    if field != 'id':
                                        setattr(existing_product, field, value)
                                existing_product.save()
                            stats['updated'] += 1
                            if row_num % 100 == 0:
                                self.stdout.write(f"Processed {row_num} rows... (Updated: {stats['updated']}, Created: {stats['created']})")
                        else:
                            product_data['name'] = product_data.get('productDisplayName', f'Product {product_id}')
                            product_data['slug'] = f"product-{product_id}"

                            if not dry_run:
                                Product(**product_data).save()
                            stats['created'] += 1
                            if row_num % 100 == 0:
                                self.stdout.write(f"Processed {row_num} rows... (Updated: {stats['updated']}, Created: {stats['created']})")

                    except Exception as e:
                        stats['errors'] += 1
                        self.stdout.write(
                            self.style.ERROR(f"Row {row_num}: Error processing product: {e}")
                        )
                        logger.exception(f"Error processing row {row_num}")

        except Exception as e:
            self.stdout.write(self.style.ERROR(f"Error reading CSV file: {e}"))
            logger.exception("Error reading CSV file")
            return

        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("Sync Summary")
        self.stdout.write("=" * 80)
        self.stdout.write(f"Total rows processed: {stats['total']}")
        self.stdout.write(f"Products created: {stats['created']}")
        self.stdout.write(f"Products updated: {stats['updated']}")
        self.stdout.write(f"Errors: {stats['errors']}")
        self.stdout.write(f"Skipped: {stats['skipped']}")

        if dry_run:
            self.stdout.write(self.style.WARNING("\n*** DRY RUN MODE - No data was saved to MongoDB ***"))
        else:
            self.stdout.write(self.style.SUCCESS("\n✓ Sync completed successfully!"))

        self.stdout.write("=" * 80)


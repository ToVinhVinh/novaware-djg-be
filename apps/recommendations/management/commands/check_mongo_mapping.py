"""Management command to check MongoDB-Django mapping for recommendations."""

from __future__ import annotations

import logging
from django.core.management.base import BaseCommand

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = "Check MongoDB-Django mapping for users, products, and interactions"

    def handle(self, *args, **options):
        self.stdout.write("=" * 80)
        self.stdout.write("Checking MongoDB-Django Mapping")
        self.stdout.write("=" * 80)
        
        # Check Users
        self.stdout.write("\n1. Checking Users...")
        self._check_users()
        
        # Check Products
        self.stdout.write("\n2. Checking Products...")
        self._check_products()
        
        # Check Interactions
        self.stdout.write("\n3. Checking Interactions...")
        self._check_interactions()
        
        self.stdout.write("\n" + "=" * 80)
        self.stdout.write("Check completed!")
        self.stdout.write("=" * 80)

    def _check_users(self):
        """Check user mapping between Django and MongoDB."""
        try:
            from apps.users.models import User as DjangoUser
            from apps.users.mongo_models import User as MongoUser
            
            django_users = DjangoUser.objects.all()
            mongo_users = MongoUser.objects.all()
            
            self.stdout.write(f"  Django users: {django_users.count()}")
            self.stdout.write(f"  MongoDB users: {mongo_users.count()}")
            
            # Check email matching
            django_emails = set(django_users.values_list('email', flat=True))
            mongo_emails = set(mongo_users.values_list('email', flat=True))
            
            matched_emails = django_emails & mongo_emails
            self.stdout.write(f"  Matched emails: {len(matched_emails)}")
            self.stdout.write(f"  Django-only emails: {len(django_emails - mongo_emails)}")
            self.stdout.write(f"  MongoDB-only emails: {len(mongo_emails - django_emails)}")
            
            if matched_emails:
                self.stdout.write(f"  ✓ Found {len(matched_emails)} users that can be mapped")
            else:
                self.stdout.write(f"  ✗ No matching users found!")
                
        except Exception as e:
            self.stdout.write(f"  ✗ Error checking users: {e}")

    def _check_products(self):
        """Check product mapping between Django and MongoDB."""
        try:
            from apps.products.models import Product as DjangoProduct
            from apps.products.mongo_models import Product as MongoProduct
            
            django_products = DjangoProduct.objects.all()
            mongo_products = MongoProduct.objects.all()
            
            self.stdout.write(f"  Django products: {django_products.count()}")
            self.stdout.write(f"  MongoDB products: {mongo_products.count()}")
            
            # Check slug matching
            django_slugs = set(django_products.exclude(slug__isnull=True).exclude(slug='').values_list('slug', flat=True))
            mongo_slugs = set()
            for mp in mongo_products:
                if hasattr(mp, 'slug') and mp.slug:
                    mongo_slugs.add(mp.slug)
            
            matched_slugs = django_slugs & mongo_slugs
            self.stdout.write(f"  Matched slugs: {len(matched_slugs)}")
            self.stdout.write(f"  Django-only slugs: {len(django_slugs - mongo_slugs)}")
            self.stdout.write(f"  MongoDB-only slugs: {len(mongo_slugs - django_slugs)}")
            
            # Check amazon_asin matching
            django_asins = set(
                django_products.exclude(amazon_asin__isnull=True)
                .exclude(amazon_asin='')
                .values_list('amazon_asin', flat=True)
            )
            mongo_asins = set()
            for mp in mongo_products:
                if hasattr(mp, 'amazon_asin') and mp.amazon_asin:
                    mongo_asins.add(mp.amazon_asin)
            
            matched_asins = django_asins & mongo_asins
            self.stdout.write(f"  Matched amazon_asins: {len(matched_asins)}")
            
            # Check name matching
            django_names = set(django_products.exclude(name__isnull=True).exclude(name='').values_list('name', flat=True))
            mongo_names = set()
            for mp in mongo_products:
                if hasattr(mp, 'name') and mp.name:
                    mongo_names.add(mp.name)
            
            matched_names = django_names & mongo_names
            self.stdout.write(f"  Matched names: {len(matched_names)}")
            
            total_matched = len(matched_slugs) + len(matched_asins) + len(matched_names)
            if total_matched > 0:
                self.stdout.write(f"  ✓ Found {total_matched} products that can be mapped")
            else:
                self.stdout.write(f"  ✗ No matching products found!")
                
        except Exception as e:
            self.stdout.write(f"  ✗ Error checking products: {e}")

    def _check_interactions(self):
        """Check interactions from both Django and MongoDB."""
        try:
            from apps.users.models import UserInteraction as DjangoInteraction
            from apps.users.mongo_models import UserInteraction as MongoInteraction
            
            django_interactions = DjangoInteraction.objects.all()
            mongo_interactions = MongoInteraction.objects.all()
            
            self.stdout.write(f"  Django interactions: {django_interactions.count()}")
            self.stdout.write(f"  MongoDB interactions: {mongo_interactions.count()}")
            
            # Check unique users in interactions
            django_user_ids = set(django_interactions.values_list('user_id', flat=True))
            mongo_user_ids = set()
            for mi in mongo_interactions:
                if mi.user_id:
                    mongo_user_ids.add(str(mi.user_id))
            
            self.stdout.write(f"  Unique Django user IDs in interactions: {len(django_user_ids)}")
            self.stdout.write(f"  Unique MongoDB user IDs in interactions: {len(mongo_user_ids)}")
            
            # Check unique products in interactions
            django_product_ids = set(django_interactions.values_list('product_id', flat=True))
            mongo_product_ids = set()
            for mi in mongo_interactions:
                if mi.product_id:
                    mongo_product_ids.add(str(mi.product_id))
            
            self.stdout.write(f"  Unique Django product IDs in interactions: {len(django_product_ids)}")
            self.stdout.write(f"  Unique MongoDB product IDs in interactions: {len(mongo_product_ids)}")
            
            if mongo_interactions.count() > 0:
                self.stdout.write(f"  ✓ Found {mongo_interactions.count()} MongoDB interactions")
            else:
                self.stdout.write(f"  ✗ No MongoDB interactions found!")
                
        except Exception as e:
            self.stdout.write(f"  ✗ Error checking interactions: {e}")


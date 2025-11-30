from __future__ import annotations

from decimal import Decimal

from django.test import TestCase

from apps.products.models import Category, Product
from apps.recommendations.common.filters import CandidateFilter
from apps.recommendations.common.outfit import OutfitBuilder
from apps.users.models import User, UserInteraction

class RecommendationPipelineTests(TestCase):

    @classmethod
    def setUpTestData(cls):
        cls.category_tops = Category.objects.create(name="Tops")
        cls.category_bottoms = Category.objects.create(name="Bottoms")
        cls.category_shoes = Category.objects.create(name="Shoes")
        cls.category_accessories = Category.objects.create(name="Accessories")

        cls.merchant = User.objects.create_user(
            email="merchant@example.com",
            password="password",
            gender=User.Gender.OTHER,
        )
        cls.customer = User.objects.create_user(
            email="customer@example.com",
            password="password",
            gender=User.Gender.FEMALE,
            age=27,
        )

        base_kwargs = {
            "user": cls.merchant,
            "description": "Test product",
            "images": [],
            "price": Decimal("100.00"),
            "outfit_tags": ["streetwear"],
            "style_tags": ["streetwear"],
            "count_in_stock": 10,
        }

        cls.current_top = Product.objects.create(
            category=cls.category_tops,
            name="Female Top Active",
            slug="female-top-active",
            gender=Product.GenderChoices.FEMALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.TOPS,
            **base_kwargs,
        )

        cls.current_shoe = Product.objects.create(
            category=cls.category_shoes,
            name="Unisex Sneaker Active",
            slug="unisex-sneaker-active",
            gender=Product.GenderChoices.UNISEX,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.SHOES,
            **base_kwargs,
        )

        cls.history_top = Product.objects.create(
            category=cls.category_tops,
            name="Female Top History",
            slug="female-top-history",
            gender=Product.GenderChoices.FEMALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.TOPS,
            **base_kwargs,
        )

        cls.candidate_top = Product.objects.create(
            category=cls.category_tops,
            name="Female Top Candidate",
            slug="female-top-candidate",
            gender=Product.GenderChoices.FEMALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.TOPS,
            **base_kwargs,
        )

        cls.candidate_bottom = Product.objects.create(
            category=cls.category_bottoms,
            name="Female Bottom Candidate",
            slug="female-bottom-candidate",
            gender=Product.GenderChoices.FEMALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.BOTTOMS,
            **base_kwargs,
        )

        cls.candidate_shoes = Product.objects.create(
            category=cls.category_shoes,
            name="Unisex Shoe Candidate",
            slug="unisex-shoe-candidate",
            gender=Product.GenderChoices.UNISEX,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.SHOES,
            **base_kwargs,
        )

        cls.candidate_accessory = Product.objects.create(
            category=cls.category_accessories,
            name="Female Accessory Candidate",
            slug="female-accessory-candidate",
            gender=Product.GenderChoices.FEMALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.ACCESSORIES,
            **base_kwargs,
        )

        cls.male_product = Product.objects.create(
            category=cls.category_bottoms,
            name="Male Bottom Exclusive",
            slug="male-bottom-exclusive",
            gender=Product.GenderChoices.MALE,
            age_group=Product.AgeGroupChoices.ADULT,
            category_type=Product.CategoryTypeChoices.BOTTOMS,
            **base_kwargs,
        )

        UserInteraction.objects.create(
            user=cls.customer,
            product=cls.history_top,
            interaction_type=UserInteraction.InteractionType.PURCHASE,
        )

    def test_candidate_pool_enforces_gender_age_and_exclusions(self):
        context = CandidateFilter.build_context(
            user_id=self.customer.id,
            current_product_id=self.current_top.id,
            top_k_personal=5,
            top_k_outfit=3,
        )

        candidate_ids = {product.id for product in context.candidate_products}

        self.assertEqual(context.resolved_gender, Product.GenderChoices.FEMALE)
        self.assertEqual(context.resolved_age_group, Product.AgeGroupChoices.ADULT)
        self.assertNotIn(self.current_top.id, candidate_ids)
        self.assertNotIn(self.history_top.id, candidate_ids)
        self.assertNotIn(self.male_product.id, candidate_ids)
        self.assertTrue(candidate_ids, "Candidate pool should not be empty")

        genders = {product.gender for product in context.candidate_products}
        age_groups = {product.age_group for product in context.candidate_products}
        self.assertTrue(genders.issubset({Product.GenderChoices.FEMALE, Product.GenderChoices.UNISEX}))
        self.assertEqual(age_groups, {Product.AgeGroupChoices.ADULT})

    def test_outfit_categories_depend_on_current_product(self):
        context_top = CandidateFilter.build_context(
            user_id=self.customer.id,
            current_product_id=self.current_top.id,
            top_k_personal=5,
            top_k_outfit=2,
        )
        scores_top = {product.id: 1.0 for product in context_top.candidate_products}
        outfit_top, _ = OutfitBuilder.build(context_top, scores_top, top_k=2)

        self.assertSetEqual(set(outfit_top.keys()), {"bottoms", "shoes", "accessories"})

        context_shoe = CandidateFilter.build_context(
            user_id=self.customer.id,
            current_product_id=self.current_shoe.id,
            top_k_personal=5,
            top_k_outfit=2,
        )
        scores_shoe = {product.id: 1.0 for product in context_shoe.candidate_products}
        outfit_shoe, _ = OutfitBuilder.build(context_shoe, scores_shoe, top_k=2)

        self.assertSetEqual(set(outfit_shoe.keys()), {"tops", "bottoms", "accessories"})


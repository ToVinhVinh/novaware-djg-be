from __future__ import annotations

from typing import Iterable

MALE_TERMS: set[str] = {"male", "man", "men", "boy", "boys"}
FEMALE_TERMS: set[str] = {"female", "woman", "women", "girl", "girls"}
UNISEX_TERMS: set[str] = {"unisex"}

FILTER_VALUE_MAP = {
    "male": ["Men", "men", "Male", "male", "Boys", "boys"],
    "female": ["Women", "women", "Woman", "woman", "Girls", "girls"],
    "unisex": ["Unisex", "unisex", "UNISEX"],
}

def normalize_gender(value: str | None) -> str:
    normalized = (value or "").strip().lower()
    if not normalized:
        return ""
    if normalized in MALE_TERMS:
        return "male"
    if normalized in FEMALE_TERMS:
        return "female"
    if normalized in UNISEX_TERMS:
        return "unisex"
    return normalized

def gender_filter_values(preferred_gender: str | None) -> list[str]:
    normalized = normalize_gender(preferred_gender)
    allowed: set[str] = set(FILTER_VALUE_MAP["unisex"])

    if normalized == "male":
        allowed.update(FILTER_VALUE_MAP["male"])
    elif normalized == "female":
        allowed.update(FILTER_VALUE_MAP["female"])
    else:
        pass

    return list(allowed)

def genders_compatible(user_gender: str | None, product_gender: str | None) -> bool:
    user_normalized = normalize_gender(user_gender)
    product_normalized = normalize_gender(product_gender)

    if not product_normalized:
        return False
    if product_normalized == "unisex":
        return True
    if not user_normalized:
        return False
    return user_normalized == product_normalized

def is_unisex_gender(value: str | None) -> bool:
    return normalize_gender(value) == "unisex"

def product_gender_matches_any(product_gender: str | None, gender_terms: Iterable[str]) -> bool:
    product_normalized = normalize_gender(product_gender)
    return product_normalized in {normalize_gender(term) for term in gender_terms}


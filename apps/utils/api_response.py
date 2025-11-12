"""Common helpers for building API responses."""

from __future__ import annotations

import math
from typing import Iterable, Tuple

from rest_framework import response, status

DEFAULT_PAGE_KEYS = ("page", "pageNumber")
DEFAULT_PAGE_SIZE_KEYS = ("page_size", "pageSize", "perPage")


def _first_query_value(request, keys: Iterable[str]) -> str | None:
    for key in keys:
        value = request.query_params.get(key)
        if value is not None:
            return value
    return None


def _coerce_positive_int(value: str | None, default: int) -> int:
    try:
        coerced = int(value) if value is not None else default
    except (TypeError, ValueError):
        coerced = default
    return max(coerced, 1)


def get_pagination_params(
    request,
    page_keys: Iterable[str] = DEFAULT_PAGE_KEYS,
    page_size_keys: Iterable[str] = DEFAULT_PAGE_SIZE_KEYS,
    default_page: int = 1,
    default_page_size: int = 20,
) -> Tuple[int, int]:
    """Extract pagination params from the request query params."""
    page_raw = _first_query_value(request, page_keys)
    page_size_raw = _first_query_value(request, page_size_keys)

    page = _coerce_positive_int(page_raw, default_page)
    page_size = _coerce_positive_int(page_size_raw, default_page_size)

    return page, page_size


def paginate_queryset(queryset, page: int, page_size: int):
    """Paginate a queryset-like object that supports slicing and count()."""
    total_count = queryset.count()
    total_pages = max(math.ceil(total_count / page_size), 1) if total_count else 1

    current_page = min(page, total_pages) if total_pages else page
    start = (current_page - 1) * page_size
    end = start + page_size
    items = queryset[start:end]

    return items, total_count, total_pages, current_page, page_size


def api_success(message: str, data, status_code: int = status.HTTP_200_OK):
    """Return a standardized success response."""
    return response.Response(
        {
            "status": "success",
            "message": message,
            "data": data,
        },
        status=status_code,
    )


def api_error(
    message: str,
    data=None,
    *,
    status_code: int = status.HTTP_400_BAD_REQUEST,
):
    """Return a standardized error response."""
    return response.Response(
        {
            "status": "error",
            "message": message,
            "data": data,
        },
        status=status_code,
    )


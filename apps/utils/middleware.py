"""Custom middleware for Novaware Django project."""

from django.middleware.common import CommonMiddleware


class NoAppendSlashForAPIMiddleware(CommonMiddleware):
    """
    Middleware that normalizes API URLs by removing trailing slashes.
    
    This ensures that both /api/v1/products/ and /api/v1/products work correctly,
    since the API router is configured with trailing_slash=False.
    """

    def process_request(self, request):
        """Normalize API URLs by removing trailing slashes before routing."""
        # For API routes, remove trailing slash to match router configuration
        if request.path_info.startswith('/api/') and request.path_info != '/api/' and request.path_info.endswith('/'):
            # Remove trailing slash except for root /api/ path
            # Modify PATH_INFO in META which is used for URL resolution
            request.META['PATH_INFO'] = request.path_info.rstrip('/')
            # Also update the path property by clearing the cached value
            if hasattr(request, '_cached_path'):
                delattr(request, '_cached_path')
        return None

    def process_response(self, request, response):
        """Override to skip APPEND_SLASH redirect for API routes."""
        # Skip APPEND_SLASH redirect for API routes
        # This prevents the RuntimeError when PUT/PATCH requests don't have trailing slashes
        if request.path.startswith('/api/'):
            # For API routes, don't try to append slash
            # This allows PUT/PATCH requests without trailing slashes to work
            return response
        # For non-API routes, use the default CommonMiddleware behavior
        return super().process_response(request, response)


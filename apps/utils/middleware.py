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
        if request.path_info.startswith('/api/') and request.path_info != '/api/' and request.path_info.endswith('/'):
            request.META['PATH_INFO'] = request.path_info.rstrip('/')
            if hasattr(request, '_cached_path'):
                delattr(request, '_cached_path')
        return None

    def process_response(self, request, response):
        """Override to skip APPEND_SLASH redirect for API routes."""
        if request.path.startswith('/api/'):
            return response
        return super().process_response(request, response)


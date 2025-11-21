"""Custom middleware for Novaware Django project."""

from django.middleware.common import CommonMiddleware

class NoAppendSlashForAPIMiddleware(CommonMiddleware):
    """
    Middleware that allows API URLs to work with or without trailing slashes.
    
    This ensures that both /api/v1/products/ and /api/v1/products work correctly.
    """

    def process_request(self, request):
        """Allow both URLs with and without trailing slashes to work."""
        return None

    def process_response(self, request, response):
        """Allow APPEND_SLASH redirect for API routes to support both formats."""
        return super().process_response(request, response)


"""Custom middleware for Novaware Django project."""

from django.urls import resolve, Resolver404
from django.utils.deprecation import MiddlewareMixin


class NoAppendSlashForAPIMiddleware(MiddlewareMixin):
    """
    Middleware that allows API URLs to work with or without trailing slashes.
    
    This ensures that both /api/v1/products/ and /api/v1/products work correctly.
    """

    def process_request(self, request):
        """Handle both URLs with and without trailing slashes for API routes."""
        # Only process API routes
        if not request.path.startswith('/api/'):
            return None
        
        path = request.path
        
        # Try to resolve the current path first
        try:
            resolve(path)
            return None  # Path resolves, continue normally
        except Resolver404:
            pass
        
        # If path doesn't resolve, try the opposite (with/without trailing slash)
        if path.endswith('/') and len(path) > 1:
            # Current path has trailing slash, try without
            alternate_path = path.rstrip('/')
        else:
            # Current path doesn't have trailing slash, try with
            alternate_path = path + '/'
        
        # Try to resolve the alternate path
        try:
            resolve(alternate_path)
            # The alternate path resolves, modify request to use it
            request.path_info = alternate_path
            request.path = alternate_path
            return None
        except Resolver404:
            pass
        
        return None

    def process_response(self, request, response):
        """Don't append slash for API routes - we handle it in process_request."""
        if request.path.startswith('/api/'):
            return response
        from django.middleware.common import CommonMiddleware
        common_middleware = CommonMiddleware()
        return common_middleware.process_response(request, response)


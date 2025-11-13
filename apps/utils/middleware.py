"""Custom middleware for Novaware Django project."""

from django.middleware.common import CommonMiddleware


class NoAppendSlashForAPIMiddleware(CommonMiddleware):
    """
    Middleware that disables APPEND_SLASH for API routes.
    
    This prevents Django from trying to redirect PUT/PATCH requests
    to add trailing slashes, which would lose the request body.
    """

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


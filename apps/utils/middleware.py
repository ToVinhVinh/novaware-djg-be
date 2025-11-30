from django.urls import resolve, Resolver404
from django.utils.deprecation import MiddlewareMixin

class NoAppendSlashForAPIMiddleware(MiddlewareMixin):

    def process_request(self, request):
        if not request.path.startswith('/api/'):
            return None

        path = request.path

        try:
            resolve(path)
            return None
        except Resolver404:
            pass

        if path.endswith('/') and len(path) > 1:
            alternate_path = path.rstrip('/')
        else:
            alternate_path = path + '/'

        try:
            resolve(alternate_path)
            request.path_info = alternate_path
            request.path = alternate_path
            return None
        except Resolver404:
            pass

        return None

    def process_response(self, request, response):
        if request.path.startswith('/api/'):
            return response
        from django.middleware.common import CommonMiddleware
        common_middleware = CommonMiddleware()
        return common_middleware.process_response(request, response)


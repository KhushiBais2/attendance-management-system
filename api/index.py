import os
import sys

# Ensure the project root directory is in the Python module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app


class VercelPathFixMiddleware:
    """
    WSGI Middleware to restore the original client request path on Vercel.
    When Vercel rewrites routes to /api/index, PATH_INFO can be overwritten,
    causing Flask to return 404 for existing routes.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        original_path = (
            environ.get("HTTP_X_NOW_PATHNAME") or
            environ.get("HTTP_X_VERCEL_PATH") or
            environ.get("HTTP_X_FORWARDED_PATH") or
            environ.get("REQUEST_URI")
        )

        if original_path:
            clean_path = original_path.split("?")[0].split("#")[0]
            if clean_path:
                environ["PATH_INFO"] = clean_path
        elif environ.get("PATH_INFO") in ("/api/index", "/api/index.py"):
            environ["PATH_INFO"] = "/"

        return self.wsgi_app(environ, start_response)


# Apply the middleware to the WSGI application
app.wsgi_app = VercelPathFixMiddleware(app.wsgi_app)

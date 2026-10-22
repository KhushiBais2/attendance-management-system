import os
import sys

# Ensure the project root directory is in the Python module search path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import app

# Vercel's Python runtime searches for the WSGI `app` object
if __name__ == "__main__":
    app.run()


web: gunicorn --workers 1 --worker-class sync --timeout 120 --graceful-timeout 90 --max-requests 1 --bind 0.0.0.0:$PORT app:app

FROM python:3.10-slim-bookworm

WORKDIR /app

# Install system dependencies for OpenCV (headless) and MediaPipe
# libgl1-mesa-glx doesn't exist in bookworm-slim — use libgl1 instead
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the API port
EXPOSE 5000

# Set environment variables for production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# Use PORT env var from Render (defaults to 5000 locally)
CMD gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:${PORT:-5000} api:app

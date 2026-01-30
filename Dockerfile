# Production Dockerfile for AI Safety Monitoring
FROM python:3.11-slim

# Prevent Python from writing bytecode and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for OpenCV and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files first for better layer caching
COPY pyproject.toml .
COPY src/ src/

# Install Python dependencies (production only, no dev dependencies)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy remaining application files
COPY main.py .
COPY README.md .

# Expose ports for backend and drone API
EXPOSE 8000 8001

# Default command runs the backend server
# For development with all services, override with: python main.py
CMD ["python", "-m", "src.backend.server"]

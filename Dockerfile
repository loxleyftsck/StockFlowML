# Base image: Python 3.11 Slim (Lightweight)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    Start_Command="python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000"

# Install system dependencies (gcc required for some python packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for caching)
COPY requirements.txt .

# Install python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ src/
COPY models/ models/
COPY data/processed/ data/processed/
# Note: In production, models usually loaded from S3/DVC, not copied.
# But for this Docker demo, we include local models.

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]

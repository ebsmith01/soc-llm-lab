# Simple single-stage Dockerfile for your SOC RAG API

FROM python:3.11-slim

# Don't buffer stdout/stderr, don't write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps for numpy / faiss / etc. (can be trimmed later if not needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy project files into the image
COPY . /app/

# Install your project and dependencies from pyproject.toml
# This assumes `pip install .` works on your machine already
RUN pip install --no-cache-dir .

# Expose FastAPI port
EXPOSE 8000

# Run the API with uvicorn
CMD ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
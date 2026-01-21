# Production Dockerfile for PolicyChat
# Uses Python 3.12 to avoid tokenizers build issues on Python 3.13

FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install minimal system dependencies if needed for scikit-learn/numpy
# These are typically needed for scipy/numpy/scikit-learn wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (leverage Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir to keep image size small
# tokenizers will install from pre-built wheels on Python 3.12
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Health check for Streamlit
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit app
CMD ["streamlit", "run", "ux.py", "--server.port=8501", "--server.address=0.0.0.0"]

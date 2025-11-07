# ASR Service Dockerfile
# Multi-stage build for optimized image size

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    wget \
    curl \
    git \
    # Audio processing libraries
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    # Required for faster-whisper
    libcublas-12-0 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install
WORKDIR /app
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install Silero VAD from git (if not available on PyPI)
RUN pip install --no-cache-dir git+https://github.com/snakers4/silero-vad.git || echo "Silero VAD install skipped"

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Set working directory
WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    libsndfile1 \
    ffmpeg \
    libcublas-12-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY server.py .
COPY run.py .

# Create directories for model cache
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /root/.cache/torch

# Set environment variables for model paths
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Expose ports
# 50051 for gRPC
# 8050 for REST API (run.py)
EXPOSE 50051
EXPOSE 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command (can be overridden)
CMD ["python", "run.py"]

# Alternative: Run as gRPC server
# CMD ["python", "server.py"]

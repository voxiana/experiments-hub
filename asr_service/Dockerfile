# ASR Service Dockerfile
# Multi-stage build for optimized image size
# Using CUDA 12.4 with cuDNN 9 (required by faster-whisper/CTranslate2)

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    build-essential \
    pkg-config \
    wget \
    curl \
    git \
    # Audio processing libraries
    libsndfile1 \
    libsndfile1-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3.10 -m pip install --upgrade pip setuptools wheel

# Create virtual environment
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set pip timeout and retry for large packages
ENV PIP_DEFAULT_TIMEOUT=100

# Copy requirements and install
WORKDIR /app
COPY requirements.txt .

# Install PyTorch first with CUDA 12.4 support
# Note: Using cu124 wheels
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies (excluding PyTorch lines)
RUN grep -v "^torch" requirements.txt | \
    grep -v "^--index-url" | \
    grep -v "^#" | \
    grep -v "^$" > requirements_filtered.txt && \
    pip install --no-cache-dir -r requirements_filtered.txt

# Install Silero VAD from git
RUN pip install --no-cache-dir git+https://github.com/snakers4/silero-vad.git

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA and cuDNN environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
# Include cuDNN library paths (both lib64 and lib for compatibility)
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${CUDA_HOME}/lib64/stubs:${LD_LIBRARY_PATH}
# cuDNN specific paths
ENV CUDNN_PATH=${CUDA_HOME}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
# Suppress cuDNN version warnings (if using cuDNN 8 with cuDNN 9 expectations)
ENV CUDNN_LOGINFO_DBG=0
ENV CUDNN_LOGDEST_DBG=""

# Set working directory
WORKDIR /app

# Install runtime system dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-venv \
    libsndfile1 \
    ffmpeg \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY server.py .
COPY run.py .

# Create directories for model cache
RUN mkdir -p /root/.cache/huggingface && \
    mkdir -p /root/.cache/torch && \
    chmod -R 777 /root/.cache

# Set environment variables for model paths
ENV HF_HOME=/root/.cache/huggingface
ENV TORCH_HOME=/root/.cache/torch
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface

# Set default model config
ENV WHISPER_MODEL=large-v3
ENV DEVICE=cuda
ENV COMPUTE_TYPE=float16

# Verify CUDA installation (runs after venv is copied)
# This helps debug CUDA/cuDNN issues during build
RUN python3.10 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')" || echo "PyTorch verification skipped (this is OK if running without GPU during build)"

# Expose ports
EXPOSE 50051 8050

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8050/health || exit 1

# Default command
CMD ["python3.10", "run.py"]

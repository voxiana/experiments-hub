# Docker Build Notes

## PyAV Build Issues (pyannote.audio, speechbrain)

The `pyannote.audio` and `speechbrain` packages depend on `av` (PyAV), which requires system libraries to build. These packages are currently **commented out** in requirements.txt files to avoid build failures.

### Why Commented Out?

PyAV requires:
- `pkg-config`
- FFmpeg development libraries (`libavcodec-dev`, `libavformat-dev`, etc.)
- Build tools

Most services don't actually use these packages, so they're disabled by default.

### Enabling Diarization and Emotion Analysis

If you need these features (mainly for the **workers** service for post-call analytics):

#### 1. Update Dockerfile

Add system dependencies before `pip install`:

```dockerfile
# Install system dependencies for PyAV
RUN apt-get update && apt-get install -y \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*
```

#### 2. Uncomment in requirements.txt

```bash
# Uncomment these lines:
pyannote.audio==3.1.1
speechbrain==0.5.16
```

### Example: Workers Service with Diarization

**workers/Dockerfile**:
```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Install Python and PyAV dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libswscale-dev \
    libswresample-dev \
    libavfilter-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy service code
COPY tasks.py /app/tasks.py

CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
```

**workers/requirements.txt**:
```python
# Uncomment these:
pyannote.audio==3.1.1
speechbrain==0.5.16
```

### Services That Might Need These

- ✅ **workers** - Post-call analytics (diarization, emotion)
- ❌ **gateway** - Not needed
- ❌ **nlu_service** - Not needed
- ❌ **tts_service** - Not needed
- ❌ **rag_service** - Not needed
- ❌ **connectors** - Not needed

### Alternative: Pre-built Wheels

To avoid build issues, you can use pre-built wheels:

```bash
# Install from pre-built wheel
pip install av --only-binary=:all:
```

Or use conda:

```bash
conda install -c conda-forge av
```

### Testing the Build

```bash
# Test if build works
docker build -t workers-test ./workers

# If successful, run container
docker run --rm workers-test python3 -c "import pyannote.audio; print('Success!')"
```

## Other Common Build Issues

### 1. CUDA Version Mismatches

**Problem**: CUDA runtime mismatch

**Solution**: Match base image CUDA version with PyTorch CUDA version

```dockerfile
# For PyTorch with CUDA 12.1
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
```

### 2. cuDNN Not Found

**Problem**: cuDNN libraries not in library path

**Solution**: Use `-cudnn8-runtime` or `-cudnn8-devel` base images

### 3. Out of Space During Build

**Problem**: Docker build runs out of space

**Solution**: Use multi-stage builds

```dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04 AS builder
# Build dependencies
RUN pip install --no-cache-dir -r requirements.txt

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
# Copy only what's needed
COPY --from=builder /usr/local/lib/python3.10 /usr/local/lib/python3.10
```

## Recommended Approach

For production:
1. Build a **base image** with all common dependencies
2. Use this base for all services
3. Only add service-specific dependencies in each Dockerfile

Example:

```dockerfile
# base.Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
RUN apt-get update && apt-get install -y python3.10 python3-pip
RUN pip install fastapi uvicorn pydantic aiohttp

# service/Dockerfile
FROM voiceai-base:latest
COPY requirements.txt .
RUN pip install -r requirements.txt
```

This reduces build time and ensures consistency across services.

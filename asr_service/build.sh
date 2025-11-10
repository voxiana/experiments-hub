#!/bin/bash
# Build script for ASR service Docker images

set -e  # Exit on error

echo "=================================================="
echo "ASR Service Docker Build Script"
echo "=================================================="

# Parse arguments
BUILD_TYPE="${1:-gpu}"  # gpu or cpu
TAG="${2:-latest}"

if [ "$BUILD_TYPE" = "gpu" ]; then
    echo "Building GPU-enabled image..."
    echo "- Dockerfile: Dockerfile"
    echo "- Requirements: requirements.txt (CUDA 12.1 + PyTorch)"
    echo ""

    docker build \
        -f Dockerfile \
        -t voiceai-asr:${TAG} \
        -t voiceai-asr:gpu-${TAG} \
        .

    echo ""
    echo "✅ GPU image built successfully!"
    echo "   Tag: voiceai-asr:${TAG}"
    echo "   Tag: voiceai-asr:gpu-${TAG}"
    echo ""
    echo "Run with:"
    echo "  docker run --gpus all -p 8050:8050 voiceai-asr:${TAG}"

elif [ "$BUILD_TYPE" = "cpu" ]; then
    echo "Building CPU-only image..."
    echo "- Dockerfile: Dockerfile.cpu"
    echo "- Requirements: requirements-cpu.txt"
    echo ""

    docker build \
        -f Dockerfile.cpu \
        -t voiceai-asr:cpu-${TAG} \
        .

    echo ""
    echo "✅ CPU image built successfully!"
    echo "   Tag: voiceai-asr:cpu-${TAG}"
    echo ""
    echo "Run with:"
    echo "  docker run -p 8050:8050 voiceai-asr:cpu-${TAG}"

else
    echo "Error: Invalid build type '$BUILD_TYPE'"
    echo "Usage: ./build.sh [gpu|cpu] [tag]"
    echo ""
    echo "Examples:"
    echo "  ./build.sh gpu latest"
    echo "  ./build.sh cpu v1.0.0"
    exit 1
fi

echo ""
echo "=================================================="
echo "Build complete!"
echo "=================================================="

# ASR Service - Streaming Speech Recognition

Standalone ASR (Automatic Speech Recognition) service using **faster-whisper** (Whisper large-v3) with **Silero VAD** for voice activity detection.

## Features

- ✅ **Streaming ASR**: Real-time transcription with 250ms chunks
- ✅ **Voice Activity Detection**: Silero VAD for turn segmentation
- ✅ **Multilingual**: Arabic, English, and 90+ languages
- ✅ **GPU Accelerated**: Optimized with CTranslate2
- ✅ **WebSocket Support**: Real-time bidirectional streaming
- ✅ **REST API**: File upload and base64 transcription
- ✅ **Batch Processing**: Process complete audio files

---

## Quick Start

### 1. Install Dependencies

**For GPU systems with CUDA 12.1:**
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    libsndfile1 \
    ffmpeg

# Install Python dependencies with CUDA support
pip install -r requirements.txt

# Install Silero VAD
pip install git+https://github.com/snakers4/silero-vad.git
```

**For CPU-only systems:**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    libsndfile1 \
    ffmpeg

# Install CPU-only dependencies (smaller, no CUDA)
pip install -r requirements-cpu.txt

# Install Silero VAD
pip install git+https://github.com/snakers4/silero-vad.git
```

**Important Notes:**
- `requirements.txt` installs PyTorch with CUDA 12.1 support (~2.5GB download)
- `requirements-cpu.txt` installs CPU-only PyTorch (~200MB download)
- The files handle dependency conflicts automatically (no onnxruntime conflicts)

### 2. Run the Service

```bash
# Run with defaults (port 8050, GPU if available)
python run.py

# Run on CPU
python run.py --device cpu

# Run with smaller model (faster, less accurate)
python run.py --model medium

# Custom port
python run.py --port 8051

# Dev mode with auto-reload
python run.py --reload
```

### 3. Test the Service

```bash
# Health check
curl http://localhost:8050/health

# Transcribe a file
curl -F "file=@test.wav" http://localhost:8050/transcribe

# Transcribe with Arabic
curl -F "file=@audio_ar.wav" -F "language=ar" http://localhost:8050/transcribe
```

---

## Docker Usage

### Build Image

**GPU Version (recommended for production):**
```bash
# Using build script
./build.sh gpu latest

# Or manually
docker build -f Dockerfile -t voiceai-asr:latest .
```

**CPU Version (no GPU required):**
```bash
# Using build script
./build.sh cpu latest

# Or manually
docker build -f Dockerfile.cpu -t voiceai-asr:cpu-latest .
```

### Run Container (GPU)

**Important**: Always use `--gpus all` to enable CUDA/cuDNN support in the container.

```bash
# Run with GPU support (default: large-v3)
docker run --gpus all \
  -p 8050:8050 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  voiceai-asr:latest

# Run with specific GPU
docker run --gpus '"device=0"' \
  -p 8050:8050 \
  voiceai-asr:latest

# Run with smaller model (faster)
docker run --gpus all \
  -p 8050:8050 \
  -e WHISPER_MODEL=medium \
  voiceai-asr:latest

# Verify CUDA is working inside container
docker run --gpus all --rm voiceai-asr:latest \
  python3.10 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

**Troubleshooting CUDA in Docker:**

If you see cuDNN errors, verify:
1. **NVIDIA Container Toolkit is installed** on the host:
   ```bash
   # Check if installed
   docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
   ```

2. **GPU is accessible**:
   ```bash
   nvidia-smi  # Should show GPU info
   ```

3. **Container has GPU access**:
   ```bash
   docker run --gpus all --rm voiceai-asr:latest nvidia-smi
   ```

### Run Container (CPU)

```bash
# CPU version with optimized settings
docker run \
  -p 8050:8050 \
  voiceai-asr:cpu-latest

# CPU with custom model
docker run \
  -p 8050:8050 \
  -e WHISPER_MODEL=small \
  -e COMPUTE_TYPE=int8 \
  voiceai-asr:cpu-latest
```

---

## API Reference

### REST Endpoints

#### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model": "large-v3",
  "device": "cuda",
  "vad_enabled": true,
  "timestamp": 1699564800.123
}
```

---

#### `POST /transcribe`
Transcribe an audio file

**Request (multipart/form-data):**
- `file`: Audio file (WAV, MP3, FLAC, OGG, etc.)
- `language` (optional): Language code (ar, en, auto)
- `task` (optional): "transcribe" or "translate"

**Example:**
```bash
curl -F "file=@audio.wav" http://localhost:8050/transcribe
```

**Response:**
```json
{
  "text": "مرحبا، كيف يمكنني مساعدتك اليوم؟",
  "language": "ar",
  "language_probability": 0.95,
  "duration_seconds": 3.5,
  "inference_time_seconds": 0.42,
  "segments": [
    {
      "start": 0.0,
      "end": 3.5,
      "text": "مرحبا، كيف يمكنني مساعدتك اليوم؟"
    }
  ]
}
```

---

#### `POST /transcribe/base64`
Transcribe base64-encoded audio

**Request (application/json):**
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAABAAEA...",
  "language": "ar",
  "task": "transcribe"
}
```

**Response:** Same as `/transcribe`

---

#### `WebSocket /ws/transcribe`
Real-time streaming transcription

**Client → Server:**
```json
{
  "language": "auto"
}
```

Then send audio chunks:
```json
{
  "audio": "<base64 PCM 16kHz mono>"
}
```

**Server → Client (interim):**
```json
{
  "type": "interim",
  "text": "Hello, how...",
  "timestamp": 1699564800.123
}
```

**Server → Client (final):**
```json
{
  "type": "final",
  "text": "Hello, how can I help you?",
  "language": "en",
  "timestamp": 1699564801.456
}
```

---

## Configuration

### Environment Variables

```bash
# Model selection
WHISPER_MODEL=large-v3          # large-v3, medium, small, base, tiny
DEVICE=cuda                      # cuda or cpu
COMPUTE_TYPE=float16             # float16, int8, float32

# VAD settings
VAD_THRESHOLD=0.5                # Speech probability threshold
VAD_FRAME_SIZE=512               # Samples per frame (32ms at 16kHz)

# Audio settings
SAMPLE_RATE=16000                # Fixed at 16kHz
CHUNK_DURATION_MS=250            # Streaming chunk size
WHISPER_BEAM_SIZE=1              # 1=fast, 5=accurate
```

### Model Options

| Model | Parameters | VRAM | Speed | Accuracy |
|-------|------------|------|-------|----------|
| **large-v3** | 1550M | 6GB | Slow | Highest |
| **large-v2** | 1550M | 6GB | Slow | Very High |
| **medium** | 769M | 4GB | Medium | High |
| **small** | 244M | 2GB | Fast | Good |
| **base** | 74M | 1GB | Very Fast | Fair |
| **tiny** | 39M | <1GB | Fastest | Low |
| **distil-large-v3** | - | 4GB | Fast | High |

---

## Examples

### Python Client

```python
import requests

# Transcribe a file
with open("audio.wav", "rb") as f:
    response = requests.post(
        "http://localhost:8050/transcribe",
        files={"file": f},
        data={"language": "ar"}
    )

result = response.json()
print(f"Transcript: {result['text']}")
print(f"Language: {result['language']}")
print(f"Duration: {result['duration_seconds']:.1f}s")
```

### WebSocket Client (Python)

```python
import asyncio
import websockets
import json
import base64

async def transcribe_stream():
    uri = "ws://localhost:8050/ws/transcribe"

    async with websockets.connect(uri) as ws:
        # Send config
        await ws.send(json.dumps({"language": "auto"}))

        # Send audio chunks
        with open("audio.wav", "rb") as f:
            while chunk := f.read(4000):  # 250ms at 16kHz
                audio_b64 = base64.b64encode(chunk).decode()
                await ws.send(json.dumps({"audio": audio_b64}))

        # Close
        await ws.send(json.dumps({"type": "close"}))

        # Receive results
        async for message in ws:
            result = json.loads(message)
            print(f"[{result['type']}] {result['text']}")

asyncio.run(transcribe_stream())
```

### cURL Examples

```bash
# Simple transcription
curl -F "file=@audio.wav" http://localhost:8050/transcribe

# Arabic transcription
curl -F "file=@audio_ar.wav" -F "language=ar" \
  http://localhost:8050/transcribe

# Translation to English
curl -F "file=@audio_ar.wav" -F "task=translate" \
  http://localhost:8050/transcribe

# Pretty print JSON
curl -F "file=@audio.wav" http://localhost:8050/transcribe | jq .
```

---

## Performance Tuning

### GPU Optimization

```bash
# Check GPU availability
nvidia-smi

# Run with specific GPU
CUDA_VISIBLE_DEVICES=0 python run.py

# Optimize for throughput (multiple workers)
python run.py --workers 4
```

### Latency vs Accuracy

**Low Latency** (< 300ms per chunk):
```bash
python run.py \
  --model medium \
  --compute-type int8
```

**Balanced** (300-400ms per chunk):
```bash
python run.py \
  --model large-v3 \
  --compute-type float16
```

**High Accuracy** (400-600ms per chunk):
```bash
# Edit server.py:
# WHISPER_BEAM_SIZE = 5

python run.py \
  --model large-v3 \
  --compute-type float16
```

### Memory Optimization

```bash
# Reduce VRAM usage
export WHISPER_MODEL=medium
export COMPUTE_TYPE=int8

# CPU fallback
python run.py --device cpu --compute-type int8
```

---

## Troubleshooting

### Dependency Conflicts (Fixed)

**Problem**: Docker build fails with package conflicts

**Solution**: The updated files fix these issues:
- ✅ Removed duplicate `onnxruntime` and `onnxruntime-gpu` (use only `-gpu` version)
- ✅ PyTorch now installs with explicit CUDA 12.1 support
- ✅ Added `python3.10-venv` package to Dockerfile
- ✅ Changed builder stage to use `-devel` image (has build tools)

**Files Updated**:
- `requirements.txt` - GPU version with CUDA 12.1 PyTorch
- `requirements-cpu.txt` - CPU-only version (no CUDA)
- `Dockerfile` - Fixed for GPU builds
- `Dockerfile.cpu` - Added for CPU-only builds

### GPU Not Detected

```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
python -c "import torch; print(torch.version.cuda)"

# Should output:
# True
# 12.1

# Install CUDA toolkit
# See: https://developer.nvidia.com/cuda-downloads

# Install NVIDIA Container Toolkit (Docker)
# See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
```

### cuDNN 9 DLL Error

**Error**: `Could not locate cudnn_ops64_9.dll` or `Unable to load any of {libcudnn_ops.so.9.1.0...}`

**Cause**: faster-whisper requires cuDNN 9, but the system has cuDNN 8 or cuDNN 9 is not in the library path.

**Solutions**:

1. **Docker (Linux)**:
   ```bash
   # The Dockerfile now uses cuDNN 9 base images
   # If build fails, try the alternative Dockerfile:
   docker build -f Dockerfile.cudnn9 -t voiceai-asr:latest .
   
   # Or manually install cuDNN 9 in the container
   ```

2. **Windows (Local)**:
   ```powershell
   # Download cuDNN 9 from NVIDIA (requires developer account)
   # https://developer.nvidia.com/cudnn
   
   # Extract and copy to CUDA directory:
   # Copy bin/*.dll to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin
   # Copy lib/*.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\lib\x64
   # Copy include/*.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\include
   
   # Add to PATH:
   $env:PATH += ";C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
   ```

3. **Fallback to CPU** (if cuDNN 9 unavailable):
   ```bash
   # Run with CPU mode
   python run.py --device cpu --compute-type int8
   
   # Or in Docker:
   docker run -e DEVICE=cpu -e COMPUTE_TYPE=int8 voiceai-asr:latest
   ```

4. **Verify cuDNN in Docker**:
   ```bash
   # Check cuDNN libraries
   docker run --gpus all --rm voiceai-asr:latest \
     find /usr/local/cuda -name "*cudnn*" -type f
   
   # Check library path
   docker run --gpus all --rm voiceai-asr:latest \
     echo $LD_LIBRARY_PATH
   ```

### Out of Memory

```bash
# Use smaller model
python run.py --model medium

# Use int8 quantization
python run.py --compute-type int8

# Reduce batch size (edit server.py)
# NUM_WORKERS = 2  # Reduce concurrent streams
```

### Slow Transcription

```bash
# Check if GPU is being used
python -c "import torch; print(torch.cuda.is_available())"

# Use GPU
python run.py --device cuda

# Use smaller model
python run.py --model medium

# Reduce beam size (edit server.py)
# WHISPER_BEAM_SIZE = 1
```

### Model Download Issues

```bash
# Pre-download models
python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3')"

# Set cache directory
export HF_HOME=/path/to/cache
export TORCH_HOME=/path/to/cache

# Use local model path
export WHISPER_MODEL=/path/to/model
```

---

## Development

### Run Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest test_asr.py -v

# With coverage
pytest test_asr.py --cov=. --cov-report=html
```

### Code Structure

```
asr_service/
├── server.py           # Core ASR service (VAD + Whisper)
├── run.py              # Standalone server (FastAPI)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
├── README.md           # This file
└── test_asr.py         # Tests (to be created)
```

---

## Benchmarks

**Hardware**: NVIDIA A10 (24GB VRAM)

| Model | Chunk Latency | Full File (10s) | VRAM | Concurrent Streams |
|-------|---------------|-----------------|------|--------------------|
| large-v3 | 250-300ms | 800ms | 6GB | 2-3 |
| medium | 150-200ms | 400ms | 4GB | 4-5 |
| small | 100-150ms | 250ms | 2GB | 8-10 |

**Note**: Latency includes VAD + inference. Real-world performance may vary.

---

## Language Support

**Primary**:
- 🇸🇦 Arabic (ar) - Gulf, MSA
- 🇬🇧 English (en)

**Other Supported Languages**:
- French (fr), Spanish (es), German (de), Italian (it)
- Chinese (zh), Japanese (ja), Korean (ko)
- Hindi (hi), Urdu (ur), Turkish (tr), Persian (fa)
- ... and 80+ more

See: https://github.com/openai/whisper#available-models-and-languages

---

## License

MIT License - see LICENSE file

---

## Credits

- **Whisper**: OpenAI - https://github.com/openai/whisper
- **faster-whisper**: Guillaume Klein - https://github.com/guillaumekln/faster-whisper
- **Silero VAD**: snakers4 - https://github.com/snakers4/silero-vad

---

## Support

- **Issues**: https://github.com/yourorg/voiceai-cx-platform/issues
- **Docs**: https://docs.voiceai-cx.example/asr
- **API Docs**: http://localhost:8050/docs (when running)

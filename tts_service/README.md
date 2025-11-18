# TTS Service

## Overview

The TTS (Text-to-Speech) Service provides neural speech synthesis for the Voice AI CX Platform. Built with Coqui XTTS v2, it delivers multilingual, expressive speech with voice cloning capabilities. The service supports both streaming and non-streaming synthesis with controllable prosody for natural-sounding Arabic and English voices.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    TTS Service                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │   Text     │  │   Voice     │  │   Prosody    │      │
│  │Preprocessor│  │   Cloner    │  │  Controller  │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │  XTTS v2    │                        │
│                    │   Model     │                        │
│                    └──────┬──────┘                        │
│                           │                                │
│         ┌─────────────────┴──────────────────┐            │
│         │                                    │             │
│    ┌────▼────┐  ┌──────────┐  ┌──────────┐ │            │
│    │Streaming│  │  Speed   │  │ Emotion  │ │            │
│    │Generator│  │Adjustment│  │ Modulation│ │            │
│    └─────────┘  └──────────┘  └──────────┘ │            │
└──────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities

- **Multilingual Synthesis**: Arabic (Gulf/MSA) and English
- **Voice Cloning**: XTTS v2 voice cloning from reference samples
- **Streaming Support**: Real-time sentence-by-sentence synthesis
- **Prosody Control**: Adjustable speed and emotion
- **High Quality**: Neural TTS with natural-sounding output
- **GPU Accelerated**: CUDA support for low latency
- **Multiple Voice Presets**: Pre-configured voices for different personas
- **Expressive Speech**: Emotion-aware synthesis

### Supported Languages

- **Arabic Gulf** (خليجي): Natural UAE/Saudi dialect voices
- **Arabic MSA** (فصحى): Modern Standard Arabic
- **English**: UAE-accented English

### Voice Presets

- `arabic_gulf_male` - Male Gulf Arabic voice
- `arabic_gulf_female` - Female Gulf Arabic voice
- `arabic_msa_male` - Male Modern Standard Arabic
- `english_uae_male` - Male UAE English
- `english_uae_female` - Female UAE English

## Technology Stack

### Core Framework

- **Python 3.10+** - Runtime environment
- **FastAPI 0.104.1** - API framework
- **Uvicorn 0.24.0** - ASGI server
- **Pydantic 2.5.0** - Data validation

### AI/ML

- **Coqui TTS 0.20.0** - XTTS v2 model
- **PyTorch 2.1.1** - Deep learning framework
- **librosa 0.10.1** - Audio processing and time-stretching
- **soundfile 0.12.1** - Audio I/O

### Audio Processing

- **NumPy 1.26.2** - Array operations
- **Sample Rate**: 24kHz (XTTS v2 native)
- **Format**: WAV, 16-bit PCM
- **Streaming**: 250ms chunks

## Configuration

### Environment Variables

```bash
# Model Configuration
MODEL_NAME="tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE="cuda"  # cuda or cpu

# Audio Settings
SAMPLE_RATE=24000
CHUNK_DURATION_MS=250
STREAMING=true

# Voice Presets Path
VOICE_PRESETS_DIR="/app/voices"

# Server
HOST="0.0.0.0"
PORT=8002

# Performance
MAX_WORKERS=4  # Thread pool size for synthesis
```

### GPU Requirements

- **Recommended**: NVIDIA A10 (24GB) or better
- **Minimum**: NVIDIA T4 (16GB)
- **VRAM Usage**: ~4GB per concurrent synthesis
- **CPU Mode**: Supported but 10-20x slower

## Installation

### Local Development

1. **Install system dependencies**:
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3.10 \
    python3.10-venv \
    libsndfile1 \
    ffmpeg
```

2. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

3. **Download XTTS v2 model** (automatic on first run):
```bash
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

4. **Run the service**:
```bash
python server.py
# Service will start on http://0.0.0.0:8002
```

### Docker Deployment

```bash
docker build -t tts-service:latest .
docker run --gpus all -p 8002:8002 tts-service:latest
```

### Docker Compose

From repository root:

```bash
docker-compose up tts-service
```

## API Reference

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "device": "cuda",
  "voices": [
    "arabic_gulf_male",
    "arabic_gulf_female",
    "arabic_msa_male",
    "english_uae_male",
    "english_uae_female"
  ]
}
```

---

### List Voices

**Endpoint**: `GET /voices`

**Response**:
```json
{
  "voices": [
    {
      "id": "arabic_gulf_male",
      "language": "ar",
      "gender": "male",
      "region": "Gulf"
    },
    {
      "id": "arabic_gulf_female",
      "language": "ar",
      "gender": "female",
      "region": "Gulf"
    },
    {
      "id": "arabic_msa_male",
      "language": "ar",
      "gender": "male",
      "region": "MSA"
    },
    {
      "id": "english_uae_male",
      "language": "en",
      "gender": "male",
      "region": "UAE"
    },
    {
      "id": "english_uae_female",
      "language": "en",
      "gender": "female",
      "region": "UAE"
    }
  ]
}
```

---

### Synthesize (Non-Streaming)

Synthesize complete audio from text.

**Endpoint**: `POST /synthesize`

**Request Body**:
```json
{
  "text": "مرحبا، كيف يمكنني مساعدتك اليوم؟",
  "voice_id": "arabic_gulf_male",
  "language": "ar",
  "speed": 1.0,
  "emotion": "neutral",
  "stream": false
}
```

**Parameters**:
- `text` (required): Text to synthesize
- `voice_id` (default: "arabic_gulf_male"): Voice preset ID
- `language` (default: "ar"): Language code (`ar`, `en`)
- `speed` (default: 1.0): Speech speed multiplier (0.5-2.0)
- `emotion` (optional): Emotion hint (`neutral`, `happy`, `sad`, `energetic`)
- `stream` (default: true): Enable streaming mode

**Response**:
```json
{
  "audio_base64": "UklGRiQAAABXQVZFZm10IBAAAAAB...",
  "duration_seconds": 3.5,
  "sample_rate": 24000,
  "format": "wav"
}
```

**Response Fields**:
- `audio_base64`: Base64-encoded WAV audio
- `duration_seconds`: Audio duration
- `sample_rate`: Sample rate (24kHz)
- `format`: Audio format (WAV)

---

### Synthesize Streaming

Stream audio chunks as they are generated.

**Endpoint**: `POST /synthesize/stream`

**Request Body**: Same as non-streaming

**Response**: Server-Sent Events (SSE) stream

**Event Format**:
```json
{
  "chunk_index": 0,
  "audio_base64": "UklGRiQAAABXQVZF...",
  "is_final": false
}
```

**Example SSE Stream**:
```
data: {"chunk_index": 0, "audio_base64": "UklGR...", "is_final": false}

data: {"chunk_index": 1, "audio_base64": "iQAAA...", "is_final": false}

data: {"chunk_index": 2, "audio_base64": "BXQVZ...", "is_final": true}

data: [DONE]
```

## Usage Examples

### Python Client (Non-Streaming)

```python
import requests
import base64
import wave

response = requests.post(
    "http://localhost:8002/synthesize",
    json={
        "text": "مرحبا، كيف يمكنني مساعدتك؟",
        "voice_id": "arabic_gulf_male",
        "language": "ar",
        "speed": 1.0
    }
)

result = response.json()

# Decode and save audio
audio_data = base64.b64decode(result["audio_base64"])
with open("output.wav", "wb") as f:
    f.write(audio_data)

print(f"Duration: {result['duration_seconds']:.2f}s")
```

---

### Python Client (Streaming)

```python
import httpx
import asyncio
import base64

async def stream_tts():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8002/synthesize/stream",
            json={
                "text": "This is a longer text that will be streamed in chunks.",
                "voice_id": "english_uae_male",
                "language": "en"
            },
            timeout=30.0
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # Remove "data: " prefix

                    if data == "[DONE]":
                        break

                    import json
                    chunk = json.loads(data)

                    # Decode audio chunk
                    audio_data = base64.b64decode(chunk["audio_base64"])

                    # Play or save chunk
                    print(f"Chunk {chunk['chunk_index']}: {len(audio_data)} bytes")

asyncio.run(stream_tts())
```

---

### cURL Examples

```bash
# Simple synthesis
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello, how can I help you?",
    "voice_id": "english_uae_male",
    "language": "en"
  }'

# Arabic synthesis with custom speed
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{
    "text": "أهلا وسهلا بكم",
    "voice_id": "arabic_gulf_female",
    "language": "ar",
    "speed": 0.9
  }'

# Streaming synthesis
curl -X POST http://localhost:8002/synthesize/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This will stream in real-time.",
    "voice_id": "english_uae_male",
    "language": "en"
  }'
```

---

### JavaScript Client

```javascript
// Non-streaming
const response = await fetch('http://localhost:8002/synthesize', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    text: 'مرحبا',
    voice_id: 'arabic_gulf_male',
    language: 'ar'
  })
});

const result = await response.json();

// Decode and play audio
const audioData = atob(result.audio_base64);
const audioBlob = new Blob([audioData], {type: 'audio/wav'});
const audioUrl = URL.createObjectURL(audioBlob);

const audio = new Audio(audioUrl);
audio.play();
```

## Voice Cloning

### Adding Custom Voices

XTTS v2 supports voice cloning from short audio samples (6-12 seconds).

**Steps**:

1. **Record reference audio**:
```bash
# Record 10 seconds of clean speech
ffmpeg -f pulse -i default -t 10 -ar 24000 -ac 1 my_voice.wav
```

2. **Add voice preset**:
```python
# In server.py
VOICE_PRESETS = {
    "my_custom_voice": "voices/my_voice.wav",
    ...
}
```

3. **Use custom voice**:
```json
{
  "text": "Test with my cloned voice",
  "voice_id": "my_custom_voice",
  "language": "en"
}
```

**Requirements**:
- Clean audio (no background noise)
- Consistent volume
- Natural speaking pace
- 6-12 seconds duration
- Sample rate: 24kHz or 16kHz
- Mono channel

## Prosody Control

### Speed Adjustment

Control speech rate from 0.5x (slow) to 2.0x (fast):

```json
{
  "text": "This will be spoken slowly",
  "speed": 0.7
}
```

**Recommended Ranges**:
- **Slow/Clear**: 0.7 - 0.9
- **Normal**: 0.9 - 1.1
- **Fast**: 1.1 - 1.5

### Emotion Hints

While XTTS v2 doesn't natively support emotion control, the service adds subtle markers:

```json
{
  "text": "I'm so happy to help you!",
  "emotion": "happy"
}
```

**Supported Emotions**:
- `neutral` - Default, natural speech
- `happy` - Adds exclamation marks
- `sad` - Adds pauses (ellipsis)
- `energetic` - Slightly faster pace

## Performance

### Benchmarks

**Hardware**: NVIDIA A10 (24GB)

| Metric | Value | Notes |
|--------|-------|-------|
| **Latency (sentence)** | 200-300ms | XTTS v2, GPU |
| **Latency (full synthesis)** | 400-600ms | 2-3 sentences |
| **Real-Time Factor (RTF)** | 0.15-0.25 | 1s audio in 150-250ms |
| **Streaming TTFB** | 200ms | Time to first byte |
| **Concurrent Requests** | 4-6 | Per GPU |
| **VRAM Usage** | 4GB | Per synthesis |

**CPU Performance**:
- RTF: 2.0-3.0 (10-20x slower)
- Not recommended for production

### Optimization Tips

1. **Use Streaming**: Reduces perceived latency
2. **Sentence Splitting**: Synthesize sentences in parallel
3. **GPU Pooling**: Load balance across multiple GPUs
4. **Batch Processing**: Queue and batch similar requests
5. **Cache Common Phrases**: Pre-synthesize greetings, confirmations
6. **Use Quantization**: INT8 quantization for 2x speedup (slight quality loss)

## Audio Format

### Output Specifications

- **Format**: WAV (PCM)
- **Sample Rate**: 24kHz
- **Bit Depth**: 16-bit
- **Channels**: Mono
- **Encoding**: Base64 (for API transport)

### Converting to Other Formats

```bash
# Convert WAV to MP3
ffmpeg -i output.wav -codec:a libmp3lame -qscale:a 2 output.mp3

# Convert WAV to Opus (WebRTC-friendly)
ffmpeg -i output.wav -c:a libopus -b:a 64k output.opus

# Resample to 16kHz (for telephony)
ffmpeg -i output.wav -ar 16000 output_16k.wav
```

## Monitoring

### Health Checks

```bash
# Check service health
curl http://localhost:8002/health

# Expected response
{
  "status": "healthy",
  "model": "tts_models/multilingual/multi-dataset/xtts_v2",
  "device": "cuda",
  "voices": [...]
}
```

### Performance Metrics

Monitor these metrics for production:

- **Synthesis Latency**: Time to synthesize
- **Real-Time Factor (RTF)**: Synthesis time / audio duration
- **Queue Depth**: Pending synthesis requests
- **GPU Utilization**: Monitor with `nvidia-smi`
- **Error Rate**: Failed synthesis attempts

## Troubleshooting

### CUDA Out of Memory

**Problem**: `CUDA out of memory` error

**Solutions**:
```bash
# Reduce concurrent workers
export MAX_WORKERS=2

# Use smaller batch sizes
# Clear GPU cache between requests

# Monitor GPU memory
watch -n 1 nvidia-smi
```

---

### Slow Synthesis

**Problem**: Synthesis taking >2 seconds

**Solutions**:
1. Verify GPU is being used:
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {DEVICE}")
```

2. Check GPU utilization:
```bash
nvidia-smi dmon -s u
```

3. Use streaming mode for perceived latency reduction

---

### Poor Audio Quality

**Problem**: Robotic or distorted output

**Solutions**:
- Use reference voice samples for voice cloning
- Ensure text is properly formatted (no special characters)
- Check sample rate matches (24kHz)
- Verify audio encoding is correct

---

### Model Download Fails

**Problem**: Cannot download XTTS v2 model

**Solutions**:
```bash
# Manually download
mkdir -p ~/.local/share/tts
cd ~/.local/share/tts

# Download from Coqui model zoo
# Or use HuggingFace mirror

# Set cache directory
export TTS_HOME=/path/to/cache
```

---

### Voice Presets Not Found

**Problem**: Voice preset returns 404

**Solution**:
```bash
# List available voices
curl http://localhost:8002/voices

# Check voice presets directory
ls /app/voices/

# Add missing voice samples
cp my_voice.wav /app/voices/custom_voice.wav
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest test_tts.py -v

# Test synthesis
python -c "
from server import TTSService
import asyncio

async def test():
    tts = TTSService()
    audio, duration = await tts.synthesize('Hello world', 'english_uae_male', 'en')
    print(f'Generated {duration:.2f}s audio')

asyncio.run(test())
"
```

### Code Structure

```
tts_service/
├── server.py           # Main TTS service
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
├── voices/             # Voice preset samples (not in repo)
└── README.md           # This file
```

### Adding New Voice Presets

1. Record or obtain voice sample (6-12s, 24kHz mono WAV)
2. Place in `voices/` directory
3. Add to `VOICE_PRESETS` dictionary:

```python
VOICE_PRESETS = {
    "my_new_voice": "voices/my_new_voice.wav",
}
```

4. Register in `/voices` endpoint

## Deployment Considerations

### Production Checklist

- [ ] Use GPU-enabled infrastructure (NVIDIA A10+ recommended)
- [ ] Set up model caching to avoid re-downloads
- [ ] Configure proper logging and monitoring
- [ ] Implement request queuing for burst traffic
- [ ] Set up health checks and auto-restart
- [ ] Use HTTPS for API endpoints
- [ ] Implement rate limiting per tenant
- [ ] Cache common phrases (greetings, etc.)
- [ ] Set up GPU autoscaling for high loads
- [ ] Monitor RTF and error rates

### Scaling Strategy

**Vertical Scaling**:
- Larger GPU (A100 for 2x throughput)
- More VRAM for concurrent requests

**Horizontal Scaling**:
- Multiple TTS service instances
- Load balancer with round-robin
- Session affinity not required

**GPU Pooling**:
```
Load Balancer
    ├── TTS Instance 1 (GPU 0)
    ├── TTS Instance 2 (GPU 1)
    └── TTS Instance 3 (GPU 2)
```

## Alternatives

### Piper TTS (Low Latency)

For ultra-low latency (< 100ms RTF), consider Piper TTS:
- Faster than XTTS v2
- Lower quality but acceptable
- Smaller models
- Good for simple confirmations

### Cloud TTS Services

For simpler deployment (no GPU required):
- Google Cloud TTS (WaveNet)
- Amazon Polly (Neural)
- Microsoft Azure TTS

**Trade-offs**:
- ✅ No infrastructure management
- ✅ High availability
- ❌ Recurring costs
- ❌ Privacy concerns (data leaves premises)
- ❌ Limited voice customization

## Best Practices

1. **Pre-warm Models**: Load model at startup, not on first request
2. **Sentence Streaming**: Stream sentence-by-sentence for lower latency
3. **Text Normalization**: Clean text before synthesis (numbers, abbreviations)
4. **Voice Consistency**: Use same voice_id throughout conversation
5. **Cache Common Phrases**: Pre-synthesize frequent responses
6. **Monitor RTF**: Alert if RTF > 0.5 (quality degradation)
7. **Graceful Degradation**: Fall back to simpler model if GPU busy
8. **Audio Validation**: Verify synthesized audio is not silent/corrupt

## Security Considerations

- **Input Validation**: Sanitize text input (no injection attacks)
- **Rate Limiting**: Prevent abuse via excessive synthesis requests
- **Resource Limits**: Cap synthesis length (e.g., 500 characters max)
- **API Authentication**: Require authentication for synthesis endpoints
- **SSML Filtering**: If supporting SSML, sanitize tags

## Roadmap

- [ ] Support for more languages (French, Spanish, Hindi)
- [ ] Real-time prosody control via SSML
- [ ] Voice mixing (blend multiple reference voices)
- [ ] Emotion classification from text
- [ ] Integration with emotion detection from ASR
- [ ] WebRTC streaming support
- [ ] Voice style transfer
- [ ] Custom pronunciation dictionaries

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- XTTS v2: [Coqui TTS](https://github.com/coqui-ai/TTS)

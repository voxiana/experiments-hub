# 🎤 Voice AI CX Platform

> **Open-source, privacy-first, real-time voice AI customer care platform optimized for UAE/MENA markets**

A production-ready conversational AI system for handling customer service calls with **streaming ASR**, **LLM-powered NLU**, **expressive TTS**, **RAG-grounded responses**, and **seamless human handoff**. Built entirely with open-source models for on-premise deployment.

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-required-blue.svg)](https://www.docker.com/)
[![NVIDIA GPU](https://img.shields.io/badge/GPU-NVIDIA-green.svg)](https://developer.nvidia.com/cuda-toolkit)

---

## 📋 Table of Contents

- [Features](#-features)
- [Architecture](#-architecture)
- [Technology Stack](#-technology-stack)
- [Quick Start](#-quick-start)
- [Deployment](#-deployment)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Documentation](#-api-documentation)
- [Development](#-development)
- [Testing](#-testing)
- [Performance Tuning](#-performance-tuning)
- [Compliance & Security](#-compliance--security)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## ✨ Features

### Core Capabilities

- **🎙️ Real-time Voice Streaming**: WebRTC & SIP integration with <1.5s E2E latency
- **🗣️ Multilingual Support**: Arabic (Gulf, MSA) + English with code-switching
- **🧠 Intelligent NLU**: Powered by Qwen2.5-72B or Llama-3.1-70B via vLLM
- **📚 RAG-grounded Responses**: Knowledge base search with bge-m3 embeddings + reranking
- **🔊 Expressive TTS**: XTTS v2 with controllable prosody and voice cloning
- **🔄 Barge-in Support**: Natural conversation interruption handling
- **👤 Human Handoff**: Seamless escalation with full context transfer
- **🔌 CRM Integration**: Salesforce, Zendesk, HubSpot, Freshdesk connectors

### Advanced Features

- **📊 Real-time Analytics**: Call metrics, sentiment tracking, intent analysis
- **🎭 Emotion Detection**: Paralinguistic analysis for escalation triggers
- **🔒 Privacy-first**: On-premise deployment, PII redaction, GDPR compliance
- **📈 Scalable**: Kubernetes-ready, GPU autoscaling, horizontal scaling
- **🔍 Observable**: OpenTelemetry, Prometheus, Grafana, Langfuse integration
- **🌍 Multi-tenant**: Isolated namespaces, per-tenant configuration

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Client Layer                                │
│  Web Browser (WebRTC) │ Phone (SIP) │ Mobile App                    │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────────┐
│                       Gateway API (FastAPI)                          │
│  Authentication │ Rate Limiting │ Session Management                │
└─────────────────────────┬───────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
┌─────────▼─────┐ ┌──────▼──────┐ ┌─────▼──────┐
│  ASR Service  │ │ NLU Service │ │ TTS Service│
│ faster-whisper│ │ vLLM + Tools│ │ XTTS v2    │
│ + silero-vad  │ │ RAG Client  │ │ Streaming  │
└───────────────┘ └──────┬──────┘ └────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
┌─────────▼─────┐ ┌─────▼──────┐ ┌────▼─────────┐
│  RAG Service  │ │ Connectors │ │  Workers     │
│ bge-m3 + Qdrant│ │ CRM + Chat │ │ Celery Batch │
│ Reranking     │ │ Handoff    │ │ Analytics    │
└───────────────┘ └────────────┘ └──────────────┘
```

**Full architecture details**: See [`docs/architecture.md`](docs/architecture.md)

---

## 🛠️ Technology Stack

### AI Models (All Open Source)

| Component | Model | Purpose |
|-----------|-------|---------|
| **ASR** | Whisper large-v3 (faster-whisper) | Streaming speech recognition |
| **VAD** | Silero VAD v4 | Voice activity detection |
| **Diarization** | pyannote.audio 3.1 | Speaker separation |
| **Emotion** | speechbrain / Emotion2Vec | Paralinguistic analysis |
| **LLM** | Qwen2.5-72B / Llama-3.1-70B (vLLM) | Intent & response generation |
| **Embeddings** | bge-m3 | Multilingual text embeddings |
| **Reranker** | bge-reranker-v2-m3 | RAG result reranking |
| **TTS** | Coqui XTTS v2 | Expressive speech synthesis |

### Infrastructure

- **API Framework**: FastAPI (async, WebSocket support)
- **Databases**: PostgreSQL, Qdrant (vectors), ClickHouse (analytics)
- **Cache/PubSub**: Redis
- **Workers**: Celery + Beat
- **Streaming**: Kafka/Redpanda (optional)
- **Observability**: OpenTelemetry, Prometheus, Grafana, Langfuse
- **Orchestration**: Docker Compose (dev), Kubernetes (prod)

---

## 🚀 Quick Start

### Prerequisites

**GPU Mode (Recommended for Production):**
- **Hardware**: NVIDIA GPU (A10, A100, or RTX 4090 recommended)
  - Minimum 24GB VRAM for dev (single GPU)
  - 160GB+ VRAM for production (multi-GPU)
- **Software**:
  - Docker 20+ with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  - Docker Compose 2.0+
  - 64GB+ RAM
  - 500GB+ SSD storage

**CPU Mode (Development/Testing):**
- **Hardware**: CPU-only (no GPU required)
  - 16GB+ RAM recommended
  - Multi-core CPU (4+ cores)
- **Software**:
  - Docker 20+
  - Docker Compose 2.0+
  - 100GB+ SSD storage
- **Note**: Services will run slower but fully functional for testing

### Installation (Development)

1. **Clone repository**:
   ```bash
   git clone https://github.com/yourorg/voiceai-cx-platform.git
   cd voiceai-cx-platform
   ```

2. **Configure environment**:
   ```bash
   cp .env.sample .env
   # Edit .env with your settings
   ```

3. **Start services**:

   **GPU Mode** (requires NVIDIA Container Toolkit):
   ```bash
   docker-compose up -d --profile gpu
   ```

   **CPU Mode** (no GPU required, Windows/Mac compatible):
   ```bash
   docker-compose up -d
   ```

   This will start:
   - Gateway API (port 8000)
   - ASR service (port 50051) - CPU mode uses smaller models
   - NLU service (port 8001) - Note: vLLM requires GPU, use external LLM API in CPU mode
   - TTS service (port 8002) - CPU mode supported
   - RAG service (port 8080) - CPU mode supported
   - vLLM server (port 8000) - **Only with `--profile gpu`**
   - PostgreSQL, Redis, Qdrant, ClickHouse
   - Prometheus (9090), Grafana (3000)
   - Web demo client (port 3001)

   **Note**: In CPU mode, services automatically use CPU-optimized settings. The `vllm` service is excluded unless you use `--profile gpu`.

4. **Verify services**:
   ```bash
   # Check all services are healthy
   docker-compose ps

   # Check gateway
   curl http://localhost:8000/health

   # Check ASR
   curl http://localhost:50051/health

   # Check vLLM
   curl http://localhost:8000/health
   ```

5. **Ingest sample knowledge base**:
   ```bash
   python scripts/ingest_docs.py sample \
     --tenant-id demo \
     --api-key demo_key
   ```

6. **Open web demo**:
   ```
   http://localhost:3001
   ```

   Click "Start Call" and begin speaking!

### First Call Test

```bash
# Get auth token
curl -X POST http://localhost:8000/auth/token \
  -d "tenant_id=demo&api_key=demo_key"

# Start call
curl -X POST http://localhost:8000/call/start \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "demo",
    "language": "en",
    "voice_id": "english_uae_male"
  }'

# Open web client and test voice interaction
```

---

## 📦 Deployment

### Development (Single GPU)

```bash
docker-compose up -d
```

**Resource allocation**:
- ASR: 6GB VRAM (2 streams)
- LLM: Qwen2.5-32B-Q4 (12GB VRAM)
- TTS: 4GB VRAM (2 streams)
- Total: ~22GB VRAM

### Production (Kubernetes)

1. **Install Kubernetes cluster** (e.g., EKS, GKE, or on-prem)

2. **Install NVIDIA GPU Operator**:
   ```bash
   kubectl apply -f https://raw.githubusercontent.com/NVIDIA/gpu-operator/master/deployments/gpu-operator.yaml
   ```

3. **Deploy with Helm** (optional):
   ```bash
   helm install voiceai ./helm/voiceai \
     --namespace voiceai \
     --create-namespace \
     --values values-production.yaml
   ```

4. **Or use kubectl**:
   ```bash
   kubectl apply -f k8s/
   ```

**Production architecture**:
- **ASR Pool**: 2x A10 (24GB each) = 8 concurrent streams
- **LLM Pool**: 2x A100 (80GB each) = Qwen2.5-72B-Q4
- **TTS Pool**: 2x A10 (24GB each) = 16 concurrent streams
- **Autoscaling**: HPA based on queue depth

See [`k8s/README.md`](k8s/README.md) for detailed Kubernetes deployment guide.

---

## ⚙️ Configuration

### Environment Variables

Key settings in `.env`:

```bash
# Model selection
WHISPER_MODEL=large-v3           # or distil-large-v3
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct
TTS_MODEL=tts_models/multilingual/multi-dataset/xtts_v2

# GPU configuration
NVIDIA_VISIBLE_DEVICES=all
ASR_GPU_MEMORY_GB=6
LLM_GPU_MEMORY_GB=40
TTS_GPU_MEMORY_GB=4

# Latency tuning
ASR_CHUNK_DURATION_MS=250        # Lower = faster, higher overhead
WHISPER_BEAM_SIZE=1              # 1=fast, 5=accurate
NLU_MAX_TOKENS=200
TTS_CHUNK_DURATION_MS=250

# Features
ENABLE_DIARIZATION=true
ENABLE_EMOTION_ANALYSIS=true
ENABLE_RERANKER=true
```

### Per-Tenant Configuration

Stored in `configs` table:

```python
{
  "escalation_keywords": ["manager", "complaint"],
  "max_turns_before_escalation": 10,
  "voice_settings": {
    "speed": 1.0,
    "emotion": "professional"
  },
  "crm": {
    "type": "salesforce",
    "credentials": {...}
  }
}
```

---

## 📖 Usage

### Web Client Demo

1. Open `http://localhost:3001`
2. Select language (Arabic/English/Auto)
3. Click "Start Call"
4. Speak naturally
5. View real-time transcript and bot responses
6. Barge-in supported (interrupt anytime)

### API Examples

#### Start a call
```python
import requests

# Authenticate
auth = requests.post(
    "http://localhost:8000/auth/token",
    data={"tenant_id": "demo", "api_key": "demo_key"}
)
token = auth.json()["access_token"]

# Start call
call = requests.post(
    "http://localhost:8000/call/start",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "tenant_id": "demo",
        "language": "ar",
        "voice_id": "arabic_gulf_male",
        "metadata": {"customer_id": "12345"}
    }
)

call_id = call.json()["call_id"]
ws_url = call.json()["ws_url"]
```

#### Query knowledge base
```python
rag = requests.post(
    "http://localhost:8000/rag/query",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "query": "What are your support hours?",
        "tenant_id": "demo",
        "top_k": 3
    }
)

results = rag.json()["results"]
for result in results:
    print(f"{result['source']}: {result['text']}")
```

#### Request handoff
```python
handoff = requests.post(
    "http://localhost:8000/handoff",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "call_id": call_id,
        "reason": "Customer requested supervisor",
        "agent_queue": "supervisors"
    }
)

print(f"Estimated wait: {handoff.json()['estimated_wait_seconds']}s")
```

---

## 📚 API Documentation

### REST Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/auth/token` | POST | Get JWT token |
| `/call/start` | POST | Initialize call session |
| `/call/{id}` | GET | Get call status |
| `/call/{id}/end` | POST | End call |
| `/handoff` | POST | Escalate to human |
| `/rag/query` | POST | Query knowledge base |
| `/ingest` | POST | Ingest document |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |

### WebSocket Protocol

Connect to `ws://gateway:8000/ws/{call_id}`

**Client → Server**:
```json
{
  "type": "audio",
  "data": "<base64 PCM 16kHz mono>"
}

{
  "type": "barge_in"
}
```

**Server → Client**:
```json
{
  "type": "transcript_interim",
  "text": "Hello, how...",
  "timestamp": 1699564800.123
}

{
  "type": "transcript_final",
  "text": "Hello, how can I help?",
  "language": "en",
  "confidence": 0.95
}

{
  "type": "response",
  "text": "I can help you with that.",
  "audio": "<base64 WAV>",
  "intent": "question",
  "timestamp": 1699564801.456
}
```

**Full API docs**: See [`docs/api.md`](docs/api.md) or visit `http://localhost:8000/docs` (Swagger UI)

---

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module
pytest tests/test_nlu.py -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Integration Tests

```bash
# Test full call flow (requires services running)
pytest tests/integration/test_call_flow.py -v
```

### E2E Tests

```bash
# Synthetic call test
python tests/e2e/test_synthetic_call.py
```

### Load Testing

```bash
# 50 concurrent calls
locust -f tests/load/locustfile.py --users 50 --host http://localhost:8000
```

---

## ⚡ Performance Tuning

### Latency Budget

| Stage | Target | Tuning Knobs |
|-------|--------|--------------|
| ASR | 250ms | Model: `large-v3` → `distil-large-v3`, Beam: 5 → 1 |
| NLU | 600ms | GPU count, Quantization: FP16 → Q4, Context length |
| RAG | 150ms | Top-K: 20 → 10, Disable reranker, Smaller embedder |
| TTS | 500ms | Model: XTTS → Piper, Chunk size |

### Accuracy vs Speed

**High Accuracy** (E2E ~2000ms):
```bash
WHISPER_MODEL=large-v3
WHISPER_BEAM_SIZE=5
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct  # FP16
ENABLE_RERANKER=true
TTS_MODEL=xtts_v2
```

**Balanced** (E2E ~1200ms):
```bash
WHISPER_MODEL=large-v3
WHISPER_BEAM_SIZE=1
VLLM_MODEL=Qwen/Qwen2.5-72B-Instruct  # Q4
ENABLE_RERANKER=true
TTS_MODEL=xtts_v2
```

**Low Latency** (E2E ~800ms):
```bash
WHISPER_MODEL=distil-large-v3
WHISPER_BEAM_SIZE=1
VLLM_MODEL=Qwen/Qwen2.5-32B-Instruct  # Q4
ENABLE_RERANKER=false
TTS_MODEL=piper
```

---

## 🔒 Compliance & Security

### Data Protection

- **PII Redaction**: Automatic masking of phone numbers, emails, credit cards
- **Encryption**: TLS 1.3 for REST, DTLS-SRTP for WebRTC, AES-256 at rest
- **Access Control**: JWT (RS256), RBAC, row-level security
- **Audit Logging**: Immutable audit trail for all mutations

### Data Retention

Configurable per tenant:
```bash
TRANSCRIPT_RETENTION_DAYS=90     # Auto-purge
RECORDING_RETENTION_DAYS=30      # Opt-in
CALL_RECORDING_ENABLED=false     # Consent required
```

### Compliance

- **GDPR**: Right to delete, data portability, consent management
- **ISO 27001**: On-premise deployment checklist included
- **SOC 2**: Access logs, change management procedures
- **HIPAA**: Optional PHI masking for healthcare use cases

### UAE/MENA Specifics

- **Data Residency**: All data in UAE data centers (configurable)
- **Timezone**: Asia/Dubai (UTC+4) for call routing
- **Calendar**: Islamic calendar support for Ramadan scheduling
- **Language**: Arabic (Gulf/MSA) + English code-switching

---

## 🗺️ Roadmap

### Phase 1: MVP (Weeks 1-6) ✅
- [x] Core services (ASR, NLU, TTS, RAG)
- [x] Web demo client
- [x] Docker Compose deployment
- [x] Sample knowledge base
- [x] Basic CRM stubs

### Phase 2: Pilot (Weeks 7-18)
- [ ] Multi-tenant production deployment
- [ ] Salesforce/Zendesk live connectors
- [ ] LiveChat human handoff
- [ ] SIP integration (Twilio)
- [ ] Arabic voice fine-tuning
- [ ] Load testing (50 concurrent calls)

### Phase 3: Scale (Months 6-12)
- [ ] Kubernetes autoscaling
- [ ] Custom voice cloning per tenant
- [ ] WhatsApp Business integration
- [ ] Real-time quality monitoring (LLM-as-judge)
- [ ] Self-serve admin portal
- [ ] ISO 27001 / SOC 2 certification

---

## 🤝 Contributing

We welcome contributions! See [`CONTRIBUTING.md`](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repo
git clone https://github.com/yourorg/voiceai-cx-platform.git
cd voiceai-cx-platform

# Create virtual environment
python3.10 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/ -v
```

---

## 📄 License

This project is licensed under the **MIT License** - see [`LICENSE`](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Open Source Models

- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Qwen2.5](https://github.com/QwenLM/Qwen2.5) - Language model
- [Llama 3.1](https://github.com/meta-llama/llama3) - Language model
- [vLLM](https://github.com/vllm-project/vllm) - LLM inference optimization
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text-to-speech
- [bge-m3](https://huggingface.co/BAAI/bge-m3) - Embeddings
- [pyannote.audio](https://github.com/pyannote/pyannote-audio) - Diarization
- [silero-vad](https://github.com/snakers4/silero-vad) - Voice activity detection

### Infrastructure

- [FastAPI](https://fastapi.tiangolo.com/) - API framework
- [Qdrant](https://qdrant.tech/) - Vector database
- [LiveKit](https://livekit.io/) - WebRTC infrastructure

---

## 📞 Support

- **Documentation**: [`/docs`](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourorg/voiceai-cx-platform/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourorg/voiceai-cx-platform/discussions)
- **Email**: support@voiceai-cx.example

---

## 📊 Project Status

- **Version**: 1.0.0-beta
- **Status**: Active Development
- **Last Updated**: 2025-11-06

---

<div align="center">

**Built with ❤️ for the UAE/MENA customer care community**

[⭐ Star us on GitHub](https://github.com/yourorg/voiceai-cx-platform) | [📖 Read the Docs](docs/) | [🐛 Report Bug](https://github.com/yourorg/voiceai-cx-platform/issues)

</div>

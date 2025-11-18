# Gateway Service

## Overview

The Gateway Service is the central API gateway and WebSocket orchestrator for the Voice AI CX Platform. It provides REST endpoints for call management, real-time WebSocket streaming for audio, RAG integration, and human handoff capabilities. Built with FastAPI for high-performance async operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Gateway Service                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  REST API    │  │  WebSocket   │  │   RAG API    │      │
│  │              │  │   Streaming  │  │              │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                  │
│         ┌─────────────────┴──────────────────┐              │
│         │                                    │               │
│    ┌────▼────┐  ┌──────────┐  ┌──────────┐ │              │
│    │  Redis  │  │PostgreSQL│  │   Auth   │ │              │
│    │ Pub/Sub │  │  Storage │  │   JWT    │ │              │
│    └─────────┘  └──────────┘  └──────────┘ │              │
│                                              │               │
└──────────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities

- **REST API**: FastAPI-based async REST endpoints for call lifecycle management
- **WebSocket Streaming**: Real-time bidirectional audio streaming with low latency
- **Multi-tenant Support**: Tenant isolation with per-tenant configuration
- **Authentication**: JWT-based authentication (RS256/HS256)
- **Rate Limiting**: Redis-backed rate limiting per tenant
- **Session Management**: Redis-based session storage with TTL
- **Event Publishing**: Redis Pub/Sub for service communication
- **Observability**: OpenTelemetry tracing + Prometheus metrics
- **CORS Support**: Configurable cross-origin resource sharing
- **Health Checks**: Health and metrics endpoints

### API Endpoints

#### Authentication

- `POST /auth/token` - Generate JWT access token

#### Call Management

- `POST /call/start` - Initialize new voice call session
- `GET /call/{call_id}` - Get call status and metadata
- `POST /call/{call_id}/end` - End active call

#### WebSocket Streaming

- `WS /ws/{call_id}` - Real-time audio streaming endpoint

#### Human Handoff

- `POST /handoff` - Escalate call to human agent

#### RAG (Knowledge Base)

- `POST /rag/query` - Query knowledge base
- `POST /ingest` - Ingest document into knowledge base

#### Monitoring

- `GET /health` - Health check endpoint
- `GET /metrics` - Prometheus metrics

## Technology Stack

### Core Framework

- **FastAPI 0.104.1** - Modern async web framework
- **Uvicorn 0.24.0** - ASGI server with uvloop
- **Pydantic 2.5.0** - Data validation and serialization
- **Python 3.10+** - Runtime environment

### Async & Networking

- **aiohttp 3.9.0** - Async HTTP client
- **aioredis 2.0.1** - Async Redis client
- **asyncpg 0.29.0** - Async PostgreSQL driver
- **websockets 12.0** - WebSocket protocol implementation

### Database

- **SQLAlchemy 2.0.23** - Async ORM
- **Alembic 1.12.1** - Database migrations
- **PostgreSQL** - Primary data store
- **Redis** - Cache, sessions, pub/sub

### Security

- **PyJWT 2.8.0** - JWT authentication
- **python-jose 3.3.0** - JOSE/JWT implementation
- **passlib 1.7.4** - Password hashing

### Observability

- **OpenTelemetry** - Distributed tracing
  - `opentelemetry-api 1.21.0`
  - `opentelemetry-sdk 1.21.0`
  - `opentelemetry-instrumentation-fastapi 0.42b0`
  - `opentelemetry-exporter-otlp 1.21.0`
- **Prometheus Client 0.19.0** - Metrics collection
- **Sentry SDK 1.39.1** - Error tracking

## Configuration

### Environment Variables

```bash
# Database
DATABASE_URL="postgresql+asyncpg://voiceai:voiceai@postgres:5432/voiceai"

# Redis
REDIS_URL="redis://redis:6379/0"

# Authentication
JWT_SECRET="your-secret-key-change-in-production"
JWT_ALGORITHM="HS256"  # or RS256 for production

# Server
HOST="0.0.0.0"
PORT=8000

# CORS
CORS_ORIGINS="*"  # Configure for production

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60  # seconds

# OpenTelemetry
OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
OTEL_SERVICE_NAME="gateway"

# Sentry
SENTRY_DSN=""  # Optional

# Logging
LOG_LEVEL="INFO"
```

### Docker Configuration

Build and run with Docker:

```bash
docker build -t gateway:latest .
docker run -p 8000:8000 \
  -e DATABASE_URL="postgresql+asyncpg://voiceai:voiceai@postgres:5432/voiceai" \
  -e REDIS_URL="redis://redis:6379/0" \
  gateway:latest
```

## Installation

### Local Development

1. **Create virtual environment**:
```bash
python3.10 -m venv venv
source venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set environment variables**:
```bash
export DATABASE_URL="postgresql+asyncpg://voiceai:voiceai@localhost:5432/voiceai"
export REDIS_URL="redis://localhost:6379/0"
export JWT_SECRET="dev-secret-key"
```

4. **Run the service**:
```bash
python main.py
# Or with uvicorn:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

Using docker-compose (from repository root):

```bash
docker-compose up gateway
```

## API Documentation

### Start Call

Initialize a new voice call session.

**Endpoint**: `POST /call/start`

**Headers**:
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body**:
```json
{
  "tenant_id": "tenant_123",
  "user_id": "user_456",
  "language": "auto",
  "voice_id": "arabic_gulf_male",
  "metadata": {
    "customer_id": "C12345",
    "order_id": "O67890"
  }
}
```

**Response**:
```json
{
  "call_id": "call_a1b2c3d4e5f6",
  "session_token": "eyJhbGciOiJIUzI1NiIs...",
  "ws_url": "ws://localhost:8000/ws/call_a1b2c3d4e5f6",
  "status": "ready"
}
```

**Parameters**:
- `tenant_id` (required): Tenant identifier
- `user_id` (optional): User/customer identifier
- `language` (default: "auto"): Language preference (`ar`, `en`, `auto`)
- `voice_id` (default: "arabic_gulf_male"): TTS voice preset
- `metadata` (optional): Additional context data

**Rate Limit**: 100 requests per minute per tenant

---

### WebSocket Protocol

Connect to real-time audio streaming.

**Endpoint**: `WS /ws/{call_id}`

**Query Parameters**:
```
?token=<session_token>
```

**Client → Server Messages**:

```json
{
  "type": "audio",
  "data": "<base64-encoded PCM audio>"
}
```

```json
{
  "type": "barge_in"
}
```

```json
{
  "type": "ping"
}
```

**Server → Client Messages**:

```json
{
  "type": "transcript_interim",
  "text": "Processing speech...",
  "timestamp": 1699999999.123
}
```

```json
{
  "type": "transcript_final",
  "text": "I need help with my order",
  "timestamp": 1699999999.456
}
```

```json
{
  "type": "response",
  "text": "I'd be happy to help with your order",
  "audio": "<base64-encoded audio>",
  "timestamp": 1700000000.789
}
```

```json
{
  "type": "pong",
  "timestamp": 1700000001.012
}
```

**Audio Format**:
- Encoding: Base64
- Sample Rate: 16kHz
- Channels: Mono
- Format: 16-bit PCM
- Chunk Size: 250ms recommended

---

### Query RAG

Query knowledge base using semantic search.

**Endpoint**: `POST /rag/query`

**Request**:
```json
{
  "query": "What are your support hours?",
  "tenant_id": "tenant_123",
  "top_k": 3,
  "filters": {
    "category": "support"
  }
}
```

**Response**:
```json
{
  "results": [
    {
      "text": "Our support hours are Sunday to Thursday, 9 AM to 6 PM UAE time.",
      "source": "FAQ - Support Hours",
      "score": 0.89,
      "metadata": {
        "page": 1,
        "url": "https://example.com/faq"
      }
    }
  ],
  "took_ms": 45.67
}
```

---

### Human Handoff

Escalate to human agent.

**Endpoint**: `POST /handoff`

**Request**:
```json
{
  "call_id": "call_a1b2c3d4e5f6",
  "reason": "customer_request",
  "agent_queue": "general"
}
```

**Response**:
```json
{
  "status": "queued",
  "queue": "general",
  "estimated_wait_seconds": 30,
  "context": {
    "call_id": "call_a1b2c3d4e5f6",
    "reason": "customer_request",
    "duration_seconds": 125,
    "transcript": ["..."],
    "intents": ["greeting", "question", "escalation"],
    "sentiment": "neutral"
  }
}
```

## Data Models

### CallStartRequest

```python
class CallStartRequest(BaseModel):
    tenant_id: str
    user_id: Optional[str] = None
    language: str = "auto"  # ar, en, auto
    voice_id: str = "arabic_gulf_male"
    metadata: Dict = Field(default_factory=dict)
```

### CallEvent

```python
class CallEvent(BaseModel):
    event_type: str  # speech_detected, transcript_interim, transcript_final, response, error
    call_id: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict = Field(default_factory=dict)
```

## Redis Pub/Sub Channels

The gateway publishes events to Redis channels for service communication:

- `call.started` - New call initialized
- `call.ended` - Call terminated
- `audio:{call_id}` - Audio chunks for ASR processing
- `barge_in:{call_id}` - User interruption detected
- `handoff.requested` - Human handoff requested
- `ingest.queued` - Document ingestion queued

## Metrics

### Prometheus Metrics

- `gateway_requests_total{method, endpoint}` - Total HTTP requests
- `gateway_request_duration_seconds` - Request latency histogram
- `gateway_ws_connections_total{status}` - WebSocket connections (connected/disconnected)
- `gateway_call_events_total{event_type}` - Call events (started, ended, handoff, barge_in)

### OpenTelemetry Traces

- Span: `start_call` - Call initialization
- Span: `handoff` - Human handoff
- Span: `rag_query` - Knowledge base query

## Security

### Authentication Flow

1. Client obtains API key from platform admin
2. Client calls `POST /auth/token` with `tenant_id` and `api_key`
3. Server validates credentials and returns JWT access token
4. Client includes token in `Authorization: Bearer <token>` header
5. Gateway validates token signature and expiration on each request

### JWT Token Structure

```json
{
  "tenant_id": "tenant_123",
  "sub": "tenant_123",
  "exp": 1700000000
}
```

**Token Expiry**: 1 hour (configurable)

### Rate Limiting

Rate limiting uses Redis with sliding window:
- Default: 100 requests per 60 seconds per tenant
- Key format: `ratelimit:{tenant_id}:{window}`
- Configurable via environment variables

### CORS Configuration

For production, configure allowed origins:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://app.example.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": 1699999999.123,
  "version": "1.0.0"
}
```

### Prometheus Metrics

```bash
curl http://localhost:8000/metrics
```

### OpenTelemetry Integration

Configure OTLP exporter:

```bash
export OTEL_EXPORTER_OTLP_ENDPOINT="http://otel-collector:4317"
export OTEL_SERVICE_NAME="gateway"
```

View traces in Jaeger: `http://localhost:16686`

## Performance

### Benchmarks

- **REST API Latency**: ~5-10ms (p95)
- **WebSocket Message Latency**: ~2-5ms
- **Concurrent Connections**: 10,000+ WebSocket connections per instance
- **Throughput**: 5,000+ requests/second

### Optimization

1. **Connection Pooling**: PostgreSQL and Redis connection pools
2. **Async I/O**: All I/O operations are async (asyncpg, aioredis)
3. **Uvloop**: High-performance event loop
4. **No Blocking Operations**: All CPU-intensive work offloaded to workers

### Scaling

**Horizontal Scaling**:
- Stateless design allows multiple instances
- WebSocket connections distributed via load balancer
- Redis Pub/Sub for cross-instance communication

**Recommended Setup**:
- 3+ Gateway instances behind load balancer
- Sticky sessions for WebSocket connections
- Redis cluster for high availability

## Error Handling

### HTTP Error Codes

- `400 Bad Request` - Invalid request payload
- `401 Unauthorized` - Missing or invalid JWT token
- `404 Not Found` - Call ID not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

### WebSocket Errors

Server sends error messages:
```json
{
  "type": "error",
  "code": "ASR_TIMEOUT",
  "message": "Speech recognition timeout",
  "timestamp": 1699999999.123
}
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio pytest-cov

# Run tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Format code
black main.py

# Lint
ruff check main.py

# Type checking
mypy main.py
```

### Hot Reload

```bash
uvicorn main:app --reload --log-level debug
```

## Troubleshooting

### Database Connection Issues

**Problem**: `asyncpg.exceptions.InvalidPasswordError`

**Solution**: Verify `DATABASE_URL` credentials and ensure PostgreSQL is running:
```bash
docker-compose up -d postgres
psql -h localhost -U voiceai -d voiceai
```

---

### Redis Connection Timeout

**Problem**: `aioredis.exceptions.ConnectionError`

**Solution**: Check Redis connectivity:
```bash
redis-cli -h localhost -p 6379 ping
# Should return: PONG
```

---

### WebSocket Connection Refused

**Problem**: WebSocket fails to connect

**Solution**:
1. Verify call was initialized via `/call/start`
2. Check `call_id` is valid
3. Ensure session token is included in query params
4. Verify CORS settings if connecting from browser

---

### Rate Limit False Positives

**Problem**: Rate limit triggered incorrectly

**Solution**: Clear Redis rate limit keys:
```bash
redis-cli KEYS "ratelimit:*" | xargs redis-cli DEL
```

## Integration Examples

### Python Client

```python
import asyncio
import aiohttp
import json
import base64

async def make_call():
    async with aiohttp.ClientSession() as session:
        # Get token
        async with session.post(
            "http://localhost:8000/auth/token",
            json={"tenant_id": "tenant_123", "api_key": "secret"}
        ) as resp:
            token_data = await resp.json()
            token = token_data["access_token"]

        # Start call
        headers = {"Authorization": f"Bearer {token}"}
        async with session.post(
            "http://localhost:8000/call/start",
            json={"tenant_id": "tenant_123", "language": "ar"},
            headers=headers
        ) as resp:
            call_data = await resp.json()
            call_id = call_data["call_id"]

        # Connect WebSocket
        async with session.ws_connect(
            f"ws://localhost:8000/ws/{call_id}"
        ) as ws:
            # Send audio
            audio_chunk = base64.b64encode(b"..." * 1000).decode()
            await ws.send_json({
                "type": "audio",
                "data": audio_chunk
            })

            # Receive messages
            async for msg in ws:
                data = json.loads(msg.data)
                print(f"Received: {data['type']}")

asyncio.run(make_call())
```

### JavaScript Client

```javascript
// Get token
const tokenResp = await fetch('http://localhost:8000/auth/token', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({tenant_id: 'tenant_123', api_key: 'secret'})
});
const {access_token} = await tokenResp.json();

// Start call
const callResp = await fetch('http://localhost:8000/call/start', {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${access_token}`,
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    tenant_id: 'tenant_123',
    language: 'auto'
  })
});
const {call_id} = await callResp.json();

// Connect WebSocket
const ws = new WebSocket(`ws://localhost:8000/ws/${call_id}`);

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Received: ${data.type}`);
};

// Send audio
const audioData = btoa(String.fromCharCode(...new Uint8Array(audioBuffer)));
ws.send(JSON.stringify({
  type: 'audio',
  data: audioData
}));
```

## Roadmap

### Planned Features

- [ ] Multi-region support with geo-routing
- [ ] GraphQL API
- [ ] gRPC endpoints for service-to-service communication
- [ ] Advanced rate limiting with tiered quotas
- [ ] WebRTC support for browser-native audio
- [ ] Call recording and playback
- [ ] Real-time analytics dashboard
- [ ] A/B testing framework

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- Architecture: [Architecture Docs](../docs/architecture.md)

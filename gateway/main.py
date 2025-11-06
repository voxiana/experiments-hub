"""
Voice AI CX Platform - Gateway API
FastAPI-based gateway with REST endpoints and WebSocket streaming.
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from uuid import uuid4

import aioredis
from fastapi import (
    FastAPI,
    WebSocket,
    WebSocketDisconnect,
    HTTPException,
    Depends,
    Header,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import jwt
from datetime import datetime, timedelta

# ============================================================================
# Configuration
# ============================================================================

# Environment variables (replace with config loader in production)
DATABASE_URL = "postgresql+asyncpg://voiceai:voiceai@postgres:5432/voiceai"
REDIS_URL = "redis://redis:6379/0"
JWT_SECRET = "your-secret-key-change-in-production"
JWT_ALGORITHM = "HS256"

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenTelemetry
tracer = trace.get_tracer(__name__)

# Prometheus metrics
REQUESTS = Counter('gateway_requests_total', 'Total requests', ['method', 'endpoint'])
LATENCY = Histogram('gateway_request_duration_seconds', 'Request latency')
WS_CONNECTIONS = Counter('gateway_ws_connections_total', 'WebSocket connections', ['status'])
CALL_EVENTS = Counter('gateway_call_events_total', 'Call events', ['event_type'])

# ============================================================================
# Database Setup
# ============================================================================

engine = create_async_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()

# ============================================================================
# Models
# ============================================================================

class CallStartRequest(BaseModel):
    tenant_id: str
    user_id: Optional[str] = None
    language: str = "auto"  # ar, en, auto
    voice_id: str = "arabic_gulf_male"
    metadata: Dict = Field(default_factory=dict)

class CallStartResponse(BaseModel):
    call_id: str
    session_token: str
    ws_url: str
    status: str = "ready"

class CallEvent(BaseModel):
    event_type: str  # speech_detected, transcript_interim, transcript_final, response, error
    call_id: str
    timestamp: float = Field(default_factory=time.time)
    data: Dict = Field(default_factory=dict)

class HandoffRequest(BaseModel):
    call_id: str
    reason: str
    agent_queue: str = "general"

class RAGQueryRequest(BaseModel):
    query: str
    tenant_id: str
    top_k: int = 3
    filters: Optional[Dict] = None

class RAGQueryResponse(BaseModel):
    results: List[Dict]
    took_ms: float

class IngestRequest(BaseModel):
    tenant_id: str
    source_url: Optional[str] = None
    source_text: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)

class IngestResponse(BaseModel):
    task_id: str
    status: str = "queued"

# ============================================================================
# Redis Connection Manager
# ============================================================================

class RedisManager:
    def __init__(self):
        self.redis: Optional[aioredis.Redis] = None
        self.pubsub = None

    async def connect(self):
        self.redis = await aioredis.from_url(REDIS_URL, decode_responses=True)
        logger.info("✅ Redis connected")

    async def disconnect(self):
        if self.redis:
            await self.redis.close()
            logger.info("Redis disconnected")

    async def publish(self, channel: str, message: dict):
        if self.redis:
            await self.redis.publish(channel, json.dumps(message))

    async def get(self, key: str) -> Optional[str]:
        if self.redis:
            return await self.redis.get(key)
        return None

    async def set(self, key: str, value: str, ex: int = None):
        if self.redis:
            await self.redis.set(key, value, ex=ex)

    async def incr(self, key: str) -> int:
        if self.redis:
            return await self.redis.incr(key)
        return 0

    async def expire(self, key: str, seconds: int):
        if self.redis:
            await self.redis.expire(key, seconds)

redis_manager = RedisManager()

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, call_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[call_id] = websocket
        logger.info(f"WebSocket connected: {call_id}")
        WS_CONNECTIONS.labels(status='connected').inc()

    def disconnect(self, call_id: str):
        if call_id in self.active_connections:
            del self.active_connections[call_id]
            logger.info(f"WebSocket disconnected: {call_id}")
            WS_CONNECTIONS.labels(status='disconnected').inc()

    async def send_json(self, call_id: str, message: dict):
        if call_id in self.active_connections:
            await self.active_connections[call_id].send_json(message)

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

manager = ConnectionManager()

# ============================================================================
# Authentication & Rate Limiting
# ============================================================================

def create_access_token(data: dict, expires_delta: timedelta = timedelta(hours=1)) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + expires_delta
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=JWT_ALGORITHM)

async def verify_token(authorization: Optional[str] = Header(None)) -> dict:
    """Verify JWT token from Authorization header"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing authorization header")

    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header")

async def check_rate_limit(tenant_id: str, limit: int = 100, window: int = 60) -> bool:
    """Simple rate limiting using Redis"""
    key = f"ratelimit:{tenant_id}:{int(time.time() // window)}"
    count = await redis_manager.incr(key)
    if count == 1:
        await redis_manager.expire(key, window)
    return count <= limit

# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting Voice AI CX Gateway...")
    await redis_manager.connect()

    # TODO: Initialize database tables
    # async with engine.begin() as conn:
    #     await conn.run_sync(Base.metadata.create_all)

    yield

    # Shutdown
    logger.info("Shutting down...")
    await redis_manager.disconnect()
    await engine.dispose()

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="Voice AI CX Platform",
    description="Real-time voice AI customer care gateway",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# ============================================================================
# Health & Metrics Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0"
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type="text/plain")

# ============================================================================
# Authentication Endpoint
# ============================================================================

@app.post("/auth/token")
async def login(tenant_id: str, api_key: str):
    """
    Generate JWT token for API access
    In production, validate api_key against database
    """
    # TODO: Validate api_key
    token = create_access_token({"tenant_id": tenant_id, "sub": tenant_id})
    return {"access_token": token, "token_type": "bearer"}

# ============================================================================
# Call Management Endpoints
# ============================================================================

@app.post("/call/start", response_model=CallStartResponse)
async def start_call(
    request: CallStartRequest,
    token: dict = Depends(verify_token),
    db: AsyncSession = Depends(get_db),
):
    """
    Initialize a new voice call session
    Returns WebSocket URL and session token
    """
    with tracer.start_as_current_span("start_call") as span:
        REQUESTS.labels(method='POST', endpoint='/call/start').inc()

        # Rate limiting
        if not await check_rate_limit(request.tenant_id, limit=100):
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        # Generate call ID and session token
        call_id = f"call_{uuid4().hex}"
        session_token = create_access_token({"call_id": call_id, "tenant_id": request.tenant_id})

        # Store call metadata in Redis
        call_data = {
            "call_id": call_id,
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "language": request.language,
            "voice_id": request.voice_id,
            "status": "initialized",
            "created_at": time.time(),
            "metadata": request.metadata,
        }
        await redis_manager.set(f"call:{call_id}", json.dumps(call_data), ex=3600)

        # TODO: Insert into PostgreSQL calls table
        # call = Call(**call_data)
        # db.add(call)
        # await db.commit()

        # Publish event
        await redis_manager.publish("call.started", call_data)
        CALL_EVENTS.labels(event_type='started').inc()

        span.set_attribute("call_id", call_id)
        span.set_attribute("tenant_id", request.tenant_id)

        logger.info(f"✅ Call started: {call_id}")

        return CallStartResponse(
            call_id=call_id,
            session_token=session_token,
            ws_url=f"ws://localhost:8000/ws/{call_id}",
            status="ready",
        )

@app.get("/call/{call_id}")
async def get_call_status(call_id: str, token: dict = Depends(verify_token)):
    """Get call status and metadata"""
    call_data = await redis_manager.get(f"call:{call_id}")
    if not call_data:
        raise HTTPException(status_code=404, detail="Call not found")

    return json.loads(call_data)

@app.post("/call/{call_id}/end")
async def end_call(call_id: str, token: dict = Depends(verify_token)):
    """End an active call"""
    call_data = await redis_manager.get(f"call:{call_id}")
    if not call_data:
        raise HTTPException(status_code=404, detail="Call not found")

    # Update status
    data = json.loads(call_data)
    data["status"] = "ended"
    data["ended_at"] = time.time()
    await redis_manager.set(f"call:{call_id}", json.dumps(data), ex=86400)  # 24h retention

    # Publish event
    await redis_manager.publish("call.ended", data)
    CALL_EVENTS.labels(event_type='ended').inc()

    # Disconnect WebSocket
    manager.disconnect(call_id)

    logger.info(f"✅ Call ended: {call_id}")

    return {"status": "ended", "call_id": call_id}

# ============================================================================
# WebSocket Endpoint (Real-time Audio Streaming)
# ============================================================================

@app.websocket("/ws/{call_id}")
async def websocket_endpoint(websocket: WebSocket, call_id: str):
    """
    WebSocket endpoint for real-time audio streaming
    Protocol:
    - Client sends: {"type": "audio", "data": "<base64 PCM>"}
    - Server sends: {"type": "transcript_interim", "text": "..."}
    - Server sends: {"type": "response", "text": "...", "audio": "<base64>"}
    """
    # TODO: Verify session token from query params

    await manager.connect(call_id, websocket)

    try:
        # Update call status
        call_data = await redis_manager.get(f"call:{call_id}")
        if call_data:
            data = json.loads(call_data)
            data["status"] = "active"
            await redis_manager.set(f"call:{call_id}", json.dumps(data), ex=3600)

        # Subscribe to Redis channels for this call
        # (In production, use separate task for pub/sub)

        while True:
            # Receive message from client
            message = await websocket.receive_json()

            msg_type = message.get("type")

            if msg_type == "audio":
                # Audio chunk received from client
                audio_data = message.get("data")  # base64-encoded PCM

                # Publish to Redis for ASR service to consume
                await redis_manager.publish(f"audio:{call_id}", json.dumps({
                    "call_id": call_id,
                    "audio": audio_data,
                    "timestamp": time.time(),
                }))

                # TODO: In production, send via gRPC to ASR service

                # Echo interim transcript (mock)
                await websocket.send_json({
                    "type": "transcript_interim",
                    "text": "Processing...",
                    "timestamp": time.time(),
                })

            elif msg_type == "barge_in":
                # User interrupted bot speech
                logger.info(f"Barge-in detected: {call_id}")
                await redis_manager.publish(f"barge_in:{call_id}", json.dumps({
                    "call_id": call_id,
                    "timestamp": time.time(),
                }))

                # Signal TTS service to stop
                CALL_EVENTS.labels(event_type='barge_in').inc()

            elif msg_type == "ping":
                # Keepalive
                await websocket.send_json({"type": "pong", "timestamp": time.time()})

            else:
                logger.warning(f"Unknown message type: {msg_type}")

    except WebSocketDisconnect:
        manager.disconnect(call_id)
        logger.info(f"Client disconnected: {call_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(call_id)

# ============================================================================
# Human Handoff Endpoint
# ============================================================================

@app.post("/handoff")
async def handoff_to_human(
    request: HandoffRequest,
    token: dict = Depends(verify_token),
):
    """
    Escalate call to human agent
    - Generates context summary
    - Posts to LiveChat/CRM
    - Returns agent info
    """
    with tracer.start_as_current_span("handoff") as span:
        call_data = await redis_manager.get(f"call:{request.call_id}")
        if not call_data:
            raise HTTPException(status_code=404, detail="Call not found")

        data = json.loads(call_data)

        # TODO: Fetch full transcript from database
        # TODO: Generate summary using LLM
        # TODO: Post to LiveChat API

        context_summary = {
            "call_id": request.call_id,
            "reason": request.reason,
            "duration_seconds": time.time() - data.get("created_at", time.time()),
            "transcript": ["User: Hello", "Bot: How can I help?", "User: I need a human"],
            "intents": ["greeting", "escalation"],
            "sentiment": "neutral",
            "metadata": data.get("metadata", {}),
        }

        # Publish handoff event
        await redis_manager.publish("handoff.requested", {
            "call_id": request.call_id,
            "queue": request.agent_queue,
            "context": context_summary,
        })

        CALL_EVENTS.labels(event_type='handoff').inc()

        logger.info(f"✅ Handoff requested: {request.call_id} → {request.agent_queue}")

        return {
            "status": "queued",
            "queue": request.agent_queue,
            "estimated_wait_seconds": 30,
            "context": context_summary,
        }

# ============================================================================
# RAG Endpoints
# ============================================================================

@app.post("/rag/query", response_model=RAGQueryResponse)
async def query_rag(
    request: RAGQueryRequest,
    token: dict = Depends(verify_token),
):
    """
    Query knowledge base using RAG
    Returns top-K relevant documents with citations
    """
    with tracer.start_as_current_span("rag_query") as span:
        start_time = time.time()

        # TODO: Call RAG service via gRPC or HTTP
        # For now, return mock results
        results = [
            {
                "text": "Our support hours are Sunday to Thursday, 9 AM to 6 PM UAE time.",
                "source": "FAQ - Support Hours",
                "score": 0.89,
                "metadata": {"page": 1, "url": "https://example.com/faq"},
            },
            {
                "text": "You can reach us at +971-4-123-4567 or support@example.ae",
                "source": "Contact Information",
                "score": 0.76,
                "metadata": {"section": "contact"},
            },
        ]

        took_ms = (time.time() - start_time) * 1000

        span.set_attribute("tenant_id", request.tenant_id)
        span.set_attribute("results_count", len(results))

        return RAGQueryResponse(results=results, took_ms=took_ms)

@app.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: IngestRequest,
    token: dict = Depends(verify_token),
):
    """
    Ingest document into knowledge base
    Queues Celery task for processing
    """
    with tracer.start_as_current_span("ingest") as span:
        task_id = f"ingest_{uuid4().hex}"

        # Queue Celery task
        # TODO: celery_app.send_task('ingest_document', ...)

        # For now, just publish event
        await redis_manager.publish("ingest.queued", {
            "task_id": task_id,
            "tenant_id": request.tenant_id,
            "source_url": request.source_url,
            "source_text": request.source_text,
            "metadata": request.metadata,
        })

        logger.info(f"✅ Ingestion queued: {task_id}")

        return IngestResponse(task_id=task_id, status="queued")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )

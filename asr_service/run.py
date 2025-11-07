#!/usr/bin/env python3
"""
ASR Service - Standalone Runner
Run the ASR service independently for testing and development

Usage:
    python run.py                    # Start server on port 8050
    python run.py --port 8051        # Custom port
    python run.py --model distil-large-v3  # Use smaller model
    python run.py --device cpu       # Use CPU instead of GPU

Test with:
    curl http://localhost:8050/health
    curl -F "file=@test.wav" http://localhost:8050/transcribe
"""

import argparse
import asyncio
import base64
import io
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path to import server module
sys.path.insert(0, str(Path(__file__).parent))

from server import ASRService, VADService

# ============================================================================
# Configuration
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance
asr_service: Optional[ASRService] = None

# ============================================================================
# Request/Response Models
# ============================================================================

class TranscribeRequest(BaseModel):
    """Request for transcription (when sending base64 audio)"""
    audio_base64: str
    language: Optional[str] = "auto"
    task: str = "transcribe"  # or "translate"

class TranscribeResponse(BaseModel):
    """Response from transcription"""
    text: str
    language: str
    language_probability: float
    duration_seconds: float
    inference_time_seconds: float
    segments: list = Field(default_factory=list)

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model: str
    device: str
    vad_enabled: bool
    timestamp: float

class StreamingTranscriptChunk(BaseModel):
    """Streaming transcript chunk"""
    type: str  # interim or final
    text: str
    language: Optional[str] = None
    timestamp: float

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="ASR Service (Standalone)",
    description="Streaming ASR with Whisper large-v3 and VAD",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize ASR service on startup"""
    global asr_service

    logger.info("🚀 Starting ASR service...")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Compute Type: {args.compute_type}")

    try:
        # Initialize ASR service
        import os
        os.environ['WHISPER_MODEL'] = args.model
        os.environ['DEVICE'] = args.device
        os.environ['COMPUTE_TYPE'] = args.compute_type

        asr_service = ASRService()

        logger.info("✅ ASR service initialized successfully!")
        logger.info(f"   GPU Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    except Exception as e:
        logger.error(f"❌ Failed to initialize ASR service: {e}", exc_info=True)
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down ASR service...")

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if asr_service else "initializing",
        model=args.model,
        device=args.device,
        vad_enabled=True,
        timestamp=time.time(),
    )

@app.get("/")
async def root():
    """Root endpoint with service info"""
    return {
        "service": "ASR Service (Standalone)",
        "version": "1.0.0",
        "model": args.model,
        "device": args.device,
        "endpoints": {
            "health": "GET /health",
            "transcribe_file": "POST /transcribe (multipart form-data)",
            "transcribe_base64": "POST /transcribe/base64 (JSON)",
            "streaming": "WebSocket /ws/transcribe",
        },
        "docs": "/docs",
    }

@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = "auto",
    task: str = "transcribe",
):
    """
    Transcribe an audio file

    Supported formats: WAV, MP3, FLAC, OGG, etc. (via soundfile/ffmpeg)

    Example:
        curl -F "file=@audio.wav" http://localhost:8050/transcribe
        curl -F "file=@audio.wav" -F "language=ar" http://localhost:8050/transcribe
    """
    if not asr_service:
        raise HTTPException(status_code=503, detail="ASR service not initialized")

    try:
        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"Transcribing file: {file.filename} ({len(content)} bytes)")

        # Transcribe
        result = await asr_service.transcribe_file(
            tmp_path,
            language=language if language != "auto" else None,
        )

        # Cleanup
        import os
        os.unlink(tmp_path)

        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            language_probability=result["language_probability"],
            duration_seconds=result["duration"],
            inference_time_seconds=result["inference_time"],
            segments=result["segments"],
        )

    except Exception as e:
        logger.error(f"Transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/base64", response_model=TranscribeResponse)
async def transcribe_base64(request: TranscribeRequest):
    """
    Transcribe base64-encoded audio

    Example:
        {
            "audio_base64": "<base64 PCM or WAV>",
            "language": "ar",
            "task": "transcribe"
        }
    """
    if not asr_service:
        raise HTTPException(status_code=503, detail="ASR service not initialized")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Try to parse as WAV first, otherwise assume PCM
        try:
            audio, sr = sf.read(io.BytesIO(audio_bytes))
        except Exception:
            # Assume raw PCM 16-bit mono 16kHz
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
            sr = 16000

        # Resample if needed
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        logger.info(f"Transcribing base64 audio ({len(audio)/16000:.1f}s)")

        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            asr_service.executor,
            asr_service._transcribe_chunk,
            audio,
            request.language if request.language != "auto" else None,
            request.task,
            True,  # final=True
        )

        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            language_probability=result["language_probability"],
            duration_seconds=result["duration"],
            inference_time_seconds=result["inference_time"],
            segments=result["segments"],
        )

    except Exception as e:
        logger.error(f"Base64 transcription failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for streaming transcription

    Protocol:
        Client → Server: {"audio": "<base64 PCM>", "language": "auto"}
        Server → Client: {"type": "interim", "text": "...", "timestamp": 123}
        Server → Client: {"type": "final", "text": "...", "language": "en"}
    """
    if not asr_service:
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await websocket.accept()
    logger.info("WebSocket client connected")

    try:
        # Get initial config
        config_msg = await websocket.receive_json()
        language = config_msg.get("language", "auto")

        logger.info(f"Starting streaming transcription (language={language})")

        # Create audio stream generator
        async def audio_generator():
            while True:
                try:
                    message = await websocket.receive_json()
                    if message.get("type") == "close":
                        break

                    audio_b64 = message.get("audio")
                    if audio_b64:
                        yield audio_b64
                except WebSocketDisconnect:
                    break

        # Stream transcription
        async for result in asr_service.transcribe_streaming(
            audio_generator(),
            language=language if language != "auto" else None,
        ):
            await websocket.send_json({
                "type": result["type"],
                "text": result["text"],
                "language": result.get("language"),
                "timestamp": result.get("timestamp"),
            })

        logger.info("Streaming transcription completed")

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        await websocket.close(code=1011, reason=str(e))

# ============================================================================
# Test Endpoint
# ============================================================================

@app.post("/test/echo")
async def test_echo():
    """
    Test endpoint that generates a simple transcript
    Useful for testing without actual audio
    """
    return {
        "text": "This is a test transcription from the ASR service.",
        "language": "en",
        "language_probability": 0.99,
        "duration_seconds": 2.5,
        "inference_time_seconds": 0.15,
    }

# ============================================================================
# CLI & Main
# ============================================================================

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="ASR Service Standalone Runner"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the service on (default: 8050)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="large-v3",
        choices=["large-v3", "large-v2", "medium", "small", "base", "tiny", "distil-large-v3"],
        help="Whisper model to use (default: large-v3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to run on (default: auto-detect)",
    )
    parser.add_argument(
        "--compute-type",
        type=str,
        default="float16" if torch.cuda.is_available() else "int8",
        choices=["float16", "int8", "float32"],
        help="Compute type for inference (default: float16 on GPU, int8 on CPU)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (dev mode)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)",
    )

    return parser.parse_args()

def main():
    """Main entry point"""
    global args
    args = parse_args()

    logger.info("=" * 60)
    logger.info("ASR Service (Standalone)")
    logger.info("=" * 60)
    logger.info(f"Port: {args.port}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Compute Type: {args.compute_type}")
    logger.info("=" * 60)

    # Check GPU availability
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("⚠️  CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"
        args.compute_type = "int8"

    # Run server
    uvicorn.run(
        "run:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
        log_level="info",
    )

if __name__ == "__main__":
    main()

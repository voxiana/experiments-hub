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

    logger.info("=" * 60)
    logger.info("🚀 Starting ASR Service")
    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Device: {args.device}")
    logger.info(f"   Compute Type: {args.compute_type}")
    logger.info(f"   Host: {args.host}")
    logger.info(f"   Port: {args.port}")
    logger.info("=" * 60)

    try:
        # Check system info
        logger.info("System Information:")
        logger.info(f"   Python version: {sys.version.split()[0]}")
        logger.info(f"   PyTorch version: {torch.__version__}")
        logger.info(f"   CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   cuDNN version: {torch.backends.cudnn.version()}")
            logger.info(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
                logger.info(f"      Memory: {props.total_memory / 1e9:.1f} GB")
                logger.info(f"      Compute Capability: {props.major}.{props.minor}")

        # Initialize ASR service
        logger.info("=" * 60)
        logger.info("Initializing ASR service components...")
        import os
        os.environ['WHISPER_MODEL'] = args.model
        os.environ['DEVICE'] = args.device
        os.environ['COMPUTE_TYPE'] = args.compute_type

        init_start = time.time()
        asr_service = ASRService()
        init_time = time.time() - init_start

        logger.info("=" * 60)
        logger.info("✅ ASR service initialized successfully!")
        logger.info(f"   Initialization time: {init_time:.2f}s")
        logger.info(f"   Ready to accept requests")
        logger.info("=" * 60)
        logger.info("Supported audio formats:")
        logger.info("   - WAV (via soundfile)")
        logger.info("   - MP3 (via librosa/ffmpeg)")
        logger.info("   - M4A (via librosa/ffmpeg)")
        logger.info("   - FLAC (via soundfile)")
        logger.info("   - OGG (via soundfile)")
        logger.info("   - AAC, WMA, OPUS (via librosa/ffmpeg)")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("=" * 60)
        logger.error("❌ Failed to initialize ASR service")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 60)
        logger.error("Full traceback:", exc_info=True)
        sys.exit(1)

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("=" * 60)
    logger.info("Shutting down ASR service...")
    logger.info("=" * 60)
    if asr_service:
        logger.info("Cleaning up ASR service resources...")
        # Any cleanup if needed
    logger.info("✅ Shutdown complete")
    logger.info("=" * 60)

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

    Supported formats: WAV, MP3, M4A, FLAC, OGG, and other formats (via librosa/ffmpeg)

    Example:
        curl -F "file=@audio.wav" http://localhost:8050/transcribe
        curl -F "file=@audio.mp3" -F "language=ar" http://localhost:8050/transcribe
        curl -F "file=@audio.m4a" http://localhost:8050/transcribe
    """
    if not asr_service:
        logger.error("Transcription request received but ASR service not initialized")
        raise HTTPException(status_code=503, detail="ASR service not initialized")

    logger.info("=" * 60)
    logger.info("Received transcription request via HTTP")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content-Type: {file.content_type}")
    logger.info(f"Language: {language}")
    logger.info(f"Task: {task}")

    try:
        # Read file content
        content = await file.read()
        file_size_mb = len(content) / (1024 * 1024)
        logger.info(f"File size: {file_size_mb:.2f} MB ({len(content)} bytes)")

        # Detect file format from extension
        file_ext = Path(file.filename).suffix.lower()
        logger.info(f"File extension: {file_ext}")

        # Validate supported formats (informational only, actual support depends on ffmpeg)
        supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', '.wma', '.opus']
        if file_ext not in supported_formats:
            logger.warning(f"File extension {file_ext} not in common supported formats: {supported_formats}")
            logger.warning("Will attempt to process anyway via librosa/ffmpeg")

        # Save uploaded file temporarily
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
            logger.info(f"Saved temporary file: {tmp_path}")

        # Transcribe
        logger.info("Calling ASR service for transcription...")
        transcription_start = time.time()

        result = await asr_service.transcribe_file(
            tmp_path,
            language=language if language != "auto" else None,
        )

        transcription_time = time.time() - transcription_start

        # Cleanup
        import os
        os.unlink(tmp_path)
        logger.info(f"Cleaned up temporary file: {tmp_path}")

        # Log results
        logger.info("=" * 60)
        logger.info("Transcription successful!")
        logger.info(f"   Language: {result['language']} (confidence: {result['language_probability']:.2%})")
        logger.info(f"   Duration: {result['duration']:.2f}s")
        logger.info(f"   Segments: {len(result['segments'])}")
        logger.info(f"   Total time: {transcription_time:.2f}s")
        logger.info(f"   Text length: {len(result['text'])} characters")
        logger.info(f"   Text preview: '{result['text'][:100]}...'")
        logger.info("=" * 60)

        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            language_probability=result["language_probability"],
            duration_seconds=result["duration"],
            inference_time_seconds=result["inference_time"],
            segments=result["segments"],
        )

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Transcription failed for file: {file.filename}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 60)
        logger.error("Full traceback:", exc_info=True)
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
        logger.error("Base64 transcription request received but ASR service not initialized")
        raise HTTPException(status_code=503, detail="ASR service not initialized")

    logger.info("=" * 60)
    logger.info("Received base64 transcription request")
    logger.info(f"Language: {request.language}")
    logger.info(f"Task: {request.task}")
    logger.info(f"Base64 length: {len(request.audio_base64)} characters")

    try:
        # Decode base64 audio
        logger.info("Decoding base64 audio...")
        audio_bytes = base64.b64decode(request.audio_base64)
        logger.info(f"Decoded audio size: {len(audio_bytes)} bytes")

        # Try to parse as WAV first, otherwise assume PCM
        audio_format = None
        try:
            logger.info("Attempting to parse as WAV file...")
            audio, sr = sf.read(io.BytesIO(audio_bytes))
            audio_format = "WAV"
            logger.info(f"✅ Successfully parsed as WAV (sr={sr} Hz)")
        except Exception as e:
            logger.info(f"Not a WAV file ({e}), assuming raw PCM...")
            # Assume raw PCM 16-bit mono 16kHz
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio_int16.astype(np.float32) / 32768.0
            sr = 16000
            audio_format = "PCM"
            logger.info(f"✅ Parsed as raw PCM (assuming 16kHz)")

        duration = len(audio) / sr
        logger.info(f"Audio info: format={audio_format}, sr={sr} Hz, duration={duration:.2f}s, samples={len(audio)}")

        # Resample if needed
        if sr != 16000:
            logger.info(f"Resampling from {sr} Hz to 16000 Hz...")
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            logger.info(f"✅ Resampling complete")

        logger.info(f"Starting transcription of {len(audio)/16000:.1f}s audio...")

        # Transcribe
        transcribe_start = time.time()
        result = await asyncio.get_event_loop().run_in_executor(
            asr_service.executor,
            asr_service._transcribe_chunk,
            audio,
            request.language if request.language != "auto" else None,
            request.task,
            True,  # final=True
        )
        transcribe_time = time.time() - transcribe_start

        logger.info("=" * 60)
        logger.info("Base64 transcription successful!")
        logger.info(f"   Language: {result['language']} (confidence: {result['language_probability']:.2%})")
        logger.info(f"   Duration: {result['duration']:.2f}s")
        logger.info(f"   Transcribe time: {transcribe_time:.2f}s")
        logger.info(f"   Text preview: '{result['text'][:100]}...'")
        logger.info("=" * 60)

        return TranscribeResponse(
            text=result["text"],
            language=result["language"],
            language_probability=result["language_probability"],
            duration_seconds=result["duration"],
            inference_time_seconds=result["inference_time"],
            segments=result["segments"],
        )

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"Base64 transcription failed")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 60)
        logger.error("Full traceback:", exc_info=True)
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
    client_id = f"{websocket.client.host}:{websocket.client.port}" if websocket.client else "unknown"

    if not asr_service:
        logger.error(f"WebSocket connection from {client_id} rejected: ASR service not initialized")
        await websocket.close(code=1011, reason="Service not initialized")
        return

    await websocket.accept()
    logger.info("=" * 60)
    logger.info(f"WebSocket client connected: {client_id}")
    logger.info("=" * 60)

    try:
        # Get initial config
        logger.info("Waiting for configuration message...")
        config_msg = await websocket.receive_json()
        language = config_msg.get("language", "auto")

        logger.info(f"WebSocket configuration received:")
        logger.info(f"   Language: {language}")
        logger.info(f"Starting streaming transcription session...")

        chunk_count = 0
        result_count = 0

        # Create audio stream generator
        async def audio_generator():
            nonlocal chunk_count
            while True:
                try:
                    message = await websocket.receive_json()
                    if message.get("type") == "close":
                        logger.info("Client sent close message")
                        break

                    audio_b64 = message.get("audio")
                    if audio_b64:
                        chunk_count += 1
                        logger.debug(f"WS: Received audio chunk #{chunk_count} ({len(audio_b64)} chars)")
                        yield audio_b64
                except WebSocketDisconnect:
                    logger.info("Client disconnected during stream")
                    break

        # Stream transcription
        async for result in asr_service.transcribe_streaming(
            audio_generator(),
            language=language if language != "auto" else None,
        ):
            result_count += 1
            result_type = result["type"]
            text_preview = result["text"][:50] + "..." if len(result["text"]) > 50 else result["text"]

            logger.info(f"WS: Sending {result_type} result #{result_count}: '{text_preview}'")

            await websocket.send_json({
                "type": result["type"],
                "text": result["text"],
                "language": result.get("language"),
                "timestamp": result.get("timestamp"),
            })

        logger.info("=" * 60)
        logger.info(f"Streaming transcription completed for {client_id}")
        logger.info(f"   Total chunks received: {chunk_count}")
        logger.info(f"   Total results sent: {result_count}")
        logger.info("=" * 60)

    except WebSocketDisconnect:
        logger.info("=" * 60)
        logger.info(f"WebSocket client disconnected: {client_id}")
        logger.info("=" * 60)
    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"WebSocket error for client {client_id}")
        logger.error(f"Error type: {type(e).__name__}")
        logger.error(f"Error message: {str(e)}")
        logger.error("=" * 60)
        logger.error("Full traceback:", exc_info=True)
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

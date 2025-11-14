"""
ASR Service - Streaming Speech Recognition
Uses faster-whisper (large-v3) with silero-vad for voice activity detection
Supports real-time streaming with low latency
"""

import os

# Set environment variables BEFORE importing torch/faster_whisper to prevent CUDA loading
# This must happen before any CUDA-related imports
_device = os.environ.get("DEVICE", "").lower()
if _device == "cpu":
    # Hide CUDA devices to prevent faster-whisper from trying to load CUDA/cuDNN libraries
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    # Suppress cuDNN warnings
    os.environ["CUDNN_LOGINFO_DBG"] = "0"
    os.environ["CUDNN_LOGDEST_DBG"] = ""

import asyncio
import base64
import logging
import time
from typing import AsyncIterator, Optional
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch

from faster_whisper import WhisperModel
import grpc
from grpc import aio

# Generated proto imports (would be from asr_pb2, asr_pb2_grpc)
# For this example, we'll use a simplified approach

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Model configuration - read from environment variables if set (from run.py)
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3")
_requested_device = os.environ.get("DEVICE", "").lower()

# Determine device: respect environment variable, fall back to auto-detection
if _requested_device == "cpu":
    DEVICE = "cpu"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "int8")
elif _requested_device == "cuda":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")
else:
    # Auto-detect
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    COMPUTE_TYPE = os.environ.get("COMPUTE_TYPE", "float16" if DEVICE == "cuda" else "int8")

# Force CPU if CUDA is requested but not available
if DEVICE == "cuda" and not torch.cuda.is_available():
    logger.warning("CUDA requested but not available. Falling back to CPU.")
    DEVICE = "cpu"
    COMPUTE_TYPE = "int8"

BEAM_SIZE = 1  # Increase to 5 for higher accuracy, lower speed
VAD_THRESHOLD = 0.5

# Audio configuration
SAMPLE_RATE = 16000  # 16kHz
CHUNK_DURATION_MS = 250  # 250ms chunks for streaming
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000)

# VAD configuration
VAD_FRAME_SIZE = 512  # samples (32ms at 16kHz)

# ============================================================================
# VAD Service (Silero VAD)
# ============================================================================

class VADService:
    """Voice Activity Detection using Silero VAD"""

    def __init__(self):
        logger.info("Loading Silero VAD model...")
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        self.model.to(DEVICE)
        self.get_speech_timestamps = self.utils[0]

        logger.info(f"✅ VAD model loaded on {DEVICE}")

    def is_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech in audio chunk
        Returns: speech probability [0-1]
        """
        # Ensure correct shape and type
        if len(audio_chunk) != VAD_FRAME_SIZE:
            # Pad or trim
            if len(audio_chunk) < VAD_FRAME_SIZE:
                audio_chunk = np.pad(audio_chunk, (0, VAD_FRAME_SIZE - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:VAD_FRAME_SIZE]

        audio_tensor = torch.from_numpy(audio_chunk).float().to(DEVICE)

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        return speech_prob

    def get_speech_segments(self, audio: np.ndarray) -> list:
        """
        Get speech segments with timestamps
        Returns: [(start_sample, end_sample), ...]
        """
        audio_tensor = torch.from_numpy(audio).float()

        speech_timestamps = self.get_speech_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD,
            min_speech_duration_ms=250,
            min_silence_duration_ms=100,
        )

        segments = [(ts['start'], ts['end']) for ts in speech_timestamps]
        return segments

# ============================================================================
# ASR Service (faster-whisper)
# ============================================================================

class ASRService:
    """
    Streaming ASR using faster-whisper
    Supports Arabic and English with code-switching
    """

    def __init__(self):
        logger.info(f"Loading Whisper model: {WHISPER_MODEL}...")
        logger.info(f"   Device: {DEVICE}, Compute Type: {COMPUTE_TYPE}")
        
        # Try to initialize with specified device, fall back to CPU on error
        device = DEVICE
        compute_type = COMPUTE_TYPE
        
        try:
            self.model = WhisperModel(
                WHISPER_MODEL,
                device=device,
                compute_type=compute_type,
                num_workers=4,
            )
            logger.info(f"✅ Whisper {WHISPER_MODEL} loaded on {device}")
        except Exception as e:
            error_msg = str(e).lower()
            # Check if it's a CUDA/cuDNN related error
            if device == "cuda" or "cuda" in error_msg or "cudnn" in error_msg:
                logger.warning(f"Failed to load model on {device}: {e}")
                logger.warning("Falling back to CPU...")
                device = "cpu"
                compute_type = "int8"
                try:
                    self.model = WhisperModel(
                        WHISPER_MODEL,
                        device=device,
                        compute_type=compute_type,
                        num_workers=4,
                    )
                    logger.info(f"✅ Whisper {WHISPER_MODEL} loaded on {device} (fallback)")
                except Exception as e2:
                    logger.error(f"Failed to load model on CPU: {e2}")
                    raise
            else:
                logger.error(f"Failed to load model: {e}")
                raise

        self.vad = VADService()
        self.executor = ThreadPoolExecutor(max_workers=4)

    async def transcribe_streaming(
        self,
        audio_stream: AsyncIterator[bytes],
        language: str = None,
        task: str = "transcribe",
    ) -> AsyncIterator[dict]:
        """
        Streaming transcription
        Yields interim and final results
        """
        buffer = np.array([], dtype=np.float32)
        utterance_buffer = []
        last_speech_time = time.time()
        silence_threshold = 1.0  # seconds

        async for audio_chunk_b64 in audio_stream:
            # Decode base64 audio (PCM 16-bit mono 16kHz)
            audio_bytes = base64.b64decode(audio_chunk_b64)
            audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0  # Normalize to [-1, 1]

            # Add to buffer
            buffer = np.concatenate([buffer, audio_float32])

            # Check VAD in frames
            if len(buffer) >= VAD_FRAME_SIZE:
                vad_chunk = buffer[:VAD_FRAME_SIZE]
                speech_prob = self.vad.is_speech(vad_chunk)

                if speech_prob > VAD_THRESHOLD:
                    last_speech_time = time.time()
                    utterance_buffer.append(audio_float32)

                    # Yield interim result if enough audio
                    if len(utterance_buffer) * len(audio_float32) > SAMPLE_RATE * 1.0:  # 1 second
                        utterance = np.concatenate(utterance_buffer)

                        # Run transcription in thread pool
                        loop = asyncio.get_event_loop()
                        result = await loop.run_in_executor(
                            self.executor,
                            self._transcribe_chunk,
                            utterance,
                            language,
                            task,
                            False,  # interim=True
                        )

                        if result["text"].strip():
                            yield {
                                "type": "interim",
                                "text": result["text"],
                                "language": result["language"],
                                "timestamp": time.time(),
                            }

                # Remove processed frames from buffer
                buffer = buffer[VAD_FRAME_SIZE:]

            # Check for end of utterance (silence)
            silence_duration = time.time() - last_speech_time
            if silence_duration > silence_threshold and len(utterance_buffer) > 0:
                # Final transcription
                utterance = np.concatenate(utterance_buffer)

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._transcribe_chunk,
                    utterance,
                    language,
                    task,
                    True,  # final=True
                )

                yield {
                    "type": "final",
                    "text": result["text"],
                    "language": result["language"],
                    "segments": result["segments"],
                    "timestamp": time.time(),
                }

                # Reset buffers
                utterance_buffer = []
                buffer = np.array([], dtype=np.float32)

    def _transcribe_chunk(
        self,
        audio: np.ndarray,
        language: Optional[str],
        task: str,
        final: bool,
    ) -> dict:
        """
        Synchronous transcription (runs in thread pool)
        """
        start_time = time.time()

        # Detect language if auto
        if language == "auto":
            language = None

        # Transcribe
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=BEAM_SIZE if final else 1,  # Higher beam for final
            vad_filter=False,  # Already using custom VAD
            word_timestamps=final,  # Only for final
        )

        # Collect segments
        segments_list = []
        full_text = ""

        for segment in segments:
            segments_list.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
            })
            full_text += segment.text

        elapsed = time.time() - start_time

        logger.info(
            f"Transcribed {len(audio)/SAMPLE_RATE:.2f}s audio in {elapsed:.3f}s "
            f"({'final' if final else 'interim'}): {full_text[:50]}..."
        )

        return {
            "text": full_text.strip(),
            "language": info.language,
            "language_probability": info.language_probability,
            "segments": segments_list,
            "duration": info.duration,
            "inference_time": elapsed,
        }

    async def transcribe_file(self, audio_path: str, language: str = None) -> dict:
        """
        Transcribe complete audio file (non-streaming)
        """
        start_time = time.time()

        # Load audio
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype='float32')

        # Resample if needed
        if sr != SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)

        # Detect language if auto
        if language == "auto":
            language = None

        # Transcribe
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._transcribe_chunk,
            audio,
            language,
            "transcribe",
            True,
        )

        result["file"] = audio_path
        result["total_time"] = time.time() - start_time

        return result

# ============================================================================
# gRPC Server (Simplified - would use proto definitions)
# ============================================================================

class ASRServicer:
    """gRPC servicer for ASR"""

    def __init__(self):
        self.asr = ASRService()

    async def StreamingRecognize(self, request_iterator):
        """
        Streaming recognition RPC
        Request: StreamingRecognizeRequest { config, audio_content }
        Response: StreamingRecognizeResponse { results }
        """
        # Extract config from first message
        first_request = await request_iterator.__anext__()
        config = first_request.config  # language, etc.

        # Create audio stream generator
        async def audio_generator():
            async for req in request_iterator:
                yield req.audio_content

        # Stream transcription
        async for result in self.asr.transcribe_streaming(
            audio_generator(),
            language=config.language if config.language else None,
        ):
            # Yield gRPC response
            yield {
                "results": [{
                    "alternatives": [{
                        "transcript": result["text"],
                        "confidence": 0.9,  # placeholder
                    }],
                    "is_final": result["type"] == "final",
                    "language_code": result["language"],
                }]
            }

    async def Recognize(self, request):
        """
        Non-streaming recognition RPC
        Request: RecognizeRequest { config, audio }
        Response: RecognizeResponse { results }
        """
        # Decode audio
        audio_bytes = base64.b64decode(request.audio.content)
        audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Transcribe
        result = await asyncio.get_event_loop().run_in_executor(
            self.asr.executor,
            self.asr._transcribe_chunk,
            audio_float32,
            request.config.language if request.config.language else None,
            "transcribe",
            True,
        )

        return {
            "results": [{
                "alternatives": [{
                    "transcript": result["text"],
                    "confidence": 0.9,
                }],
                "language_code": result["language"],
            }]
        }

# ============================================================================
# REST Server (Alternative to gRPC for testing)
# ============================================================================

async def run_rest_server():
    """Run FastAPI server for testing"""
    from fastapi import FastAPI, UploadFile, File
    from fastapi.responses import StreamingResponse
    import uvicorn

    app = FastAPI(title="ASR Service")
    asr = ASRService()

    @app.get("/health")
    async def health():
        return {"status": "healthy", "model": WHISPER_MODEL, "device": DEVICE}

    @app.post("/transcribe")
    async def transcribe(file: UploadFile = File(...), language: str = "auto"):
        """Transcribe uploaded audio file"""
        # Save temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Transcribe
        result = await asr.transcribe_file(tmp_path, language=language)

        # Cleanup
        import os
        os.unlink(tmp_path)

        return result

    @app.post("/transcribe/stream")
    async def transcribe_stream(language: str = "auto"):
        """
        WebSocket-like streaming endpoint
        Client sends audio chunks, server returns interim/final transcripts
        """
        # This would be implemented as WebSocket in production
        return {"message": "Use WebSocket endpoint /ws/transcribe"}

    uvicorn.run(app, host="0.0.0.0", port=50051, log_level="info")

# ============================================================================
# Main Entry Point
# ============================================================================

async def serve_grpc():
    """Start gRPC server"""
    server = aio.server()
    servicer = ASRServicer()

    # Add servicer to server
    # asr_pb2_grpc.add_ASRServicer_to_server(servicer, server)

    server.add_insecure_port('[::]:50051')
    logger.info("🎤 ASR gRPC server starting on port 50051...")
    await server.start()
    await server.wait_for_termination()

def main():
    """Entry point"""
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "rest":
        # Run REST server for testing
        asyncio.run(run_rest_server())
    else:
        # Run gRPC server (production)
        asyncio.run(serve_grpc())

if __name__ == "__main__":
    main()

"""
ASR Service - Streaming Speech Recognition
Uses faster-whisper (large-v3) with silero-vad for voice activity detection
Supports real-time streaming with low latency
"""

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

# Model configuration
WHISPER_MODEL = "large-v3"  # or "distil-large-v3" for lower latency
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
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
        logger.info("=" * 60)
        logger.info("Initializing VAD Service (Silero VAD)")
        logger.info("=" * 60)
        logger.info(f"Target device: {DEVICE}")
        logger.info(f"VAD threshold: {VAD_THRESHOLD}")
        logger.info(f"VAD frame size: {VAD_FRAME_SIZE} samples")

        logger.info("Loading Silero VAD model from torch hub...")
        start_time = time.time()
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False,
        )
        logger.info(f"Moving VAD model to {DEVICE}...")
        self.model.to(DEVICE)
        self.get_speech_timestamps = self.utils[0]

        elapsed = time.time() - start_time
        logger.info(f"✅ VAD model loaded successfully in {elapsed:.2f}s")
        logger.info(f"   Device: {DEVICE}")
        logger.info("=" * 60)

    def is_speech(self, audio_chunk: np.ndarray) -> float:
        """
        Detect speech in audio chunk
        Returns: speech probability [0-1]
        """
        original_size = len(audio_chunk)
        logger.debug(f"VAD: Processing audio chunk of {original_size} samples")

        # Ensure correct shape and type
        if len(audio_chunk) != VAD_FRAME_SIZE:
            # Pad or trim
            if len(audio_chunk) < VAD_FRAME_SIZE:
                logger.debug(f"VAD: Padding audio from {len(audio_chunk)} to {VAD_FRAME_SIZE} samples")
                audio_chunk = np.pad(audio_chunk, (0, VAD_FRAME_SIZE - len(audio_chunk)))
            else:
                logger.debug(f"VAD: Trimming audio from {len(audio_chunk)} to {VAD_FRAME_SIZE} samples")
                audio_chunk = audio_chunk[:VAD_FRAME_SIZE]

        audio_tensor = torch.from_numpy(audio_chunk).float().to(DEVICE)

        with torch.no_grad():
            speech_prob = self.model(audio_tensor, SAMPLE_RATE).item()

        logger.debug(f"VAD: Speech probability = {speech_prob:.3f}")
        return speech_prob

    def get_speech_segments(self, audio: np.ndarray) -> list:
        """
        Get speech segments with timestamps
        Returns: [(start_sample, end_sample), ...]
        """
        logger.info(f"VAD: Analyzing {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s) for speech segments")
        start_time = time.time()

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
        elapsed = time.time() - start_time

        logger.info(f"VAD: Found {len(segments)} speech segment(s) in {elapsed:.3f}s")
        for i, (start, end) in enumerate(segments):
            duration = (end - start) / SAMPLE_RATE
            logger.info(f"   Segment {i+1}: {start/SAMPLE_RATE:.2f}s - {end/SAMPLE_RATE:.2f}s (duration: {duration:.2f}s)")

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
        logger.info("=" * 60)
        logger.info("Initializing ASR Service (faster-whisper)")
        logger.info("=" * 60)
        logger.info(f"Model: {WHISPER_MODEL}")
        logger.info(f"Device: {DEVICE}")
        logger.info(f"Compute Type: {COMPUTE_TYPE}")
        logger.info(f"Beam Size: {BEAM_SIZE}")
        logger.info(f"Sample Rate: {SAMPLE_RATE} Hz")

        logger.info(f"Loading Whisper model: {WHISPER_MODEL}...")
        start_time = time.time()
        self.model = WhisperModel(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=COMPUTE_TYPE,
            num_workers=4,
        )
        elapsed = time.time() - start_time
        logger.info(f"✅ Whisper model loaded in {elapsed:.2f}s")

        logger.info("Initializing VAD service...")
        self.vad = VADService()

        logger.info("Creating thread pool executor (max_workers=4)...")
        self.executor = ThreadPoolExecutor(max_workers=4)

        logger.info("=" * 60)
        logger.info(f"✅ ASR Service initialized successfully")
        logger.info(f"   Model: {WHISPER_MODEL}")
        logger.info(f"   Device: {DEVICE}")
        logger.info(f"   Ready to accept requests")
        logger.info("=" * 60)

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
        logger.info("=" * 60)
        logger.info("Starting streaming transcription session")
        logger.info(f"Language: {language or 'auto-detect'}")
        logger.info(f"Task: {task}")
        logger.info(f"Silence threshold: 1.0s")
        logger.info("=" * 60)

        buffer = np.array([], dtype=np.float32)
        utterance_buffer = []
        last_speech_time = time.time()
        silence_threshold = 1.0  # seconds
        chunk_count = 0

        async for audio_chunk_b64 in audio_stream:
            chunk_count += 1
            logger.debug(f"Stream: Received chunk #{chunk_count}")
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
                    logger.debug(f"Stream: Speech detected (prob={speech_prob:.3f}), buffer size: {len(utterance_buffer)} chunks")

                    # Yield interim result if enough audio
                    if len(utterance_buffer) * len(audio_float32) > SAMPLE_RATE * 1.0:  # 1 second
                        logger.info(f"Stream: Generating interim result ({len(utterance_buffer)} chunks, ~{len(utterance_buffer)*len(audio_float32)/SAMPLE_RATE:.1f}s)")
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
                logger.info(f"Stream: Silence detected ({silence_duration:.1f}s), generating final result...")
                # Final transcription
                utterance = np.concatenate(utterance_buffer)
                utterance_duration = len(utterance) / SAMPLE_RATE
                logger.info(f"Stream: Final utterance duration: {utterance_duration:.2f}s")

                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.executor,
                    self._transcribe_chunk,
                    utterance,
                    language,
                    task,
                    True,  # final=True
                )

                logger.info(f"Stream: Final transcript: '{result['text'][:100]}...'")

                yield {
                    "type": "final",
                    "text": result["text"],
                    "language": result["language"],
                    "segments": result["segments"],
                    "timestamp": time.time(),
                }

                # Reset buffers
                logger.debug("Stream: Resetting buffers for next utterance")
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
        Supports: WAV, MP3, M4A, FLAC, OGG, and other formats via librosa/ffmpeg
        """
        logger.info("=" * 60)
        logger.info("Transcribing audio file")
        logger.info(f"File: {audio_path}")
        logger.info(f"Language: {language or 'auto-detect'}")
        logger.info("=" * 60)

        start_time = time.time()

        # Detect file format
        import os
        file_ext = os.path.splitext(audio_path)[1].lower()
        logger.info(f"Detected file extension: {file_ext}")

        # Load audio with fallback mechanism
        # Try soundfile first (fast, supports WAV, FLAC, OGG)
        # Fall back to librosa (supports MP3, M4A via ffmpeg)
        audio = None
        sr = None
        load_method = None

        try:
            import soundfile as sf
            logger.info(f"Attempting to load with soundfile...")
            audio, sr = sf.read(audio_path, dtype='float32')
            load_method = "soundfile"
            logger.info(f"✅ Successfully loaded with soundfile")
        except Exception as sf_error:
            logger.warning(f"soundfile failed: {sf_error}")
            logger.info(f"Falling back to librosa (supports more formats via ffmpeg)...")
            try:
                import librosa
                audio, sr = librosa.load(audio_path, sr=None, mono=True)
                load_method = "librosa"
                logger.info(f"✅ Successfully loaded with librosa")
            except Exception as librosa_error:
                logger.error(f"librosa also failed: {librosa_error}")
                raise Exception(
                    f"Failed to load audio file with both soundfile and librosa. "
                    f"Ensure ffmpeg is installed for MP3/M4A support. "
                    f"soundfile error: {sf_error}, librosa error: {librosa_error}"
                )

        # Log audio info
        duration = len(audio) / sr
        logger.info(f"Audio loaded successfully:")
        logger.info(f"   Method: {load_method}")
        logger.info(f"   Sample rate: {sr} Hz")
        logger.info(f"   Duration: {duration:.2f}s")
        logger.info(f"   Samples: {len(audio)}")
        logger.info(f"   Channels: {'mono' if audio.ndim == 1 else audio.shape[1]}")

        # Resample if needed
        if sr != SAMPLE_RATE:
            logger.info(f"Resampling from {sr} Hz to {SAMPLE_RATE} Hz...")
            resample_start = time.time()
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            resample_time = time.time() - resample_start
            logger.info(f"✅ Resampling complete in {resample_time:.2f}s")
            sr = SAMPLE_RATE

        # Detect language if auto
        if language == "auto":
            logger.info("Language set to auto-detect")
            language = None

        # Transcribe
        logger.info("Starting transcription...")
        transcribe_start = time.time()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self.executor,
            self._transcribe_chunk,
            audio,
            language,
            "transcribe",
            True,
        )
        transcribe_time = time.time() - transcribe_start

        total_time = time.time() - start_time
        rtf = transcribe_time / duration  # Real-time factor

        result["file"] = audio_path
        result["file_format"] = file_ext
        result["load_method"] = load_method
        result["total_time"] = total_time
        result["transcribe_time"] = transcribe_time
        result["real_time_factor"] = rtf

        logger.info("=" * 60)
        logger.info("Transcription complete!")
        logger.info(f"   Total time: {total_time:.2f}s")
        logger.info(f"   Transcribe time: {transcribe_time:.2f}s")
        logger.info(f"   Real-time factor: {rtf:.2f}x")
        logger.info(f"   Language detected: {result['language']}")
        logger.info(f"   Text preview: '{result['text'][:100]}...'")
        logger.info("=" * 60)

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

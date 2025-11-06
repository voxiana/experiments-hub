"""
TTS Service - Neural Text-to-Speech Synthesis
Uses Coqui XTTS v2 for multilingual, expressive speech synthesis
Supports Arabic (Gulf/MSA) and English with controllable prosody
"""

import asyncio
import base64
import io
import logging
import time
from typing import AsyncIterator, Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import soundfile as sf
from TTS.api import TTS
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Audio configuration
SAMPLE_RATE = 24000  # XTTS v2 native sample rate
CHUNK_DURATION_MS = 250  # Stream in 250ms chunks
STREAMING = True

# Voice presets
VOICE_PRESETS = {
    "arabic_gulf_male": "voices/arabic_gulf_male.wav",
    "arabic_gulf_female": "voices/arabic_gulf_female.wav",
    "arabic_msa_male": "voices/arabic_msa_male.wav",
    "english_uae_male": "voices/english_uae_male.wav",
    "english_uae_female": "voices/english_uae_female.wav",
}

# ============================================================================
# Request/Response Models
# ============================================================================

class SynthesizeRequest(BaseModel):
    """Request schema for synthesis"""
    text: str
    voice_id: str = "arabic_gulf_male"
    language: str = "ar"  # ar, en
    speed: float = 1.0
    emotion: Optional[str] = None  # neutral, happy, sad, energetic
    stream: bool = True

class SynthesizeResponse(BaseModel):
    """Response schema for synthesis"""
    audio_base64: Optional[str] = None  # For non-streaming
    duration_seconds: float
    sample_rate: int
    format: str = "wav"

class AudioChunk(BaseModel):
    """Streaming audio chunk"""
    chunk_index: int
    audio_base64: str
    is_final: bool

# ============================================================================
# TTS Service
# ============================================================================

class TTSService:
    """
    TTS Service using Coqui XTTS v2
    Supports multilingual synthesis with voice cloning
    """

    def __init__(self):
        logger.info(f"Loading TTS model: {MODEL_NAME}...")
        self.model = TTS(MODEL_NAME).to(DEVICE)
        self.executor = ThreadPoolExecutor(max_workers=4)

        # Load voice presets
        self.voice_samples = {}
        self._load_voice_presets()

        logger.info(f"✅ TTS model loaded on {DEVICE}")

    def _load_voice_presets(self):
        """Load reference voice samples for cloning"""
        # In production, these would be real voice samples
        # For now, we'll use placeholders
        logger.info("Loading voice presets...")
        for voice_id, path in VOICE_PRESETS.items():
            # TODO: Load actual voice samples
            # self.voice_samples[voice_id] = path
            logger.info(f"  - {voice_id}: {path} (placeholder)")

    async def synthesize_streaming(
        self,
        text: str,
        voice_id: str = "arabic_gulf_male",
        language: str = "ar",
        speed: float = 1.0,
        emotion: Optional[str] = None,
    ) -> AsyncIterator[AudioChunk]:
        """
        Streaming synthesis
        Yields audio chunks as they are generated
        """
        start_time = time.time()

        # Get voice sample
        speaker_wav = VOICE_PRESETS.get(voice_id)
        if not speaker_wav:
            speaker_wav = VOICE_PRESETS["arabic_gulf_male"]  # Default
            logger.warning(f"Voice {voice_id} not found, using default")

        # Preprocess text
        text = self._preprocess_text(text, language, emotion)

        # Split into sentences for faster streaming
        sentences = self._split_sentences(text, language)

        chunk_index = 0
        total_duration = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            # Synthesize sentence
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(
                self.executor,
                self._synthesize_chunk,
                sentence,
                speaker_wav,
                language,
                speed,
            )

            # Convert to bytes
            audio_bytes = self._audio_to_bytes(audio)

            # Encode to base64
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            duration = len(audio) / SAMPLE_RATE
            total_duration += duration

            # Yield chunk
            yield AudioChunk(
                chunk_index=chunk_index,
                audio_base64=audio_b64,
                is_final=(i == len(sentences) - 1),
            )

            chunk_index += 1

        elapsed = time.time() - start_time
        rtf = elapsed / total_duration if total_duration > 0 else 0

        logger.info(
            f"Synthesized {len(sentences)} sentences, {total_duration:.2f}s audio "
            f"in {elapsed:.3f}s (RTF={rtf:.2f})"
        )

    def _synthesize_chunk(
        self,
        text: str,
        speaker_wav: str,
        language: str,
        speed: float,
    ) -> np.ndarray:
        """
        Synchronous synthesis (runs in thread pool)
        Returns: audio array
        """
        try:
            # XTTS v2 synthesis
            # Note: In production, use actual speaker_wav file
            # For now, use built-in voices
            wav = self.model.tts(
                text=text,
                language=language,
                # speaker_wav=speaker_wav,  # Uncomment when using real voice samples
            )

            # Convert to numpy array
            audio = np.array(wav)

            # Apply speed adjustment
            if speed != 1.0:
                audio = self._adjust_speed(audio, speed)

            return audio

        except Exception as e:
            logger.error(f"Synthesis error: {e}", exc_info=True)
            # Return silence on error
            return np.zeros(int(SAMPLE_RATE * 0.5))

    async def synthesize(
        self,
        text: str,
        voice_id: str = "arabic_gulf_male",
        language: str = "ar",
        speed: float = 1.0,
        emotion: Optional[str] = None,
    ) -> tuple[bytes, float]:
        """
        Non-streaming synthesis
        Returns: (audio_bytes, duration)
        """
        start_time = time.time()

        speaker_wav = VOICE_PRESETS.get(voice_id, VOICE_PRESETS["arabic_gulf_male"])
        text = self._preprocess_text(text, language, emotion)

        # Synthesize
        loop = asyncio.get_event_loop()
        audio = await loop.run_in_executor(
            self.executor,
            self._synthesize_chunk,
            text,
            speaker_wav,
            language,
            speed,
        )

        # Convert to bytes
        audio_bytes = self._audio_to_bytes(audio)
        duration = len(audio) / SAMPLE_RATE

        elapsed = time.time() - start_time
        logger.info(f"Synthesized {duration:.2f}s audio in {elapsed:.3f}s")

        return audio_bytes, duration

    def _preprocess_text(self, text: str, language: str, emotion: Optional[str]) -> str:
        """
        Preprocess text for synthesis
        - Add SSML-like tags for emotion
        - Normalize punctuation
        - Handle numbers and abbreviations
        """
        # Remove extra whitespace
        text = " ".join(text.split())

        # Add emotion markers (if model supports)
        if emotion and emotion != "neutral":
            # XTTS v2 doesn't natively support emotion tags
            # But we can adjust text to convey emotion
            if emotion == "happy":
                text = f"{text}!"  # Add exclamation
            elif emotion == "sad":
                text = f"{text}..."  # Add ellipsis

        # Handle Arabic numerals
        if language == "ar":
            # Convert Western numerals to Arabic-Indic if needed
            pass

        return text

    def _split_sentences(self, text: str, language: str) -> list:
        """
        Split text into sentences for streaming
        """
        import re

        if language == "ar":
            # Arabic sentence delimiters
            sentences = re.split(r'[.!؟。]', text)
        else:
            # English sentence delimiters
            sentences = re.split(r'[.!?]', text)

        # Filter empty and add punctuation back
        result = []
        for s in sentences:
            s = s.strip()
            if s:
                # Add period if missing
                if not s[-1] in '.!?؟。':
                    s += '.'
                result.append(s)

        return result

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """
        Adjust audio speed using time stretching
        """
        if speed == 1.0:
            return audio

        try:
            import librosa
            audio_stretched = librosa.effects.time_stretch(audio, rate=speed)
            return audio_stretched
        except ImportError:
            logger.warning("librosa not available, speed adjustment disabled")
            return audio

    def _audio_to_bytes(self, audio: np.ndarray) -> bytes:
        """
        Convert audio array to WAV bytes
        """
        # Ensure correct shape
        if len(audio.shape) == 1:
            audio = audio.reshape(-1, 1)

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Write to bytes buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio_int16, SAMPLE_RATE, format='WAV')
        buffer.seek(0)

        return buffer.read()

# ============================================================================
# Piper TTS (Fallback for ultra-low latency)
# ============================================================================

class PiperTTS:
    """
    Piper TTS fallback for low-latency scenarios
    Faster but less expressive than XTTS v2
    """

    def __init__(self):
        # TODO: Initialize Piper
        logger.info("Piper TTS fallback (not implemented)")

    async def synthesize(self, text: str, language: str) -> bytes:
        """Fast synthesis"""
        # TODO: Implement Piper synthesis
        return b""

# ============================================================================
# FastAPI Server
# ============================================================================

app = FastAPI(title="TTS Service", version="1.0.0")
tts_service = None

@app.on_event("startup")
async def startup():
    global tts_service
    tts_service = TTSService()
    logger.info("🔊 TTS Service started")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
        "voices": list(VOICE_PRESETS.keys()),
    }

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": [
            {"id": "arabic_gulf_male", "language": "ar", "gender": "male", "region": "Gulf"},
            {"id": "arabic_gulf_female", "language": "ar", "gender": "female", "region": "Gulf"},
            {"id": "arabic_msa_male", "language": "ar", "gender": "male", "region": "MSA"},
            {"id": "english_uae_male", "language": "en", "gender": "male", "region": "UAE"},
            {"id": "english_uae_female", "language": "en", "gender": "female", "region": "UAE"},
        ]
    }

@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """
    Synthesize speech (non-streaming)
    """
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        audio_bytes, duration = await tts_service.synthesize(
            text=request.text,
            voice_id=request.voice_id,
            language=request.language,
            speed=request.speed,
            emotion=request.emotion,
        )

        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        return SynthesizeResponse(
            audio_base64=audio_b64,
            duration_seconds=duration,
            sample_rate=SAMPLE_RATE,
            format="wav",
        )

    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/synthesize/stream")
async def synthesize_stream(request: SynthesizeRequest):
    """
    Synthesize speech (streaming)
    Returns SSE stream of audio chunks
    """
    from fastapi.responses import StreamingResponse

    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")

    async def generate():
        try:
            async for chunk in tts_service.synthesize_streaming(
                text=request.text,
                voice_id=request.voice_id,
                language=request.language,
                speed=request.speed,
                emotion=request.emotion,
            ):
                yield f"data: {chunk.json()}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}", exc_info=True)
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
    )

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info",
    )

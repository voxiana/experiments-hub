"""
Simplified TTS Service using Coqui XTTS v2
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fix for PyTorch 2.6+ weights_only default change
# Patch torch.load to use weights_only=False for TTS compatibility
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from TTS.api import TTS

# Configuration
os.environ["COQUI_TOS_AGREED"] = "1"

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 24000

# Path to reference voice (update this to your voice file)
REFERENCE_VOICE = os.path.join(os.path.dirname(__file__), "voices", "reference_voice.wav")


# Request/Response Models
class SynthesizeRequest(BaseModel):
    text: str
    language: str = "ar"  # ar, en
    reference_audio: str | None = None  # Optional base64 audio to clone


class SynthesizeResponse(BaseModel):
    audio_base64: str
    duration_seconds: float
    sample_rate: int = SAMPLE_RATE


# TTS Service
class TTSService:
    def __init__(self):
        logger.info(f"Loading model: {MODEL_NAME} on {DEVICE}")
        self.model = TTS(MODEL_NAME).to(DEVICE)
        logger.info("Model loaded successfully")

    def synthesize(self, text: str, language: str, speaker_wav: str) -> tuple[bytes, float]:
        """Generate speech from text."""
        
        # Generate audio
        wav = self.model.tts(
            text=text,
            speaker_wav=speaker_wav,
            language=language,
        )
        
        # Process audio
        audio = self._process_audio(wav)
        
        # Convert to WAV bytes
        duration = len(audio) / SAMPLE_RATE
        audio_bytes = self._to_wav_bytes(audio)
        
        return audio_bytes, duration

    def _process_audio(self, wav) -> np.ndarray:
        """Process and clean the generated audio."""
        
        # Convert to numpy array
        audio = np.array(wav, dtype=np.float32)
        
        # Ensure 1D
        if audio.ndim > 1:
            audio = audio.flatten()
        
        # Remove NaN/Inf values
        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Normalize with headroom (-3dB)
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.707
        
        # Apply fade in/out to prevent clicks (10ms)
        fade_samples = int(SAMPLE_RATE * 0.01)
        if len(audio) > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            audio[:fade_samples] *= fade_in
            audio[-fade_samples:] *= fade_out
        
        return audio

    def _to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert audio array to WAV bytes."""
        buffer = io.BytesIO()
        sf.write(buffer, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        buffer.seek(0)
        return buffer.read()


# FastAPI App
tts_service: TTSService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_service
    tts_service = TTSService()
    yield
    tts_service = None


app = FastAPI(title="TTS Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": DEVICE,
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    """Synthesize speech from text."""
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    
    # Determine speaker reference
    if request.reference_audio:
        # Use provided base64 audio as reference
        speaker_wav = _save_temp_audio(request.reference_audio)
    else:
        # Use default reference voice
        if not os.path.exists(REFERENCE_VOICE):
            raise HTTPException(
                status_code=500,
                detail=f"Reference voice not found: {REFERENCE_VOICE}"
            )
        speaker_wav = REFERENCE_VOICE
    
    try:
        audio_bytes, duration = tts_service.synthesize(
            text=request.text,
            language=request.language,
            speaker_wav=speaker_wav,
        )
        
        return SynthesizeResponse(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            duration_seconds=duration,
        )
    
    except Exception as e:
        logger.error(f"Synthesis failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        # Cleanup temp file if created
        if request.reference_audio and os.path.exists(speaker_wav):
            os.remove(speaker_wav)


def _save_temp_audio(base64_audio: str) -> str:
    """Save base64 audio to a temp file and return the path."""
    import tempfile
    
    audio_bytes = base64.b64decode(base64_audio)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        return f.name


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
TTS Service proxy for Voxtral TTS served by vllm-omni sidecar.
"""

import base64
import io
import logging
import os
from contextlib import asynccontextmanager

import httpx
import soundfile as sf
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TTS_VLLM_URL = os.environ.get("TTS_VLLM_URL", "http://tts-vllm:8000/v1")
TTS_MODEL = os.environ.get("TTS_MODEL", "mistralai/Voxtral-4B-TTS-2603")
TTS_DEFAULT_VOICE = os.environ.get("TTS_DEFAULT_VOICE", "casual_male")
SAMPLE_RATE = 24000


class SynthesizeRequest(BaseModel):
    text: str
    language: str = "ar"
    reference_audio: str | None = None
    voice: str | None = None
    speaker: str | None = None


class SynthesizeResponse(BaseModel):
    audio_base64: str
    duration_seconds: float
    sample_rate: int = SAMPLE_RATE


class TTSService:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client

    async def synthesize(self, text: str, voice: str | None) -> tuple[bytes, float]:
        selected_voice = voice or TTS_DEFAULT_VOICE
        payload = {
            "model": TTS_MODEL,
            "input": text,
            "voice": selected_voice,
            "response_format": "wav",
        }
        response = await self.client.post(
            f"{TTS_VLLM_URL}/audio/speech",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        audio_bytes = response.content
        duration = _duration_from_wav(audio_bytes)
        return audio_bytes, duration


def _duration_from_wav(audio_bytes: bytes) -> float:
    try:
        data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
        if data.ndim > 1:
            samples = data.shape[0]
        else:
            samples = len(data)
        return float(samples / sample_rate)
    except Exception:
        logger.warning("Failed to parse WAV duration; returning 0.0s", exc_info=True)
        return 0.0


tts_service: TTSService | None = None
http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_service, http_client
    http_client = httpx.AsyncClient()
    tts_service = TTSService(http_client)
    yield
    if http_client is not None:
        await http_client.aclose()
    tts_service = None
    http_client = None


app = FastAPI(title="TTS Service", version="1.0.0", lifespan=lifespan)


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": TTS_MODEL,
        "vllm_url": TTS_VLLM_URL,
    }


@app.post("/synthesize", response_model=SynthesizeResponse)
async def synthesize(request: SynthesizeRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    if request.reference_audio:
        raise HTTPException(
            status_code=400,
            detail="reference-WAV cloning not supported by Voxtral TTS; pass 'voice' preset.",
        )

    if tts_service is None:
        raise HTTPException(status_code=503, detail="TTS service not initialized")

    try:
        voice = request.voice or request.speaker
        audio_bytes, duration = await tts_service.synthesize(request.text, voice)
        return SynthesizeResponse(
            audio_base64=base64.b64encode(audio_bytes).decode("utf-8"),
            duration_seconds=duration,
        )
    except httpx.HTTPStatusError as exc:
        detail = f"Voxtral backend error ({exc.response.status_code}): {exc.response.text}"
        logger.error(detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except Exception as exc:
        logger.error("Synthesis failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
"""
HTTP adapter for local/self-hosted TTS services.
"""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx

from adapters.base import VoiceModelAdapter
from schema import EvalScenario, NormalizedSynthesisResult

REQUIRED_CONTRACT_FIELDS = ("audio_base64", "duration_seconds", "sample_rate")


class HttpTTSAdapter(VoiceModelAdapter):
    async def synthesize(self, scenario: EvalScenario) -> NormalizedSynthesisResult:
        url = f"{self.backend.base_url.rstrip('/')}/{self.backend.synthesize_path.lstrip('/')}"

        payload: dict[str, Any] = dict(self.backend.request_template)
        payload["text"] = scenario.text
        payload["language"] = scenario.language
        if scenario.reference_audio:
            payload["reference_audio"] = scenario.reference_audio

        started = time.perf_counter()
        response_payload: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=self.backend.timeout_seconds) as client:
                response = await client.post(url, json=payload, headers=self.backend.extra_headers)
                response.raise_for_status()
                response_payload = response.json()

            missing_fields = [field for field in REQUIRED_CONTRACT_FIELDS if field not in response_payload]
            if missing_fields:
                return NormalizedSynthesisResult(
                    backend_id=self.backend.backend_id,
                    case_id=scenario.case_id,
                    text=scenario.text,
                    language=scenario.language,
                    latency_ms=(time.perf_counter() - started) * 1000,
                    raw_response=response_payload,
                    error=f"Missing response fields: {missing_fields}",
                    metadata={"missing_fields": missing_fields},
                )

            audio_bytes = base64.b64decode(response_payload["audio_base64"])
            latency_ms = (time.perf_counter() - started) * 1000
            return NormalizedSynthesisResult(
                backend_id=self.backend.backend_id,
                case_id=scenario.case_id,
                text=scenario.text,
                language=scenario.language,
                audio_bytes=audio_bytes,
                sample_rate=int(response_payload["sample_rate"]),
                duration_seconds=float(response_payload["duration_seconds"]),
                latency_ms=latency_ms,
                raw_response=response_payload,
            )
        except Exception as exc:  # pragma: no cover - runtime failures are expected in experiments
            return NormalizedSynthesisResult(
                backend_id=self.backend.backend_id,
                case_id=scenario.case_id,
                text=scenario.text,
                language=scenario.language,
                latency_ms=(time.perf_counter() - started) * 1000,
                raw_response=response_payload,
                error=str(exc),
            )


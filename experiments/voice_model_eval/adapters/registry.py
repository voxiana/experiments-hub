"""
Adapter registry/factory.
"""

from __future__ import annotations

from adapters.base import VoiceModelAdapter
from adapters.http_tts import HttpTTSAdapter
from schema import BackendConfig


def build_adapter(backend: BackendConfig) -> VoiceModelAdapter:
    if backend.adapter == "http_tts":
        return HttpTTSAdapter(backend)
    raise ValueError(f"Unsupported adapter type: {backend.adapter}")


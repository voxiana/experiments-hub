"""
Typed schemas for voice model evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class EvalScenario:
    case_id: str
    text: str
    language: str = "ar"
    reference_audio: str | None = None
    tags: list[str] = field(default_factory=list)


@dataclass(slots=True)
class BackendConfig:
    backend_id: str
    adapter: str
    base_url: str
    synthesize_path: str = "/synthesize"
    timeout_seconds: float = 120.0
    request_template: dict[str, Any] = field(default_factory=dict)
    extra_headers: dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass(slots=True)
class BaselineConfig:
    enabled: bool = True
    base_url: str = "http://tts-service:8002"
    synthesize_path: str = "/synthesize"
    timeout_seconds: float = 120.0


@dataclass(slots=True)
class EvalRunConfig:
    run_name: str
    backends: list[BackendConfig]
    scenarios: list[EvalScenario]
    concurrency: int = 1
    baseline: BaselineConfig = field(default_factory=BaselineConfig)


@dataclass(slots=True)
class NormalizedSynthesisResult:
    backend_id: str
    case_id: str
    text: str
    language: str
    audio_bytes: bytes = b""
    sample_rate: int | None = None
    duration_seconds: float | None = None
    latency_ms: float | None = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.error is None


@dataclass(slots=True)
class EvalRecord:
    timestamp_utc: str
    backend_id: str
    case_id: str
    text: str
    language: str
    success: bool
    latency_ms: float | None
    duration_seconds: float | None
    sample_rate: int | None
    audio_bytes_len: int
    real_time_factor: float | None
    error: str | None
    baseline_latency_ms: float | None = None
    baseline_duration_seconds: float | None = None
    baseline_contract_compatible: bool | None = None
    duration_delta_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


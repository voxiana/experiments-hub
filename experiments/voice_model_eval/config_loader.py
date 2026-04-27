"""
YAML configuration loader for evaluation runs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from schema import BackendConfig, BaselineConfig, EvalRunConfig, EvalScenario


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {path}, got: {type(data).__name__}")
    return data


def load_run_config(
    models_path: str | Path,
    scenarios_path: str | Path,
    run_name: str,
    concurrency: int,
    baseline_url: str | None = None,
    baseline_enabled: bool = True,
) -> EvalRunConfig:
    models_data = _read_yaml(Path(models_path))
    scenarios_data = _read_yaml(Path(scenarios_path))

    backend_entries = models_data.get("models", [])
    if not isinstance(backend_entries, list):
        raise ValueError("`models` must be a list in the models config file")

    scenario_entries = scenarios_data.get("scenarios", [])
    if not isinstance(scenario_entries, list):
        raise ValueError("`scenarios` must be a list in the scenarios config file")

    backends = [
        BackendConfig(
            backend_id=str(entry["backend_id"]),
            adapter=str(entry.get("adapter", "http_tts")),
            base_url=str(entry["base_url"]),
            synthesize_path=str(entry.get("synthesize_path", "/synthesize")),
            timeout_seconds=float(entry.get("timeout_seconds", 120)),
            request_template=dict(entry.get("request_template", {})),
            extra_headers=dict(entry.get("extra_headers", {})),
            enabled=bool(entry.get("enabled", True)),
        )
        for entry in backend_entries
    ]

    scenarios = [
        EvalScenario(
            case_id=str(entry["case_id"]),
            text=str(entry["text"]),
            language=str(entry.get("language", "ar")),
            reference_audio=entry.get("reference_audio"),
            tags=list(entry.get("tags", [])),
        )
        for entry in scenario_entries
    ]

    baseline_section = models_data.get("baseline", {})
    if not isinstance(baseline_section, dict):
        raise ValueError("`baseline` must be a mapping when present")

    if baseline_url:
        baseline_base_url = baseline_url
    else:
        baseline_base_url = str(baseline_section.get("base_url", "http://tts-service:8002"))

    baseline = BaselineConfig(
        enabled=bool(baseline_section.get("enabled", baseline_enabled)),
        base_url=baseline_base_url,
        synthesize_path=str(baseline_section.get("synthesize_path", "/synthesize")),
        timeout_seconds=float(baseline_section.get("timeout_seconds", 120)),
    )

    if not backends:
        raise ValueError("No backends configured in models file")
    if not scenarios:
        raise ValueError("No scenarios configured in scenarios file")

    return EvalRunConfig(
        run_name=run_name,
        backends=[backend for backend in backends if backend.enabled],
        scenarios=scenarios,
        concurrency=max(1, concurrency),
        baseline=baseline,
    )


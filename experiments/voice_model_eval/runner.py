"""
Evaluation runner for local voice model backends.
"""

from __future__ import annotations

import asyncio
import csv
import json
import statistics
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from adapters.registry import build_adapter
from schema import BackendConfig, BaselineConfig, EvalRecord, EvalRunConfig, EvalScenario, utc_now_iso
from telemetry import eval_span, record_metrics


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    index = max(0, min(len(sorted_values) - 1, int(round(quantile * (len(sorted_values) - 1)))))
    return sorted_values[index]


def _coerce_contract(raw_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    required = ("audio_base64", "duration_seconds", "sample_rate")
    missing = [key for key in required if key not in raw_payload]
    return len(missing) == 0, missing


class BaselineEvaluator:
    def __init__(self, config: BaselineConfig):
        self._backend = BackendConfig(
            backend_id="baseline",
            adapter="http_tts",
            base_url=config.base_url,
            synthesize_path=config.synthesize_path,
            timeout_seconds=config.timeout_seconds,
        )
        self._adapter = build_adapter(self._backend)
        self._enabled = config.enabled

    async def evaluate(self, scenario: EvalScenario) -> dict[str, Any]:
        if not self._enabled:
            return {
                "enabled": False,
                "latency_ms": None,
                "duration_seconds": None,
                "contract_compatible": None,
                "missing_fields": [],
                "error": None,
            }

        baseline_result = await self._adapter.synthesize(scenario)
        contract_compatible, missing_fields = _coerce_contract(baseline_result.raw_response)
        return {
            "enabled": True,
            "latency_ms": baseline_result.latency_ms,
            "duration_seconds": baseline_result.duration_seconds,
            "contract_compatible": contract_compatible,
            "missing_fields": missing_fields,
            "error": baseline_result.error,
        }


async def run_evaluation(config: EvalRunConfig) -> tuple[list[EvalRecord], dict[str, Any]]:
    semaphore = asyncio.Semaphore(config.concurrency)
    baseline = BaselineEvaluator(config.baseline)

    async def _evaluate_case(backend: BackendConfig, scenario: EvalScenario) -> EvalRecord:
        adapter = build_adapter(backend)
        with eval_span(
            "voice_eval_case",
            {"backend_id": backend.backend_id, "case_id": scenario.case_id},
        ):
            async with semaphore:
                result = await adapter.synthesize(scenario)
                baseline_result = await baseline.evaluate(scenario)

                real_time_factor = None
                if result.duration_seconds and result.latency_ms:
                    real_time_factor = (result.latency_ms / 1000) / result.duration_seconds

                duration_delta = None
                baseline_duration = baseline_result.get("duration_seconds")
                if result.duration_seconds is not None and baseline_duration is not None:
                    duration_delta = result.duration_seconds - baseline_duration

                metadata = dict(result.metadata)
                metadata["baseline_missing_fields"] = baseline_result.get("missing_fields", [])
                metadata["baseline_error"] = baseline_result.get("error")

                record = EvalRecord(
                    timestamp_utc=utc_now_iso(),
                    backend_id=backend.backend_id,
                    case_id=scenario.case_id,
                    text=scenario.text,
                    language=scenario.language,
                    success=result.is_success,
                    latency_ms=result.latency_ms,
                    duration_seconds=result.duration_seconds,
                    sample_rate=result.sample_rate,
                    audio_bytes_len=len(result.audio_bytes),
                    real_time_factor=real_time_factor,
                    error=result.error,
                    baseline_latency_ms=baseline_result.get("latency_ms"),
                    baseline_duration_seconds=baseline_result.get("duration_seconds"),
                    baseline_contract_compatible=baseline_result.get("contract_compatible"),
                    duration_delta_seconds=duration_delta,
                    metadata=metadata,
                )
                record_metrics(
                    backend_id=backend.backend_id,
                    case_id=scenario.case_id,
                    success=record.success,
                    latency_ms=record.latency_ms,
                )
                return record

    tasks = [
        _evaluate_case(backend, scenario)
        for backend in config.backends
        for scenario in config.scenarios
    ]
    records = await asyncio.gather(*tasks)
    summary = summarize(records)
    return records, summary


def summarize(records: list[EvalRecord]) -> dict[str, Any]:
    grouped: dict[str, list[EvalRecord]] = {}
    for record in records:
        grouped.setdefault(record.backend_id, []).append(record)

    leaderboard: list[dict[str, Any]] = []
    for backend_id, backend_records in grouped.items():
        latencies = [item.latency_ms for item in backend_records if item.latency_ms is not None]
        durations = [item.duration_seconds for item in backend_records if item.duration_seconds is not None]
        rtfs = [item.real_time_factor for item in backend_records if item.real_time_factor is not None]
        successes = [item.success for item in backend_records]
        compatible_contract = [
            item.baseline_contract_compatible
            for item in backend_records
            if item.baseline_contract_compatible is not None
        ]

        entry = {
            "backend_id": backend_id,
            "runs": len(backend_records),
            "success_rate": sum(successes) / len(successes) if successes else 0.0,
            "latency_p50_ms": _percentile(latencies, 0.5),
            "latency_p95_ms": _percentile(latencies, 0.95),
            "avg_duration_seconds": statistics.mean(durations) if durations else 0.0,
            "avg_real_time_factor": statistics.mean(rtfs) if rtfs else 0.0,
            "baseline_contract_compatibility_rate": (
                sum(1 for state in compatible_contract if state) / len(compatible_contract)
                if compatible_contract
                else None
            ),
        }
        leaderboard.append(entry)

    leaderboard.sort(key=lambda item: (-item["success_rate"], item["latency_p50_ms"]))
    return {"leaderboard": leaderboard, "total_cases": len(records)}


def write_outputs(output_dir: str | Path, records: list[EvalRecord], summary: dict[str, Any]) -> Path:
    run_dir = Path(output_dir) / datetime.utcnow().strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    records_jsonl = run_dir / "records.jsonl"
    with records_jsonl.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(asdict(record), ensure_ascii=True) + "\n")

    records_csv = run_dir / "records.csv"
    if records:
        with records_csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))

    summary_file = run_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, ensure_ascii=True)

    leaderboard_csv = run_dir / "leaderboard.csv"
    leaderboard_rows = summary.get("leaderboard", [])
    if leaderboard_rows:
        with leaderboard_csv.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(leaderboard_rows[0].keys()))
            writer.writeheader()
            writer.writerows(leaderboard_rows)

    return run_dir


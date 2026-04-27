"""
CLI entrypoint for model evaluation runs.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from config_loader import load_run_config
from runner import run_evaluation, write_outputs
from telemetry import init_telemetry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate local voice model backends")
    parser.add_argument(
        "--models",
        default="config/models.yml",
        help="Path to models/backends YAML file",
    )
    parser.add_argument(
        "--scenarios",
        default="config/scenarios.yml",
        help="Path to scenarios YAML file",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to write run artifacts",
    )
    parser.add_argument(
        "--run-name",
        default="voice_model_eval",
        help="Human-readable run name",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent synth requests across backend/case matrix",
    )
    parser.add_argument(
        "--baseline-url",
        default=None,
        help="Override baseline TTS URL (ex: http://tts-service:8002)",
    )
    parser.add_argument(
        "--disable-baseline",
        action="store_true",
        help="Skip baseline compatibility checks",
    )
    return parser.parse_args()


async def _main() -> int:
    args = parse_args()
    init_telemetry()

    config = load_run_config(
        models_path=args.models,
        scenarios_path=args.scenarios,
        run_name=args.run_name,
        concurrency=args.concurrency,
        baseline_url=args.baseline_url,
        baseline_enabled=not args.disable_baseline,
    )

    records, summary = await run_evaluation(config)
    run_dir = write_outputs(args.output_dir, records, summary)
    print(f"Run artifacts: {run_dir}")
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))


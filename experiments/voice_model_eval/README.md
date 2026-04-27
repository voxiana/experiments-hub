# Voice Model Evaluation

This module runs compatibility-first evaluations for local/self-hosted TTS backends.

## What it provides

- A backend adapter interface with a normalized synthesis result schema.
- Config-driven evaluation matrix (`models.yml` x `scenarios.yml`).
- Baseline regression check against the existing `tts-service` contract.
- Artifacts per run (`records.jsonl`, `records.csv`, `summary.json`, `leaderboard.csv`).
- Optional OpenTelemetry and Prometheus metrics.

## Directory map

- `adapters/`: backend adapters (`http_tts` included).
- `config/`: model matrix and scenario files.
- `data/`: optional prompt/reference source files.
- `scripts/`: CLI entrypoints.
- `results/`: run outputs.

## Contract compatibility

The default adapter expects the same response fields as current production TTS:

- `audio_base64`
- `duration_seconds`
- `sample_rate`

Each run records whether baseline contract fields are present.

## Local run (host)

```bash
cd experiments/voice_model_eval
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python scripts/run_eval.py --models config/models.yml --scenarios config/scenarios.yml --output-dir results
```

Generate markdown leaderboard:

```bash
python scripts/generate_report.py --summary results/<run_dir>/summary.json
```

## Docker run

Use the compose profile added in the root `docker-compose.yml`:

```bash
docker compose --profile eval up --build eval-runner
```

## Config examples

- Add another backend in `config/models.yml` with `adapter: http_tts` and a different `base_url`.
- Add representative prompts/cases in `config/scenarios.yml`.
- Optionally disable baseline checks with `--disable-baseline`.

## Optional telemetry

- `ENABLE_OTEL=true` and `OTEL_EXPORTER_OTLP_ENDPOINT=http://otel-collector:4317`
- `ENABLE_PROMETHEUS=true` and `PROMETHEUS_PORT=9108`

Telemetry is optional; artifact files remain the source of truth.


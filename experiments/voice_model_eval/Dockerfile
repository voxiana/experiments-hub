FROM python:3.11-slim

WORKDIR /app/experiments/voice_model_eval

COPY ./experiments/voice_model_eval/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

COPY ./experiments/voice_model_eval /app/experiments/voice_model_eval

ENV PYTHONUNBUFFERED=1

CMD ["python", "scripts/run_eval.py", "--models", "config/models.yml", "--scenarios", "config/scenarios.yml", "--output-dir", "results"]


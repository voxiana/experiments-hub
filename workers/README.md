# Workers Service

## Overview

The Workers Service provides background task processing for the Voice AI CX Platform using Celery. It handles asynchronous operations like document ingestion, post-call analytics, diarization, emotion analysis, and scheduled maintenance tasks.

## Features

### Task Types

**Ingestion Tasks** (Queue: `ingest`):
- Document ingestion into RAG system
- Batch document processing
- Embedding generation
- Vector indexing

**Analytics Tasks** (Queue: `analytics`):
- Post-call analytics processing
- Speaker diarization (pyannote.audio)
- Emotion analysis (speechbrain)
- Call summarization
- Metrics calculation (AHT, FCR, sentiment)

**Report Tasks** (Queue: `reports`):
- Daily analytics reports
- Tenant-specific reporting
- Email delivery

**Scheduled Tasks** (Celery Beat):
- Daily metrics aggregation (midnight UTC)
- Expired session cleanup (1 AM UTC)
- Model performance audits

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                  Workers Service                          │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │  Celery    │  │   Celery    │  │    Task      │      │
│  │  Workers   │  │    Beat     │  │   Queues     │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │    Redis    │                        │
│                    │   Broker    │                        │
│                    └─────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Technology Stack

- **Celery 5.3.4** - Distributed task queue
- **Redis 5.0.1** - Message broker and result backend
- **Python 3.10+** - Runtime environment

### Dependencies

- **pyannote.audio 3.1.1** - Speaker diarization
- **speechbrain 0.5.16** - Emotion recognition
- **PostgreSQL** - Call data storage
- **ClickHouse** - Analytics database

## Configuration

### Environment Variables

```bash
# Redis
CELERY_BROKER_URL="redis://redis:6379/1"
CELERY_RESULT_BACKEND="redis://redis:6379/1"

# Timezone
CELERY_TIMEZONE="Asia/Dubai"

# Performance
WORKER_CONCURRENCY=4
WORKER_MAX_TASKS_PER_CHILD=100
TASK_TIME_LIMIT=3600  # 1 hour
TASK_SOFT_TIME_LIMIT=3300  # 55 minutes

# Services
RAG_SERVICE_URL="http://rag-service:8080"
POSTGRES_URL="postgresql://voiceai:voiceai@postgres:5432/voiceai"
CLICKHOUSE_URL="http://clickhouse:8123"
```

## Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start Redis**:
```bash
docker run -p 6379:6379 redis:latest
```

3. **Run worker**:
```bash
celery -A tasks worker --loglevel=info --queues=ingest,analytics,reports
```

4. **Run beat scheduler** (in separate terminal):
```bash
celery -A tasks beat --loglevel=info
```

### Docker Deployment

```bash
docker-compose up workers worker-beat
```

## Tasks

### ingest_document

Ingest a document into the RAG system.

**Parameters**:
- `document_id` (str): Unique document identifier
- `tenant_id` (str): Tenant identifier
- `source_path` (str): Path to document file

**Example**:
```python
from workers.tasks import ingest_document

result = ingest_document.delay(
    document_id="doc_123",
    tenant_id="tenant_456",
    source_path="/tmp/document.pdf"
)

print(result.get())  # Wait for result
```

---

### process_call_analytics

Process post-call analytics.

**Parameters**:
- `call_id` (str): Unique call identifier

**Analytics Performed**:
- Speaker diarization
- Emotion analysis
- Intent classification
- Summary generation
- Metrics calculation

**Example**:
```python
from workers.tasks import process_call_analytics

result = process_call_analytics.delay(call_id="call_abc123")
metrics = result.get()
```

---

### diarize_call

Speaker diarization using pyannote.audio.

**Parameters**:
- `call_id` (str): Call identifier
- `audio_path` (str): Path to audio file

**Returns**:
- Speaker segments with timestamps
- Number of speakers detected

**Example**:
```python
from workers.tasks import diarize_call

result = diarize_call.delay(
    call_id="call_xyz",
    audio_path="/tmp/call_audio.wav"
)

diarization = result.get()
print(f"Speakers: {diarization['num_speakers']}")
```

---

### aggregate_daily_metrics

Aggregate daily analytics metrics.

**Scheduled**: Midnight UTC (Celery Beat)

**Metrics Calculated**:
- Total calls
- Average handling time (AHT)
- First call resolution rate (FCR)
- Average sentiment (CSAT proxy)
- Top intents

---

### cleanup_expired_sessions

Clean up expired sessions and temporary data.

**Scheduled**: 1 AM UTC (Celery Beat)

**Operations**:
- Delete expired Redis keys
- Archive old call data
- Clean up temporary files

## Queue Configuration

### Queue Routing

```python
{
    'workers.tasks.ingest_document': {'queue': 'ingest'},
    'workers.tasks.process_call_analytics': {'queue': 'analytics'},
    'workers.tasks.generate_daily_report': {'queue': 'reports'},
}
```

### Running Specific Queues

```bash
# Ingest queue only
celery -A tasks worker -Q ingest

# Analytics and reports
celery -A tasks worker -Q analytics,reports

# All queues
celery -A tasks worker -Q ingest,analytics,reports
```

## Monitoring

### Flower (Web UI)

```bash
# Install Flower
pip install flower

# Start Flower
celery -A tasks flower --port=5555

# Access UI at http://localhost:5555
```

### Task Status

```python
from celery.result import AsyncResult

result = AsyncResult(task_id)

print(f"State: {result.state}")
print(f"Result: {result.result}")
print(f"Traceback: {result.traceback}")
```

### Worker Stats

```bash
# Inspect active tasks
celery -A tasks inspect active

# Inspect registered tasks
celery -A tasks inspect registered

# Worker stats
celery -A tasks inspect stats
```

## Performance

### Benchmarks

| Task | Duration | Notes |
|------|----------|-------|
| Document Ingestion | 2-5s | Depends on size |
| Call Analytics | 5-10s | Full processing |
| Diarization | 3-5s | 10s audio |
| Daily Aggregation | 10-30s | 1000 calls |

### Optimization

1. **Concurrency**: Adjust worker concurrency based on CPU cores
2. **Prefetch**: Use `worker_prefetch_multiplier=1` for long tasks
3. **Task Routing**: Separate queues for different task types
4. **Retries**: Implement exponential backoff
5. **Batching**: Batch similar tasks together

## Error Handling

### Retry Logic

```python
@app.task(bind=True, max_retries=3)
def my_task(self):
    try:
        # Task logic
        pass
    except Exception as e:
        # Retry after 60 seconds
        raise self.retry(exc=e, countdown=60)
```

### Task Callbacks

```python
class CallbackTask(Task):
    def on_success(self, retval, task_id, args, kwargs):
        logger.info(f"Task {task_id} succeeded")

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"Task {task_id} failed: {exc}")

    def on_retry(self, exc, task_id, args, kwargs, einfo):
        logger.warning(f"Task {task_id} retrying")
```

## Best Practices

1. **Idempotency**: Make tasks idempotent (safe to retry)
2. **Timeouts**: Set appropriate task time limits
3. **Logging**: Log task execution for debugging
4. **Monitoring**: Use Flower or Prometheus for monitoring
5. **Error Handling**: Always catch and log exceptions
6. **Resource Limits**: Set memory and CPU limits
7. **Queue Separation**: Use separate queues for different priorities

## Troubleshooting

### Worker Not Processing Tasks

**Solution**:
```bash
# Check worker is running
celery -A tasks inspect active_queues

# Check Redis connection
redis-cli ping

# Restart worker
celery -A tasks worker --loglevel=debug
```

### Tasks Stuck in Pending

**Solution**:
- Verify worker is consuming from correct queue
- Check Redis memory usage
- Review task routing configuration

### High Memory Usage

**Solution**:
```bash
# Limit tasks per worker
export WORKER_MAX_TASKS_PER_CHILD=50

# Reduce concurrency
celery -A tasks worker --concurrency=2
```

## Development

### Adding New Tasks

```python
@app.task(base=CallbackTask, bind=True, max_retries=3)
def my_new_task(self, arg1, arg2):
    """
    Task description
    """
    try:
        # Task logic
        result = do_something(arg1, arg2)
        return result
    except Exception as e:
        raise self.retry(exc=e, countdown=60)
```

### Testing Tasks

```python
# Run task synchronously (for testing)
result = my_task.apply(args=['arg1', 'arg2'])

# Or use eager mode
app.conf.task_always_eager = True
```

## Deployment

### Production Checklist

- [ ] Configure proper Redis persistence
- [ ] Set up Flower for monitoring
- [ ] Configure log aggregation
- [ ] Set resource limits (CPU, memory)
- [ ] Enable task result expiration
- [ ] Set up alerting for failed tasks
- [ ] Configure auto-scaling for workers

### Scaling

**Vertical**: Increase worker concurrency
**Horizontal**: Add more worker instances

```bash
# Multiple workers
celery multi start worker1 worker2 -A tasks -Q analytics

# Auto-scaling
celery -A tasks worker --autoscale=10,3
```

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- Celery: [Documentation](https://docs.celeryq.dev/)

.PHONY: help dev up down clean test lint format ingest docs

help: ## Show this help message
	@echo "Voice AI CX Platform - Make Commands"
	@echo "===================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

dev: ## Start development environment
	@echo "🚀 Starting development environment..."
	docker-compose up -d
	@echo "✅ Services started!"
	@echo "   Gateway:    http://localhost:8000"
	@echo "   Web Client: http://localhost:3001"
	@echo "   Grafana:    http://localhost:3000 (admin/admin)"
	@echo "   Prometheus: http://localhost:9090"

up: dev ## Alias for dev

down: ## Stop all services
	@echo "🛑 Stopping services..."
	docker-compose down

stop: down ## Alias for down

restart: ## Restart all services
	@echo "♻️  Restarting services..."
	docker-compose restart

clean: ## Remove all containers, volumes, and cached data
	@echo "🧹 Cleaning up..."
	docker-compose down -v --remove-orphans
	@echo "✅ Cleanup complete!"

logs: ## Show logs from all services
	docker-compose logs -f

logs-gateway: ## Show gateway logs
	docker-compose logs -f gateway

logs-asr: ## Show ASR service logs
	docker-compose logs -f asr-service

logs-nlu: ## Show NLU service logs
	docker-compose logs -f nlu-service

logs-tts: ## Show TTS service logs
	docker-compose logs -f tts-service

test: ## Run tests
	@echo "🧪 Running tests..."
	pytest tests/ -v --cov=. --cov-report=term-missing

test-unit: ## Run unit tests only
	@echo "🧪 Running unit tests..."
	pytest tests/ -v -m "not integration and not e2e"

test-integration: ## Run integration tests
	@echo "🧪 Running integration tests..."
	pytest tests/integration/ -v

lint: ## Lint code
	@echo "🔍 Linting code..."
	black --check .
	ruff check .
	mypy .

format: ## Format code
	@echo "✨ Formatting code..."
	black .
	ruff check --fix .
	isort .

ingest-sample: ## Ingest sample knowledge base
	@echo "📚 Ingesting sample knowledge base..."
	python scripts/ingest_docs.py sample --tenant-id demo --api-key demo_key
	@echo "✅ Sample KB ingested!"

ingest-file: ## Ingest a file (usage: make ingest-file FILE=path/to/file.pdf)
	@echo "📄 Ingesting file: $(FILE)"
	python scripts/ingest_docs.py file $(FILE) --tenant-id demo --api-key demo_key

ingest-dir: ## Ingest a directory (usage: make ingest-dir DIR=path/to/dir)
	@echo "📁 Ingesting directory: $(DIR)"
	python scripts/ingest_docs.py directory $(DIR) --tenant-id demo --api-key demo_key

psql: ## Connect to PostgreSQL
	docker-compose exec postgres psql -U voiceai -d voiceai

redis-cli: ## Connect to Redis
	docker-compose exec redis redis-cli

db-migrate: ## Run database migrations
	@echo "🗄️  Running database migrations..."
	alembic upgrade head

db-rollback: ## Rollback last migration
	@echo "⏪ Rolling back migration..."
	alembic downgrade -1

db-reset: ## Reset database (WARNING: destructive)
	@echo "⚠️  Resetting database..."
	docker-compose exec postgres psql -U voiceai -c "DROP DATABASE IF EXISTS voiceai;"
	docker-compose exec postgres psql -U voiceai -c "CREATE DATABASE voiceai;"
	$(MAKE) db-migrate

health: ## Check health of all services
	@echo "🏥 Checking service health..."
	@echo "\nGateway:"
	@curl -s http://localhost:8000/health | jq . || echo "❌ Gateway not responding"
	@echo "\nASR Service:"
	@curl -s http://localhost:50051/health | jq . || echo "❌ ASR not responding"
	@echo "\nNLU Service:"
	@curl -s http://localhost:8001/health | jq . || echo "❌ NLU not responding"
	@echo "\nTTS Service:"
	@curl -s http://localhost:8002/health | jq . || echo "❌ TTS not responding"
	@echo "\nRAG Service:"
	@curl -s http://localhost:8080/health | jq . || echo "❌ RAG not responding"
	@echo "\nvLLM:"
	@curl -s http://localhost:8000/health | jq . || echo "❌ vLLM not responding"

metrics: ## View Prometheus metrics
	@echo "📊 Opening Prometheus..."
	open http://localhost:9090 || xdg-open http://localhost:9090

dashboard: ## View Grafana dashboards
	@echo "📈 Opening Grafana..."
	open http://localhost:3000 || xdg-open http://localhost:3000

web: ## Open web demo client
	@echo "🌐 Opening web client..."
	open http://localhost:3001 || xdg-open http://localhost:3001

docs: ## Generate documentation
	@echo "📖 Generating documentation..."
	mkdocs build
	@echo "✅ Documentation built in site/"

docs-serve: ## Serve documentation locally
	@echo "📖 Serving documentation..."
	mkdocs serve

install-dev: ## Install development dependencies
	@echo "📦 Installing development dependencies..."
	pip install -r requirements-dev.txt
	pre-commit install
	@echo "✅ Development environment ready!"

build: ## Build all Docker images
	@echo "🔨 Building Docker images..."
	docker-compose build

pull: ## Pull latest Docker images
	@echo "⬇️  Pulling Docker images..."
	docker-compose pull

ps: ## Show running containers
	docker-compose ps

stats: ## Show container stats
	docker stats --no-stream

gpu-check: ## Check GPU availability
	@echo "🎮 Checking GPU availability..."
	nvidia-smi

backup: ## Backup databases
	@echo "💾 Backing up databases..."
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U voiceai voiceai > backups/postgres_$(shell date +%Y%m%d_%H%M%S).sql
	@echo "✅ Backup saved to backups/"

load-test: ## Run load tests (requires locust)
	@echo "⚡ Running load tests..."
	locust -f tests/load/locustfile.py --users 50 --spawn-rate 10 --host http://localhost:8000

benchmark: ## Benchmark latency
	@echo "⏱️  Benchmarking latency..."
	python scripts/benchmark_latency.py

version: ## Show version info
	@echo "Voice AI CX Platform"
	@echo "Version: 1.0.0-beta"
	@echo "Python: $(shell python --version)"
	@echo "Docker: $(shell docker --version)"
	@echo "Docker Compose: $(shell docker-compose --version)"

# RAG Service

## Overview

The RAG (Retrieval-Augmented Generation) Service provides semantic search and document ingestion capabilities for the Voice AI CX Platform. It combines embedding models (bge-m3), vector databases (Qdrant), and reranking (bge-reranker-v2-m3) to enable accurate, multilingual information retrieval from knowledge bases.

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    RAG Service                            │
│  ┌────────────┐  ┌─────────────┐  ┌──────────────┐      │
│  │ Document   │  │  Embedding  │  │  Reranker    │      │
│  │ Processor  │  │  Service    │  │  Service     │      │
│  └──────────────┘  └─────────────┘  └──────────────┘     │
│         │                 │                  │             │
│         │                 │                  │             │
│    ┌────▼────┐       ┌────▼────┐      ┌─────▼──────┐    │
│    │  Load   │       │ bge-m3  │      │bge-reranker│    │
│    │  Parse  │       │ (1024d) │      │   -v2-m3   │    │
│    │  Chunk  │       │Embedding│      │            │    │
│    └────┬────┘       └────┬────┘      └─────┬──────┘    │
│         │                 │                  │             │
│         └─────────────────┴──────────────────┘             │
│                           │                                │
│                    ┌──────▼──────┐                        │
│                    │   Qdrant    │                        │
│                    │   Vector    │                        │
│                    │   Database  │                        │
│                    └─────────────┘                        │
└──────────────────────────────────────────────────────────┘
```

## Features

### Core Capabilities

- **Document Ingestion**: PDF, DOCX, Markdown, and text files
- **Multilingual Embeddings**: bge-m3 supports Arabic, English, and 100+ languages
- **Vector Search**: Qdrant for efficient similarity search
- **Reranking**: Cross-encoder reranking for improved relevance
- **Multi-tenant**: Isolated collections per tenant
- **Chunk Management**: Intelligent text splitting with overlap
- **Metadata Filtering**: Filter results by custom metadata
- **High Performance**: GPU-accelerated embedding generation

### Supported Document Formats

- PDF (`.pdf`) - PyPDFLoader
- Word (`.docx`, `.doc`) - UnstructuredWordDocumentLoader
- Markdown (`.md`, `.markdown`) - UnstructuredMarkdownLoader
- Plain Text (`.txt`) - Native Python

### Chunking Strategy

- **Chunk Size**: 512 tokens
- **Overlap**: 128 tokens
- **Separators**: Paragraph breaks, sentences, punctuation (multilingual)
- **Metadata Preservation**: Document source, page numbers, custom fields

## Technology Stack

### Core Framework

- **Python 3.10+** - Runtime environment
- **FastAPI 0.104.1** - API framework
- **Pydantic 2.5.0** - Data validation

### AI/ML

- **sentence-transformers 2.2.2** - Embedding framework
- **bge-m3** - Multilingual embedding model (1024-dim)
- **bge-reranker-v2-m3** - Cross-encoder reranker
- **transformers 4.36.0** - Model utilities
- **PyTorch 2.1.1** - Deep learning framework

### Vector Database

- **Qdrant 1.7.0** - Vector database client
- **Distance Metric**: Cosine similarity
- **Indexing**: HNSW for fast search

### Document Processing

- **LangChain 0.1.0** - Document loading and splitting
- **PyPDF 3.17.1** - PDF parsing
- **python-docx 1.1.0** - Word document parsing
- **unstructured 0.11.2** - Unified document loader
- **tiktoken 0.5.2** - Token counting

## Configuration

### Environment Variables

```bash
# Embedding Model
EMBEDDING_MODEL="BAAI/bge-m3"
DEVICE="cuda"  # cuda or cpu

# Reranker Model
RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
ENABLE_RERANKER=true

# Qdrant
QDRANT_URL="http://qdrant:6333"
COLLECTION_PREFIX="kb"

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=128

# Server
HOST="0.0.0.0"
PORT=8080
```

### GPU Requirements

- **Recommended**: NVIDIA T4 (16GB) or better
- **Minimum**: 4GB VRAM
- **VRAM Usage**:
  - bge-m3 embedding: ~2GB
  - bge-reranker-v2-m3: ~1GB
- **CPU Mode**: Supported but 5-10x slower

## Installation

### Local Development

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Start Qdrant** (via Docker):
```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant
```

3. **Run the service**:
```bash
python ingest.py
# Service will start on http://0.0.0.0:8080
```

### Docker Deployment

```bash
docker build -t rag-service:latest .
docker run -p 8080:8080 \
  -e QDRANT_URL="http://qdrant:6333" \
  rag-service:latest
```

### Docker Compose

From repository root:

```bash
docker-compose up rag-service qdrant
```

## API Reference

### Health Check

**Endpoint**: `GET /health`

**Response**:
```json
{
  "status": "healthy",
  "embedding_model": "BAAI/bge-m3",
  "reranker_model": "BAAI/bge-reranker-v2-m3",
  "device": "cuda"
}
```

---

### Ingest Document (File Upload)

Upload and ingest a document into the knowledge base.

**Endpoint**: `POST /ingest`

**Content-Type**: `multipart/form-data`

**Parameters**:
- `tenant_id` (required): Tenant identifier
- `file` (required if no text): Document file
- `text` (required if no file): Raw text content
- `title` (optional): Document title

**Example (cURL)**:
```bash
curl -X POST http://localhost:8080/ingest \
  -F "tenant_id=tenant_123" \
  -F "file=@knowledge_base.pdf" \
  -F "title=Product Documentation"
```

**Response**:
```json
{
  "document_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "chunks_created": 42,
  "status": "completed",
  "took_ms": 2341.5
}
```

---

### Ingest Document (Text)

Ingest raw text without file upload.

**Endpoint**: `POST /ingest`

**Request Body**:
```json
{
  "tenant_id": "tenant_123",
  "text": "Our return policy allows returns within 30 days of purchase...",
  "title": "Return Policy"
}
```

**Response**: Same as file upload

---

### Query Knowledge Base

Search for relevant information.

**Endpoint**: `POST /query`

**Request Body**:
```json
{
  "query": "What is your return policy?",
  "tenant_id": "tenant_123",
  "top_k": 3,
  "filters": {
    "category": "policies"
  },
  "use_reranker": true
}
```

**Parameters**:
- `query` (required): Search query text
- `tenant_id` (required): Tenant identifier
- `top_k` (default: 3): Number of results to return
- `filters` (optional): Metadata filters
- `use_reranker` (default: true): Enable cross-encoder reranking

**Response**:
```json
{
  "results": [
    {
      "text": "Items can be returned within 30 days of purchase with original receipt...",
      "score": 0.92,
      "source": "Return Policy Documentation",
      "metadata": {
        "document_id": "a1b2c3d4-...",
        "title": "Return Policy",
        "chunk_index": 0,
        "total_chunks": 5,
        "page": 1
      }
    },
    {
      "text": "For international orders, return shipping is customer's responsibility...",
      "score": 0.87,
      "source": "Return Policy Documentation",
      "metadata": {
        "document_id": "a1b2c3d4-...",
        "title": "Return Policy",
        "chunk_index": 2,
        "total_chunks": 5
      }
    }
  ],
  "took_ms": 145.8
}
```

## Usage Examples

### Python Client - Ingest File

```python
import requests

# Upload PDF
with open("product_manual.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8080/ingest",
        data={"tenant_id": "tenant_123", "title": "Product Manual"},
        files={"file": f}
    )

result = response.json()
print(f"Document ID: {result['document_id']}")
print(f"Chunks created: {result['chunks_created']}")
```

---

### Python Client - Ingest Text

```python
import requests

response = requests.post(
    "http://localhost:8080/ingest",
    json={
        "tenant_id": "tenant_123",
        "text": "Our support hours are Sunday to Thursday, 9 AM to 6 PM UAE time.",
        "title": "Support Hours"
    }
)

result = response.json()
print(f"Ingested: {result['document_id']}")
```

---

### Python Client - Query

```python
import requests

response = requests.post(
    "http://localhost:8080/query",
    json={
        "query": "What are the support hours?",
        "tenant_id": "tenant_123",
        "top_k": 3,
        "use_reranker": True
    }
)

result = response.json()

for i, hit in enumerate(result["results"]):
    print(f"\n--- Result {i+1} (score: {hit['score']:.2f}) ---")
    print(f"Source: {hit['source']}")
    print(f"Text: {hit['text']}")
```

---

### cURL Examples

```bash
# Ingest PDF
curl -X POST http://localhost:8080/ingest \
  -F "tenant_id=tenant_123" \
  -F "file=@faq.pdf"

# Ingest text
curl -X POST http://localhost:8080/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "tenant_id": "tenant_123",
    "text": "We ship worldwide via DHL and FedEx.",
    "title": "Shipping Info"
  }'

# Query
curl -X POST http://localhost:8080/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Do you ship internationally?",
    "tenant_id": "tenant_123",
    "top_k": 3
  }'
```

## Embedding Models

### bge-m3 (Primary)

- **Dimensions**: 1024
- **Languages**: 100+ including Arabic, English
- **Context Length**: 8192 tokens
- **Performance**: State-of-the-art multilingual retrieval
- **Special Features**:
  - Multi-lingual support without translation
  - Code-switching aware (Arabic + English)
  - Dense + sparse hybrid retrieval

**Supported Languages**: Arabic (ar), English (en), Chinese (zh), French (fr), German (de), Spanish (es), Hindi (hi), Japanese (ja), Korean (ko), Russian (ru), and 90+ more

### Alternative Models

```python
# For English-only (faster)
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"  # 768-dim

# For smaller footprint
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"  # 384-dim

# For maximum multilingual
EMBEDDING_MODEL = "intfloat/multilingual-e5-large"  # 1024-dim
```

## Reranking

### Two-Stage Retrieval

1. **Stage 1 - Vector Search**: Retrieve top-K*5 candidates using bge-m3 embeddings
2. **Stage 2 - Reranking**: Cross-encoder reranks candidates to top-K

**Benefits**:
- Higher precision (10-20% improvement)
- Better handling of complex queries
- More accurate relevance scoring

**Trade-offs**:
- ~100ms added latency
- Higher GPU usage
- Can be disabled for low-latency use cases

**Disable Reranking**:
```json
{
  "query": "...",
  "use_reranker": false
}
```

## Multi-Tenancy

### Tenant Isolation

Each tenant gets a separate Qdrant collection:

```
kb_tenant_123
kb_tenant_456
kb_tenant_789
```

**Benefits**:
- Data isolation
- Independent scaling
- Per-tenant backups
- Easy tenant deletion

### Querying Across Tenants

Not supported by default. For cross-tenant search:
1. Query multiple tenants separately
2. Merge and re-sort results
3. Or use a shared collection with tenant_id filtering

## Metadata Filtering

### Filter by Custom Fields

```python
# Query with filters
{
  "query": "product specifications",
  "tenant_id": "tenant_123",
  "filters": {
    "category": "technical",
    "product_id": "PROD-123"
  }
}
```

### Metadata Structure

Documents automatically include:
- `document_id` - Unique document identifier
- `title` - Document title
- `source` - File path or "text_input"
- `tenant_id` - Tenant identifier
- `created_at` - Timestamp
- `chunk_index` - Chunk position
- `total_chunks` - Total chunks in document

### Custom Metadata

Add custom fields during ingestion:
```python
metadata = {
    "category": "technical",
    "product_id": "PROD-123",
    "version": "2.0",
    "language": "en"
}
```

## Performance

### Benchmarks

**Hardware**: NVIDIA T4 (16GB)

| Operation | Latency | Throughput | Notes |
|-----------|---------|------------|-------|
| **Embed query** | 10-20ms | 50-100 queries/s | Single query |
| **Embed batch (32)** | 100-150ms | 200-300 docs/s | Ingestion |
| **Vector search** | 5-15ms | - | Qdrant HNSW |
| **Rerank (15 docs)** | 80-120ms | - | Cross-encoder |
| **Full query (with rerank)** | 100-150ms | - | End-to-end |
| **Ingest 100-page PDF** | 2-3s | - | Includes chunking |

**CPU Performance**:
- Embedding: 5-10x slower
- Reranking: 8-12x slower
- Not recommended for production

### Optimization Tips

1. **Batch Embedding**: Embed multiple chunks together
2. **Disable Reranker**: For low-latency scenarios
3. **Reduce top_k**: Retrieve fewer candidates
4. **GPU Batching**: Process multiple queries in parallel
5. **Cache Embeddings**: Cache common query embeddings
6. **Quantization**: Use INT8 for 2x speedup

## Troubleshooting

### Qdrant Connection Failed

**Problem**: Cannot connect to Qdrant

**Solutions**:
```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Verify URL in config
export QDRANT_URL="http://qdrant:6333"
```

---

### Model Download Issues

**Problem**: Cannot download bge-m3 or reranker models

**Solutions**:
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# Set HuggingFace cache
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Use mirror (if blocked)
export HF_ENDPOINT=https://hf-mirror.com
```

---

### Poor Search Quality

**Problem**: Search results not relevant

**Solutions**:
1. Enable reranker: `use_reranker: true`
2. Increase top_k for more results
3. Check query language matches documents
4. Verify proper text chunking (not too large/small)
5. Add metadata filters to narrow search
6. Ensure documents are properly ingested

---

### Slow Ingestion

**Problem**: Document ingestion taking too long

**Solutions**:
```bash
# Use GPU
export DEVICE=cuda

# Increase batch size (edit ingest.py)
# embeddings = self.model.encode(texts, batch_size=64)

# Process smaller chunks
export CHUNK_SIZE=256
```

---

### Out of Memory

**Problem**: CUDA out of memory during embedding

**Solutions**:
```bash
# Reduce batch size
# embeddings = self.model.encode(texts, batch_size=16)

# Use CPU for embedding
export DEVICE=cpu

# Use smaller model
export EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
```

## Development

### Running Tests

```bash
# Install dev dependencies
pip install pytest pytest-asyncio httpx

# Run tests
pytest test_rag.py -v

# Test ingestion
python -c "
from ingest import RAGService
import asyncio

async def test():
    rag = RAGService()
    result = await rag.ingest_document(
        tenant_id='test',
        text='This is a test document.',
        title='Test'
    )
    print(f'Ingested: {result.document_id}')

asyncio.run(test())
"
```

### Code Structure

```
rag_service/
├── ingest.py           # Main RAG service
├── requirements.txt    # Python dependencies
├── Dockerfile          # Container image
└── README.md           # This file
```

### Adding Custom Document Loaders

```python
from langchain_community.document_loaders import SomeLoader

# In DocumentProcessor class
elif extension == '.custom':
    loader = SomeLoader(file_path)
    docs = loader.load()
    text = "\n\n".join([d.page_content for d in docs])
```

## Deployment Considerations

### Production Checklist

- [ ] Deploy Qdrant with persistent storage
- [ ] Set up Qdrant backups (snapshots)
- [ ] Configure GPU for embedding service
- [ ] Enable authentication on Qdrant
- [ ] Set up monitoring for query latency
- [ ] Implement rate limiting per tenant
- [ ] Cache frequently accessed embeddings
- [ ] Set up index optimization schedule
- [ ] Monitor vector database size
- [ ] Configure resource limits

### Qdrant Production Setup

```yaml
# docker-compose.yml
qdrant:
  image: qdrant/qdrant
  ports:
    - "6333:6333"
  volumes:
    - ./qdrant_storage:/qdrant/storage
  environment:
    - QDRANT_API_KEY=your-secret-key
  restart: unless-stopped
```

### Scaling Strategy

**Vertical Scaling**:
- Larger GPU for faster embedding
- More RAM for larger batches

**Horizontal Scaling**:
- Read-heavy: Multiple RAG service instances + Qdrant cluster
- Write-heavy: Queue ingestion tasks with Celery

**Qdrant Clustering**:
```
RAG Service 1 ──┐
RAG Service 2 ──┼──→ Qdrant Cluster
RAG Service 3 ──┘       ├── Node 1 (Shard 1)
                        ├── Node 2 (Shard 2)
                        └── Node 3 (Shard 3)
```

## Best Practices

1. **Chunk Size**: Keep 256-512 tokens for balanced context
2. **Overlap**: Use 10-25% overlap to preserve context
3. **Metadata**: Add rich metadata for better filtering
4. **Reranking**: Always enable for production (quality > latency)
5. **Batch Ingestion**: Ingest documents in batches for efficiency
6. **Index Optimization**: Rebuild Qdrant indexes periodically
7. **Monitor Quality**: Track query relevance metrics
8. **Version Control**: Track document versions in metadata
9. **Cleanup**: Remove old documents to maintain quality
10. **Test Queries**: Maintain test queries for quality benchmarks

## Security Considerations

- **Tenant Isolation**: Ensure queries cannot access other tenant data
- **Input Validation**: Sanitize file uploads and text inputs
- **Rate Limiting**: Prevent abuse via excessive ingestion/queries
- **File Size Limits**: Cap upload size (e.g., 50MB max)
- **Qdrant Authentication**: Enable API key in production
- **Data Retention**: Implement document TTL and cleanup policies

## Roadmap

- [ ] Hybrid search (dense + sparse + keyword)
- [ ] Document versioning and updates
- [ ] Automatic index optimization
- [ ] Multi-vector search (different embedding models)
- [ ] Semantic caching for common queries
- [ ] Document summarization before chunking
- [ ] Graph-based retrieval (entities + relations)
- [ ] Query expansion and reformulation

## Contributing

See main repository [CONTRIBUTING.md](../CONTRIBUTING.md)

## License

See main repository [LICENSE](../LICENSE)

## Support

- Issues: [GitHub Issues](https://github.com/voxiana/experiments-hub/issues)
- Docs: [Main README](../README.md)
- bge-m3: [HuggingFace](https://huggingface.co/BAAI/bge-m3)
- Qdrant: [Documentation](https://qdrant.tech/documentation/)

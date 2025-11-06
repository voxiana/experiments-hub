"""
RAG Service - Ingestion & Retrieval
Document ingestion, chunking, embedding, and semantic search
Uses bge-m3 for embeddings and Qdrant for vector storage
"""

import asyncio
import hashlib
import logging
import time
from typing import List, Dict, Optional, Any
from pathlib import Path
import uuid

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Document loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

# Embedding model
EMBEDDING_MODEL = "BAAI/bge-m3"  # Multilingual, 1024-dim
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Reranker model
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Qdrant
QDRANT_URL = "http://qdrant:6333"
COLLECTION_PREFIX = "kb"

# Chunking
CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 128

# ============================================================================
# Request/Response Models
# ============================================================================

class IngestRequest(BaseModel):
    """Document ingestion request"""
    tenant_id: str
    source_url: Optional[str] = None
    source_text: Optional[str] = None
    title: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class IngestResponse(BaseModel):
    """Document ingestion response"""
    document_id: str
    chunks_created: int
    status: str
    took_ms: float

class QueryRequest(BaseModel):
    """RAG query request"""
    query: str
    tenant_id: str
    top_k: int = 3
    filters: Optional[Dict] = None
    use_reranker: bool = True

class QueryResponse(BaseModel):
    """RAG query response"""
    results: List[Dict]
    took_ms: float

class SearchResult(BaseModel):
    """Single search result"""
    text: str
    score: float
    source: str
    metadata: Dict

# ============================================================================
# Embedding Service
# ============================================================================

class EmbeddingService:
    """
    Embedding service using bge-m3
    Supports multilingual text (Arabic, English, code-switched)
    """

    def __init__(self):
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(f"✅ Embedding model loaded on {DEVICE}, dim={self.dimension}")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Embed multiple texts
        Returns: (n_texts, dimension) array
        """
        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # For cosine similarity
        )
        return embeddings

    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query
        Returns: (dimension,) array
        """
        embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            normalize_embeddings=True,
        )[0]
        return embedding

# ============================================================================
# Reranker Service
# ============================================================================

class RerankerService:
    """
    Cross-encoder reranker using bge-reranker-v2-m3
    Re-scores query-document pairs for better relevance
    """

    def __init__(self):
        logger.info(f"Loading reranker model: {RERANKER_MODEL}...")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
        self.model.to(DEVICE)
        self.model.eval()

        logger.info(f"✅ Reranker model loaded on {DEVICE}")

    def rerank(self, query: str, documents: List[str]) -> List[float]:
        """
        Rerank documents for a query
        Returns: relevance scores (higher is better)
        """
        if not documents:
            return []

        # Prepare pairs
        pairs = [[query, doc] for doc in documents]

        # Tokenize
        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512,
            ).to(DEVICE)

            # Get scores
            scores = self.model(**inputs, return_dict=True).logits.view(-1).float()
            scores = scores.cpu().numpy().tolist()

        return scores

# ============================================================================
# Document Processor
# ============================================================================

class DocumentProcessor:
    """
    Document loading, parsing, and chunking
    """

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", "؟", "。", " ", ""],
        )

    def load_document(self, file_path: str) -> Dict:
        """
        Load document from file
        Returns: {text, metadata}
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        try:
            if extension == '.pdf':
                loader = PyPDFLoader(file_path)
                pages = loader.load()
                text = "\n\n".join([p.page_content for p in pages])
                metadata = {"pages": len(pages)}

            elif extension in ['.docx', '.doc']:
                loader = UnstructuredWordDocumentLoader(file_path)
                docs = loader.load()
                text = "\n\n".join([d.page_content for d in docs])
                metadata = {}

            elif extension in ['.md', '.markdown']:
                loader = UnstructuredMarkdownLoader(file_path)
                docs = loader.load()
                text = "\n\n".join([d.page_content for d in docs])
                metadata = {}

            elif extension == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                metadata = {}

            else:
                raise ValueError(f"Unsupported file type: {extension}")

            return {
                "text": text,
                "metadata": metadata,
                "source": file_path,
            }

        except Exception as e:
            logger.error(f"Error loading document: {e}", exc_info=True)
            raise

    def chunk_text(self, text: str, metadata: Dict = None) -> List[Dict]:
        """
        Split text into chunks
        Returns: list of {text, metadata}
        """
        chunks = self.text_splitter.split_text(text)

        result = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = {
                **(metadata or {}),
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            result.append({
                "text": chunk,
                "metadata": chunk_metadata,
            })

        return result

# ============================================================================
# RAG Service
# ============================================================================

class RAGService:
    """
    RAG Service combining embeddings, vector search, and reranking
    """

    def __init__(self):
        self.embedder = EmbeddingService()
        self.reranker = RerankerService()
        self.processor = DocumentProcessor()

        # Initialize Qdrant client
        self.qdrant = QdrantClient(url=QDRANT_URL)

        logger.info("✅ RAG Service initialized")

    def _get_collection_name(self, tenant_id: str) -> str:
        """Get tenant-specific collection name"""
        return f"{COLLECTION_PREFIX}_{tenant_id}"

    def _ensure_collection(self, tenant_id: str):
        """Create collection if it doesn't exist"""
        collection_name = self._get_collection_name(tenant_id)

        collections = self.qdrant.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            logger.info(f"Creating collection: {collection_name}")
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedder.dimension,
                    distance=Distance.COSINE,
                ),
            )

    async def ingest_document(
        self,
        tenant_id: str,
        file_path: Optional[str] = None,
        text: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> IngestResponse:
        """
        Ingest a document into the knowledge base
        """
        start_time = time.time()

        # Generate document ID
        document_id = str(uuid.uuid4())

        # Load or use provided text
        if file_path:
            doc = self.processor.load_document(file_path)
            text = doc["text"]
            doc_metadata = {**doc["metadata"], **(metadata or {})}
            doc_metadata["source"] = file_path
        elif text:
            doc_metadata = metadata or {}
            doc_metadata["source"] = "text_input"
        else:
            raise ValueError("Either file_path or text must be provided")

        # Add document metadata
        doc_metadata["document_id"] = document_id
        doc_metadata["title"] = title or "Untitled"
        doc_metadata["tenant_id"] = tenant_id
        doc_metadata["created_at"] = time.time()

        # Chunk text
        chunks = self.processor.chunk_text(text, doc_metadata)

        # Embed chunks
        chunk_texts = [c["text"] for c in chunks]
        embeddings = self.embedder.embed_texts(chunk_texts)

        # Ensure collection exists
        self._ensure_collection(tenant_id)
        collection_name = self._get_collection_name(tenant_id)

        # Prepare points
        points = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = hashlib.md5(
                f"{document_id}_{i}".encode()
            ).hexdigest()

            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload={
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                },
            ))

        # Upsert to Qdrant
        self.qdrant.upsert(
            collection_name=collection_name,
            points=points,
        )

        took_ms = (time.time() - start_time) * 1000

        logger.info(
            f"✅ Ingested document {document_id}: {len(chunks)} chunks, {took_ms:.0f}ms"
        )

        return IngestResponse(
            document_id=document_id,
            chunks_created=len(chunks),
            status="completed",
            took_ms=took_ms,
        )

    async def query(
        self,
        query: str,
        tenant_id: str,
        top_k: int = 3,
        filters: Optional[Dict] = None,
        use_reranker: bool = True,
    ) -> QueryResponse:
        """
        Query the knowledge base
        Returns top-k most relevant chunks
        """
        start_time = time.time()

        # Ensure collection exists
        collection_name = self._get_collection_name(tenant_id)

        # Embed query
        query_vector = self.embedder.embed_query(query)

        # Search Qdrant
        # Retrieve more candidates for reranking
        retrieve_k = top_k * 5 if use_reranker else top_k

        search_result = self.qdrant.search(
            collection_name=collection_name,
            query_vector=query_vector.tolist(),
            limit=retrieve_k,
            query_filter=self._build_filter(filters) if filters else None,
        )

        # Extract results
        candidates = []
        for hit in search_result:
            candidates.append({
                "text": hit.payload["text"],
                "score": hit.score,
                "metadata": hit.payload["metadata"],
            })

        # Rerank if enabled
        if use_reranker and len(candidates) > 0:
            texts = [c["text"] for c in candidates]
            rerank_scores = self.reranker.rerank(query, texts)

            # Update scores
            for candidate, score in zip(candidates, rerank_scores):
                candidate["rerank_score"] = score

            # Sort by rerank score
            candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        # Take top-k
        results = candidates[:top_k]

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"],
                "score": result.get("rerank_score", result["score"]),
                "source": result["metadata"].get("title", "Unknown"),
                "metadata": result["metadata"],
            })

        took_ms = (time.time() - start_time) * 1000

        logger.info(f"Query returned {len(formatted_results)} results in {took_ms:.0f}ms")

        return QueryResponse(
            results=formatted_results,
            took_ms=took_ms,
        )

    def _build_filter(self, filters: Dict) -> Filter:
        """Build Qdrant filter from dict"""
        conditions = []

        for key, value in filters.items():
            conditions.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )

        if len(conditions) == 1:
            return Filter(must=[conditions[0]])
        else:
            return Filter(must=conditions)

# ============================================================================
# FastAPI Server
# ============================================================================

app = FastAPI(title="RAG Service", version="1.0.0")
rag_service = None

@app.on_event("startup")
async def startup():
    global rag_service
    rag_service = RAGService()
    logger.info("📚 RAG Service started")

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "device": DEVICE,
    }

@app.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    tenant_id: str,
    file: Optional[UploadFile] = File(None),
    text: Optional[str] = None,
    title: Optional[str] = None,
):
    """
    Ingest a document
    Can accept either a file upload or raw text
    """
    try:
        if file:
            # Save uploaded file temporarily
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            result = await rag_service.ingest_document(
                tenant_id=tenant_id,
                file_path=tmp_path,
                title=title or file.filename,
            )

            # Cleanup
            import os
            os.unlink(tmp_path)

        elif text:
            result = await rag_service.ingest_document(
                tenant_id=tenant_id,
                text=text,
                title=title,
            )

        else:
            raise HTTPException(status_code=400, detail="Either file or text required")

        return result

    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """
    Query the knowledge base
    """
    try:
        result = await rag_service.query(
            query=request.query,
            tenant_id=request.tenant_id,
            top_k=request.top_k,
            filters=request.filters,
            use_reranker=request.use_reranker,
        )
        return result

    except Exception as e:
        logger.error(f"Query failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info",
    )

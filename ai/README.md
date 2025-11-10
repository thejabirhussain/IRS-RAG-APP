# IRS RAG Chat Service

A production-ready Retrieval-Augmented Generation (RAG) chat service that answers questions using content exclusively from `https://www.irs.gov/` and its direct subpages. The service provides verifiable citations for every answer and supports both local and cloud deployment.

## Overview

This RAG system:
- **Scrapes and indexes** only IRS.gov content (HTML pages and PDFs)
- **Provides citations** with URLs, page titles, section headings, and character offsets
- **Enforces guardrails** to prevent hallucination and scope creep
- **Supports multiple providers** for embeddings (OpenAI/local) and LLMs (OpenAI/Ollama)
- **Runs locally** with Docker Compose
- **Cloud-ready** for Azure/AWS deployment

## Features

- ✅ **Respectful crawling**: Follows robots.txt, rate limits, retries with backoff
- ✅ **Multi-format support**: HTML pages and PDFs with page number tracking
- ✅ **Smart chunking**: Section-aware chunking with overlap (800-1600 chars, 20-30% overlap)
- ✅ **Vector search**: Qdrant with HNSW indexing (M=64, ef_construct=128)
- ✅ **Reranking**: Optional cross-encoder reranking for improved retrieval
- ✅ **Citation system**: Every answer includes source URL, title, section, excerpt, and character offsets
- ✅ **Guardrails**: Returns "no verifiable information" when unsupported; adds legal disclaimer for advice queries
- ✅ **Type safety**: Full type hints with mypy checking
- ✅ **Testing**: Unit and integration tests with pytest
- ✅ **CI/CD**: GitHub Actions for lint, type-check, and tests

## Tech Stack

- **Language**: Python 3.11+
- **Backend**: FastAPI
- **Vector DB**: Qdrant (self-hosted Docker, cloud-ready)
- **Embeddings**: OpenAI `text-embedding-3-small` (default) or local `sentence-transformers/all-MiniLM-L6-v2`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-12-v2` (optional, default-on)
- **LLM**: OpenAI `gpt-4o-mini` (default) or Ollama (`llama3`, `mistral`)
- **PDF parsing**: PyMuPDF
- **HTML parsing**: BeautifulSoup4 + readability-lxml
- **Task queue**: Simple background threads (optional `arq` for production)
- **Auth**: API key header + `slowapi` rate limiting
- **Testing**: pytest
- **Formatting**: ruff, black, mypy

## Prerequisites

- **Docker** and **Docker Compose** (for local development)
- **Python 3.11+** (for local development without Docker)
- **OpenAI API key** (optional, for hosted embeddings/LLM)
- **Ollama** (optional, for local LLM)

## Quickstart

### 1. Setup Environment

```bash
# Clone and navigate to project
cd "IRS RAG POC 02"

# Copy environment file
cp .env.example .env

# Edit .env and set your API keys
# OPENAI_API_KEY=sk-...
# API_KEY=your-secret-api-key
```

### 2. Start Services with Docker

```bash
# Start Qdrant and API services
docker-compose up -d

# Check services are healthy
curl http://localhost:8000/health
curl http://localhost:6333/health
```

### 3. Ingest IRS.gov Content

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Run ingestion (crawl -> parse -> chunk -> embed -> upsert)
python -m app.scripts.ingest \
  --seed https://www.irs.gov/ \
  --max-pages 1500 \
  --concurrency 4 \
  --allow-pdf true
```

This will:
1. Discover URLs from sitemaps and robots.txt
2. Crawl IRS.gov pages (respecting rate limits)
3. Parse HTML and PDFs
4. Chunk text (section-aware with overlap)
5. Generate embeddings
6. Upsert to Qdrant

**Note**: Initial ingestion may take 30-60 minutes depending on `--max-pages` and `--concurrency`.

### 4. Start API Server

```bash
# If using Docker (already running)
# API is available at http://localhost:8000

# Or run locally
uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Query the API

```bash
# Simple query
curl -X POST "http://localhost:8000/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the 2024 standard deduction for single filers?",
    "json": true
  }'
```

**Response:**
```json
{
  "answer_text": "The standard deduction for single filers in 2024 is $14,600...",
  "sources": [
    {
      "url": "https://www.irs.gov/...",
      "title": "Standard Deduction",
      "section": "Filing Status",
      "snippet": "The standard deduction for single filers...",
      "char_start": 1024,
      "char_end": 1350,
      "score": 0.91
    }
  ],
  "confidence": "high",
  "query_embedding_similarity": [0.91, 0.88, 0.84]
}
```

## Configuration

### Environment Variables

Edit `.env` to configure:

```bash
# API Security
API_KEY=dev-secret  # Change in production!

# Embeddings Provider
EMBEDDINGS_PROVIDER=openai  # or "local"
OPENAI_EMBED_MODEL=text-embedding-3-small
OPENAI_API_KEY=sk-...

# LLM Provider
LLM_PROVIDER=openai  # or "ollama"
OPENAI_CHAT_MODEL=gpt-4o-mini
OLLAMA_MODEL=llama3
OLLAMA_HOST=http://localhost:11434

# Qdrant
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=  # Optional
COLLECTION_NAME=irs_rag_v1

# Crawling
CRAWL_BASE=https://www.irs.gov
RATE_LIMIT_RPS=0.5  # Requests per second

# Retrieval
SIMILARITY_CUTOFF=0.22  # Cosine similarity threshold
TOP_K=40  # Initial retrieval count
TOP_N=3   # Final chunks after reranking

# Legal Disclaimer
LEGAL_DISCLAIMER=I am not a lawyer; for legal or tax-filing advice consult a qualified tax professional or the IRS.
```

## Provider Options

### OpenAI (Hosted)

```bash
export OPENAI_API_KEY=sk-...
export EMBEDDINGS_PROVIDER=openai
export LLM_PROVIDER=openai
```

### Local Embeddings + OpenAI LLM

```bash
export EMBEDDINGS_PROVIDER=local
export LLM_PROVIDER=openai
export OPENAI_API_KEY=sk-...
```

### Fully Local (Ollama + Sentence Transformers)

```bash
# Install Ollama: https://ollama.ai
ollama pull llama3

export EMBEDDINGS_PROVIDER=local
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama3
export OLLAMA_HOST=http://localhost:11434
```

## Usage

### API Endpoints

#### Chat Endpoint

```bash
POST /v1/chat
Content-Type: application/json

{
  "query": "When are estimated tax payments due?",
  "filters": {
    "content_type": "html"  # Optional
  },
  "json": true  # Optional: return JSON format
}
```

#### Admin Endpoints

```bash
# Get statistics
GET /admin/stats
X-API-Key: your-api-key

# Trigger reindex
POST /admin/reindex
X-API-Key: your-api-key
Content-Type: application/json

{
  "force": false
}

# Trigger ingestion
POST /admin/ingest
X-API-Key: your-api-key
Content-Type: application/json

{
  "max_pages": 1000,
  "concurrency": 4
}
```

### CLI Scripts

```bash
# Ingest content
python -m app.scripts.ingest \
  --seed https://www.irs.gov/ \
  --max-pages 2000 \
  --concurrency 4 \
  --allow-pdf true

# Rebuild index
python -m app.scripts.rebuild_index --force

# Run evaluation suite
python -m app.scripts.eval_suite --output-file eval_results.json

# Export snapshot
python -m app.scripts.export_snapshot --output-dir snapshots
```

## Citation System

Every answer includes verifiable citations:

- **Source URL**: Direct link to IRS.gov page
- **Page Title**: Title of the source page
- **Section Heading**: Relevant section (if available)
- **Excerpt**: Snippet of source text
- **Character Offsets**: `char_start` and `char_end` for exact location
- **Similarity Score**: Embedding similarity score

Example:
```
[1] Standard Deduction — Filing Status — https://www.irs.gov/... — 
    excerpt: "The standard deduction for single filers in 2024 is $14,600" 
    (char_start: 1024, char_end: 1350)
```

## Guardrails

### No Verifiable Information

When the knowledge base doesn't contain supporting information, the system returns:

> "I don't have verifiable information in the knowledge base for that query."

And optionally offers closest matches with excerpts.

### Legal Disclaimer

For queries that imply tax/legal advice, a disclaimer is automatically prepended:

> "I am not a lawyer; for legal or tax-filing advice consult a qualified tax professional or the IRS."

Keywords that trigger the disclaimer: "should I file", "what should I claim", "advice", "deduct", "penalty strategy", etc.

### Scope Limiting

The system only answers from IRS.gov content. Queries outside this scope are marked as such.

## Testing

```bash
# Run all tests
pytest app/tests/ -v

# Run specific test suite
pytest app/tests/test_chunker.py -v
pytest app/tests/test_embed_and_upsert.py -v
pytest app/tests/test_rag_pipeline.py -v

# With coverage
pytest app/tests/ --cov=app --cov-report=html
```

## Development

### Code Quality

```bash
# Format code
make fmt
# or
black app/
ruff check --fix app/

# Lint
make lint
# or
ruff check app/

# Type check
make typecheck
# or
mypy app/ --ignore-missing-imports
```

### Makefile Targets

```bash
make fmt          # Format code
make lint         # Lint code
make typecheck    # Type check
make test         # Run tests
make up           # Start docker-compose
make down         # Stop docker-compose
make ingest       # Run ingestion
make reindex      # Rebuild index
make eval         # Run evaluation suite
make clean        # Clean cache files
```

## Deployment

### Local Development

```bash
docker-compose up -d
python -m app.scripts.ingest
uvicorn app.api.main:app --reload
```

### Cloud Deployment (Azure)

1. **Container Registry**: Build and push Docker image
   ```bash
   docker build -f Dockerfile.api -t your-registry.azurecr.io/irs-rag-api:latest .
   docker push your-registry.azurecr.io/irs-rag-api:latest
   ```

2. **Azure Container Apps**: Deploy API service
   - Use managed Qdrant Cloud or Azure Cosmos DB for Vector Search
   - Store secrets in Azure Key Vault
   - Configure TLS/HTTPS

3. **Environment Variables**:
   ```bash
   QDRANT_URL=https://your-qdrant-cloud-url
   QDRANT_API_KEY=...
   OPENAI_API_KEY=...
   API_KEY=...
   ```

### Cloud Deployment (AWS)

1. **ECS/Fargate**: Deploy API container
   ```bash
   # Build and push to ECR
   aws ecr get-login-password | docker login --username AWS --password-stdin your-account.dkr.ecr.region.amazonaws.com
   docker build -f Dockerfile.api -t your-account.dkr.ecr.region.amazonaws.com/irs-rag-api:latest .
   docker push your-account.dkr.ecr.region.amazonaws.com/irs-rag-api:latest
   ```

2. **Secrets**: Store in AWS Secrets Manager
3. **Qdrant**: Use Qdrant Cloud or Amazon OpenSearch Serverless (with migration notes)
4. **Load Balancer**: Configure ALB with TLS termination

### Migration Notes

**Vector DB Alternatives:**
- **Milvus**: Similar API, update `qdrant_client.py` to use Milvus client
- **Pinecone**: REST API, update retriever to use Pinecone SDK
- **Weaviate**: GraphQL API, update client implementation
- **RedisVector**: Redis extension, update client to use Redis client

**LLM Alternatives:**
- **vLLM**: Expose OpenAI-compatible server, set `LLM_PROVIDER=vllm` and `VLLM_BASE_URL`
- **Azure OpenAI**: Use `AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_API_KEY`

## Evaluation

Run the evaluation suite with 20 IRS queries:

```bash
python -m app.scripts.eval_suite --output-file eval_results.json --verbose
```

**Acceptance Criteria:**
- **Correctness**: ≥80% overlap with source text
- **Attribution**: At least 1 source per claim/answer
- **Non-hallucination**: Must output "no verifiable info" when unsupported
- **Latency**: Retrieval ≤1.2s, generation ≤1.5s (configurable)

## Limitations

- **Scope**: Only IRS.gov content (no other domains)
- **No live web**: Answers are based on indexed content only
- **No legal advice**: System provides information, not legal/tax advice
- **Update frequency**: Depends on re-crawl schedule (daily/weekly recommended)

## Security & Compliance

- ✅ **TLS/HTTPS**: Required in production (TLS termination at load balancer)
- ✅ **API Key Auth**: Required for admin endpoints
- ✅ **Rate Limiting**: Per-IP rate limits via `slowapi`
- ✅ **Secrets Management**: Use Azure Key Vault or AWS Secrets Manager
- ✅ **PII Masking**: Logs mask SSNs, EINs, etc.
- ✅ **Dependency Scanning**: GitHub Dependabot enabled
- ✅ **Encryption**: Qdrant at-rest encryption (if managed)

## Maintenance

### Re-crawl Schedule

Set up a cron job or scheduled task to re-crawl IRS.gov:

```bash
# Daily re-crawl (example cron)
0 2 * * * cd /path/to/project && python -m app.scripts.ingest --max-pages 1000
```

The crawler respects `If-Modified-Since` and `ETag` headers to avoid re-downloading unchanged content.

### Versioning Chunks

Chunks are versioned: never deleted, marked `is_latest=false` when obsolete. This preserves history and enables rollback.

### Backup

Export snapshots regularly:

```bash
python -m app.scripts.export_snapshot --output-dir snapshots
```

## Troubleshooting

### Qdrant Connection Issues

```bash
# Check Qdrant is running
curl http://localhost:6333/health

# Check docker-compose
docker-compose ps
docker-compose logs qdrant
```

### Embedding Issues

```bash
# Test local embeddings
python -c "from app.vector.embeddings import LocalEmbeddingProvider; p = LocalEmbeddingProvider(); print(p.get_embedding('test'))"
```

### LLM Issues

```bash
# Test Ollama
curl http://localhost:11434/api/generate -d '{"model":"llama3","prompt":"test"}'
```

## License

MIT

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Run `make fmt lint typecheck test`
5. Submit a pull request

## Support

For issues and questions, please open an issue on GitHub.



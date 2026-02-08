# Triton Semantic Search

GPU-accelerated semantic search using NVIDIA Triton Inference Server with ONNX Runtime and TensorRT optimization. Serves three embedding models simultaneously and powers a code search system indexed against all portfolio repositories.

**GPU:** NVIDIA GTX 1080 | **Latency:** 2–5ms | **Throughput:** 500–1000 sentences/sec

## Models

Three embedding models served concurrently, each optimized for different use cases:

| Model | Backend | Dimensions | Max Batch | Use Case |
|:------|:--------|:-----------|:----------|:---------|
| `all-MiniLM-L6-v2` | ONNX Runtime | 384 | 32 | Fast general-purpose embeddings |
| `e5-large-v2` | ONNX Runtime | 1024 | 16 | Higher-quality semantic search |
| `bge_embeddings` | Python (ONNX) | 768 | 32 | Gateway default, balanced quality/speed |

All models use TensorRT FP16 optimization on GPU with dynamic batching.

## Quick Start

```bash
# 1. Download models
./setup.sh

# 2. Start Triton server
docker compose up -d

# 3. Wait for TensorRT compilation (2-5 min first time)
docker compose logs -f triton

# 4. Test with Python client
cd client
pip install -r requirements.txt
python semantic_search.py
```

## Ports

| Port | Protocol | Purpose |
|:-----|:---------|:--------|
| 8020 | HTTP | REST inference API |
| 8021 | gRPC | High-performance inference |
| 8022 | HTTP | Prometheus metrics |

## Code Search System

A complete semantic search pipeline for querying code across all portfolio repositories using natural language.

### Indexing

The indexer scans repositories, extracts code chunks (functions, classes, files), generates embeddings via Triton, and stores them in PostgreSQL with pgvector.

**Supported languages:** Python, JavaScript, TypeScript, Go, Rust, Java, C/C++, Ruby, PHP, SQL, Bash, Markdown

```bash
# Index a single repository
python code_search/indexer.py /path/to/repo \
  --repo-name my-repo \
  --db-url postgresql://user:pass@host/db \
  --model minilm

# Index all portfolio repositories
./code_search/index-all.sh
```

Python files are indexed at the function/class level. Other languages are indexed at the file level.

### Search

```bash
# Natural language code search
./code_search/search-code "authentication middleware with JWT"

# With filters
python code_search/search.py "database connection pooling" \
  --repo bookmarked --language python --limit 5 --model e5
```

The `search-code` wrapper handles connectivity automatically: checks Triton health, tries VPS PostgreSQL, falls back to Neon if unreachable.

### Database Schema

Two tables for different embedding dimensions, both using HNSW indexes for fast cosine similarity search:

```sql
CREATE TABLE code_embeddings (
    id SERIAL PRIMARY KEY,
    repo_name TEXT NOT NULL,
    file_path TEXT NOT NULL,
    chunk_type TEXT NOT NULL,     -- function, class, file
    name TEXT,
    content TEXT NOT NULL,
    start_line INTEGER,
    end_line INTEGER,
    embedding vector(384) NOT NULL,
    language TEXT,
    indexed_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(repo_name, file_path, chunk_type, name, start_line)
);

CREATE INDEX ON code_embeddings
    USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
```

An equivalent `code_embeddings_e5` table exists with `vector(1024)` for E5 embeddings.

## Python Client

```python
from semantic_search import SemanticSearchClient

client = SemanticSearchClient(triton_url="localhost:8020")

# Generate embeddings
embeddings = client.encode(["React hooks tutorial", "Kubernetes deployment"])

# Similarity search
results = client.similarity(
    query="How to use useState",
    documents=["React hooks guide", "CSS flexbox tutorial", "useState examples"]
)
```

**Pipeline:** Tokenize (HuggingFace) → Triton inference → Mean pooling → L2 normalize → Cosine similarity

## Integration with AI Gateway

The [Shared AI Gateway](https://github.com/maxjeffwell/shared-ai-gateway) calls Triton as its embedding backend:

- **Tier 1:** VPS CPU Triton — `bge_embeddings` model, always available
- **Tier 2:** Local GPU Triton — GTX 1080 via Cloudflare tunnel

Uses the KServe V2 REST protocol for inference requests.

## Project Structure

```
triton-semantic-search/
├── setup.sh                    # Download models
├── docker-compose.yml          # Triton deployment (GPU)
├── models/
│   ├── all-minilm-l6-v2/      # 384-dim, ONNX Runtime
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx
│   ├── e5-large-v2/            # 1024-dim, ONNX Runtime
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx
│   └── bge_embeddings/         # 768-dim, Python backend
│       ├── config.pbtxt
│       ├── 1/model.py          # Custom Triton Python backend
│       └── requirements.txt
├── client/
│   ├── semantic_search.py      # Python client library
│   ├── benchmark.py            # Latency & throughput benchmarks
│   └── requirements.txt
└── code_search/
    ├── indexer.py               # Repository indexer
    ├── search.py                # Semantic code search
    ├── schema.sql               # pgvector table definitions
    ├── index-all.sh             # Batch index all repos
    ├── index-redundant.sh       # Dual-database indexing (VPS + Neon)
    ├── search-code              # CLI wrapper with auto-failover
    └── start-tunnel.sh          # Cloudflare tunnel to VPS
```

## Docker Deployment

```yaml
services:
  triton:
    image: nvcr.io/nvidia/tritonserver:24.01-py3
    ports:
      - "8020:8000"
      - "8021:8001"
      - "8022:8002"
    volumes:
      - ./models:/models
    command: tritonserver --model-repository=/models --log-verbose=1
    runtime: nvidia
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/v2/health/ready"]
      interval: 30s
      start_period: 120s
    restart: unless-stopped
```

The 120-second start period accounts for TensorRT engine compilation on first launch. Subsequent starts use cached engines in `trt_cache/`.

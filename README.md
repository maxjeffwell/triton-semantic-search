# Triton Semantic Search

GPU-accelerated semantic search using NVIDIA Triton Inference Server with ONNX Runtime + TensorRT.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Triton Inference Server                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                  ONNX Runtime Backend                  │  │
│  │  ┌──────────────────────────────────────────────────┐  │  │
│  │  │           TensorRT Execution Provider            │  │  │
│  │  │              (FP16 Optimization)                 │  │  │
│  │  └──────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │                               │
│                              ▼                               │
│                    GTX 1080 (GPU 1)                          │
└──────────────────────────────────────────────────────────────┘
```

## Model

- **all-MiniLM-L6-v2**: 22.7M parameters, 384-dim embeddings
- Trained on 1B+ sentence pairs for semantic similarity
- [HuggingFace Model Card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

## Quick Start

### 1. Download the model

```bash
./setup.sh
```

### 2. Start Triton server

```bash
docker-compose up
```

> **Note**: First startup takes 2-5 minutes while TensorRT compiles optimized engines.
> Subsequent starts use cached engines and are much faster.

### 3. Test with Python client

```bash
cd client
pip install -r requirements.txt
python semantic_search.py
```

## Endpoints

| Port | Protocol | Description |
|------|----------|-------------|
| 8000 | HTTP | REST API |
| 8001 | gRPC | High-performance RPC |
| 8002 | HTTP | Prometheus metrics |

## API Example

```python
from semantic_search import SemanticSearchClient

client = SemanticSearchClient()

# Get embeddings
embeddings = client.encode(["Hello world", "How are you?"])

# Semantic search
results = client.similarity(
    query="machine learning deployment",
    documents=["Triton deploys ML models", "The weather is nice"]
)
```

## Configuration

Edit `models/all-minilm-l6-v2/config.pbtxt` to:
- Change GPU (`gpus: [ 1 ]` → different index)
- Adjust batch sizes
- Modify TensorRT settings

## Performance

Expected performance on GTX 1080 with TensorRT FP16:
- Latency: ~2-5ms per batch
- Throughput: ~500-1000 sentences/sec (depends on sequence length)

## Files

```
triton-semantic-search/
├── setup.sh                    # Download model
├── docker-compose.yml          # Triton deployment
├── models/
│   └── all-minilm-l6-v2/
│       ├── config.pbtxt        # Triton model config
│       ├── 1/
│       │   └── model.onnx      # ONNX model (downloaded)
│       └── trt_cache/          # TensorRT engine cache
└── client/
    ├── requirements.txt
    └── semantic_search.py      # Python client example
```

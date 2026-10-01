"""
OVMS embedding client (OpenVINO Model Server, OpenAI-compatible /v3/embeddings).

Since 2026-10-01 the e5 code index is embedded by the in-cluster OVMS on the
Intel iGPU (model "e5-large", 1024 dims, CLS pooling, L2-normalized by the
server). The same OVMS embeds bookmarked, firebook and intervalai data, so all
1024-d vectors in the homelab come from one model. Triton was retired on
2026-09-30.

From a workstation the default path is the AI gateway's public embed endpoint
(https://ai-gateway.el-jefe.me/api/ai/embed), which forwards to the same OVMS
e5-large; no port-forward needed. Set EMBED_URL to an OVMS base URL
(e.g. http://ovms-embeddings.ovms:8000 in-cluster) to call OVMS directly.
"""

import json
import os
import urllib.request
from typing import List

import numpy as np

DEFAULT_OVMS_URL = os.environ.get("EMBED_URL", "https://ai-gateway.el-jefe.me/api/ai/embed")


class OvmsEmbedder:
    """Embed texts with OVMS. `prefix` follows the e5 convention:
    'passage: ' for documents, 'query: ' for search queries."""

    def __init__(self, ovms_url: str = DEFAULT_OVMS_URL, model: str = "e5-large",
                 prefix: str = "", dims: int = 1024, timeout: int = 60):
        url = ovms_url.rstrip("/")
        # Gateway mode: .../api/ai/embed takes {"texts": [...]} -> {"embeddings": [...]}
        self.gateway = url.endswith("/api/ai/embed")
        self.url = url if self.gateway else url + "/v3/embeddings"
        self.model = model
        self.prefix = prefix
        self.dims = dims
        self.timeout = timeout

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        inputs = [self.prefix + t for t in texts]
        body_obj = {"texts": inputs} if self.gateway else {"model": self.model, "input": inputs}
        payload = json.dumps(body_obj).encode("utf-8")
        # Cloudflare (in front of ai-gateway.el-jefe.me) answers error 1010 / 403 to
        # Python's default "Python-urllib" user agent: always send our own.
        req = urllib.request.Request(
            self.url, data=payload, method="POST",
            headers={
                "Content-Type": "application/json",
                "User-Agent": "code-search-indexer/1.0 (+https://github.com/maxjeffwell/triton-semantic-search)",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            body = json.loads(resp.read())

        if self.gateway:
            vectors = body.get("embeddings") or []
        else:
            vectors = [d["embedding"] for d in sorted(body["data"], key=lambda d: d["index"])]
        if len(vectors) != len(texts):
            raise RuntimeError(f"embedder returned {len(vectors)} embeddings for {len(texts)} inputs")

        embeddings = np.asarray(vectors, dtype=np.float32)
        if embeddings.shape[1] != self.dims:
            raise RuntimeError(f"expected {self.dims} dims, got {embeddings.shape[1]}")

        # OVMS normalizes already; normalize again defensively so cosine == dot.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return embeddings / norms

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

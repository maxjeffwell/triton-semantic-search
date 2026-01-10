#!/usr/bin/env python3
"""
Semantic Search Client for Triton + all-MiniLM-L6-v2

This client demonstrates:
1. Tokenizing text with HuggingFace transformers
2. Sending inference requests to Triton
3. Mean pooling to get sentence embeddings
4. Cosine similarity for semantic search
"""

import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from typing import List


class SemanticSearchClient:
    def __init__(self, triton_url: str = "localhost:8020", model_name: str = "all-minilm-l6-v2"):
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

        # Check if model is ready
        if not self.client.is_model_ready(model_name):
            raise RuntimeError(f"Model '{model_name}' is not ready on Triton server")
        print(f"[OK] Connected to Triton, model '{model_name}' is ready")

    def encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """
        Encode texts into 384-dimensional embeddings.

        Args:
            texts: List of strings to encode
            normalize: Whether to L2-normalize embeddings (recommended for cosine similarity)

        Returns:
            numpy array of shape (len(texts), 384)
        """
        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="np"
        )

        # Prepare inputs for Triton
        input_ids = httpclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
        attention_mask = httpclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
        token_type_ids = httpclient.InferInput("token_type_ids", encoded["input_ids"].shape, "INT64")

        input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
        token_type_ids.set_data_from_numpy(np.zeros_like(encoded["input_ids"], dtype=np.int64))

        # Request output
        output = httpclient.InferRequestedOutput("last_hidden_state")

        # Run inference
        response = self.client.infer(
            model_name=self.model_name,
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[output]
        )

        # Get token embeddings
        token_embeddings = response.as_numpy("last_hidden_state")

        # Mean pooling (average over tokens, weighted by attention mask)
        mask = encoded["attention_mask"][:, :, np.newaxis].astype(np.float32)
        embeddings = np.sum(token_embeddings * mask, axis=1) / np.sum(mask, axis=1)

        # L2 normalize
        if normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def similarity(self, query: str, documents: List[str]) -> List[tuple]:
        """
        Find most similar documents to a query.

        Returns:
            List of (document, score) tuples, sorted by similarity (descending)
        """
        # Encode query and documents
        query_embedding = self.encode([query])[0]
        doc_embeddings = self.encode(documents)

        # Cosine similarity (embeddings are normalized, so dot product = cosine sim)
        scores = np.dot(doc_embeddings, query_embedding)

        # Sort by score
        results = [(doc, float(score)) for doc, score in zip(documents, scores)]
        results.sort(key=lambda x: x[1], reverse=True)

        return results


def main():
    """Demo: Semantic search over sample documents"""

    print("=" * 60)
    print("Semantic Search Demo with Triton + TensorRT")
    print("=" * 60)
    print()

    # Initialize client
    client = SemanticSearchClient()

    # Sample documents
    documents = [
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models can be deployed using Triton Inference Server",
        "Python is a popular programming language for data science",
        "NVIDIA GPUs accelerate deep learning inference",
        "Semantic search finds documents by meaning, not just keywords",
        "The weather today is sunny with a high of 75 degrees",
        "Docker containers package applications with their dependencies",
        "TensorRT optimizes neural networks for faster inference",
    ]

    # Sample queries
    queries = [
        "How do I speed up neural network inference?",
        "What programming language is good for ML?",
        "How do I deploy machine learning models?",
    ]

    for query in queries:
        print(f"Query: \"{query}\"")
        print("-" * 50)

        results = client.similarity(query, documents)

        for i, (doc, score) in enumerate(results[:3]):
            print(f"  {i+1}. [{score:.3f}] {doc}")
        print()


if __name__ == "__main__":
    main()

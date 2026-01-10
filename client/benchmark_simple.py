#!/usr/bin/env python3
"""Simple sequential throughput benchmark"""

import time
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer

def main():
    print("=" * 60)
    print("Throughput Benchmark (Sequential)")
    print("=" * 60)

    client = httpclient.InferenceServerClient(url="localhost:8020")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    # Test sentences
    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can be deployed using Triton.",
        "Python is great for data science and AI.",
        "GPUs accelerate deep learning inference significantly.",
    ] * 250  # 1000 total sentences

    batch_size = 16
    num_batches = len(sentences) // batch_size

    print(f"\nEncoding {len(sentences)} sentences in batches of {batch_size}...")
    print()

    start = time.perf_counter()

    for i in range(num_batches):
        batch = sentences[i * batch_size:(i + 1) * batch_size]

        encoded = tokenizer(batch, padding=True, truncation=True, max_length=128, return_tensors="np")

        input_ids = httpclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
        attention_mask = httpclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
        token_type_ids = httpclient.InferInput("token_type_ids", encoded["input_ids"].shape, "INT64")

        input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
        token_type_ids.set_data_from_numpy(np.zeros_like(encoded["input_ids"], dtype=np.int64))

        output = httpclient.InferRequestedOutput("last_hidden_state")

        client.infer(model_name="all-minilm-l6-v2", inputs=[input_ids, attention_mask, token_type_ids], outputs=[output])

    elapsed = time.perf_counter() - start

    print(f"Total sentences:     {len(sentences)}")
    print(f"Batch size:          {batch_size}")
    print(f"Total time:          {elapsed:.2f} sec")
    print(f"Throughput:          {len(sentences) / elapsed:.1f} sentences/sec")
    print(f"Avg latency/batch:   {(elapsed / num_batches) * 1000:.2f} ms")
    print()
    print("=" * 60)

if __name__ == "__main__":
    main()

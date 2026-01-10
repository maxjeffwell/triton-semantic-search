#!/usr/bin/env python3
"""
Benchmark for Triton Semantic Search

Measures:
- Latency (ms per request)
- Throughput (sentences per second)
- Batch size impact
"""

import time
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics


class TritonBenchmark:
    def __init__(self, triton_url: str = "localhost:8020", model_name: str = "all-minilm-l6-v2"):
        self.client = httpclient.InferenceServerClient(url=triton_url)
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

    def encode_batch(self, texts: list) -> float:
        """Encode a batch and return latency in ms"""
        start = time.perf_counter()

        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="np"
        )

        input_ids = httpclient.InferInput("input_ids", encoded["input_ids"].shape, "INT64")
        attention_mask = httpclient.InferInput("attention_mask", encoded["attention_mask"].shape, "INT64")
        token_type_ids = httpclient.InferInput("token_type_ids", encoded["input_ids"].shape, "INT64")

        input_ids.set_data_from_numpy(encoded["input_ids"].astype(np.int64))
        attention_mask.set_data_from_numpy(encoded["attention_mask"].astype(np.int64))
        token_type_ids.set_data_from_numpy(np.zeros_like(encoded["input_ids"], dtype=np.int64))

        output = httpclient.InferRequestedOutput("last_hidden_state")

        self.client.infer(
            model_name=self.model_name,
            inputs=[input_ids, attention_mask, token_type_ids],
            outputs=[output]
        )

        return (time.perf_counter() - start) * 1000  # ms

    def warmup(self, n: int = 10):
        """Warmup the model"""
        print("Warming up...", end=" ", flush=True)
        texts = ["This is a warmup sentence."] * 4
        for _ in range(n):
            self.encode_batch(texts)
        print("done")

    def benchmark_latency(self, batch_size: int, num_iterations: int = 100):
        """Measure latency for a given batch size"""
        texts = ["The quick brown fox jumps over the lazy dog."] * batch_size
        latencies = []

        for _ in range(num_iterations):
            latency = self.encode_batch(texts)
            latencies.append(latency)

        return {
            "batch_size": batch_size,
            "iterations": num_iterations,
            "mean_ms": statistics.mean(latencies),
            "median_ms": statistics.median(latencies),
            "p95_ms": np.percentile(latencies, 95),
            "p99_ms": np.percentile(latencies, 99),
            "min_ms": min(latencies),
            "max_ms": max(latencies),
            "throughput": (batch_size * 1000) / statistics.mean(latencies)  # sentences/sec
        }

    def benchmark_concurrent(self, num_workers: int = 4, requests_per_worker: int = 50):
        """Measure throughput with concurrent requests"""
        texts = ["The quick brown fox jumps over the lazy dog."] * 4  # batch of 4

        def worker():
            latencies = []
            for _ in range(requests_per_worker):
                latencies.append(self.encode_batch(texts))
            return latencies

        start = time.perf_counter()
        all_latencies = []

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker) for _ in range(num_workers)]
            for future in as_completed(futures):
                all_latencies.extend(future.result())

        total_time = time.perf_counter() - start
        total_sentences = num_workers * requests_per_worker * 4

        return {
            "workers": num_workers,
            "total_requests": num_workers * requests_per_worker,
            "total_sentences": total_sentences,
            "total_time_sec": total_time,
            "throughput": total_sentences / total_time,
            "mean_latency_ms": statistics.mean(all_latencies),
            "p95_latency_ms": np.percentile(all_latencies, 95)
        }


def main():
    print("=" * 60)
    print("Triton Semantic Search Benchmark")
    print("Model: all-MiniLM-L6-v2 on GTX 1080")
    print("=" * 60)
    print()

    bench = TritonBenchmark()
    bench.warmup()
    print()

    # Latency by batch size
    print("=" * 60)
    print("LATENCY BY BATCH SIZE")
    print("=" * 60)
    print(f"{'Batch':<8} {'Mean':>10} {'P95':>10} {'P99':>10} {'Throughput':>15}")
    print(f"{'Size':<8} {'(ms)':>10} {'(ms)':>10} {'(ms)':>10} {'(sent/sec)':>15}")
    print("-" * 60)

    for batch_size in [1, 4, 8, 16, 32]:
        result = bench.benchmark_latency(batch_size, num_iterations=100)
        print(f"{result['batch_size']:<8} {result['mean_ms']:>10.2f} {result['p95_ms']:>10.2f} {result['p99_ms']:>10.2f} {result['throughput']:>15.1f}")

    print()

    # Concurrent throughput
    print("=" * 60)
    print("CONCURRENT THROUGHPUT")
    print("=" * 60)
    print(f"{'Workers':<10} {'Requests':>10} {'Time':>10} {'Throughput':>15} {'P95 Lat':>12}")
    print(f"{'':10} {'':>10} {'(sec)':>10} {'(sent/sec)':>15} {'(ms)':>12}")
    print("-" * 60)

    for workers in [1, 2, 4, 8]:
        result = bench.benchmark_concurrent(num_workers=workers, requests_per_worker=50)
        print(f"{result['workers']:<10} {result['total_requests']:>10} {result['total_time_sec']:>10.2f} {result['throughput']:>15.1f} {result['p95_latency_ms']:>12.2f}")

    print()
    print("=" * 60)
    print("Benchmark complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

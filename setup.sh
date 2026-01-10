#!/bin/bash
# Setup script for Triton Semantic Search with all-MiniLM-L6-v2
# Uses TensorRT optimization on GTX 1080 (GPU 1)

set -e

MODEL_DIR="models/all-minilm-l6-v2/1"
TRT_CACHE_DIR="models/all-minilm-l6-v2/trt_cache"

echo "=== Triton Semantic Search Setup ==="
echo "Target GPU: GTX 1080 (index 1)"
echo ""

# Create directories
mkdir -p "$MODEL_DIR"
mkdir -p "$TRT_CACHE_DIR"

# Check if model already exists
if [ -f "$MODEL_DIR/model.onnx" ]; then
    echo "[OK] Model already downloaded"
else
    echo "[*] Downloading all-MiniLM-L6-v2 ONNX model..."

    wget -q --show-progress -O "$MODEL_DIR/model.onnx" \
        "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"

    echo "[OK] Model downloaded"
fi

# Show model info
SIZE=$(du -h "$MODEL_DIR/model.onnx" | cut -f1)
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Model: all-MiniLM-L6-v2 ($SIZE)"
echo "Backend: ONNX Runtime + TensorRT"
echo "GPU: GTX 1080 (index 1)"
echo ""
echo "To start Triton server:"
echo "  docker-compose up"
echo ""
echo "NOTE: First startup will take a few minutes while TensorRT"
echo "      builds optimized engines. Subsequent starts use cached engines."

FROM nvcr.io/nvidia/tritonserver:24.01-py3

# Install CUDA 11.8 runtime libraries for pip onnxruntime-gpu compatibility
# (pip ORT 1.16.3 is built against CUDA 11.8, but this container ships CUDA 12.3)
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends \
        libcufft-11-8 \
        libcublas-11-8 \
        cuda-cudart-11-8 && \
    rm -rf /var/lib/apt/lists/* && \
    echo '/usr/local/cuda-11.8/targets/x86_64-linux/lib' > /etc/ld.so.conf.d/cuda-11-8.conf && \
    ldconfig

# Install Python dependencies for the bge_embeddings Python backend model
RUN pip install --no-cache-dir \
        onnxruntime-gpu==1.16.3 \
        transformers>=4.35.0

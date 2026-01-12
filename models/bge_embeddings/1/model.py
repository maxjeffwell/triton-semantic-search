"""
BGE Embeddings Model for Triton Python Backend

Loads the ONNX-exported bge-base-en-v1.5 model and provides
sentence embeddings via Triton's Python backend.

Input: text strings
Output: 768-dim normalized embeddings
"""

import os
import json
import numpy as np
from pathlib import Path

import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    """Triton Python backend for BGE embeddings"""

    def initialize(self, args):
        """Load ONNX model and tokenizer"""
        self.logger = pb_utils.Logger

        # Get model directory
        # model_repository is already the model's directory (e.g., /models/bge_embeddings)
        model_config = json.loads(args['model_config'])
        model_dir = Path(args['model_repository'])
        version = args['model_version']
        model_path = model_dir / version

        self.logger.log_info(f"Loading BGE embeddings from {model_path}")

        # Import dependencies
        try:
            import onnxruntime as ort
            from transformers import AutoTokenizer
        except ImportError as e:
            self.logger.log_error(f"Missing dependency: {e}")
            raise

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.logger.log_info("Tokenizer loaded")

        # Detect execution providers
        available_providers = ort.get_available_providers()
        self.logger.log_info(f"Available ONNX providers: {available_providers}")

        # TensorRT cache directory for compiled engines
        trt_cache_path = str(model_path / "trt_cache")
        os.makedirs(trt_cache_path, exist_ok=True)

        # GPU device ID (GTX 1080 is device 0 inside container since we use device_ids: ['1'])
        gpu_device_id = 0
        self.logger.log_info(f"Using GPU device: {gpu_device_id}")

        # Provider priority: TensorRT > CUDA > CPU
        # Configure with optimization options for best performance
        providers = []

        if 'TensorrtExecutionProvider' in available_providers:
            trt_options = {
                'device_id': gpu_device_id,
                # FP16 precision - ~2x speedup on GTX 1080 (Pascal supports FP16)
                'trt_fp16_enable': True,
                # Workspace memory for TensorRT optimization algorithms (2GB)
                'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                # Cache compiled TensorRT engines to avoid rebuild on restart
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': trt_cache_path,
                # Optimization level: 3 balances build time vs inference speed
                'trt_builder_optimization_level': 3,
                # Timing cache for faster engine building
                'trt_timing_cache_enable': True,
                'trt_timing_cache_path': trt_cache_path,
            }
            providers.append(('TensorrtExecutionProvider', trt_options))
            self.logger.log_info(f"TensorRT enabled with FP16, cache at {trt_cache_path}")

        if 'CUDAExecutionProvider' in available_providers:
            cuda_options = {
                'device_id': gpu_device_id,
                'arena_extend_strategy': 'kSameAsRequested',
                'cudnn_conv_algo_search': 'EXHAUSTIVE',
            }
            providers.append(('CUDAExecutionProvider', cuda_options))
            self.logger.log_info("CUDA execution provider configured")

        providers.append('CPUExecutionProvider')
        self.logger.log_info(f"Provider order: {[p[0] if isinstance(p, tuple) else p for p in providers]}")

        # Load ONNX model
        onnx_path = model_path / "model.onnx"
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.logger.log_info(f"ONNX model loaded: {onnx_path}")

        # Get model info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.logger.log_info(f"Model inputs: {self.input_names}")
        self.logger.log_info(f"Model outputs: {self.output_names}")

    def execute(self, requests):
        """Process embedding requests"""
        responses = []

        for request in requests:
            try:
                # Get input text
                text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
                texts = text_tensor.as_numpy()

                # Decode byte strings to text
                decoded_texts = []
                for text in texts.flatten():
                    if isinstance(text, bytes):
                        decoded_texts.append(text.decode('utf-8'))
                    else:
                        decoded_texts.append(str(text))

                # Tokenize
                inputs = self.tokenizer(
                    decoded_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="np"
                )

                # Prepare ONNX inputs
                onnx_inputs = {
                    "input_ids": inputs["input_ids"].astype(np.int64),
                    "attention_mask": inputs["attention_mask"].astype(np.int64)
                }

                # Run inference
                outputs = self.session.run(None, onnx_inputs)

                # Get sentence embeddings (already pooled by the model)
                # sentence_embedding is the second output (index 1)
                if 'sentence_embedding' in self.output_names:
                    idx = self.output_names.index('sentence_embedding')
                    embeddings = outputs[idx]
                else:
                    # Fallback: use token embeddings with mean pooling
                    token_embeddings = outputs[0]
                    attention_mask = inputs["attention_mask"]
                    mask_expanded = np.expand_dims(attention_mask, -1)
                    sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
                    sum_mask = np.clip(mask_expanded.sum(axis=1), 1e-9, None)
                    embeddings = sum_embeddings / sum_mask

                # Normalize embeddings (BGE uses L2 normalization)
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.clip(norms, 1e-9, None)

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "embeddings",
                    embeddings.astype(np.float32)
                )

                response = pb_utils.InferenceResponse(output_tensors=[output_tensor])
                responses.append(response)

            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                error = pb_utils.TritonError(f"Embedding failed: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=error))

        return responses

    def finalize(self):
        """Cleanup"""
        self.logger.log_info("BGE embeddings model finalized")
        del self.session
        del self.tokenizer

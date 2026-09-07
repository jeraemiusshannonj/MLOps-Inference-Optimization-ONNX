# PyTorch to ONNX FP16 Inference Optimization

A minimal benchmark testing model inference acceleration on NVIDIA GPUs using ONNX graph tracing and half-precision (FP16) casting.

Standard FP32 PyTorch models are often bottlenecked by VRAM bandwidth during real-time serving. Converting the computational graph to an ONNX intermediate representation and executing in FP16 precision leverages hardware Tensor Cores, effectively doubling throughput without modifying the underlying model architecture.

## Benchmark Results

Tested on an **NVIDIA T4 GPU** (Batch Size: 32, 200 execution passes after 50 warmup runs):

| Execution Mode | Precision | Batch Size | Latency (ms) | Throughput (QPS) | Latency Reduction | Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PyTorch Baseline** | FP32 | 32 | 35.79 ms | 894.1 QPS | Baseline | 1.00x |
| **Quantized / Tensor Core** | **FP16** | **32** | **17.24 ms** | **1,855.7 QPS** | **-51.8%** | **2.08x** |

## Key Technical Steps

1. **Graph Tracing:** Exported a pre-trained `torchvision` ResNet-18 model to an ONNX intermediate representation (`resnet18.onnx`).
2. **Half-Precision Casting:** Converted model parameters and input tensors from 32-bit floating point to 16-bit half precision (`.half()`), halving memory bandwidth demands.
3. **CUDA Measurement:** Profiled execution timing directly on the GPU stream using `torch.cuda.Event` to eliminate CPU-GPU synchronization overhead.

## Repository Files

* `inference_benchmark.ipynb` - Complete Google Colab notebook containing the export and profiling pipeline.
* `resnet18.onnx` - Exported intermediate representation file.

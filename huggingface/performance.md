# Moondream2 HF Inference Performance on Jetson Orin Nano

## Test Environment

| Item | Specification |
|------|--------------|
| Hardware | NVIDIA Jetson Orin Nano (8 GB) |
| JetPack / L4T | 6.x / 36.4.7 |
| CUDA | 12.6 |
| GPU Driver | 540.4.0 |
| GPU | Orin (nvgpu), 1024 CUDA cores, 64 Tensor cores |
| CPU | 6-core ARM Cortex-A78AE |
| RAM | 8 GB LPDDR5 (CPU/GPU unified) |
| Storage | 64 GB SD card (rootfs) |
| OS | Ubuntu 22.04.5 LTS (aarch64) |

## Software Stack

| Component | Version | Notes |
|-----------|---------|-------|
| Python | 3.10.12 | System |
| PyTorch | 2.5.0a0 (NVIDIA Jetson build) | `torch-2.5.0a0+872d972e41.nv24.08` |
| torchvision | 0.20.1 | `--no-deps`, NMS patched |
| transformers | 4.40.0 | Pinned for moondream2 compat |
| accelerate | 1.14.0 | `--no-deps` |
| Model | vikhyatk/moondream2 | revision `2024-04-02` |
| Precision | FP16 | `torch_dtype=torch.float16` |

## Model Architecture & Parameter Count

Model: **vikhyatk/moondream2**, revision `2024-04-02`

Architecture: SigLIP Vision Transformer → Vision Projection → Phi-2 Text Decoder

| Sub-model | Component | Parameters |
|-----------|-----------|-------------|
| **Vision Encoder** | SigLIP ViT (27 layers, dim=1152) | 412,987,248 |
| ├─ Patch embedding | Linear(588→1152) | 678,528 |
| ├─ Position embedding | 729×1152 (27×27 patches) | 839,808 |
| ├─ 27× VitBlock | Attn(QKV+Proj) + MLP(4304) + 2×LN | 411,466,608 |
| └─ Final LayerNorm | | 2,304 |
| **Vision Projection** | MLP(1152→8192→2048) | 26,224,640 |
| **Text Decoder** | Phi-2 based LLM (24 layers, dim=2048) | 1,313,417,216 |
| ├─ Token Embedding | 51200×2048 | 104,857,600 |
| ├─ 24× DecoderLayer | Parallel Attn+MLP, 1×LN shared | 1,208,500,224 |
| ├─ Final LayerNorm | | 4,096 |
| ├─ LM head LayerNorm | | 4,096 |
| └─ LM head bias | Weight tied to embedding | 51,200 |
| | | |
| **Total** | | **1,752,629,104 (~1.75B)** |

### Architecture Details

**Vision Encoder (SigLIP ViT):**
- 27 transformer blocks, embedding dimension 1152
- MLP hidden dimension 4304, 16 attention heads
- Input: 378×378 RGB image → 27×27 patches (14×14 px each)
- Patch feature dimension: 3×14×14 = 588 → projected to 1152
- Output: 729 tokens × 1152 dim

**Vision Projection:**
- 2-layer MLP: 1152 → 8192 (GELU) → 2048
- Maps vision tokens to text decoder's hidden dimension

**Text Decoder (Phi-2 based LLM):**
- 24 transformer layers, hidden size 2048, intermediate 8192
- 32 attention heads, head dimension 64, partial RoPE (factor 0.5)
- Parallel attention + MLP architecture (single input LayerNorm shared)
- Combined QKV projection (not separate Q, K, V)
- Vocabulary size: 51200, LM head weight tied to token embedding
- Max context length: 2048 tokens

### Parameter Distribution

| Component | Parameters | % of Total | Memory @ BF16 |
|-----------|-----------|------------|---------------|
| Vision Encoder (ViT) | 413.0M | 23.6% | 0.79 GB |
| Vision Projection | 26.2M | 1.5% | 0.05 GB |
| Text Decoder (Phi-2) | 1,313.4M | 74.9% | 2.51 GB |
| **Total** | **1,752.6M** | **100%** | **3.35 GB** |

> Note: Actual safetensors file is ~3.85 GB (BF16 weights + safetensors metadata overhead).
> Runtime memory is higher (~4.36 GB) due to KV cache, activations, and CUDA context.

## Benchmark Results

### Test 1: Short Prompt

```
Prompt:    "Describe this image in one sentence."
Output:    37 tokens
```

| Metric | Value |
|--------|-------|
| Model load | 48.8 s |
| Image encode (3840×2160) | 2.9 s |
| First token latency (TTFT) | 2.6 s |
| Total inference time | 6.2 s |
| Decode-only time | 3.6 s |
| Throughput | **6.0 tok/s** |
| GPU memory peak | 4.35 GB |

### Test 2: Long Prompt

```
Prompt:    "Describe this image in detail, including the animal, its pose,
            the environment, the lighting, and the overall mood."
Output:    153 tokens
```

| Metric | Value |
|--------|-------|
| Model load | 53.4 s |
| Image encode (3840×2160) | 2.1 s |
| First token latency (TTFT) | 2.9 s |
| Total inference time | 14.7 s |
| Decode-only time | 11.8 s |
| Throughput | **10.4 tok/s** |
| GPU memory peak | 4.36 GB |

### Latency Breakdown

```
[Image encode] [Prompt/image prefill] [Token 1] [Token 2] ... [Token N]
    ~2.5 s            ~2.7 s           ~0.1 s    ~0.1 s       ~0.1 s
    |______________ TTFT ______________|
    |____________________ Total inference time ____________________|
```

- **Image encode**: ~2.5 s — SigLIP vision encoder processes the image once
- **Prefill** (TTFT − encode): ~0.2 s — prompt tokens + image embeddings fed through the decoder in one forward pass
- **Decode per token**: ~0.10 s (short prompt) / ~0.08 s (long prompt)
- Longer outputs benefit from amortized prefill: 6.0 → 10.4 tok/s

## Memory Analysis

| Component | Approximate Size |
|-----------|-----------------|
| Model weights (FP16) | ~3.8 GB |
| KV cache + activations | ~0.5 GB |
| **Peak GPU memory** | **4.36 GB** |
| System available after load | ~3.6 GB |

The 8 GB Orin Nano has ~3.6 GB remaining after model load — sufficient for the OS and other processes, but no room for another large model simultaneously.

## Comparison with Other Setups

| Setup | tok/s | Notes |
|-------|-------|-------|
| Jetson Orin Nano (HF, FP16) | 6–10 | This benchmark |
| Desktop RTX 3060 (HF, FP16) | ~30–40 | Estimated |
| Desktop RTX 4090 (HF, FP16) | ~80–100 | Estimated |
| Jetson AGX Orin 64GB (HF, FP16) | ~15–20 | Estimated (2× GPU cores) |

## Key Findings

1. **FP16 is the only viable precision** on Orin Nano — INT8/INT4 quantization is not supported in the standard HF pipeline
2. **Throughput scales with output length** — prefill overhead (~2.7s) is fixed; longer generations amortize it
3. **Peak memory is predictable** — consistently ~4.36 GB regardless of output length, as memory is dominated by model weights
4. **Image resolution has minor impact** — a 3840×2160 image encodes in ~2.5s; smaller images would be faster
5. **First token latency is acceptable** — ~2.7s for interactive use cases
6. **Model load time is significant** — ~50s to load from disk; this is a one-time cost per session

## Recommendations

1. **Keep the model loaded** across multiple queries to avoid the 50s reload penalty
2. **Resize images** to ≤1024px before inference — the vision encoder downsamples anyway, and smaller images encode faster
3. **Use batch inference** where possible — the model supports `batch_answer()` for multiple prompts on the same image
4. **For production** — consider TensorRT INT4 quantization (estimated ~1 GB RAM, ~20 tok/s), but requires 1–2 weeks of engineering

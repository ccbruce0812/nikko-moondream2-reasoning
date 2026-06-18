# Moondream2 on Jetson Orin Nano — Unified Performance Report

## Test Environment

| Item | Specification |
|---|---|
| Hardware | NVIDIA Jetson Orin Nano (8 GB) |
| System | JetPack 6.x / L4T 36.4.7 |
| CUDA | 12.6 |
| GPU Driver | 540.4.0 |
| GPU | Orin (nvgpu), 1024 CUDA cores, 64 Tensor cores |
| CPU | 6-core ARM Cortex-A78AE |
| RAM | 8 GB LPDDR5 (CPU/GPU unified) |
| Storage | 64 GB SD card (rootfs) |
| OS | Ubuntu 22.04.5 LTS (aarch64) |
| Python | 3.10.12 |

---

## Model Architecture

Model: **vikhyatk/moondream2**, revision `2024-04-02`  
Architecture: SigLIP Vision Transformer → Vision Projection → Phi-2 Text Decoder

### Parameter Count

| Component | Sub-component | Parameters | % of Total |
|---|---|---|---|
| **Vision Encoder** | SigLIP ViT (27 layers, dim=1152) | 413.0M | 23.6% |
| ├ | Patch embedding | 0.68M | |
| ├ | Position embedding (729×1152) | 0.84M | |
| └ | 27× VitBlock (Attn + MLP 4304 + 2×LN) | 411.5M | |
| **Vision Projection** | MLP (1152→8192→2048) | 26.2M | 1.5% |
| **Text Decoder** | Phi-2 based LLM (24 layers, dim=2048) | 1,313.4M | 74.9% |
| ├ | Token Embedding (51200×2048) | 104.9M | |
| ├ | 24× DecoderLayer (Parallel Attn+MLP) | 1,208.5M | |
| └ | LM head (weight-tied to embedding) | 0.05M | |
| **Total** | | **1,752.6M (~1.75B)** | **100%** |

### Architecture Details

**Vision Encoder (SigLIP ViT):**
- 27 transformer blocks, embedding dimension 1152
- MLP hidden dimension 4304, 16 attention heads
- Input: 378×378 RGB image → 27×27 patches (14×14 px each)
- Patch feature dimension: 3×14×14 = 588 → projected to 1152
- Output: 729 tokens × 1152 dim

**Text Decoder (Phi-2 based):**
- 24 transformer layers, hidden size 2048, intermediate 8192
- 32 attention heads, head dim 64, partial RoPE (factor 0.5)
- Parallel attention + MLP architecture (single input LayerNorm shared)
- Combined QKV projection (not separate Q, K, V)
- Vocabulary: 51200, LM head weight tied to token embedding
- Max context: 2048 tokens

---

## Benchmarks

### GGUF q4_k (llama-cpp-python CUDA)

**Software Stack:**
| Component | Version | Notes |
|---|---|---|
| llama-cpp-python | 0.3.30 | CUDA enabled (CMAKE_ARGS="-DGGML_CUDA=on") |
| Chat handler | `MoondreamChatHandler` | Required — Llava15ChatHandler produces garbled output |
| Model repo | `salivosa/moondream2-gguf` | |
| Vision encoder | `moondream2-mmproj-f16.gguf` (868 MB) | Always FP16 |
| Text model | `moondream2-q4_k.gguf` (877 MB) | 4-bit quantized |
| GPU offload | `n_gpu_layers=-1` | All layers on GPU |
| Context size | 2048 | |

**Results:**
| Metric | Value |
|---|---|
| **Model load** | **1.41 s** |
| Model files on disk | 1,745 MB |
| **GPU memory footprint** | **~2.0 GB** |
| First token latency (TTFT) | 3.57 s |
| — Image encode (CLIP) | ~0.80 s |
| — Image decode (text model) | ~0.75 s |
| — Prompt prefill | ~2.02 s |
| Total inference (65 tokens) | 5.52 s |
| Throughput (incl. TTFT) | 11.8 tok/s |
| **Generation speed (excl. TTFT)** | **32.8 tok/s** |
| Process RSS | 259 MB (model is memory-mapped) |

> **TTFT breakdown**: The 3.57s first-token delay comprises ~0.8s CLIP image encoding, ~0.75s text-model image decoding (2 batches: 512+217 tokens), and ~2s prompt prefill. Pure text generation after the first token runs at 32.8 tok/s.

---

### GGUF f16 (llama-cpp-python CUDA) — estimated from prior runs

| Metric | GGUF f16 | GGUF q4_k | Ratio |
|---|---|---|---|
| Load time | ~2 s | 1.41 s | — |
| **Generation speed** | **~17 tok/s** | **32.8 tok/s** | **1.9×** |
| GPU memory | ~4.0 GB | ~2.0 GB | **2×** |
| Model file size | ~3.5 GB | 877 MB | 4× |

> q4_k is roughly **2× faster** and uses **half the memory** of f16, while the model file is 4× smaller. The larger speed gap vs file-size ratio comes from reduced memory bandwidth pressure — the Orin Nano's LPDDR5 is the primary bottleneck for LLM inference.

---

### HuggingFace Transformers FP16

**Software Stack:**
| Component | Version | Notes |
|---|---|---|
| PyTorch | 2.5.0a0 (NVIDIA Jetson) | `torch-2.5.0a0+872d972e41.nv24.08` |
| torchvision | 0.20.1 | `--no-deps`, NMS patched |
| transformers | 4.40.0 | Pinned for moondream2 compat |
| accelerate | 1.14.0 | `--no-deps` |
| Model | `vikhyatk/moondream2` | revision `2024-04-02` |
| Precision | `torch.float16` | FP16 throughout |

**Test 1 — Short prompt** ("Describe this image in one sentence.", 37 tokens):
| Metric | Value |
|---|---|
| Model load | 48.8 s |
| Image encode (3840×2160) | 2.9 s |
| First token latency (TTFT) | 2.6 s |
| Total inference | 6.2 s |
| Decode-only time | 3.6 s |
| **Throughput** | **6.0 tok/s** |
| GPU memory peak | 4.35 GB |

**Test 2 — Long prompt** ("Describe this image in detail...", 153 tokens):
| Metric | Value |
|---|---|
| Model load | 53.4 s |
| Image encode (3840×2160) | 2.1 s |
| First token latency (TTFT) | 2.9 s |
| Total inference | 14.7 s |
| Decode-only time | 11.8 s |
| **Throughput** | **10.4 tok/s** |
| GPU memory peak | 4.36 GB |

**Latency breakdown:**
```
[Image encode] [Prompt/image prefill] [Token 1] [Token 2] ... [Token N]
    ~2.5 s            ~2.7 s           ~0.1 s    ~0.1 s       ~0.1 s
    |______________ TTFT ______________|
    |____________________ Total inference time ____________________|
```

---

## Full Comparison Matrix

| Metric | HF FP16 | GGUF f16 | GGUF q4_k |
|---|---|---|---|
| **Generation speed** | 6–10 tok/s | ~17 tok/s | **32.8 tok/s** |
| **Throughput (incl. TTFT)** | 6–10 tok/s | — | 11.8 tok/s |
| **GPU memory** | ~4.36 GB | ~4.0 GB | **~2.0 GB** |
| **Model load time** | ~50 s | ~2 s | **1.4 s** |
| **File size (total)** | ~3.85 GB | ~4.4 GB | **1.75 GB** |
| **First token (TTFT)** | 2.6–2.9 s | — | 3.57 s |
| **Per-token decode** | ~100 ms | ~60 ms | **~30 ms** |
| **Precision loss** | None (FP16) | None (FP16) | Minor (4-bit) |
| **Install complexity** | Medium | Medium | Medium |
| **Disk usage** | ~3.85 GB | ~4.4 GB | ~1.75 GB |
| **System RAM remaining** | ~3.6 GB | ~4.0 GB | **~6.0 GB** |

---

## Memory Analysis

### HF FP16
| Component | Approximate Size |
|---|---|
| Model weights (FP16) | ~3.8 GB |
| KV cache + activations | ~0.5 GB |
| **Peak GPU memory** | **4.36 GB** |
| System available after load | ~3.6 GB |

### GGUF q4_k
| Component | Approximate Size |
|---|---|
| Vision encoder (mmproj, FP16) | ~0.85 GB |
| Text model (q4_k) | ~0.85 GB |
| KV cache + runtime | ~0.3 GB |
| **Total GPU memory** | **~2.0 GB** |
| System available after load | ~6.0 GB |

> The Orin Nano's 8 GB unified memory is the primary constraint. GGUF q4_k leaves ~6 GB free — enough for the OS, camera pipeline, and other processes. HF FP16 leaves only ~3.6 GB, which risks OOM when combined with a desktop environment or camera capture.

---

## Performance Over Time (per-token latency)

```
Token-by-token generation speed comparison:

HF FP16:    ████████████████████████████████████████████  ~100 ms/tok  (10 tok/s)
GGUF f16:   ██████████████████████                        ~60 ms/tok   (17 tok/s)
GGUF q4_k:  ██████████                                    ~30 ms/tok   (33 tok/s)

0           25          50          75          100 ms
```

---

## Comparison with Other Setups

| Setup | tok/s | Notes |
|---|---|---|
| Jetson Orin Nano (GGUF q4_k) | **32.8** | This report — recommended |
| Jetson Orin Nano (GGUF f16) | ~17 | 2× memory of q4_k |
| Jetson Orin Nano (HF, FP16) | 6–10 | Baseline, 50s load |
| Desktop RTX 3060 (HF, FP16) | ~30–40 | Estimated |
| Desktop RTX 4090 (HF, FP16) | ~80–100 | Estimated |
| Jetson AGX Orin 64GB (HF, FP16) | ~15–20 | Estimated (2× GPU cores) |

---

## Key Findings

1. **GGUF q4_k is the clear winner** — 32.8 tok/s generation, ~2 GB memory, 1.4s load time. No other path comes close on Jetson Orin Nano.

2. **GGUF f16 is viable but inferior** — ~17 tok/s uses 2× memory for negligible quality gain over q4_k. Only use f16 if you absolutely need lossless weights.

3. **HF FP16 is functional but slow** — 6–10 tok/s with 50s load makes it impractical for interactive use. Only suitable for one-shot validation or when transformers ecosystem integration is required.

4. **Memory-mapped I/O is transformational** — llama.cpp's mmap reduces model load from 50s (HF) to 1.4s (GGUF). This matters for any workflow that loads/unloads the model.

5. **Chat handler matters critically** — `MoondreamChatHandler` is required for moondream2 GGUF. Using `Llava15ChatHandler` produces zero-token or garbled output due to incompatible chat template (`<image>\n\nQuestion: ...\n\nAnswer:` vs LLaVA format).

6. **kestrel SDK is not viable on Jetson** — Official moondream SDK requires CUDA 13 (closed-source `kestrel_kernels`), but Jetson is locked to CUDA 12.6. No workaround exists.

7. **TensorRT INT4 is the theoretical ceiling** — Estimated ~20 tok/s at ~1 GB, but requires 1–2 weeks of manual ONNX export and engine building. Only worth it for production deployments needing maximum throughput.

8. **Memory bandwidth is the bottleneck** — The Orin Nano's LPDDR5 unified memory is the limiting factor. q4_k's 2× speedup over f16 comes primarily from reduced weight transfer, not compute savings.

---

## Recommendations

| Use Case | Recommended Path |
|---|---|
| Interactive chat with images | **GGUF q4_k** |
| Batch processing (many images) | **GGUF q4_k** |
| One-time experiment / validation | HF FP16 (no extra setup needed) |
| Maximum quality (lossless weights) | GGUF f16 |
| Production with >20 tok/s needed | TensorRT INT4 (1–2 week engineering) |
| Camera + real-time VLM | GGUF q4_k (leave ~6 GB for camera pipeline) |

### Best Practices

1. **Pre-load the model** — GGUF mmap is fast enough (<2s) that you can load on-demand per request. HF requires keeping the model resident (~50s reload penalty).

2. **Resize images** — The vision encoder downsamples to 378×378 regardless. Pre-resizing large images saves encode time without quality loss.

3. **Use drop_caches before GGUF inference** — If other processes have consumed memory, `echo 3 > /proc/sys/vm/drop_caches` helps avoid OOM.

4. **Monitor memory** — Use `free -h` and `nvidia-smi` (or `tegrastats`) to track unified memory usage. Keep >500 MB headroom for the OS.

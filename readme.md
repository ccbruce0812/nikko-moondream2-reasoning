# Moondream2 on Jetson Orin Nano — Unified Deployment Guide

## Overview

Two proven inference paths for running moondream2 (1.75B VLM) on NVIDIA Jetson Orin Nano:

| | HuggingFace Transformers | llama.cpp GGUF |
|---|---|---|
| **Model format** | HF safetensors (FP16) | GGUF q4_k quantized |
| **Speed** | 6–10 tok/s | **32.8 tok/s** |
| **Memory** | ~4.36 GB | **~2.0 GB** |
| **Load time** | ~50 s | **~1.4 s** (mmap) |
| **Best for** | One-shot validation, prototyping | Frequent inference, production |

**Recommendation: GGUF q4_k** — 3–5× faster, 2× less memory, near-instant load.

---

## Prerequisites

| Item | Spec |
|---|---|
| Hardware | NVIDIA Jetson Orin Nano (8 GB) |
| System | JetPack 6.x (L4T 36.4), CUDA 12.6 |
| Python | 3.10.12 |
| Storage | >6 GB free (HF) / >2 GB free (GGUF) |

### SSH & SCP setup (for remote deployment)

```bash
ssh-keygen -t ed25519
ssh-copy-id brucehsu@192.168.1.119
```

### Sudo without password

```bash
sudo visudo -f /etc/sudoers.d/brucehsu
# Add: brucehsu  ALL=(ALL:ALL) NOPASSWD: ALL
```

---

## Path A: llama.cpp GGUF (Recommended)

### 1. Create venv and install dependencies

```bash
cd ~/project
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install "numpy<2"
```

### 2. Install llama-cpp-python with CUDA support

```bash
# Install build tools first
pip install scikit-build-core cmake ninja

# Set CUDA path (Jetson default: /usr/local/cuda)
export PATH=/usr/local/cuda/bin:$PATH
export CUDACXX=/usr/local/cuda/bin/nvcc

# Install with CUDA acceleration
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

### 3. Verify CUDA

```bash
python cuda_check_script.py
# Should show: ✅ PyTorch CUDA + ✅ llama-cpp CUDA
```

### 4. Download model files

```bash
pip install huggingface_hub

# Option A: Fixed download from salivosa/moondream2-gguf (q4_k)
python gguf/download_gguf.py

# Option B: Smart search (auto-finds best q4/int8 repo, avoids f16 OOM)
python gguf/moondream_gguf_download_q4.py
```

Files downloaded to `~/project/gguf/moondream2/`:
- `moondream2-mmproj-f16.gguf` (~868 MB) — vision encoder (SigLIP, always FP16)
- `moondream2-q4_k.gguf` (~877 MB) — text model (4-bit quantized)

### 5. Run inference

```bash
python gguf/moondream2_gguf.py test.jpg "Describe this image in detail."
```

### 6. Memory cleanup before inference (if OOM)

```bash
sudo sync && sudo sysctl -w vm.drop_caches=3
sudo sysctl -w vm.compact_memory=1
```

---

## Path B: HuggingFace Transformers

### 1. System packages + libcusparseLt + venv

```bash
sudo apt-get update -y && sudo apt-get upgrade -y
sudo apt-get install -y python3-pip python3-venv curl wget

# libcusparseLt MUST be installed BEFORE PyTorch
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb
sudo dpkg -i libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb
echo "/usr/lib/aarch64-linux-gnu/libcusparseLt/12" | sudo tee /etc/ld.so.conf.d/cusparselt.conf
sudo ldconfig

cd ~/project
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### 2. Install Jetson PyTorch

```bash
# DO NOT use pip install torch — must use NVIDIA's Jetson wheel
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

python3 -c "import torch; assert torch.cuda.is_available(); print('CUDA OK')"
```

### 3. Install base packages

```bash
pip install numpy==1.26.4
pip install einops Pillow
pip install --no-deps accelerate       # avoid pulling generic PyTorch
pip install psutil pyyaml huggingface-hub safetensors
```

### 4. Install torchvision + NMS patch

```bash
pip install --no-deps torchvision==0.20.1

# Patch _meta_registrations.py: wraps nms register_fake in try/except
# (the C++ nms operator doesn't exist in Jetson's torchvision build)
cp huggingface/_meta_registrations.py venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py
```

### 5. Install transformers (pinned version)

```bash
pip install transformers==4.40.0
# DO NOT use transformers >= 5.x — incompatible with torch 2.5.0a0
```

### 6. Download model & run inference

```bash
python huggingface/moondream2.py test.jpg "Describe this image in detail."
```

---

## Troubleshooting

| Problem | Cause | Solution |
|---|---|---|
| `libcusparseLt.so.0: cannot open` | CUDA 12.6 lacks cuSPARSELt | Install `libcusparselt0-cuda-12` deb (Path B Step 1) |
| `operator torchvision::nms does not exist` | C++ extension mismatch | Apply NMS patch (Path B Step 4) |
| `module 'torch' has no attribute 'float8_e8m0fnu'` | transformers 5.x + torch 2.5 | Pin `transformers==4.40.0` |
| GGUF produces 0-token / garbled output | Wrong chat handler | Use `MoondreamChatHandler`, NOT `Llava15ChatHandler` |
| OOM during GGUF inference | Not enough free memory | Reboot + drop_caches + compact_memory (Path A Step 6) |
| `llama-cpp-python` install fails | Missing build tools | Install cmake, ninja, scikit-build-core first |
| `pip install` permission error | pip cache on external drive | Set `TMPDIR=~/project/tmp` or remove symlinked cache |

---

## File Reference

### `gguf/`
| File | Purpose |
|---|---|
| `README.md` | CUDA verification + install + OOM troubleshooting |
| `download_gguf.py` | Fixed download from `salivosa/moondream2-gguf` (q4_k) |
| `moondream_gguf_download_q4.py` | Smart search — prioritizes q4/int8, excludes f16 |
| `moondream_gguf_download_f16.py` | Smart search — any quantization, Orin-optimized |
| `moondream2_gguf.py` | Inference with full performance report (streaming) |
| `test_moondream_display.py` | Camera capture + VLM inference |
| `cuda_check_script.py` | PyTorch + llama-cpp CUDA detection |

### `huggingface/`
| File | Purpose |
|---|---|
| `readme.md` | Original 8-step deployment guide |
| `install.sh` | Self-contained one-shot installation script |
| `moondream2.py` | Inference with performance benchmarking |
| `performance.md` | HF FP16 detailed performance report |
| `comparison.md` | Four-way comparison (kestrel/HF/compile/TensorRT) |
| `log.md` | Full deployment log |
| `test.jpg` | Test image |

---

## Key Architectural Notes

- **Model**: vikhyatk/moondream2, revision `2024-04-02`, ~1.75B params
- **Vision encoder**: SigLIP ViT (27 layers, dim=1152, 413M params)
- **Text decoder**: Phi-2 based (24 layers, dim=2048, 1.31B params)
- **Chat template** (GGUF): `<image>\n\nQuestion: ...\n\nAnswer:`
- **GGUF handler**: MUST use `MoondreamChatHandler` — `Llava15ChatHandler` produces garbled output
- **Precision**: mmproj is always FP16; text model is q4_k (GGUF) or FP16 (HF)

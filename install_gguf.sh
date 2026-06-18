#!/bin/bash
# ============================================================
# Moondream2 GGUF Deployment for Jetson Orin Nano
# Based on actual deployment at brucehsu@192.168.1.119
# Uses: llama-cpp-python CUDA + q4_k quantized GGUF
# ============================================================
set -euo pipefail

PROJECT_DIR="${HOME}/project"
GGUF_DIR="${PROJECT_DIR}/gguf"
MODEL_DIR="${GGUF_DIR}/moondream2"
VENV_DIR="${PROJECT_DIR}/venv"
MODEL_REPO="salivosa/moondream2-gguf"
MMPROJ_FILE="moondream2-mmproj-f16.gguf"
TEXT_MODEL_FILE="moondream2-q4_k.gguf"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
step()  { echo -e "\n${GREEN}=== Step $1: $2 ===${NC}"; }

# ============================================================
# Step 1: Environment check
# ============================================================
step "1" "Environment check"

echo "Hostname: $(hostname)"
echo "Arch:     $(uname -m)"
echo "HOME:     ${HOME}"
echo "Project:  ${PROJECT_DIR}"

if [ -f /sys/module/tegra_fuse ]; then
    info "Running on Jetson Tegra"
else
    warn "Not on Jetson Tegra — continuing anyway (CUDA must be available)"
fi

# Check CUDA
if ! command -v nvcc &>/dev/null; then
    if [ -d /usr/local/cuda/bin ]; then
        export PATH=/usr/local/cuda/bin:$PATH
    fi
fi
if command -v nvcc &>/dev/null; then
    info "CUDA compiler: $(nvcc --version | grep 'release' | awk '{print $5, $6}')"
else
    warn "nvcc not found — will rely on pre-built llama-cpp-python or system CUDA"
fi

# Check disk space (need ~2 GB for model + ~500 MB for venv)
AVAIL_KB=$(df --output=avail "${HOME}" | tail -1)
AVAIL_GB=$((AVAIL_KB / 1024 / 1024))
echo "Disk free: ${AVAIL_GB} GB (need ~3 GB)"
if [ "$AVAIL_GB" -lt 3 ]; then
    warn "Low disk space (<3 GB). Model download may fail."
    echo "Consider running: sudo apt-get clean"
fi

# Check memory
FREE_MB=$(free -m | awk '/^Mem:/{print $7}')
echo "Memory available: ${FREE_MB} MB (need ~3 GB for inference)"

mkdir -p "${PROJECT_DIR}" "${GGUF_DIR}" "${MODEL_DIR}"

# ============================================================
# Step 2: System packages + venv
# ============================================================
step "2" "System packages + Python venv"

sudo apt-get update -y -qq
sudo apt-get install -y -qq python3-pip python3-venv curl wget cmake build-essential

# Fresh venv
rm -rf "${VENV_DIR}"
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel

info "venv created at ${VENV_DIR}"
echo "pip: $(pip --version)"

# ============================================================
# Step 3: Install numpy (must be < 2 for llama-cpp-python compat)
# ============================================================
step "3" "Install numpy<2 + build tools"

pip install "numpy<2"

# Build tools required for llama-cpp-python compilation
pip install scikit-build-core ninja

info "Build tools ready"

# ============================================================
# Step 4: Install llama-cpp-python with CUDA
# ============================================================
step "4" "Install llama-cpp-python with CUDA acceleration"

# Set CUDA paths (Jetson default: /usr/local/cuda)
if [ -d /usr/local/cuda/bin ]; then
    export PATH=/usr/local/cuda/bin:$PATH
fi
if [ -f /usr/local/cuda/bin/nvcc ]; then
    export CUDACXX=/usr/local/cuda/bin/nvcc
elif command -v nvcc &>/dev/null; then
    export CUDACXX=$(which nvcc)
fi

echo "PATH:    ${PATH}"
echo "CUDACXX: ${CUDACXX:-<not set>}"

info "Compiling llama-cpp-python with CUDA (this takes 2-5 minutes on Orin Nano)..."

CMAKE_ARGS="-DGGML_CUDA=on" \
    pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir

# ============================================================
# Step 5: Verify CUDA support
# ============================================================
step "5" "Verify CUDA support in llama-cpp-python"

python3 << 'PYEOF'
import sys

# Check PyTorch CUDA (if torch is installed)
try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
except ImportError:
    print("PyTorch: not installed (not required for GGUF path)")

# Check llama-cpp-python CUDA
import llama_cpp
from llama_cpp import llama_cpp as llama_cpp_lib
sys_info = llama_cpp_lib.llama_print_system_info().decode("utf-8", errors="replace")

print(f"\nllama-cpp-python: {llama_cpp.__version__}")
print("llama.cpp build info:")
for line in sys_info.strip().split("\n"):
    line = line.strip()
    if line:
        print(f"  {line}")

# Verify CUDA is active
if "CUDA" in sys_info or "CUBLAS" in sys_info:
    print("\n✅ llama-cpp-python CUDA acceleration is ACTIVE")
else:
    print("\n❌ CUDA NOT detected in llama-cpp-python!")
    print("   Re-run with:")
    print("   CMAKE_ARGS=\"-DGGML_CUDA=on\" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir")
    sys.exit(1)
PYEOF

# ============================================================
# Step 6: Install model download dependencies
# ============================================================
step "6" "Install huggingface_hub for model download"

pip install huggingface_hub

# ============================================================
# Step 7: Download model files
# ============================================================
step "7" "Download moondream2 GGUF model files"

echo "Repository: ${MODEL_REPO}"
echo "Target dir: ${MODEL_DIR}"
echo ""

python3 << PYEOF
import os
import sys
from huggingface_hub import hf_hub_download

repo_id = "${MODEL_REPO}"
target_dir = "${MODEL_DIR}"
files = ["${MMPROJ_FILE}", "${TEXT_MODEL_FILE}"]

os.makedirs(target_dir, exist_ok=True)

for fname in files:
    local_path = os.path.join(target_dir, fname)
    if os.path.exists(local_path):
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ {fname} already exists ({size_mb:.1f} MB) — skipping")
        continue

    print(f"⬇  Downloading {fname} ...")
    try:
        hf_hub_download(
            repo_id=repo_id,
            filename=fname,
            local_dir=target_dir,
        )
        size_mb = os.path.getsize(local_path) / (1024 * 1024)
        print(f"✅ {fname} downloaded ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"❌ Failed to download {fname}: {e}")
        sys.exit(1)

# Verify files are real GGUF (not Git LFS pointers)
for fname in files:
    local_path = os.path.join(target_dir, fname)
    size_mb = os.path.getsize(local_path) / (1024 * 1024)
    if size_mb < 1:
        print(f"❌ {fname} is only {size_mb:.1f} MB — likely a Git LFS pointer, not real weights!")
        print("   Delete the file and re-run, or check your internet connection.")
        sys.exit(1)

print(f"\n✅ All model files ready in {target_dir}")
for fname in sorted(os.listdir(target_dir)):
    size_mb = os.path.getsize(os.path.join(target_dir, fname)) / (1024 * 1024)
    print(f"   {fname} ({size_mb:.1f} MB)")
PYEOF

# ============================================================
# Step 8: Run test inference
# ============================================================
step "8" "Run test inference"

# Create a test image if none exists
TEST_IMG="${PROJECT_DIR}/test.jpg"
if [ ! -f "${TEST_IMG}" ]; then
    warn "No test.jpg found at ${TEST_IMG}"
    info "Creating a simple test image..."

    python3 << 'PYEOF'
from PIL import Image
import os
img = Image.new("RGB", (378, 378), color=(100, 149, 237))
img.save(os.path.expanduser("~/project/test.jpg"))
print("Created ~/project/test.jpg (378x378 cornflower blue)")
PYEOF
fi

info "Loading model + running inference..."
echo ""

python3 << PYEOF
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

MODEL_DIR = os.path.expanduser("${MODEL_DIR}")
TEST_IMG = os.path.expanduser("${TEST_IMG}")

# Find model files
ggufs = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
mmproj = next((os.path.join(MODEL_DIR, f) for f in ggufs if "mmproj" in f.lower()), None)
text_model = next((os.path.join(MODEL_DIR, f) for f in ggufs if "mmproj" not in f.lower()), None)

if not mmproj or not text_model:
    print("❌ Model files not found!")
    sys.exit(1)

print(f"Vision encoder: {os.path.basename(mmproj)}")
print(f"Text model:     {os.path.basename(text_model)}")
print(f"Test image:     {TEST_IMG}")
print()

# Load model
print("Loading model (mmap, ~1-2 seconds)...", flush=True)
t0 = time.time()

chat_handler = MoondreamChatHandler(clip_model_path=mmproj, verbose=False)
llm = Llama(
    model_path=text_model,
    chat_handler=chat_handler,
    n_gpu_layers=-1,
    n_ctx=2048,
    verbose=False,
)
print(f"✅ Model loaded in {time.time() - t0:.1f}s")

# Run inference
print("\nPrompt: 'Describe this image in detail.'")
print("Answer: ", end="", flush=True)

t_start = time.time()
first_token_time = None
token_count = 0

stream = llm.create_chat_completion(
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(TEST_IMG)}"}},
            {"type": "text", "text": "Describe this image in detail."},
        ],
    }],
    max_tokens=256,
    stream=True,
)

answer = ""
for chunk in stream:
    if "choices" in chunk and chunk["choices"]:
        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if content:
            if first_token_time is None:
                first_token_time = time.time()
            answer += content
            token_count += 1
            print(content, end="", flush=True)

total_time = time.time() - t_start
ttft = (first_token_time - t_start) if first_token_time else 0

print("\n")
print("=" * 50)
print("Test Result")
print("=" * 50)
print(f"  Load time:        {time.time() - t0:.1f}s")
print(f"  First token:      {ttft:.1f}s")
print(f"  Total time:       {total_time:.1f}s")
print(f"  Tokens:           {token_count}")
if total_time > 0:
    print(f"  Throughput:       {token_count/total_time:.1f} tok/s")
if ttft > 0 and token_count > 1:
    gen_time = total_time - ttft
    print(f"  Gen speed:        {(token_count-1)/gen_time:.1f} tok/s (excl. TTFT)")
print("=" * 50)
print()
print("✅ Inference test PASSED")
PYEOF

# ============================================================
# Done
# ============================================================
echo ""
info "========================================"
info " Installation complete!"
info "========================================"
echo ""
info "Model files:   ${MODEL_DIR}"
info "  ${MMPROJ_FILE}"
info "  ${TEXT_MODEL_FILE}"
info ""
info "Usage:"
info "  cd ${PROJECT_DIR}"
info "  source venv/bin/activate"
info "  python3 ${GGUF_DIR}/moondream2_gguf.py test.jpg 'Describe this image'"
info ""
info "Expected performance (Jetson Orin Nano):"
info "  Load time:    ~1.5 s"
info "  Gen speed:    ~33 tok/s (q4_k)"
info "  Memory:       ~2.0 GB"
info ""

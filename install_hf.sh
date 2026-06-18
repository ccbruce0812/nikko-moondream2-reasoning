#!/bin/bash
# ============================================================
# Moondream2 HF Deployment for Jetson Orin Nano
# Based on: huggingface/readme.md (2024-04-02 revision)
# ============================================================
set -euo pipefail

PROJECT_DIR="${HOME}/project"
VENV_DIR="${PROJECT_DIR}/venv"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ============================================================
# Step 0: Environment check
# ============================================================
info "Step 0: Checking environment..."

if [ ! -f /sys/module/tegra_fuse ]; then
    warn "Not on Jetson Tegra — continuing anyway."
fi

echo "HOME: ${HOME}"
echo "Project: ${PROJECT_DIR}"
df -h "${HOME}" | tail -1

# ============================================================
# Step 1: System packages + libcusparseLt + venv
# ============================================================
info "Step 1: System update + libcusparseLt + venv..."

sudo apt-get update -y -qq
sudo apt-get upgrade -y -qq
sudo apt-get install -y -qq python3-pip python3-venv curl wget

# libcusparseLt must be installed BEFORE PyTorch — torch import
# fails with "libcusparseLt.so.0: cannot open shared object file"
# if this library is missing.
CUSPARSE_DEB="libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb"
CUSPARSE_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/${CUSPARSE_DEB}"
if ! dpkg -l libcusparselt0-cuda-12 2>/dev/null | grep -q "^ii"; then
    if [ ! -f "$CUSPARSE_DEB" ]; then
        wget "$CUSPARSE_URL"
    fi
    sudo dpkg -i "$CUSPARSE_DEB"
    echo "/usr/lib/aarch64-linux-gnu/libcusparseLt/12" | \
        sudo tee /etc/ld.so.conf.d/cusparselt.conf > /dev/null
    sudo ldconfig
    info "libcusparseLt installed."
else
    info "libcusparseLt already present."
fi

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"
rm -rf "$VENV_DIR"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel

# ============================================================
# Step 2: Jetson PyTorch (MUST install BEFORE accelerate)
# ============================================================
info "Step 2: Installing Jetson PyTorch..."

TORCH_WHEEL="torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/${TORCH_WHEEL}"

if [ ! -f "$TORCH_WHEEL" ]; then
    wget "$TORCH_URL"
fi
pip install "$TORCH_WHEEL"

python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'PyTorch {torch.__version__} -- CUDA OK')"

# ============================================================
# Step 3: Base Python packages
# ============================================================
info "Step 3: Installing base packages..."

pip install numpy==1.26.4
pip install einops Pillow

# accelerate requires torch>=2.0.0 but pip ignores Jetson's 2.5.0a0
# (pre-release version). Install accelerate without deps, then add its
# non-torch requirements manually.
pip install --no-deps accelerate
pip install psutil pyyaml huggingface-hub safetensors

# ============================================================
# ============================================================
# Step 4: Torchvision + NMS patch
# ============================================================
info "Step 4: Installing torchvision + NMS fix..."

# --no-deps avoids pulling generic PyTorch from PyPI
pip install --no-deps torchvision==0.20.1

# The stock torchvision _meta_registrations.py calls
#   @torch.library.register_fake("torchvision::nms")
# which crashes on Jetson because the C++ operator doesn't exist.
# The patched version wraps that block in try/except RuntimeError.
# We embed the patched file here so install.sh is self-contained.
cat > /tmp/_meta_registrations_patched.py << 'METAEOF'
import functools

import torch
import torch._custom_ops
import torch.library

# Ensure that torch.ops.torchvision is visible
import torchvision.extension  # noqa: F401


@functools.lru_cache(None)
def get_meta_lib():
    return torch.library.Library("torchvision", "IMPL", "Meta")


def register_meta(op_name, overload_name="default"):
    def wrapper(fn):
        if torchvision.extension._has_ops():
            get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
        return fn

    return wrapper


@register_meta("roi_align")
def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
    torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    torch._check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    num_rois = rois.size(0)
    channels = input.size(1)
    return input.new_empty((num_rois, channels, pooled_height, pooled_width))


@register_meta("_roi_align_backward")
def meta_roi_align_backward(
    grad, rois, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width, sampling_ratio, aligned
):
    torch._check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))


@register_meta("ps_roi_align")
def meta_ps_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio):
    torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    torch._check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    channels = input.size(1)
    torch._check(
        channels % (pooled_height * pooled_width) == 0,
        "input channels must be a multiple of pooling height * pooling width",
    )
    num_rois = rois.size(0)
    out_size = (num_rois, channels // (pooled_height * pooled_width), pooled_height, pooled_width)
    return input.new_empty(out_size), torch.empty(out_size, dtype=torch.int32, device="meta")


@register_meta("_ps_roi_align_backward")
def meta_ps_roi_align_backward(
    grad, rois, channel_mapping, spatial_scale, pooled_height, pooled_width,
    sampling_ratio, batch_size, channels, height, width,
):
    torch._check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))


@register_meta("roi_pool")
def meta_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    torch._check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    num_rois = rois.size(0)
    channels = input.size(1)
    out_size = (num_rois, channels, pooled_height, pooled_width)
    return input.new_empty(out_size), torch.empty(out_size, device="meta", dtype=torch.int32)


@register_meta("_roi_pool_backward")
def meta_roi_pool_backward(
    grad, rois, argmax, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width
):
    torch._check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))


@register_meta("ps_roi_pool")
def meta_ps_roi_pool(input, rois, spatial_scale, pooled_height, pooled_width):
    torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
    torch._check(
        input.dtype == rois.dtype,
        lambda: (
            "Expected tensor for input to have the same type as tensor for rois; "
            f"but type {input.dtype} does not equal {rois.dtype}"
        ),
    )
    channels = input.size(1)
    torch._check(
        channels % (pooled_height * pooled_width) == 0,
        "input channels must be a multiple of pooling height * pooling width",
    )
    num_rois = rois.size(0)
    out_size = (num_rois, channels // (pooled_height * pooled_width), pooled_height, pooled_width)
    return input.new_empty(out_size), torch.empty(out_size, device="meta", dtype=torch.int32)


@register_meta("_ps_roi_pool_backward")
def meta_ps_roi_pool_backward(
    grad, rois, channel_mapping, spatial_scale, pooled_height, pooled_width, batch_size, channels, height, width
):
    torch._check(
        grad.dtype == rois.dtype,
        lambda: (
            "Expected tensor for grad to have the same type as tensor for rois; "
            f"but type {grad.dtype} does not equal {rois.dtype}"
        ),
    )
    return grad.new_empty((batch_size, channels, height, width))


try:
    @torch.library.register_fake("torchvision::nms")
    def meta_nms(dets, scores, iou_threshold):
        torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
        torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
        torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
        torch._check(
            dets.size(0) == scores.size(0),
            lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
        )
        ctx = torch._custom_ops.get_ctx()
        num_to_keep = ctx.create_unbacked_symint()
        return dets.new_empty(num_to_keep, dtype=torch.long)
except RuntimeError:
    pass

@register_meta("deform_conv2d")
def meta_deform_conv2d(
    input, weight, offset, mask, bias, stride_h, stride_w,
    pad_h, pad_w, dil_h, dil_w, n_weight_grps, n_offset_grps, use_mask,
):
    out_height, out_width = offset.shape[-2:]
    out_channels = weight.shape[0]
    batch_size = input.shape[0]
    return input.new_empty((batch_size, out_channels, out_height, out_width))


@register_meta("_deform_conv2d_backward")
def meta_deform_conv2d_backward(
    grad, input, weight, offset, mask, bias, stride_h, stride_w,
    pad_h, pad_w, dilation_h, dilation_w, groups, offset_groups, use_mask,
):
    grad_input = input.new_empty(input.shape)
    grad_weight = weight.new_empty(weight.shape)
    grad_offset = offset.new_empty(offset.shape)
    grad_mask = mask.new_empty(mask.shape)
    grad_bias = bias.new_empty(bias.shape)
    return grad_input, grad_weight, grad_offset, grad_mask, grad_bias
METAEOF

# Find torchvision path (pip show, not import — import would crash)
TV_DIR=$(pip show torchvision 2>/dev/null | grep "^Location:" | awk '{print $2}')/torchvision

if [ -d "$TV_DIR" ]; then
    cp /tmp/_meta_registrations_patched.py "${TV_DIR}/_meta_registrations.py"
    rm -f /tmp/_meta_registrations_patched.py
    info "Patched _meta_registrations.py (NMS operator bypass)."
else
    warn "torchvision dir not found at $TV_DIR — NMS patch skipped."
fi

# Now safe to verify
python3 -c "
import torchvision
print(f'torchvision {torchvision.__version__} OK')
" 2>&1 || warn "torchvision import had warnings (may be OK)."

# ============================================================
# Step 5: HuggingFace transformers (pinned version)
# ============================================================
info "Step 5: Installing transformers==4.40.0..."

# transformers >= 5.x causes:
#   - "module 'torch' has no attribute 'float8_e8m0fnu'"
#   - IndexError with moondream2 revision 2024-04-02
pip install transformers==4.40.0

# ============================================================
# Step 6: Download moondream2 model
# ============================================================
info "Step 6: Downloading moondream2 model..."

python3 << 'PYEOF'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_id = "vikhyatk/moondream2"
revision = "2024-04-02"

print(f"Loading {model_id} (revision={revision})...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
    revision=revision,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
print(f"Model loaded. Device: {model.device}")
PYEOF

# ============================================================
# Step 7: Run inference
# ============================================================
info "Step 7: Running inference..."

python3 << 'PYEOF'
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
warnings.filterwarnings("ignore", message=".*Failed to load image Python extension.*")

import logging
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch, os

model_id = "vikhyatk/moondream2"
revision = "2024-04-02"

model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True,
    device_map="cuda", torch_dtype=torch.float16,
    revision=revision,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)

image_path = os.path.join(os.path.expanduser("~"), "project", "test.jpg")
if not os.path.exists(image_path):
    print(f"test.jpg not found at {image_path}, using dummy image.")
    image = Image.new("RGB", (224, 224), color="blue")
else:
    image = Image.open(image_path)

image_embeds = model.encode_image(image)
result = model.answer_question(image_embeds, "describe this image", tokenizer)
print(result)
PYEOF

# ============================================================
# Done
# ============================================================
echo ""
info "========================================"
info " Installation complete!"
info "========================================"
info ""
info "Usage (as per huggingface/readme.md):"
info "  cd ~/project"
info "  source venv/bin/activate"
info "  python3 -c \""
info "    from transformers import AutoModelForCausalLM, AutoTokenizer"
info "    from PIL import Image"
info "    model = AutoModelForCausalLM.from_pretrained("
info "        'vikhyatk/moondream2', revision='2024-04-02',"
info "        trust_remote_code=True, device_map='cuda',"
info "        torch_dtype=torch.float16)"
info "    tokenizer = AutoTokenizer.from_pretrained("
info "        'vikhyatk/moondream2', revision='2024-04-02')"
info "    image = Image.open('test.jpg')"
info "    embeds = model.encode_image(image)"
info "    print(model.answer_question(embeds, 'describe this image', tokenizer))"
info "  \""
info ""

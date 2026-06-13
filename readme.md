# Moondream2 在 Jetson Orin Nano 上的部署與推論流程

## 系統需求

| 項目 | 規格 |
|------|------|
| 硬體 | NVIDIA Jetson Orin Nano |
| 系統 | JetPack 6.2.1 (r36.4.7, L4T) |
| CUDA | 12.6 |
| Python | 3.10.12 |
| 記憶體 | 建議 8GB+（含 swap） |

## 前置準備

### 1. SSH 免密碼登入

```bash
ssh-keygen -t ed25519
ssh-copy-id brucehsu@192.168.1.119
```

### 2. SCP 免密碼傳輸（與 SSH 共用金鑰）

```bash
ssh brucehsu@192.168.1.119 'mkdir ~/project'
scp ~/project/* brucehsu@192.168.1.119:~/project/
```

### 3. sudo 免密碼

```bash
sudo visudo -f /etc/sudoers.d/brucehsu
# 加入：
# brucehsu  ALL=(ALL:ALL) NOPASSWD: ALL
```

### 4. SD 卡掛載與 symlink

外接 SD 卡掛載於 `/media/brucehsu/e5ff97fd-3586-4584-942a-34bd2d978d27`

需要建立 symlink 將常用目錄指向 SD 卡（避免 rootfs 爆滿）：

```bash
SD=/media/brucehsu/e5ff97fd-3586-4584-942a-34bd2d978d27
for d in .config .cache .local project; do
    rm -rf ~/"$d"
    mkdir -p "$SD/$d"
    ln -s "$SD/$d" ~/"$d"
done
```

## 步驟 1：建立 venv

```bash
sudo apt-get update -y
sudo apt-get upgrade -y
sudo apt-get install python-pip python3-venv curl wget -y
cd ~/project
rm -rf venv                    # 清除舊環境（如有）
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

## 步驟 2：安裝基礎依賴

```bash
pip install numpy==1.26.4
pip install einops Pillow accelerate
```

## 步驟 3：安裝 Jetson 專用 Torch

PyTorch 必須使用 NVIDIA 為 Jetson 編譯的版本（aarch64 + CUDA），
不可使用 PyPI 通用版。

```bash
# 下載 Jetson Torch wheel (JetPack 6.x)
wget https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl

# 安裝
pip install torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl
```

## 步驟 4：安裝 libcusparseLt（CUDA 稀疏矩陣庫）

Jetson CUDA 12.6 預設不含 libcusparseLt，需手動安裝：

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb
sudo dpkg -i libcusparselt0-cuda-12_0.8.1.1-1_arm64.deb

# 加入 ld 搜尋路徑
echo "/usr/lib/aarch64-linux-gnu/libcusparseLt/12" | sudo tee /etc/ld.so.conf.d/cusparselt.conf
sudo ldconfig
```

## 步驟 5：安裝 torchvision

由於通用版 torchvision 與 Jetson Torch 不相容，需安裝預編 binary 後手動修補：

```bash
# 以 --no-deps 安裝（避免拉入通用版 torch）
pip install --no-deps torchvision==0.20.1

# 修補 _meta_registrations.py，繞過 nms operator 不相容問題
# 找到 '@torch.library.register_fake("torchvision::nms")'
# 改成如下
#
#try:
#    @torch.library.register_fake("torchvision::nms")
#    def meta_nms(dets, scores, iou_threshold):
#        torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
#        torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")
#        torch._check(scores.dim() == 1, lambda: f"scores should be a 1d tensor, got {scores.dim()}")
#        torch._check(
#            dets.size(0) == scores.size(0),
#            lambda: f"boxes and scores should have same number of elements in dimension 0, got {dets.size(0)} and {scores.size(0)}",
#        )
#        ctx = torch._custom_ops.get_ctx()
#        num_to_keep = ctx.create_unbacked_symint()
#        return dets.new_empty(num_to_keep, dtype=torch.long)
#except RuntimeError:
#    pass
cp _meta_registrations.py  venv/lib/python3.10/site-packages/torchvision/_meta_registrations.py
```

## 步驟 6：安裝 transformers（指定版本）

必須使用與 moondream2 模型 revision `2024-04-02` 相容的版本：

```bash
pip install transformers==4.40.0
```

## 步驟 7：準備推論程式

`moondream2.py`：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch

model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    device_map="cuda",
    torch_dtype=torch.float16,
    revision="2024-04-02"
)
tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2024-04-02")

image = Image.open("test.jpg")
image_embeds = model.encode_image(image)

result = model.answer_question(image_embeds, "describe this image", tokenizer)
print(result)
```

## 步驟 8：執行推論

```bash
cd ~/project
source venv/bin/activate
python moondream2.py test.jpg
```

## 預期輸出

```
A cheetah, with its distinctive brown and black spotted coat, is perched on a rock
in a dry grassland. The cheetah's body is oriented towards the right side of the
image, and its tail is slightly raised, suggesting alertness or anticipation.
The background is a vast expanse of dry grassland, with a few trees and bushes
visible in the distance.
```

## 驗收方式

```bash
# 1. 刪除環境
rm -rf ~/project/venv

# 2. 重建
python3 -m venv ~/project/venv
source ~/project/venv/bin/activate

# 3. 依照以上步驟 2~8 重新執行

# 4. 執行推論確認成功
python moondream2-1.py test.jpg
```

## 常見問題與解法

| 問題 | 原因 | 解法 |
|------|------|------|
| `libcusparseLt.so.0: cannot open` | CUDA 12.6 未含 cuSPARSELt | 手動安裝 `libcusparselt0-cuda-12` deb |
| `operator torchvision::nms does not exist` | torchvision binary 與 Jetson Torch 不相容 | 修補 `_meta_registrations.py` 跳過 nms |
| `module 'torch' has no attribute 'float8_e8m0fnu'` | transformers 5.x 不相容 torch 2.5 | 降級至 `transformers==4.40.0` |
| `torch._C._distributed_c10d` | Jetson Torch 缺分布式模組 | 修補 `fsdp.py` 中的 `is_fsdp_managed_module` |
| `IndexError: index is out of bounds` | transformers 版本過新 | 使用 `transformers==4.40.0` |
| 磁碟空間不足 | pip 快取占用 rootfs | 設定 `TMPDIR=~/project/tmp` 指向 SD 卡 |

## 已安裝套件清單

參考 `pip freeze` 輸出（已記錄在執行環境中）。

# Check CUDA Available or NOT in Your Environment

Make sure CUDA being available for PyTorch and llama-cpp-python

```sh
# Make sure you are in the environment where you can run nikko-pyside6 UI
$ python cuda_check_script.py
=== 1. PyTorch CUDA 狀態 (YOLO 依賴) ===
CUDA 是否可用: ✅ 是
裝置名稱: Orin
CUDA 版本 (PyTorch 編譯): 12.6

=== 2. llama-cpp-python CUDA 狀態 (Moondream2 依賴) ===
ggml_cuda_init: found 1 CUDA devices (Total VRAM: 7619 MiB):
  Device 0: Orin, compute capability 8.7, VMM: yes, VRAM: 7619 MiB
✅ 成功：llama-cpp-python 已啟用 CUDA 加速！

[詳細編譯資訊]
CUDA : ARCHS = 870 | USE_GRAPHS = 1 | PEER_MAX_BATCH_SIZE = 128 | CPU : NEON = 1 | ARM_FMA = 1 | FP16_VA = 1 | DOTPROD = 1 | LLAMAFILE = 1 | OPENMP = 1 | REPACK = 1 | 
```

## Please follow the instructions in the **Troble shooting** if you firstly run llama-cpp-python in your device.

# Step-by-step to test moondream2

## Install llama-cpp-python 

```sh
# Make sure you are in the environment where you can run nikko-pyside6 UI
$ CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## Download Moondream GGUF package

### Install huggingface_hub
```sh
$ pip install huggingface_hub
```

### Download moondream2-q4_k.gguf & moondream2-mmproj-f16.gguf

```sh
$ python moondream_gguf_download_q4.py
```

## Run test_moondream_display.py

```sh
# restart nvargus-daemon for video streaming
$ sudo systemctl restart nvargus-daemon

# 強制清理 Linux 頁面快取與記憶體碎片
$ sudo sync && sudo sysctl -w vm.drop_caches=3
$ sudo sysctl -w vm.compact_memory=1

# Ensure you just running test_moondream_display.py only without other application due to memory lack.
$ python test_moondream_display.py
```


# **Troble shooting**

## 若發生 `Preparing metadata (pyproject.toml) did not run successfully`

代表沒有安裝過 llama-cpp-python

### 升級 Python 基礎打包工具

```sh
$ pip install --upgrade pip setuptools wheel
```

### 安裝 llama-cpp-python 專用的編譯後端套件

```sh
$ pip install scikit-build-core cmake ninja
```

### Rollback the version of numpy

```sh
$ pip install "numpy<2"
```

### 重新執行安裝指令

注意：在 Jetson 上，CUDA 通常預設安裝在 `/usr/local/cuda/` 底下。

```sh
$ export PATH=/usr/local/cuda/bin:$PATH
$ export CUDACXX=/usr/local/cuda/bin/nvcc
$ CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

## Run test_moondream_display.py OOM

```sh
$ Reboot
# Don't launch any other application except terminal

# 強制清理 Linux 頁面快取與記憶體碎片
$ sudo sync && sudo sysctl -w vm.drop_caches=3
$ sudo sysctl -w vm.compact_memory=1

# 增加 NVMap 的分配額度限制 (僅限本次開機有效)
$ sudo bash -c 'echo 1024 > /sys/kernel/debug/tegra_nvmap/ext_pool_size' 2>/dev/null || true

# Run test_moondream_display.py
$ python test_moondream_display.py
```

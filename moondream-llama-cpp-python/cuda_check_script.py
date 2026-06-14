import sys

def check_pytorch():
    print("=== 1. PyTorch CUDA 狀態 (YOLO 依賴) ===")
    try:
        import torch
        is_available = torch.cuda.is_available()
        print(f"CUDA 是否可用: {'✅ 是' if is_available else '❌ 否'}")
        if is_available:
            print(f"裝置名稱: {torch.cuda.get_device_name(0)}")
            print(f"CUDA 版本 (PyTorch 編譯): {torch.version.cuda}")
    except ImportError:
        print("❌ 未安裝 PyTorch")

def check_llamacpp():
    print("\n=== 2. llama-cpp-python CUDA 狀態 (Moondream2 依賴) ===")
    try:
        import llama_cpp
        # 取得 llama.cpp 底層的編譯系統資訊
        from llama_cpp import llama_cpp as llama_cpp_lib
        system_info = llama_cpp_lib.llama_print_system_info().decode('utf-8')
        
        # 尋找關鍵字 CUBLAS = 1 或 CUDA = 1 或最新版格式的 CUDA : ARCHS
        if "CUBLAS = 1" in system_info or "CUDA = 1" in system_info or "CUDA :" in system_info or "CUDA :" in system_info:
            print("✅ 成功：llama-cpp-python 已啟用 CUDA 加速！")
        else:
            print("❌ 失敗：llama-cpp-python 未啟用 CUDA (目前僅使用 CPU)")
            print("   提示: 請在終端機重新執行: CMAKE_ARGS=\"-DLLAMA_CUDA=on\" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir")
            
        print(f"\n[詳細編譯資訊]\n{system_info}")
        
    except ImportError:
        print("❌ 未安裝 llama-cpp-python")

if __name__ == "__main__":
    check_pytorch()
    check_llamacpp()
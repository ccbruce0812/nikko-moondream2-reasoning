#!/usr/bin/env python3
"""
Moondream2 GGUF (llama-cpp-python) 推論腳本 — 含完整效能指標
Usage: python3 moondream2_gguf.py <image_path> <prompt>
"""
import sys
import os
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

# ─── 模型路徑 (相對於腳本所在目錄) ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "moondream2")
N_CTX = 2048
MAX_TOKENS = 512


def find_model_files():
    """在 MODEL_DIR 中尋找 mmproj 和 text model GGUF 檔案。"""
    if not os.path.isdir(MODEL_DIR):
        return None, None

    ggufs = [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]
    mmproj = next((os.path.join(MODEL_DIR, f) for f in ggufs if "mmproj" in f.lower()), None)
    text_model = next((os.path.join(MODEL_DIR, f) for f in ggufs if "mmproj" not in f.lower()), None)
    return mmproj, text_model


def format_time(seconds):
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    return f"{seconds:.2f}s"


def main():
    # ─── CLI args ───
    if len(sys.argv) < 3:
        print("Usage: python3 moondream2_gguf.py <image_path> <prompt>")
        print("Example: python3 moondream2_gguf.py test.jpg 'Describe this image'")
        sys.exit(1)

    image_path = sys.argv[1]
    prompt = sys.argv[2]

    if not os.path.exists(image_path):
        print(f"❌ 圖片不存在: {image_path}")
        sys.exit(1)

    # ─── 尋找模型檔案 ───
    mmproj_path, model_path = find_model_files()
    if not mmproj_path:
        print(f"❌ 找不到 mmproj GGUF，請先執行 download_gguf.py")
        print(f"   搜尋目錄: {MODEL_DIR}")
        sys.exit(1)
    if not model_path:
        print(f"❌ 找不到文字模型 GGUF，請先執行 download_gguf.py")
        sys.exit(1)

    mmproj_size = os.path.getsize(mmproj_path) / (1024 * 1024)
    model_size = os.path.getsize(model_path) / (1024 * 1024)

    print("=" * 60)
    print(" Moondream2 GGUF (llama-cpp-python CUDA) 推論")
    print("=" * 60)
    print(f"  視覺編碼器: {os.path.basename(mmproj_path)} ({mmproj_size:.1f} MB)")
    print(f"  文字模型:   {os.path.basename(model_path)} ({model_size:.1f} MB)")
    print(f"  輸入圖片:   {image_path} ({os.path.getsize(image_path)/1024:.1f} KB)")
    print(f"  Prompt:     {prompt}")
    print()

    # ─── 1. 載入模型 ───
    print("[1/3] 載入模型到 GPU ...", flush=True)
    t_load_start = time.time()

    try:
        chat_handler = MoondreamChatHandler(clip_model_path=mmproj_path, verbose=False)
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1,   # 全部 offload 到 GPU
            n_ctx=N_CTX,
            verbose=False,
        )
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        sys.exit(1)

    t_load = time.time() - t_load_start
    print(f"  載入完成: {format_time(t_load)}")
    print()

    # ─── 2. 推論 (含圖片編碼) ───
    print("[2/3] 執行 VLM 推論 (圖片編碼 + 文字生成) ...", flush=True)
    print(f"  Prompt: {prompt}")
    print("  Answer: ", end="", flush=True)

    token_times = []
    first_token_time = None
    total_start = time.time()

    try:
        # 使用 streaming 來追蹤 per-token 時間
        stream = llm.create_chat_completion(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            max_tokens=MAX_TOKENS,
            stream=True,
        )

        full_answer = ""
        for chunk in stream:
            if "choices" in chunk and len(chunk["choices"]) > 0:
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    now = time.time()
                    token_times.append(now)
                    if first_token_time is None:
                        first_token_time = now
                    full_answer += content
                    print(content, end="", flush=True)

        total_time = time.time() - total_start

    except Exception as e:
        print(f"\n❌ 推論失敗: {e}")
        sys.exit(1)

    print()
    print()

    # ─── 3. 效能報告 ───
    # 估算 token 數 (粗略: 每 4 字元 ≈ 1 token，這是英文估算)
    # 更準確的方式: 使用 llama.cpp tokenize
    try:
        tokens_encoded = llm.tokenize(full_answer.encode("utf-8"), add_bos=False, special=False)
        num_tokens = len(tokens_encoded)
    except Exception:
        num_tokens = len(full_answer) // 4  # fallback 估算

    ttft = (first_token_time - total_start) if first_token_time else 0

    # 取得 llama.cpp 系統資訊 (含 CUDA 狀態)
    try:
        from llama_cpp import llama_cpp as llama_cpp_lib
        sys_info = llama_cpp_lib.llama_print_system_info().decode("utf-8", errors="replace")
    except Exception:
        sys_info = "(無法取得)"

    print("=" * 60)
    print(" Performance Report")
    print("=" * 60)
    print(f"  Model load:          {t_load:8.2f}s")
    print(f"  First token (TTFT):  {ttft:8.2f}s")
    print(f"  Total inference:     {total_time:8.2f}s")
    print(f"  Output tokens:       {num_tokens:8d}")
    if num_tokens > 0 and total_time > 0:
        print(f"  Tokens/second:       {num_tokens/total_time:8.1f} tok/s")
    if ttft > 0 and num_tokens > 1:
        gen_time = total_time - ttft
        gen_tokens = num_tokens - 1
        if gen_time > 0:
            print(f"  Gen tok/s (excl TTFT): {gen_tokens/gen_time:8.1f} tok/s")
    print(f"  Context size:        {N_CTX:8d}")
    print(f"  Total answer length: {len(full_answer):8d} chars")
    print("=" * 60)

    # ─── llama.cpp 系統資訊 (確認 CUDA) ───
    print()
    print("[llama.cpp 系統資訊]")
    for line in sys_info.strip().split("\n"):
        line = line.strip()
        if line:
            print(f"  {line}")


if __name__ == "__main__":
    main()

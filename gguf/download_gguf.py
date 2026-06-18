#!/usr/bin/env python3
"""
Moondream2 GGUF 下載器 — 從 HuggingFace Hub 下載 f16 GGUF 檔案。
目標倉庫: moondream/moondream2-gguf (官方)
"""
import os
import sys
from huggingface_hub import hf_hub_download

REPO_ID = "salivosa/moondream2-gguf"
FILES = [
    "moondream2-q4_k.gguf",
    "moondream2-mmproj-f16.gguf",
]

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_dir = os.path.join(script_dir, "moondream2")
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 60)
    print(" Moondream2 GGUF 下載器")
    print(f" 倉庫: {REPO_ID}")
    print(f" 目標目錄: {target_dir}")
    print("=" * 60)

    for fname in FILES:
        local_path = os.path.join(target_dir, fname)
        if os.path.exists(local_path):
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"\n  {fname} 已存在 ({size_mb:.1f} MB)，跳過下載")
            continue

        print(f"\n[下載] {fname} ...")
        try:
            hf_hub_download(
                repo_id=REPO_ID,
                filename=fname,
                local_dir=target_dir,
            )
            size_mb = os.path.getsize(local_path) / (1024 * 1024)
            print(f"  完成 ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  失敗: {e}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print(" 所有檔案下載完成！")
    print(f" 模型目錄: {target_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

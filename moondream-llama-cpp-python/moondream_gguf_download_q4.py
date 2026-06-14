import os
from huggingface_hub import HfApi, hf_hub_download

def download_optimized_gguf_model():
    api = HfApi()
    
    print("=" * 60)
    print(" 🚀 Moondream2 GGUF (INT8/Q4) 智慧精準下載器")
    print("=" * 60)
    
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moondream2")
    os.makedirs(target_dir, exist_ok=True)
    
    print("\n[1/3] 正在尋找公開且包含輕量級 (INT8/Q4) 版本的倉庫...")
    try:
        # 搜尋前 15 名的 moondream gguf 倉庫
        models = list(api.list_models(search="moondream gguf", sort="downloads", limit=15))
    except Exception as e:
        print(f"❌ 搜尋失敗: {e}")
        return

    best_repo = None
    best_text_file = None
    best_mmproj_file = None

    for repo in models:
        try:
            # 取得該倉庫內的所有檔案
            repo_files = api.list_repo_files(repo_id=repo.id)
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            # 1. 尋找視覺編碼器 (必須要有)
            mmproj_file = next((f for f in gguf_files if 'mmproj' in f.lower()), None)
            
            if not mmproj_file:
                continue # 這個倉庫沒有提供視覺編碼器，跳過換下一個
                
            # 2. 嚴格尋找輕量級文字模型 (優先順序: int8 > q4_ > q5_)
            # 絕對排除 f16，避免 Orin Nano 發生 OOM 記憶體崩潰
            text_file = next((f for f in gguf_files if 'int8' in f.lower()), None)
            if not text_file:
                text_file = next((f for f in gguf_files if 'q4_' in f.lower()), None)
            if not text_file:
                text_file = next((f for f in gguf_files if 'q5_' in f.lower()), None)
            
            # 如果找到了完美符合條件的組合
            if text_file:
                best_repo = repo.id
                best_text_file = text_file
                best_mmproj_file = mmproj_file
                break # 找到了就停止搜尋
                
        except Exception as e:
            continue # 讀取失敗就換下一個倉庫
            
    if not best_repo:
        print("\n❌ 找不到包含 INT8 或 Q4 版本的公開倉庫。")
        print("💡 替代方案：請確保網路正常，或稍後再試。")
        return
        
    print(f"✅ 找到完美匹配的公開倉庫: {best_repo}")
    print(f"  👉 視覺編碼器: {best_mmproj_file}")
    print(f"  👉 輕量文字模型: {best_text_file}")
    
    print(f"\n[2/3] 開始下載視覺編碼器... (這可能需要幾分鐘)")
    try:
        hf_hub_download(repo_id=best_repo, filename=best_mmproj_file, local_dir=target_dir)
        print("✅ 視覺編碼器下載成功！")
    except Exception as e:
        print(f"❌ 視覺編碼器下載失敗: {e}")
        return
        
    print(f"\n[3/3] 開始下載輕量文字模型... (這可能需要幾分鐘)")
    try:
        hf_hub_download(repo_id=best_repo, filename=best_text_file, local_dir=target_dir)
        print("✅ 文字模型下載成功！")
    except Exception as e:
        print(f"❌ 文字模型下載失敗: {e}")
        return

    print("\n🎉 下載程序結束！請重新執行您的 test_moondream_gguf.py 驗證腳本。")

if __name__ == "__main__":
    download_optimized_gguf_model()
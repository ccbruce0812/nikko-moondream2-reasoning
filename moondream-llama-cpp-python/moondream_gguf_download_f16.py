import os
from huggingface_hub import HfApi, hf_hub_download

def download_best_orin_nano_model():
    api = HfApi()
    
    print("=" * 60)
    print(" 🚀 Moondream2 GGUF Orin Nano 專用全自動下載器")
    print("=" * 60)
    
    print("\n[1/3] 正在自動尋找社群中最穩定、下載量最高的 GGUF 倉庫...")
    try:
        # 尋找關鍵字 moondream2 gguf，依下載量排序 (最新版 API 已移除 direction 參數)
        models = list(api.list_models(search="moondream2 gguf", sort="downloads", limit=5))
    except Exception as e:
        print(f"❌ 搜尋失敗，請檢查網路連線: {e}")
        return

    if not models:
        print("❌ 找不到任何相關的 GGUF 倉庫。")
        return

    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moondream2")
    os.makedirs(target_dir, exist_ok=True)
    
    # 遍歷熱門倉庫，尋找同時包含 mmproj 與合適量化版本的倉庫
    for repo in models:
        try:
            repo_files = api.list_repo_files(repo_id=repo.id)
            gguf_files = [f for f in repo_files if f.endswith('.gguf')]
            
            # 尋找視覺編碼器 (mmproj)
            mmproj_file = next((f for f in gguf_files if 'mmproj' in f.lower()), None)
            
            # 尋找適合 Orin Nano 的文字模型 (優先順序: q4_k_m > int8 > q8_0 > 任何其他版本)
            text_file = next((f for f in gguf_files if 'q4_k_m' in f.lower()), None)
            if not text_file:
                text_file = next((f for f in gguf_files if 'int8' in f.lower()), None)
            if not text_file:
                text_file = next((f for f in gguf_files if 'q8_0' in f.lower()), None)
            if not text_file:
                # 如果都找不到，找一個不是 mmproj 的 gguf 當作文字模型
                text_file = next((f for f in gguf_files if 'mmproj' not in f.lower()), None)

            if mmproj_file and text_file:
                print(f"✅ 找到最佳倉庫: {repo.id}")
                print(f"\n[2/3] 準備下載模型檔案至: {target_dir}")
                print(f"  👉 視覺編碼器: {mmproj_file}")
                print(f"  👉 文字模型: {text_file} (適合 8GB 記憶體)")
                
                print(f"\n[3/3] 開始下載... (檔案約 1.5GB ~ 2.5GB，請耐心等候)")
                
                print(f"下載 {mmproj_file} 中...")
                hf_hub_download(repo_id=repo.id, filename=mmproj_file, local_dir=target_dir)
                
                print(f"下載 {text_file} 中...")
                hf_hub_download(repo_id=repo.id, filename=text_file, local_dir=target_dir)
                
                print("\n🎉 所有模型檔案下載完成！")
                print(f"檔案已成功儲存於: {target_dir}")
                
                # 自動更新 moondream_worker.py 路徑的提示
                print("-" * 60)
                print("💡 下一步提醒：")
                print("請開啟您的 src/modules/moondream_worker.py 檔案，確認路徑名稱是否與下方一致：")
                print(f"self.model_path = os.path.join(project_root, \"moondream2\", \"{text_file}\")")
                print(f"self.mmproj_path = os.path.join(project_root, \"moondream2\", \"{mmproj_file}\")")
                print("-" * 60)
                return
                
        except Exception as e:
            print(f"讀取倉庫 {repo.id} 時發生錯誤: {e}，嘗試下一個倉庫...")
            continue
            
    print("\n❌ 在熱門倉庫中找不到完整的 Moondream2 GGUF 檔案組合 (需包含 mmproj 與文字模型)。")

if __name__ == "__main__":
    download_best_orin_nano_model()
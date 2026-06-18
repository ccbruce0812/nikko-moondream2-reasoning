import os
import cv2
import tempfile
import time
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler

def open_camera_nikko_style():
    """
    對標 nikko-pyside6 的 VideoWorker 架構：
    使用 nvarguscamerasrc 探測硬體加速模式，避免 OpenCV 預設管線錯誤。
    """
    # nikko-pyside6 採用的 GStreamer 管道字串，確保輸出為 BGR 以便 OpenCV 處理
    gst_pipeline = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv ! "
        "video/x-raw, width=640, height=480, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    print("  👉 正在初始化 VideoWorker 硬體加速管線...")
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("❌ VideoWorker 初始化失敗，請檢查 sensor-id 或 CSI 連線。")
        return None
        
    return cap

def open_camera():
    """
    嘗試多種攝影機開啟方式，確保在 Orin/Jetson 平台上能成功連線
    """
    # 嘗試 1: 強制使用 V4L2 後端 (解決 GStreamer 預設管線錯誤的最有效方法)
    print("  👉 嘗試使用 V4L2 後端開啟攝影機...")
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        # 手動設定解析度，避免預設解析度過高導致資源問題
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("  ✅ 成功使用 V4L2 後端開啟攝影機！")
        return cap
    cap.release()

    # 嘗試 2: 標準 GStreamer USB 攝影機管線
    print("  👉 嘗試使用 GStreamer (USB) 管線...")
    pipeline_usb = "v4l2src device=/dev/video0 ! video/x-raw, width=640, height=480, format=YUY2 ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(pipeline_usb, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("  ✅ 成功使用 GStreamer (USB) 管線開啟攝影機！")
        return cap
    cap.release()

    # 嘗試 3: Jetson CSI 攝影機管線 (nvarguscamerasrc)
    print("  👉 嘗試使用 GStreamer (CSI) 管線...")
    pipeline_csi = "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! nvvidconv flip-method=0 ! video/x-raw, width=640, height=480, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink"
    cap = cv2.VideoCapture(pipeline_csi, cv2.CAP_GSTREAMER)
    if cap.isOpened():
        print("  ✅ 成功使用 GStreamer (CSI) 管線開啟攝影機！")
        return cap
    cap.release()

    # 嘗試 4: 預設開啟 (最後的手段)
    print("  👉 嘗試使用 OpenCV 預設參數...")
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("  ✅ 成功使用預設後端開啟攝影機！")
        return cap

    return None

def test_moondream_gguf():
    print("=" * 60)
    print(" 🚀 Moondream2 GGUF (CUDA) 單機驗證腳本")
    print("=" * 60)

    # 1. 自動偵測 moondream2 資料夾與模型檔案
    project_root = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(project_root, "moondream2")
    
    if not os.path.exists(model_dir):
        print(f"❌ 找不到模型資料夾: {model_dir}")
        print("   請確認您已經成功執行過下載腳本。")
        return

    # 尋找視覺編碼器 (mmproj) 和 文字模型
    gguf_files = [f for f in os.listdir(model_dir) if f.endswith('.gguf')]
    mmproj_path = next((os.path.join(model_dir, f) for f in gguf_files if 'mmproj' in f.lower()), None)
    
    # 優先尋找 int8 版本，若無則找其他版本
    text_model_file = next((f for f in gguf_files if 'int8' in f.lower()), None)
    if not text_model_file:
        text_model_file = next((f for f in gguf_files if 'mmproj' not in f.lower()), None)
    
    if not mmproj_path or not text_model_file:
        print("❌ 找不到完整的模型檔案。")
        print(f"   目前資料夾內的檔案: {gguf_files}")
        return
        
    model_path = os.path.join(model_dir, text_model_file)

    print(f"\n[1/4] 檢查模型路徑與檔案大小...")
    mmproj_size = os.path.getsize(mmproj_path) / (1024 * 1024)
    text_size = os.path.getsize(model_path) / (1024 * 1024)
    
    print(f"  👉 視覺編碼器: {os.path.basename(mmproj_path)} ({mmproj_size:.2f} MB)")
    print(f"  👉 文字模型: {os.path.basename(model_path)} ({text_size:.2f} MB)")

    if text_size < 100 or mmproj_size < 10:
        print("\n❌ 錯誤：檔案大小異常！這表示您下載到的只是 Git LFS 指標檔 (幾 KB)，而不是真實的權重檔。")
        print("   解決方案：刪除 moondream2 資料夾，然後嘗試手動去 Hugging Face 下載真實檔案。")
        return

    # 2. 載入模型 (開啟 CUDA 加速)
    print("\n[2/4] 正在將模型載入 Orin Nano GPU (這可能需要 5-10 秒)...")
    start_load = time.time()
    try:
        chat_handler = Llava15ChatHandler(clip_model_path=mmproj_path)
        
        llm = Llama(
            model_path=model_path,
            chat_handler=chat_handler,
            n_gpu_layers=-1, 
            n_ctx=2048,
            verbose=False # 關閉底層詳細日誌，讓終端機乾淨一點，若需除錯可改回 True
        )
        print(f"✅ 模型載入成功！(耗時: {time.time() - start_load:.2f} 秒)")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    #3. 啟動攝影機並顯示預覽框
    print("\n[3/4] 啟動攝影機預覽視窗 (按 'q' 鍵或 'Esc' 鍵確認截取)...")
    cap = open_camera_nikko_style()

    if cap is None or not cap.isOpened():
        print("❌ 無法從攝影機擷取畫面。")
        return

    frame = None
    while True:
        ret, temp_frame = cap.read()
        if not ret:
            print("❌ 讀取畫面失敗。")
            break
        
        # 顯示畫面
        cv2.imshow("Camera Preview - Press 'q' to Confirm", temp_frame)
        
        # 偵測按鍵 (q 或 ESC 關閉)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            frame = temp_frame # 鎖定當前畫面
            break
            
    cap.release()
    cv2.destroyAllWindows() # 關閉視窗，流程繼續

    if frame is None:
        print("❌ 未捕獲到影像。")
        return

    # 存檔供 AI 使用
    temp_dir = tempfile.gettempdir()
    temp_image_path = os.path.join(temp_dir, "test_moondream_frame.jpg")
    cv2.imwrite(temp_image_path, frame)
    print(f"✅ 畫面已確認並儲存於: {temp_image_path}")

    # 4. 執行 Moondream2 推論
    prompt = "Please describe this image in detail." 
    print(f"\n[4/4] 執行 VLM 推論...")
    print(f"  👉 提示詞: '{prompt}'")
    
    print("\n🤖 Moondream2 思考中 (CUDA 加速中)...\n")
    start_infer = time.time()
    
    try:
        response = llm.create_chat_completion(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"file://{temp_image_path}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
        )
        
        answer = response["choices"][0]["message"]["content"]
        
        print("================== 回答結果 ==================")
        print(answer.strip())
        print("==============================================")
        print(f"\n⏱️ 推論總耗時: {time.time() - start_infer:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 推論失敗: {e}")

if __name__ == "__main__":
    test_moondream_gguf()
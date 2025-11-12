# 🏐 Volleyball Court Ball Detection Pipeline (Windows PowerShell)

本專案提供完整的排球落地點偵測流程，包括：
1. 模型推論取得球軌跡（ONNX Runtime）
2. 手動描繪球場外框 (`click_court`)
3. 自動偵測落地點並產生可視化影片 (`postprocess_inout_patched_v3`)

---

## ⚙️ 第一步：環境安裝

開啟 **Windows PowerShell**，並安裝必要套件：

```powershell
pip install onnxruntime opencv-python pandas tqdm
```

若要使用其他模型格式（例如 PyTorch），可額外安裝：
```powershell
pip install torch torchvision
```

---

## 🎯 第二步：使用模型偵測球軌跡

執行 ONNX 推論腳本，從影片中偵測排球位置。

```powershell
python src/inference_onnx_seq9_gray_v2.py `
  --video_path output/test30sec.mp4 `
  --model_path models/VballNetFastV1_seq9_grayscale_233_h288_w512.onnx `
  --output_dir output `
  --track_length 10 `
  --visualize
```

執行完後，會在 `output/` 目錄下生成：
- `*_predict_ball.csv`：包含每一幀的球中心座標 (x, y)
- `*_visualized.mp4`：可視化偵測結果影片（若啟用 `--visualize`）

---

## 🏟️ 第三步：描繪球場與偵測落地點

這一步會先描出球場四個角 (`click_court`)，然後自動判斷每次落地點的 **IN / OUT** 狀態。

> ⚠️ 若 `output/court.json` 不存在，程式會自動開啟互動視窗讓你點四個角。
> 點選順序：**左上 → 右上 → 右下 → 左下**  
> 按下 `S` 儲存後會自動關閉視窗並繼續落地判定。

執行指令如下：

```powershell
python src/postprocess_inout_patched_v3.py `
  --csv_path   output/"模型輸出檔名_predict_ball.csv" `
  --court_json output/court.json `
  --out_csv    output/landings_inout.csv `
  --video_path output/"測試影片檔.mp4" `
  --out_video  output/landings_overlay.mp4 `
  --curve_filter_on `
  --min-x-drift-px 250 `
  --hud `
  --verbose
```

執行完成後，`output/` 資料夾會包含：
- `court.json`：你手動描出的球場外框資訊  
- `landings_inout.csv`：每次偵測到的落地座標與 IN / OUT 標籤  
- `landings_overlay.mp4`：加上 HUD（右側落地紀錄表）的可視化影片  

---

## 📁 檔案結構建議

```
project/
├─ models/
│  └─ VballNetFastV1_seq9_grayscale_233_h288_w512.onnx
├─ output/
│  ├─ test30sec.mp4
│  ├─ test30sec_predict_ball.csv
│  ├─ court.json
│  ├─ landings_inout.csv
│  └─ landings_overlay.mp4
├─ src/
│  ├─ inference_onnx_seq9_gray_v2.py
│  └─ postprocess_inout_patched_v3.py
└─ tools/
   └─ click_court.py
```

---

## 💡 小技巧
- 若想更嚴格過濾落地事件，可調整 `--min-x-drift-px`（建議範圍 150–300）。
- `--hud` 參數會在輸出影片右側顯示落地紀錄表。
- `--verbose` 會顯示詳細除錯資訊，建議偵錯時開啟。

---

## 🧩 範例影片輸出
執行完成後，你將得到類似下圖效果的影片：
- 球場外框（黃色）
- 每次落地位置的圓點與標籤（IN/OUT）
- 右側落地紀錄 HUD 面板

---

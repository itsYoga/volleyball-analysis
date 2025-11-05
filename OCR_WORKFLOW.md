# OCR 球衣號碼檢測工作流程

## 1. OCR 檢測頻率

**不是每一幀都檢測**，而是**每10幀檢測一次**：

```python
# 在 _get_stable_player_id 函數中
if EASYOCR_AVAILABLE and frame is not None and track_id % 10 == 0:
    jersey_num = self._detect_jersey_number(frame, bbox, track_id)
```

這意味著：
- Track ID 0, 10, 20, 30... 會觸發 OCR 檢測
- Track ID 1, 2, 3... 9, 11... 不會觸發 OCR 檢測
- 這是為了**性能優化**，因為 OCR 比較慢

## 2. OCR 確實有 Bounding Box！

OCR **使用玩家檢測的 bounding box** 來知道要檢測哪個區域：

```python
def _detect_jersey_number(self, frame: np.ndarray, bbox: List[float], track_id: int = None):
    # bbox = [x1, y1, x2, y2] - 玩家的邊界框
    x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    
    # 提取上半身區域（球衣號碼通常在胸部）
    height = y2 - y1
    roi_top = max(0, y1)
    roi_bottom = min(frame.shape[0], y1 + int(height * 0.6))  # 上半身60%
    roi_left = max(0, x1)
    roi_right = min(frame.shape[1], x2)
    
    # 提取 ROI（Region of Interest）
    roi = frame[roi_top:roi_bottom, roi_left:roi_right].copy()
    
    # 對 ROI 進行 OCR 識別
    results = self.jersey_number_model.readtext(roi)
```

## 3. 如何知道檢測到哪個球員？

通過 **track_id** 來識別：

1. **玩家檢測階段**：
   - YOLO 模型檢測到玩家，給出 bounding box
   - 每個檢測到的玩家有一個臨時 ID

2. **玩家追蹤階段**：
   - Norfair 追蹤器為每個玩家分配一個 **track_id**
   - track_id 在同一個玩家被追蹤的過程中保持不變

3. **OCR 檢測階段**：
   - 使用 track_id 來識別這是哪個玩家
   - OCR 檢測結果會記錄到 `track_id_to_jersey_history[track_id]`

## 4. 完整流程圖

```
每一幀：
├─ 玩家檢測 (YOLO)
│  └─ 輸出：多個玩家的 bounding box [x1, y1, x2, y2]
│
├─ 玩家追蹤 (Norfair)
│  ├─ 輸入：玩家 bounding box
│  ├─ 輸出：track_id（每個玩家一個唯一的追蹤ID）
│  └─ track_id 在追蹤過程中保持不變
│
└─ OCR 檢測（僅每10幀執行一次）
   ├─ 檢查：track_id % 10 == 0？
   ├─ 如果是：
   │  ├─ 提取該玩家的 bounding box
   │  ├─ 裁剪上半身區域（ROI）
   │  ├─ 圖像預處理（CLAHE、銳化）
   │  ├─ EasyOCR 識別數字
   │  └─ 記錄結果到 track_id_to_jersey_history[track_id]
   └─ 如果不是：跳過 OCR 檢測
```

## 5. 多幀融合機制

為了提高準確性，系統使用**多幀融合**：

```python
# 記錄歷史檢測結果
self.track_id_to_jersey_history[track_id].extend(detected_numbers)

# 投票：返回最常見的號碼（如果出現次數 >= 2）
counter = Counter(self.track_id_to_jersey_history[track_id])
most_common = counter.most_common(1)[0]
if most_common[1] >= 2:  # 至少出現2次才認為可靠
    return most_common[0]
```

這意味著：
- 同一個 track_id 的多次 OCR 檢測結果會被記錄
- 系統會選擇**出現次數最多**的號碼作為最終結果
- 這可以減少單次 OCR 錯誤的影響

## 6. 為什麼會檢測到多個號碼？

在你的數據中，Track ID 10 檢測到了 #9, #18, #13：

**可能原因：**
1. **OCR 誤識別**：同一幀中，OCR 可能識別出多個數字（例如背景中的數字、其他球員的號碼等）
2. **Track ID 切換**：如果同一個實際球員在不同時間被分配了不同的 track_id，每個 track_id 可能檢測到不同的號碼
3. **視角問題**：從不同角度看到的球衣號碼可能不同

**當前解決方案：**
- 系統選擇**最頻繁出現**的號碼（#9 出現50次，最頻繁）
- 但這可能不是最準確的，因為：
  - Track ID 10 可能實際上是兩個不同的球員（一個是 #9，一個是 #18）
  - 但由於追蹤器的限制，它們被合併成一個 track_id

## 7. 改進建議

如果要提高準確性，可以考慮：

1. **更頻繁的 OCR 檢測**：改為每5幀或每3幀檢測一次
2. **更嚴格的閾值**：要求號碼至少出現5次或10次才認為可靠
3. **空間一致性檢查**：如果同一個 track_id 在不同位置檢測到不同的號碼，可能需要拆分
4. **手動標記**：讓用戶手動標記無法自動識別的玩家


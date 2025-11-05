# 整合 Jersey Number Pipeline 方案

## 優勢分析

### Jersey Number Pipeline vs EasyOCR

| 特性 | Jersey Number Pipeline | EasyOCR |
|------|------------------------|---------|
| **專門性** | ✅ 專門為運動場景設計 | ❌ 通用OCR |
| **準確率** | ✅ 在運動數據集上訓練，應該更高 | ⚠️ 通用場景，可能較低 |
| **處理能力** | ✅ 處理遮擋、運動模糊、視角變化 | ⚠️ 對模糊圖像有限支持 |
| **姿態引導** | ✅ 使用姿態估計引導RoI裁剪 | ❌ 無 |
| **軌跡整合** | ✅ 跨幀整合提高準確率 | ❌ 無 |
| **設置複雜度** | ❌ 需要多個模型和依賴 | ✅ 簡單，只需pip install |
| **性能** | ⚠️ 可能較慢（多步驟） | ✅ 較快 |
| **排球適配** | ⚠️ 在曲棍球/足球訓練，需微調 | ✅ 可直接使用 |

## 建議方案

### 方案 1：混合方案（推薦）
1. **優先使用 Jersey Number Pipeline**（如果可用）
2. **降級到 EasyOCR**（如果 Pipeline 失敗或未安裝）
3. **用戶手動標記**（最終備選）

### 方案 2：完整整合 Jersey Number Pipeline
優點：
- 最高準確率
- 專門針對運動場景
- 處理複雜情況（遮擋、模糊）

缺點：
- 需要下載多個模型（SAM, Centroid-ReID, ViTPose, PARSeq）
- 設置複雜
- 可能需要針對排球數據微調

### 方案 3：輕量級整合（僅使用 PARSeq）
優點：
- 相對簡單
- 比 EasyOCR 更準確
- 不需要完整的 Pipeline

缺點：
- 無法利用姿態引導和多幀整合

## 實現建議

### 階段 1：快速改進（當前）
- 繼續使用 EasyOCR
- 優化 RoI 提取（上半身區域）
- 添加圖像預處理（對比度增強、銳化）

### 階段 2：中期改進
- 整合 PARSeq（場景文本識別）
- 實現多幀融合（同一玩家多次識別結果投票）
- 添加可讀性分類器（過濾模糊/不可讀的區域）

### 階段 3：完整整合（如果準確率仍不足）
- 整合完整的 Jersey Number Pipeline
- 添加姿態估計引導的 RoI 裁剪
- 實現軌跡級別的識別整合

## 代碼結構建議

```python
class JerseyNumberDetector:
    def __init__(self):
        self.easyocr_reader = None
        self.parseq_model = None
        self.legibility_classifier = None
        self.use_parseq = False  # 開關
    
    def detect(self, frame, bbox):
        # 1. 提取RoI（上半身區域）
        roi = self.extract_roi(frame, bbox)
        
        # 2. 可讀性檢查（可選）
        if not self.is_legible(roi):
            return None
        
        # 3. 嘗試 PARSeq（如果可用）
        if self.use_parseq and self.parseq_model:
            result = self.parseq_detect(roi)
            if result:
                return result
        
        # 4. 降級到 EasyOCR
        return self.easyocr_detect(roi)
```

## 立即可做的改進（無需外部依賴）

1. **更好的 RoI 提取**：
   - 使用姿態估計（如果可用）定位胸部區域
   - 或使用更智能的裁剪策略

2. **圖像預處理**：
   - 對比度增強
   - 銳化
   - 二值化（嘗試）

3. **多幀融合**：
   - 對同一追蹤ID的多次識別結果進行投票
   - 使用最常見的結果

4. **後處理**：
   - 過濾不合理的數字（如 > 99）
   - 使用上下文信息（同一隊的號碼通常在特定範圍）

## 結論

**短期（立即）**：
- 繼續使用 EasyOCR，但優化 RoI 提取和圖像預處理
- 實現多幀融合提高準確率

**中期（1-2週）**：
- 考慮整合 PARSeq（相對簡單，效果提升明顯）
- 實現可讀性分類器過濾低質量圖像

**長期（如果準確率仍不足）**：
- 整合完整的 Jersey Number Pipeline
- 在排球數據上微調模型

**建議**：先嘗試優化當前的 EasyOCR 實現（圖像預處理、多幀融合），如果準確率仍不足，再考慮整合 Jersey Number Pipeline。


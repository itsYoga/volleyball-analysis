# 排球分析系統 Web App

基於現有的ball detection和action classification模型的完整網頁應用程式。

## 系統架構

```
volleyball_analysis_webapp/
├── backend/           # 後端API服務 (FastAPI)
├── frontend/          # 前端介面 (React)
├── ai_core/           # AI模型處理核心
├── data/              # 數據存儲
├── models/            # 預訓練模型
├── static/            # 靜態文件
└── templates/         # 模板文件
```

## 功能特色

- 🎥 **影片上傳與處理**: 支援多種影片格式上傳
- ⚽ **球軌跡追蹤**: 基於VballNet的球位置檢測
- 🏐 **動作識別**: 基於YOLOv11的球員動作分類
- 📊 **互動式播放器**: 即時標註與時間軸導航
- 🔍 **智能篩選**: 按動作類型、球員、時間段篩選
- 📈 **數據分析**: 統計圖表與熱區圖
- 👥 **團隊協作**: 播放清單與分享功能

## 技術棧

### 後端
- **FastAPI**: 現代化Python Web框架
- **PostgreSQL**: 結構化數據存儲
- **Celery**: 異步任務處理
- **Redis**: 任務佇列與快取

### 前端
- **React**: 現代化UI框架
- **TypeScript**: 類型安全
- **Tailwind CSS**: 響應式設計
- **Video.js**: 影片播放器

### AI核心
- **PyTorch**: 深度學習框架
- **OpenCV**: 電腦視覺處理
- **Ultralytics**: YOLO模型
- **ONNX**: 模型優化

## 快速開始

### 1. 環境設置
```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt
```

### 2. 啟動服務
```bash
# 啟動後端API
cd backend
uvicorn main:app --reload

# 啟動前端 (新終端)
cd frontend
npm start

# 啟動AI處理器 (新終端)
cd ai_core
python worker.py
```

### 3. 訪問應用
- 前端: http://localhost:3000
- 後端API: http://localhost:8000
- API文檔: http://localhost:8000/docs

## 開發階段

- [x] 項目結構設置
- [ ] 離線腳本驗證
- [ ] MVP後端API
- [ ] 基礎前端介面
- [ ] AI模型整合
- [ ] 資料庫設計
- [ ] 異步處理
- [ ] 進階功能

## 授權

MIT License

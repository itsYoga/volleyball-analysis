# 排球分析系統 - 開發指南

## 🎯 項目概述

這是一個基於您現有ball detection和action classification模型的完整網頁應用程式。系統採用現代化的微服務架構，提供影片上傳、AI分析、互動式播放和數據可視化功能。

## 🏗️ 系統架構

```
volleyball_analysis_webapp/
├── backend/              # 後端API服務 (FastAPI)
│   ├── main.py          # 主API服務
│   ├── models/          # 數據模型
│   ├── routers/         # API路由
│   └── database/        # 數據庫配置
├── frontend/            # 前端介面 (React + TypeScript)
│   ├── src/
│   │   ├── components/  # React組件
│   │   ├── services/    # API服務
│   │   ├── hooks/       # 自定義Hooks
│   │   └── utils/       # 工具函數
│   └── public/          # 靜態文件
├── ai_core/             # AI處理核心
│   ├── processor.py     # 主要分析器
│   ├── worker.py        # Celery工作器
│   └── models/          # 模型載入器
├── data/                # 數據存儲
│   ├── uploads/         # 上傳的影片
│   └── results/         # 分析結果
└── models/              # 預訓練模型
```

## 🚀 快速開始

### 1. 環境準備

```bash
# 克隆項目
cd /Users/jesse/Documents/專題/volleyball_analysis_webapp

# 運行快速啟動腳本
./start.sh
```

### 2. 手動啟動 (可選)

```bash
# 1. 創建虛擬環境
python3 -m venv venv
source venv/bin/activate

# 2. 安裝依賴
pip install -r requirements.txt

# 3. 啟動Redis (如果沒有)
redis-server

# 4. 啟動後端
cd backend
uvicorn main:app --reload

# 5. 啟動AI工作器 (新終端)
cd ai_core
celery -A worker worker --loglevel=info

# 6. 啟動前端 (新終端)
cd frontend
npm install
npm start
```

## 🔧 開發階段

### 階段1: 離線腳本驗證 ✅

```bash
# 測試您的模型
python offline_test.py --video /path/to/your/video.mp4
```

**目標**: 確保ball detection和action classification模型能正常工作

### 階段2: MVP後端API 🚧

**已完成**:
- [x] 基本FastAPI服務
- [x] 影片上傳API
- [x] 分析任務API
- [x] 結果查詢API

**待完成**:
- [ ] 數據庫集成 (PostgreSQL)
- [ ] 用戶認證系統
- [ ] 文件管理優化

### 階段3: 基礎前端介面 🚧

**已完成**:
- [x] 影片上傳組件
- [x] API服務層
- [x] 基本路由

**待完成**:
- [ ] 影片庫組件
- [ ] 播放器組件
- [ ] 結果可視化

### 階段4: AI模型整合 🚧

**已完成**:
- [x] 分析器框架
- [x] 模型載入器
- [x] 離線測試腳本

**待完成**:
- [ ] 模型路徑配置
- [ ] 性能優化
- [ ] 錯誤處理

### 階段5: 進階功能 📋

- [ ] 互動式播放器
- [ ] 數據篩選與搜索
- [ ] 統計圖表
- [ ] 播放清單功能
- [ ] 團隊協作

## 🛠️ 技術細節

### 後端技術棧

- **FastAPI**: 現代化Python Web框架
- **PostgreSQL**: 關係型數據庫
- **Celery**: 異步任務處理
- **Redis**: 任務隊列與快取
- **SQLAlchemy**: ORM框架

### 前端技術棧

- **React 18**: 現代化UI框架
- **TypeScript**: 類型安全
- **Tailwind CSS**: 響應式設計
- **Video.js**: 影片播放器
- **Recharts**: 數據可視化

### AI核心技術

- **PyTorch**: 深度學習框架
- **Ultralytics**: YOLO模型
- **OpenCV**: 電腦視覺
- **ONNX Runtime**: 模型推理

## 📊 數據流程

1. **影片上傳**: 用戶上傳影片 → 後端接收 → 存儲到文件系統
2. **任務創建**: 後端創建分析任務 → 加入Celery隊列
3. **AI處理**: 工作器從隊列取任務 → 載入模型 → 分析影片
4. **結果存儲**: 分析結果 → 存儲到數據庫 → 更新任務狀態
5. **前端展示**: 前端查詢結果 → 渲染播放器 → 顯示分析數據

## 🔍 調試指南

### 後端調試

```bash
# 查看API文檔
http://localhost:8000/docs

# 查看日誌
tail -f backend/logs/app.log

# 測試API
curl -X POST "http://localhost:8000/upload" -F "file=@test.mp4"
```

### 前端調試

```bash
# 開發模式
npm start

# 構建生產版本
npm run build

# 代碼檢查
npm run lint
```

### AI核心調試

```bash
# 測試分析器
python ai_core/processor.py

# 查看工作器日誌
celery -A ai_core.worker worker --loglevel=debug
```

## 🚀 部署指南

### Docker部署

```bash
# 構建並啟動所有服務
docker-compose up -d

# 查看日誌
docker-compose logs -f

# 停止服務
docker-compose down
```

### 生產環境

1. **服務器配置**: 至少4核CPU, 8GB RAM, GPU (可選)
2. **數據庫**: PostgreSQL 15+
3. **緩存**: Redis 7+
4. **反向代理**: Nginx
5. **SSL證書**: Let's Encrypt

## 📈 性能優化

### 模型優化

- 使用ONNX格式減少推理時間
- 模型量化降低內存使用
- 批處理提高吞吐量

### 系統優化

- Redis緩存熱點數據
- 數據庫索引優化
- CDN加速靜態文件

## 🐛 常見問題

### Q: 模型載入失敗
A: 檢查模型路徑和格式，確保依賴已安裝

### Q: 影片上傳失敗
A: 檢查文件大小限制和格式支持

### Q: 分析任務卡住
A: 檢查Celery工作器狀態和Redis連接

### Q: 前端無法連接後端
A: 檢查CORS設置和API URL配置

## 📞 支持

如有問題，請檢查：
1. 日誌文件
2. API文檔 (http://localhost:8000/docs)
3. 系統狀態 (http://localhost:8000/health)

## 🎉 下一步

1. 完成離線腳本測試
2. 配置模型路徑
3. 測試完整流程
4. 添加更多功能
5. 優化性能
6. 部署上線

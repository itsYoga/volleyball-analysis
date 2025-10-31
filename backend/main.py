"""
排球分析系統 - 後端API服務
基於FastAPI的RESTful API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import uuid
import json
from datetime import datetime
from typing import List, Optional
import asyncio
from pathlib import Path

# 連結到 ai_core 分析器
import sys
BACKEND_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = BACKEND_DIR.parent
sys.path.append(str(PROJECT_ROOT / "ai_core"))
from processor import VolleyballAnalyzer  # type: ignore

# 創建FastAPI應用
app = FastAPI(
    title="排球分析系統 API",
    description="基於AI的排球影片分析系統",
    version="1.0.0"
)

# CORS設置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 靜態文件服務
# app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/static", StaticFiles(directory=(PROJECT_ROOT / "static")), name="static")

# 數據存儲目錄
UPLOAD_DIR = "data/uploads"
RESULTS_DIR = "data/results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 模擬數據庫 (實際應用中應使用PostgreSQL)
videos_db = []
analysis_tasks = {}

@app.get("/")
async def root():
    """根路徑"""
    return {"message": "排球分析系統 API 服務運行中"}

@app.get("/health")
async def health_check():
    """健康檢查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """上傳影片文件"""
    try:
        # 生成唯一ID
        video_id = str(uuid.uuid4())
        
        # 保存上傳文件
        file_extension = file.filename.split('.')[-1]
        filename = f"{video_id}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # 串流寫入，避免一次載入整個大檔到記憶體
        # with open(file_path, "wb") as buffer:
        #     content = await file.read()
        #     buffer.write(content)
        bytes_written = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                bytes_written += len(chunk)
        
        # 記錄到數據庫
        video_data = {
            "id": video_id,
            "filename": file.filename,
            "file_path": file_path,
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "file_size": bytes_written
        }
        videos_db.append(video_data)
        
        return {
            "video_id": video_id,
            "message": "影片上傳成功",
            "filename": file.filename,
            "file_size": bytes_written
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")

@app.post("/analyze/{video_id}")
async def start_analysis(video_id: str, background_tasks: BackgroundTasks):
    """開始分析影片"""
    try:
        # 查找影片
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if not video:
            raise HTTPException(status_code=404, detail="影片不存在")
        
        # 創建分析任務
        task_id = str(uuid.uuid4())
        analysis_tasks[task_id] = {
            "video_id": video_id,
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "progress": 0
        }
        
        # 更新影片狀態
        video["status"] = "processing"
        video["task_id"] = task_id
        
        # 添加背景任務 (實際應用中應使用Celery)
        background_tasks.add_task(process_video, video_id, task_id)
        
        return {
            "task_id": task_id,
            "message": "分析任務已開始",
            "video_id": video_id
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"開始分析失敗: {str(e)}")

@app.get("/videos")
async def get_videos():
    """獲取所有影片列表"""
    return {"videos": videos_db}

@app.get("/videos/{video_id}")
async def get_video(video_id: str):
    """獲取特定影片信息"""
    video = next((v for v in videos_db if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="影片不存在")
    return video

@app.get("/analysis/{task_id}")
async def get_analysis_status(task_id: str):
    """獲取分析任務狀態"""
    task = analysis_tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任務不存在")
    return task

@app.get("/results/{video_id}")
async def get_analysis_results(video_id: str):
    """獲取分析結果"""
    try:
        results_file = os.path.join(RESULTS_DIR, f"{video_id}_results.json")
        if not os.path.exists(results_file):
            raise HTTPException(status_code=404, detail="分析結果不存在")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"獲取結果失敗: {str(e)}")

@app.get("/play/{video_id}")
async def play_video(video_id: str):
    """播放影片文件"""
    video = next((v for v in videos_db if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="影片不存在")
    
    if not os.path.exists(video["file_path"]):
        raise HTTPException(status_code=404, detail="影片文件不存在")
    
    return FileResponse(video["file_path"])

async def process_video(video_id: str, task_id: str):
    """處理影片的後台任務 (實際執行分析器)"""
    try:
        # 取得影片路徑
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if not video:
            raise FileNotFoundError("影片不存在")

        video_path = video["file_path"]

        # 準備分析器與模型路徑
        models_dir = (PROJECT_ROOT / "models").resolve()
        ball_model = str(models_dir / "VballNetV1_seq9_grayscale_148_h288_w512.onnx")
        action_model = str(models_dir / "action_recognition_yv11m.pt")
        player_model = str(models_dir / "player_detection_yv8.pt")

        analyzer = VolleyballAnalyzer(
            ball_model_path=ball_model if os.path.exists(ball_model) else None,
            action_model_path=action_model if os.path.exists(action_model) else None,
            player_model_path=player_model if os.path.exists(player_model) else None,
            device="cpu"
        )

        # 執行分析（同步呼叫，簡單以進度條表示）
        analysis_tasks[task_id]["progress"] = 5
        await asyncio.sleep(0)  # 讓事件循環有機會更新

        results_path = PROJECT_ROOT / "data" / "results" / f"{video_id}_results.json"
        os.makedirs(results_path.parent, exist_ok=True)

        # 實際分析（添加錯誤處理）
        try:
            results = analyzer.analyze_video(video_path, str(results_path))
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ 分析錯誤詳情:\n{error_detail}")
            raise
        
        # 保存結果
        results_file = os.path.join(RESULTS_DIR, f"{video_id}_results.json")
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 更新任務狀態
        analysis_tasks[task_id]["status"] = "completed"
        analysis_tasks[task_id]["progress"] = 100
        analysis_tasks[task_id]["end_time"] = datetime.now().isoformat()
        
        # 更新影片狀態
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if video:
            video["status"] = "completed"
            video["analysis_time"] = datetime.now().isoformat()
    
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["error"] = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

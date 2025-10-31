"""
排球分析系統 - 後端API服務
基於FastAPI的RESTful API
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import os
import uuid
import json
from datetime import datetime
from typing import List, Optional
import asyncio
from pathlib import Path
from pydantic import BaseModel

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
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# 模擬數據庫 (實際應用中應使用PostgreSQL)
videos_db = []
analysis_tasks = {}

class VideoUpdateRequest(BaseModel):
    new_filename: str

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
        file_path = str(UPLOAD_DIR / filename)
        
        # 串流寫入，避免一次載入整個大檔到記憶體
        bytes_written = 0
        chunk_size = 1024 * 1024  # 1MB
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                buffer.write(chunk)
                bytes_written += len(chunk)
        
        # 記錄到數據庫（使用相對路徑，方便存儲）
        relative_path = str(Path(file_path).relative_to(PROJECT_ROOT))
        video_data = {
            "id": video_id,
            "filename": file.filename,
            "file_path": relative_path,  # 使用相對路徑
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

@app.put("/videos/{video_id}")
async def update_video(video_id: str, request: VideoUpdateRequest):
    """更新視頻文件名"""
    video = next((v for v in videos_db if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="影片不存在")
    
    video["filename"] = request.new_filename
    return {"message": "視頻名稱已更新", "video": video}

@app.get("/play/{video_id}")
async def play_video(video_id: str, request: Request):
    """播放影片文件（支持 Range 请求以支持视频跳转）"""
    video = next((v for v in videos_db if v["id"] == video_id), None)
    if not video:
        print(f"❌ 視頻不存在: video_id={video_id}, 數據庫中有 {len(videos_db)} 個視頻")
        raise HTTPException(status_code=404, detail=f"影片不存在 (ID: {video_id})")
    
    video_path = video.get("file_path")
    if not video_path:
        print(f"❌ 視頻路徑不存在: video_id={video_id}, video={video}")
        raise HTTPException(status_code=404, detail="影片路徑不存在")
    
    # 確保路徑是絕對路徑
    if not os.path.isabs(video_path):
        video_path = str(PROJECT_ROOT / video_path)
    
    # 標準化路徑（處理相對路徑和絕對路徑）
    video_path = os.path.normpath(video_path)
    
    print(f"🔍 檢查視頻文件: video_id={video_id}, video_path={video_path}, exists={os.path.exists(video_path)}")
    
    if not os.path.exists(video_path):
        # 嘗試其他可能的路徑
        alt_paths = [
            str(UPLOAD_DIR / os.path.basename(video_path)),
            str(PROJECT_ROOT / "data" / "uploads" / os.path.basename(video_path)),
            video.get("file_path"),  # 原始路徑
        ]
        for alt_path in alt_paths:
            if alt_path and os.path.exists(alt_path):
                video_path = alt_path
                print(f"✅ 找到替代路徑: {video_path}")
                break
        else:
            print(f"❌ 影片文件不存在: video_path={video_path}, PROJECT_ROOT={PROJECT_ROOT}")
            print(f"   嘗試的路徑: {alt_paths}")
            raise HTTPException(status_code=404, detail=f"影片文件不存在: {video_path}")
    
    # 確定媒體類型
    file_extension = video_path.split('.')[-1].lower()
    media_type_map = {
        'mp4': 'video/mp4',
        'avi': 'video/x-msvideo',
        'mov': 'video/quicktime',
        'mkv': 'video/x-matroska',
        'webm': 'video/webm'
    }
    media_type = media_type_map.get(file_extension, 'video/mp4')
    
    # 獲取文件大小
    file_size = os.path.getsize(video_path)
    
    # 處理 Range 請求（支持視頻跳轉和緩衝）
    range_header = request.headers.get('range')
    if range_header:
        # 解析 Range 頭
        range_match = range_header.replace('bytes=', '').split('-')
        start = int(range_match[0]) if range_match[0] else 0
        end = int(range_match[1]) if range_match[1] else file_size - 1
        
        # 確保範圍有效
        start = max(0, start)
        end = min(file_size - 1, end)
        length = end - start + 1
        
        # 打開文件並讀取指定範圍
        def generate():
            with open(video_path, 'rb') as f:
                f.seek(start)
                remaining = length
                while remaining:
                    chunk_size = min(8192, remaining)  # 8KB chunks
                    chunk = f.read(chunk_size)
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk
        
        headers = {
            'Content-Range': f'bytes {start}-{end}/{file_size}',
            'Accept-Ranges': 'bytes',
            'Content-Length': str(length),
            'Content-Type': media_type,
        }
        
        return StreamingResponse(
            generate(),
            status_code=206,  # Partial Content
            headers=headers,
            media_type=media_type
        )
    else:
        # 沒有 Range 請求，返回整個文件
        return FileResponse(
            video_path,
            media_type=media_type,
            filename=video.get("filename", f"{video_id}.{file_extension}")
        )

async def process_video(video_id: str, task_id: str):
    """處理影片的後台任務 (實際執行分析器)"""
    try:
        # 取得影片路徑
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if not video:
            raise FileNotFoundError("影片不存在")

        video_path = video["file_path"]
        # 確保路徑是絕對路徑
        if not os.path.isabs(video_path):
            video_path = str(PROJECT_ROOT / video_path)
        
        # 檢查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片文件不存在: {video_path}")

        # 準備分析器與模型路徑
        models_dir = (PROJECT_ROOT / "models").resolve()
        ball_model = str(models_dir / "VballNetV1_seq9_grayscale_148_h288_w512.onnx")
        action_model = str(models_dir / "action_recognition_yv11m.pt")
        player_model = str(models_dir / "player_detection_yv8.pt")

        # 更新進度
        analysis_tasks[task_id]["progress"] = 5
        await asyncio.sleep(0)  # 讓事件循環有機會更新，允許其他請求處理

        results_path = RESULTS_DIR / f"{video_id}_results.json"
        os.makedirs(results_path.parent, exist_ok=True)

        # 定義一個內部函數來執行所有阻塞操作（包括分析器初始化和分析）
        def run_analysis():
            """在執行緒池中運行的阻塞操作"""
            # 創建進度回調函數來更新任務進度
            def update_progress(progress: float, frame_count: int, total_frames: int):
                """更新進度（在線程中執行，需要安全地更新共享狀態）"""
                # 進度範圍：5-95%（5%用於初始化，95%用於分析，100%完成）
                # 5% + (progress * 0.90) 將視頻分析的進度映射到 5-95%
                mapped_progress = 5 + (progress * 0.90)
                analysis_tasks[task_id]["progress"] = min(95, mapped_progress)
            
            analyzer = VolleyballAnalyzer(
                ball_model_path=ball_model if os.path.exists(ball_model) else None,
                action_model_path=action_model if os.path.exists(action_model) else None,
                player_model_path=player_model if os.path.exists(player_model) else None,
                device="cpu"
            )
            return analyzer.analyze_video(video_path, str(results_path), progress_callback=update_progress)

        # 實際分析（在執行緒池中執行，避免阻塞事件循環）
        try:
            # 使用 run_in_executor 在執行緒池中執行阻塞操作
            # 這可以確保不會阻塞 FastAPI 的事件循環，讓其他請求（如 /videos）可以正常處理
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            
            results = await loop.run_in_executor(None, run_analysis)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ 分析錯誤詳情:\n{error_detail}")
            raise
        
        # 保存結果
        results_file = RESULTS_DIR / f"{video_id}_results.json"
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

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
from typing import List, Optional, Dict
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
# 也檢查 backend/data 目錄（兼容舊版本）
BACKEND_UPLOAD_DIR = BACKEND_DIR / "data" / "uploads"
BACKEND_RESULTS_DIR = BACKEND_DIR / "data" / "results"
DB_FILE = PROJECT_ROOT / "data" / "videos_db.json"  # 數據庫文件
JERSEY_MAPPINGS_FILE = PROJECT_ROOT / "data" / "jersey_mappings.json"  # 球衣號碼映射文件
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DB_FILE.parent, exist_ok=True)

# 球衣號碼映射存儲
jersey_mappings = {}  # {video_id: {track_id: jersey_number}}

def load_jersey_mappings():
    """載入球衣號碼映射"""
    global jersey_mappings
    if JERSEY_MAPPINGS_FILE.exists():
        try:
            with open(JERSEY_MAPPINGS_FILE, 'r', encoding='utf-8') as f:
                jersey_mappings = json.load(f)
            print(f"✅ 載入球衣號碼映射: {len(jersey_mappings)} 個視頻")
        except Exception as e:
            print(f"⚠️  載入映射失敗: {e}")
            jersey_mappings = {}
    else:
        jersey_mappings = {}

def save_jersey_mappings():
    """保存球衣號碼映射"""
    try:
        with open(JERSEY_MAPPINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(jersey_mappings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  保存映射失敗: {e}")

# 啟動時載入映射
load_jersey_mappings()

# 模擬數據庫 (實際應用中應使用PostgreSQL)
videos_db = []
analysis_tasks = {}

def load_videos_db():
    """從文件載入視頻數據庫"""
    global videos_db
    if DB_FILE.exists():
        try:
            with open(DB_FILE, 'r', encoding='utf-8') as f:
                videos_db = json.load(f)
            print(f"✅ 載入 {len(videos_db)} 個視頻記錄")
        except Exception as e:
            print(f"⚠️  載入數據庫失敗: {e}")
            videos_db = []
    else:
        videos_db = []

def save_videos_db():
    """保存視頻數據庫到文件"""
    try:
        with open(DB_FILE, 'w', encoding='utf-8') as f:
            json.dump(videos_db, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️  保存數據庫失敗: {e}")

def scan_existing_videos():
    """掃描 data 文件夾，自動恢復已存在的視頻記錄"""
    existing_ids = {v["id"] for v in videos_db}
    
    # 掃描 uploads 文件夾（檢查兩個可能的位置）
    upload_dirs = [UPLOAD_DIR]
    if BACKEND_UPLOAD_DIR.exists():
        upload_dirs.append(BACKEND_UPLOAD_DIR)
    
    for upload_dir in upload_dirs:
        if upload_dir.exists():
            for file_path in upload_dir.iterdir():
                if file_path.is_file() and file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.webm']:
                    # 從文件名提取 video_id（格式：{video_id}.{ext}）
                    video_id = file_path.stem
                    
                    if video_id not in existing_ids:
                        # 使用相對於 PROJECT_ROOT 的路徑
                        relative_path = str(file_path.relative_to(PROJECT_ROOT))
                        # 檢查是否有對應的結果文件（檢查兩個可能的位置）
                        results_file = RESULTS_DIR / f"{video_id}_results.json"
                        if not results_file.exists() and BACKEND_RESULTS_DIR.exists():
                            results_file = BACKEND_RESULTS_DIR / f"{video_id}_results.json"
                        
                        status = "completed" if results_file.exists() else "uploaded"
                        
                        # 嘗試從文件名中提取有意義的名稱（如果文件名是 UUID，使用默認名稱）
                        display_filename = file_path.name
                        # 如果文件名看起來像 UUID（36個字符，包含連字符），使用一個更友好的名稱
                        if len(file_path.stem) == 36 and file_path.stem.count('-') == 4:
                            # 使用文件大小和日期來生成一個友好的名稱
                            file_size_mb = file_path.stat().st_size / (1024 * 1024)
                            date_str = datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d')
                            display_filename = f"Video_{date_str}_{file_size_mb:.0f}MB{file_path.suffix}"
                        
                        video_data = {
                            "id": video_id,
                            "filename": display_filename,  # 使用更友好的文件名
                            "original_filename": display_filename,  # 如果沒有原始文件名，使用當前文件名
                            "file_path": relative_path,
                            "upload_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                            "status": status,
                            "file_size": file_path.stat().st_size
                        }
                        
                        # 如果數據庫中已經有這個視頻記錄，保留其 original_filename
                        existing_video = next((v for v in videos_db if v["id"] == video_id), None)
                        if existing_video and existing_video.get("original_filename"):
                            video_data["original_filename"] = existing_video["original_filename"]
                            video_data["filename"] = existing_video["original_filename"]  # 優先使用原始文件名
                        
                        if status == "completed":
                            video_data["analysis_time"] = datetime.fromtimestamp(results_file.stat().st_mtime).isoformat()
                        
                        videos_db.append(video_data)
                        existing_ids.add(video_id)
                        print(f"✅ 恢復視頻記錄: {file_path.name} ({status})")
    
    # 掃描 results 文件夾（檢查兩個可能的位置）
    results_dirs = [RESULTS_DIR]
    if BACKEND_RESULTS_DIR.exists():
        results_dirs.append(BACKEND_RESULTS_DIR)
    
    for results_dir in results_dirs:
        if results_dir.exists():
            for results_file in results_dir.iterdir():
                if results_file.is_file() and results_file.suffix == '.json':
                    video_id = results_file.stem.replace('_results', '')
                    
                    if video_id not in existing_ids:
                        # 檢查是否有對應的上傳文件（檢查兩個可能的位置）
                        upload_file = None
                        for upload_dir in upload_dirs:
                            if upload_dir.exists():
                                upload_file = upload_dir / f"{video_id}.mp4"
                                if not upload_file.exists():
                                    # 嘗試其他擴展名
                                    for ext in ['.avi', '.mov', '.mkv', '.webm']:
                                        upload_file = upload_dir / f"{video_id}{ext}"
                                        if upload_file.exists():
                                            break
                                if upload_file and upload_file.exists():
                                    break
                        
                        if upload_file and upload_file.exists():
                            relative_path = str(upload_file.relative_to(PROJECT_ROOT))
                            
                            # 嘗試從文件名中提取有意義的名稱
                            display_filename = upload_file.name
                            if len(upload_file.stem) == 36 and upload_file.stem.count('-') == 4:
                                file_size_mb = upload_file.stat().st_size / (1024 * 1024)
                                date_str = datetime.fromtimestamp(upload_file.stat().st_mtime).strftime('%Y-%m-%d')
                                display_filename = f"Video_{date_str}_{file_size_mb:.0f}MB{upload_file.suffix}"
                            
                            video_data = {
                                "id": video_id,
                                "filename": display_filename,
                                "original_filename": display_filename,  # 如果沒有原始文件名，使用當前文件名
                                "file_path": relative_path,
                                "upload_time": datetime.fromtimestamp(upload_file.stat().st_mtime).isoformat(),
                                "status": "completed",
                                "file_size": upload_file.stat().st_size,
                                "analysis_time": datetime.fromtimestamp(results_file.stat().st_mtime).isoformat()
                            }
                            
                            # 如果數據庫中已經有這個視頻記錄，保留其 original_filename
                            existing_video = next((v for v in videos_db if v["id"] == video_id), None)
                            if existing_video and existing_video.get("original_filename"):
                                video_data["original_filename"] = existing_video["original_filename"]
                                video_data["filename"] = existing_video["original_filename"]  # 優先使用原始文件名
                            
                            videos_db.append(video_data)
                            existing_ids.add(video_id)
                            print(f"✅ 恢復視頻記錄（從結果文件）: {upload_file.name}")
    
    # 保存更新後的數據庫
    if videos_db:
        save_videos_db()

# 啟動時載入數據
load_videos_db()
scan_existing_videos()

class VideoUpdateRequest(BaseModel):
    new_filename: str

class JerseyNumberMappingRequest(BaseModel):
    video_id: str
    track_id: int
    jersey_number: int
    frame: int  # 可選：標記時的幀號
    bbox: List[float]  # 可選：標記時的邊界框

class JerseyNumberMappingResponse(BaseModel):
    success: bool
    message: str
    mapping: Optional[Dict] = None

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
        original_filename = file.filename  # 保存原始文件名
        video_data = {
            "id": video_id,
            "filename": original_filename,  # 顯示用的文件名（使用原始文件名）
            "original_filename": original_filename,  # 原始文件名（永遠不會改變）
            "file_path": relative_path,  # 使用相對路徑
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "file_size": bytes_written
        }
        videos_db.append(video_data)
        save_videos_db()  # 保存到文件
        
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
        save_videos_db()  # 保存到文件
        
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
        # 檢查兩個可能的位置
        results_file = RESULTS_DIR / f"{video_id}_results.json"
        if not results_file.exists() and BACKEND_RESULTS_DIR.exists():
            results_file = BACKEND_RESULTS_DIR / f"{video_id}_results.json"
        
        if not results_file.exists():
            raise HTTPException(status_code=404, detail="分析結果不存在")
        
        with open(results_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"獲取結果失敗: {str(e)}")

@app.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """刪除視頻及其相關文件"""
    try:
        # 查找視頻
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if not video:
            raise HTTPException(status_code=404, detail="影片不存在")
        
        # 刪除視頻文件
        video_path = video.get("file_path")
        if video_path:
            # 確保路徑是絕對路徑
            if not os.path.isabs(video_path):
                video_path = str(PROJECT_ROOT / video_path)
            
            video_path = os.path.normpath(video_path)
            
            # 嘗試刪除視頻文件（如果存在）
            if os.path.exists(video_path):
                try:
                    os.remove(video_path)
                    print(f"✅ 已刪除視頻文件: {video_path}")
                except Exception as e:
                    print(f"⚠️  刪除視頻文件失敗: {e}")
            
            # 也嘗試刪除 backend/data 目錄中的文件（如果存在）
            backend_video_path = str(BACKEND_UPLOAD_DIR / os.path.basename(video_path))
            if os.path.exists(backend_video_path):
                try:
                    os.remove(backend_video_path)
                    print(f"✅ 已刪除備份視頻文件: {backend_video_path}")
                except Exception as e:
                    print(f"⚠️  刪除備份視頻文件失敗: {e}")
        
        # 刪除結果文件（檢查兩個可能的位置）
        results_file = RESULTS_DIR / f"{video_id}_results.json"
        if results_file.exists():
            try:
                results_file.unlink()
                print(f"✅ 已刪除結果文件: {results_file}")
            except Exception as e:
                print(f"⚠️  刪除結果文件失敗: {e}")
        
        backend_results_file = BACKEND_RESULTS_DIR / f"{video_id}_results.json"
        if backend_results_file.exists():
            try:
                backend_results_file.unlink()
                print(f"✅ 已刪除備份結果文件: {backend_results_file}")
            except Exception as e:
                print(f"⚠️  刪除備份結果文件失敗: {e}")
        
        # 從數據庫中移除
        videos_db[:] = [v for v in videos_db if v["id"] != video_id]
        save_videos_db()  # 保存到文件
        
        # 刪除相關的分析任務
        task_ids_to_remove = [task_id for task_id, task in analysis_tasks.items() if task.get("video_id") == video_id]
        for task_id in task_ids_to_remove:
            del analysis_tasks[task_id]
        
        return {
            "message": "視頻已成功刪除",
            "video_id": video_id
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除視頻失敗: {str(e)}")

@app.post("/videos/{video_id}/jersey-mapping")
async def set_jersey_mapping(video_id: str, request: JerseyNumberMappingRequest):
    """設置玩家球衣號碼映射（用戶手動標記）"""
    try:
        # 驗證視頻存在
        video = next((v for v in videos_db if v["id"] == video_id), None)
        if not video:
            raise HTTPException(status_code=404, detail="影片不存在")
        
        # 初始化該視頻的映射字典
        if video_id not in jersey_mappings:
            jersey_mappings[video_id] = {}
        
        # 保存映射：track_id -> jersey_number
        jersey_mappings[video_id][str(request.track_id)] = {
            "jersey_number": request.jersey_number,
            "frame": request.frame,
            "bbox": request.bbox,
            "timestamp": datetime.now().isoformat()
        }
        
        save_jersey_mappings()
        
        return {
            "success": True,
            "message": f"已設置追蹤ID {request.track_id} 的球衣號碼為 {request.jersey_number}",
            "mapping": jersey_mappings[video_id][str(request.track_id)]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"設置映射失敗: {str(e)}")

@app.get("/videos/{video_id}/jersey-mappings")
async def get_jersey_mappings(video_id: str):
    """獲取視頻的所有球衣號碼映射"""
    if video_id not in jersey_mappings:
        return {"mappings": {}}
    
    return {"mappings": jersey_mappings[video_id]}

@app.delete("/videos/{video_id}/jersey-mapping/{track_id}")
async def delete_jersey_mapping(video_id: str, track_id: str):
    """刪除球衣號碼映射"""
    try:
        if video_id in jersey_mappings and track_id in jersey_mappings[video_id]:
            del jersey_mappings[video_id][track_id]
            save_jersey_mappings()
            return {"success": True, "message": f"已刪除追蹤ID {track_id} 的映射"}
        else:
            raise HTTPException(status_code=404, detail="映射不存在")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"刪除映射失敗: {str(e)}")

@app.put("/videos/{video_id}")
async def update_video(video_id: str, request: VideoUpdateRequest):
    """更新視頻文件名"""
    video = next((v for v in videos_db if v["id"] == video_id), None)
    if not video:
        raise HTTPException(status_code=404, detail="影片不存在")
    
    # 更新顯示文件名，但保留 original_filename（如果存在）
    video["filename"] = request.new_filename
    # 如果沒有 original_filename，設置它為當前文件名（第一次設置）
    if "original_filename" not in video or not video.get("original_filename"):
        video["original_filename"] = request.new_filename
    save_videos_db()  # 保存到文件
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
        jersey_number_model = str(models_dir / "jersey_number_detection.pt")

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
                jersey_number_model_path=jersey_number_model if os.path.exists(jersey_number_model) else None,
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
            save_videos_db()  # 保存到文件
    
    except Exception as e:
        analysis_tasks[task_id]["status"] = "failed"
        analysis_tasks[task_id]["error"] = str(e)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

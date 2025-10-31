"""
排球分析系統 - Celery工作器
處理異步影片分析任務
"""

from celery import Celery
import os
import json
import sys
from pathlib import Path
from datetime import datetime
from processor import VolleyballAnalyzer

# 添加項目根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

# Celery配置
app = Celery('volleyball_analyzer')
app.conf.broker_url = 'redis://localhost:6379/0'
app.conf.result_backend = 'redis://localhost:6379/0'

# 任務配置
app.conf.task_serializer = 'json'
app.conf.accept_content = ['json']
app.conf.result_serializer = 'json'
app.conf.timezone = 'Asia/Taipei'
app.conf.enable_utc = True

# 模型路徑配置
BALL_MODEL_PATH = os.getenv('BALL_MODEL_PATH', '../models/VballNetV1_seq9_grayscale_148_h288_w512.onnx')
ACTION_MODEL_PATH = os.getenv('ACTION_MODEL_PATH', '../models/action_recognition_yv11m.pt')
DEVICE = os.getenv('DEVICE', 'cpu')

# 全局分析器實例
analyzer = None

def get_analyzer():
    """獲取分析器實例 (單例模式)"""
    global analyzer
    if analyzer is None:
        analyzer = VolleyballAnalyzer(
            ball_model_path=BALL_MODEL_PATH,
            action_model_path=ACTION_MODEL_PATH,
            device=DEVICE
        )
    return analyzer

@app.task(bind=True)
def analyze_video_task(self, video_id: str, video_path: str):
    """
    分析影片的Celery任務
    
    Args:
        video_id: 影片唯一ID
        video_path: 影片文件路徑
        
    Returns:
        分析結果字典
    """
    try:
        # 更新任務狀態
        self.update_state(
            state='PROGRESS',
            meta={'status': 'initializing', 'progress': 0}
        )
        
        # 獲取分析器
        analyzer = get_analyzer()
        
        # 檢查文件是否存在
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"影片文件不存在: {video_path}")
        
        # 更新狀態
        self.update_state(
            state='PROGRESS',
            meta={'status': 'analyzing', 'progress': 10}
        )
        
        # 執行分析
        results_dir = Path(__file__).parent.parent / "data" / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = results_dir / f"{video_id}_results.json"
        
        # 分析影片
        results = analyzer.analyze_video(video_path, str(output_path))
        
        # 添加任務信息
        results['task_id'] = self.request.id
        results['video_id'] = video_id
        results['status'] = 'completed'
        
        # 更新最終狀態
        self.update_state(
            state='SUCCESS',
            meta={
                'status': 'completed',
                'progress': 100,
                'results': results
            }
        )
        
        return results
        
    except Exception as e:
        # 更新錯誤狀態
        self.update_state(
            state='FAILURE',
            meta={
                'status': 'failed',
                'error': str(e),
                'progress': 0
            }
        )
        raise e

@app.task
def cleanup_old_files():
    """清理舊文件任務"""
    try:
        import time
        from datetime import datetime, timedelta
        
        # 清理超過7天的文件
        cutoff_time = time.time() - (7 * 24 * 60 * 60)
        
        uploads_dir = Path(__file__).parent.parent / "data" / "uploads"
        results_dir = Path(__file__).parent.parent / "data" / "results"
        
        cleaned_files = 0
        
        # 清理上傳文件
        if uploads_dir.exists():
            for file_path in uploads_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_files += 1
        
        # 清理結果文件
        if results_dir.exists():
            for file_path in results_dir.iterdir():
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    cleaned_files += 1
        
        return {
            'status': 'success',
            'cleaned_files': cleaned_files,
            'message': f'清理了 {cleaned_files} 個舊文件'
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

@app.task
def health_check():
    """健康檢查任務"""
    try:
        analyzer = get_analyzer()
        
        # 檢查模型是否載入
        ball_model_loaded = analyzer.ball_model is not None
        action_model_loaded = analyzer.action_model is not None
        
        return {
            'status': 'healthy',
            'ball_model_loaded': ball_model_loaded,
            'action_model_loaded': action_model_loaded,
            'device': DEVICE,
            'timestamp': str(datetime.now())
        }
        
    except Exception as e:
        return {
            'status': 'unhealthy',
            'error': str(e)
        }

if __name__ == '__main__':
    # 啟動Celery工作器
    app.start()

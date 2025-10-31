"""
排球分析系統 - AI處理核心
整合ball detection和action classification模型
"""

import cv2
import numpy as np
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import torch
import onnxruntime as ort
from ultralytics import YOLO
import time
import norfair

# 添加項目根目錄到路徑
sys.path.append(str(Path(__file__).parent.parent))

class VolleyballAnalyzer:
    """排球分析器 - 整合球追蹤和動作識別"""
    
    def __init__(self, 
                 ball_model_path: str = None,
                 action_model_path: str = None,
                 player_model_path: str = None,
                 device: str = "cpu"):
        """
        初始化分析器
        
        Args:
            ball_model_path: 球追蹤模型路徑 (ONNX格式)
            action_model_path: 動作識別模型路徑 (YOLO格式)
            device: 運行設備 ('cpu', 'cuda', 'mps')
        """
        self.device = device
        self.ball_model = None
        self.action_model = None
        self.player_model = None
        
        # 載入球追蹤模型
        if ball_model_path and os.path.exists(ball_model_path):
            self.load_ball_model(ball_model_path)
        
        # 載入動作識別模型
        if action_model_path and os.path.exists(action_model_path):
            self.load_action_model(action_model_path)

        # 載入球員偵測模型
        if player_model_path and os.path.exists(player_model_path):
            self.load_player_model(player_model_path)
        
        # 新增追蹤器實例
        self.tracker = norfair.Tracker(distance_function="euclidean", distance_threshold=50, initialization_delay=1, hit_counter_max=10)
    
    def load_ball_model(self, model_path: str):
        """載入球追蹤模型 (ONNX)"""
        try:
            self.ball_model = ort.InferenceSession(model_path)
            print(f"✅ 球追蹤模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 球追蹤模型載入失敗: {e}")
            self.ball_model = None
    
    def load_action_model(self, model_path: str):
        """載入動作識別模型 (YOLO)"""
        try:
            self.action_model = YOLO(model_path)
            print(f"✅ 動作識別模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 動作識別模型載入失敗: {e}")
            self.action_model = None

    def load_player_model(self, model_path: str):
        """載入球員偵測模型 (YOLOv8/YOLO 系列 .pt)"""
        try:
            self.player_model = YOLO(model_path)
            print(f"✅ 球員偵測模型載入成功: {model_path}")
        except Exception as e:
            print(f"❌ 球員偵測模型載入失敗: {e}")
            self.player_model = None
    
    def detect_ball(self, frame: np.ndarray) -> Optional[Dict]:
        """
        檢測球的位置
        
        Args:
            frame: 輸入幀 (BGR格式)
            
        Returns:
            球的位置信息或None
        """
        if self.ball_model is None:
            return None
        
        try:
            # 預處理幀
            input_frame = self.preprocess_ball_frame(frame)
            
            # 模型推理
            input_name = self.ball_model.get_inputs()[0].name
            output = self.ball_model.run(None, {input_name: input_frame})
            
            # 後處理結果
            ball_info = self.postprocess_ball_output(output, frame.shape)
            return ball_info
            
        except Exception as e:
            print(f"球檢測錯誤: {e}")
            return None
    
    def detect_actions(self, frame: np.ndarray) -> List[Dict]:
        """
        檢測球員動作
        
        Args:
            frame: 輸入幀 (BGR格式)
            
        Returns:
            動作檢測結果列表
        """
        if self.action_model is None:
            return []
        
        try:
            # YOLO模型推理
            results = self.action_model(frame, verbose=False)
            # 保證可迭代
            if not isinstance(results, (list, tuple)):
                results = [results]
            
            # 解析結果
            actions = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # 獲取邊界框座標
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # 獲取類別名稱
                        class_name = self.action_model.names[class_id]
                        
                        actions.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(confidence),
                            "class_id": class_id,
                            "action": class_name
                        })
            
            return actions
            
        except Exception as e:
            print(f"動作檢測錯誤: {e}")
            return []

    def detect_players(self, frame: np.ndarray) -> List[Dict]:
        """
        偵測球員框 (目標為人/球員)
        Returns: 每個偵測包含 {bbox, confidence, class_id, label}
        """
        if self.player_model is None:
            return []
        try:
            results = self.player_model(frame, verbose=False)
            players: List[Dict] = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                        label = self.player_model.names.get(class_id, "player") if hasattr(self.player_model, 'names') else "player"
                        players.append({
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": confidence,
                            "class_id": class_id,
                            "label": label
                        })
            return players
        except Exception as e:
            print(f"球員偵測錯誤: {e}")
            return []
    
    def preprocess_ball_frame(self, frame: np.ndarray) -> np.ndarray:
        """預處理球檢測幀，輸出形狀符合 VballNet: (1, 9, 288, 512) 灰階序列。
        目前以單幀複製9次作為替代，後續可接入滑動視窗。"""
        # 調整大小到 (W,H) = (512, 288)
        target_size = (512, 288)
        resized = cv2.resize(frame, target_size)
        # 轉灰階 (H,W)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # 正規化到 0-1，float32
        gray_f = gray.astype(np.float32) / 255.0
        # 疊成 9 個時間步: (9, H, W)
        seq = np.stack([gray_f] * 9, axis=0)
        # 添加 batch 維度 -> (1, 9, H, W)
        input_tensor = np.expand_dims(seq, axis=0).astype(np.float32)
        return input_tensor
    
    def postprocess_ball_output(self, output: List, frame_shape: Tuple) -> Optional[Dict]:
        """後處理球檢測輸出"""
        try:
            # 這裡需要根據您的具體模型輸出格式調整
            # 假設輸出包含球的位置和置信度
            predictions = output[0]  # 假設第一個輸出是預測結果
            
            # 找到最高置信度的檢測
            max_conf_idx = np.argmax(predictions[0, :, 4])  # 假設第5列是置信度
            max_confidence = predictions[0, max_conf_idx, 4]
            
            if max_confidence > 0.5:  # 置信度閾值
                x, y, w, h = predictions[0, max_conf_idx, :4]
                
                # 轉換回原始幀座標
                orig_h, orig_w = frame_shape[:2]
                x = int(x * orig_w)
                y = int(y * orig_h)
                w = int(w * orig_w)
                h = int(h * orig_h)
                
                return {
                    "center": [x, y],
                    "bbox": [x - w//2, y - h//2, x + w//2, y + h//2],
                    "confidence": float(max_confidence)
                }
            
            return None
            
        except Exception as e:
            print(f"球檢測後處理錯誤: {e}")
            return None
    
    def track_players(self, players):
        # players = [{bbox:..., confidence:...}]
        norfair_dets = []
        for d in players:
            cx = (d['bbox'][0]+d['bbox'][2])/2
            cy = (d['bbox'][1]+d['bbox'][3])/2
            norfair_dets.append(norfair.Detection(points=np.array([cx, cy]), scores=np.array([d['confidence']])))
        tracked = self.tracker.update(norfair_dets)
        output = []
        for t in tracked:
            # 預設20*20 bbox，實際可根據模型微調判斷
            est = t.estimate
            output.append({
                'id': int(t.id),
                'bbox': [float(est[0]-20), float(est[1]-20), float(est[0]+20), float(est[1]+20)],
                'confidence': float(max(t.last_detection.scores))
            })
        return output

    def _iou(self, boxA, boxB):
        # 標準IOU計算
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def assign_action_to_player(self, action_bbox, tracked_players):
        # action_bbox: [x1,y1,x2,y2]; tracked_players含id/bbox
        max_iou, player_id = 0, None
        for p in tracked_players:
            iou = self._iou(action_bbox, p['bbox'])
            if iou > max_iou:
                max_iou, player_id = iou, p['id']
        return player_id if max_iou > 0.2 else None

    def analyze_video(self, video_path: str, output_path: str = None) -> dict:
        """
        分析整個影片
        
        Args:
            video_path: 輸入影片路徑
            output_path: 輸出結果路徑
            
        Returns:
            分析結果字典
        """
        print(f"🎬 開始分析影片: {video_path}")
        
        # 打開影片
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"無法打開影片: {video_path}")
        
        # 獲取影片信息
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 影片信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 幀")
        
        # 初始化結果
        results = {
            "video_info": {
                "width": width,
                "height": height,
                "fps": fps,
                "total_frames": total_frames,
                "duration": total_frames / fps
            },
            "player_detection": {
                "detections": [],  # 每幀的球員偵測彙整
                "total_players_detected": 0
            },
            "ball_tracking": {
                "trajectory": [],
                "detected_frames": 0,
                "total_frames": total_frames
            },
            "action_recognition": {
                "actions": [],
                "action_counts": {},
                "total_actions": 0
            },
            " Ghost: 球員追蹤數據",
            "players_tracking": [],
            "scores": [],
            "game_states": [],  # 遊戲狀態（Play/No-Play/Timeout等）
            "analysis_time": time.time()
        }
        
        frame_count = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # ----- 球員偵測 + 追蹤 -----
                players = self.detect_players(frame)
                tracked_players = self.track_players(players)
                if tracked_players:
                    results["players_tracking"].append({
                        "frame": frame_count,
                        "timestamp": frame_count / fps,
                        "players": tracked_players
                    })
                    results["player_detection"]["total_players_detected"] += len(tracked_players)

                # ----- 球偵測 -----
                ball_info = self.detect_ball(frame)
                if ball_info:
                    results["ball_tracking"]["trajectory"].append({
                        "frame": frame_count,
                        "timestamp": frame_count / fps,
                        "center": ball_info["center"],
                        "bbox": ball_info["bbox"],
                        "confidence": ball_info["confidence"]
                    })
                    results["ball_tracking"]["detected_frames"] += 1
                
                # ----- 動作偵測並關聯球員id -----
                actions = self.detect_actions(frame)
                for action in actions:
                    pid = self.assign_action_to_player(action["bbox"], tracked_players)
                    action_data = {
                        "frame": frame_count,
                        "timestamp": frame_count / fps,
                        "bbox": action["bbox"],
                        "confidence": action["confidence"],
                        "action": action["action"],
                        "player_id": int(pid) if pid is not None else None
                    }
                    results["action_recognition"]["actions"].append(action_data)
                    # 若此action=得分，可加score event
                    if action["action"] in ["score", "spike_score", "attack_score"]:
                        results["scores"].append({
                            "player_id": action_data["player_id"],
                            "frame": frame_count,
                            "timestamp": frame_count / fps,
                            "score_type": action["action"]
                        })
                    
                    # 統計動作數量
                    action_name = action["action"]
                    if action_name not in results["action_recognition"]["action_counts"]:
                        results["action_recognition"]["action_counts"][action_name] = 0
                    results["action_recognition"]["action_counts"][action_name] += 1
                
                # ----- 簡單的遊戲狀態判斷：有動作時為Play，否則為No-Play -----
                # 這是一個簡化實現，實際可以根據動作類型、球位置等更精確判斷
                has_action = len(actions) > 0 or ball_info is not None
                current_state = "Play" if has_action else "No-Play"
                
                # 更新遊戲狀態（簡單邏輯：如果狀態改變，記錄新狀態段）
                if not results["game_states"] or results["game_states"][-1]["state"] != current_state:
                    results["game_states"].append({
                        "state": current_state,
                        "start_frame": frame_count,
                        "end_frame": frame_count,  # 將在下次狀態改變時更新
                        "start_timestamp": frame_count / fps,
                        "end_timestamp": frame_count / fps
                    })
                else:
                    # 更新當前狀態段的結束時間
                    results["game_states"][-1]["end_frame"] = frame_count
                    results["game_states"][-1]["end_timestamp"] = frame_count / fps
                
                # 進度顯示
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    print(f"⏳ 進度: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.1f}s")
        
        finally:
            cap.release()
        
        # 完成統計
        results["action_recognition"]["total_actions"] = len(results["action_recognition"]["actions"])
        results["analysis_time"] = time.time() - start_time
        
        print(f"✅ 分析完成!")
        print(f"⏱️  總耗時: {results['analysis_time']:.2f} 秒")
        print(f"👥 球員偵測: 總框數 {results['player_detection']['total_players_detected']}")
        print(f"⚽ 球追蹤: {results['ball_tracking']['detected_frames']}/{total_frames} 幀")
        print(f"🏐 動作識別: {results['action_recognition']['total_actions']} 個動作")
        
        # 保存結果
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"💾 結果已保存: {output_path}")
        
        return results

def main():
    """主函數 - 用於測試"""
    # 模型路徑 (請根據實際路徑調整)
    ball_model_path = "../models/VballNetV1_seq9_grayscale_148_h288_w512.onnx"
    action_model_path = "../models/action_recognition_yv11m.pt"
    player_model_path = "../models/player_detection_yv8.pt"
    
    # 創建分析器
    analyzer = VolleyballAnalyzer(
        ball_model_path=ball_model_path,
        action_model_path=action_model_path,
        player_model_path=player_model_path,
        device="cpu"
    )
    
    # 測試影片路徑
    test_video = "../data/test_video.mp4"
    if os.path.exists(test_video):
        results = analyzer.analyze_video(test_video, "../data/results.json")
        print("🎉 測試完成!")
    else:
        print("❌ 測試影片不存在，請提供有效的影片路徑")

if __name__ == "__main__":
    main()

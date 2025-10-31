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
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 只保留置信度 >= 0.6 的動作檢測
                        if confidence < 0.6:
                            continue
                        
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
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # 只保留置信度 >= 0.5 的球員檢測
                        if confidence < 0.5:
                            continue
                        
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
        """後處理球檢測輸出 - VballNet 模型輸出格式處理"""
        try:
            # VballNet 輸出格式檢查
            predictions = output[0] if output else None
            if predictions is None:
                return None
            
            # 檢查輸出形狀
            pred_shape = predictions.shape
            orig_h, orig_w = frame_shape[:2]
            
            # VballNet 可能輸出多種格式，嘗試常見格式
            # 格式1: (batch, height, width) - 熱力圖格式
            if len(pred_shape) == 3:
                # 形狀: (1, H, W) 或 (batch, H, W)
                heatmap = predictions[0] if pred_shape[0] == 1 else predictions
                if len(heatmap.shape) == 2:
                    # 找到熱力圖中的最大值位置
                    max_val = float(np.max(heatmap))
                    if max_val > 0.1:  # 降低置信度閾值到 0.1，提高檢測率
                        max_pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
                        # 轉換到原始座標（確保是標量）
                        y_norm = float(max_pos[0]) / float(heatmap.shape[0])
                        x_norm = float(max_pos[1]) / float(heatmap.shape[1])
                        x = int(x_norm * orig_w)
                        y = int(y_norm * orig_h)
                        
                        # 估算球的大小（假設為固定比例）
                        ball_size = min(orig_w, orig_h) * 0.02  # 約2%的畫面大小
                        w = h = int(ball_size)
                        
                        return {
                            "center": [x, y],
                            "bbox": [max(0, x - w//2), max(0, y - h//2), 
                                    min(orig_w, x + w//2), min(orig_h, y + h//2)],
                            "confidence": max_val
                        }
            
            # 格式2: (batch, num_detections, features) - 檢測框格式
            elif len(pred_shape) == 3 and pred_shape[2] >= 4:
                # 形狀: (1, N, 4) 或 (batch, N, 4) - 可能是 [x, y, w, h] 或 [x1, y1, x2, y2]
                detections = predictions[0] if pred_shape[0] == 1 else predictions
                # 找到置信度最高的檢測
                if pred_shape[2] >= 5:  # 包含置信度
                    confidences = detections[:, 4] if detections.shape[1] > 4 else None
                    if confidences is not None:
                        max_idx = np.argmax(confidences)
                        max_conf = float(confidences[max_idx])
                        if max_conf > 0.1:  # 降低置信度閾值
                            det = detections[max_idx]
                            # 假設格式為 [x, y, w, h] 或 [x1, y1, x2, y2]
                            if len(det) >= 4:
                                x1, y1, x2, y2 = float(det[0]), float(det[1]), float(det[2]), float(det[3])
                                # 如果看起來是 [x, y, w, h] 格式，轉換
                                if x2 < x1 or y2 < y1:
                                    w, h = x2, y2
                                    x1, y1 = x1 - w/2, y1 - h/2
                                    x2, y2 = x1 + w, y1 + h
                                
                                return {
                                    "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                                    "bbox": [max(0, int(x1)), max(0, int(y1)), 
                                            min(orig_w, int(x2)), min(orig_h, int(y2))],
                                    "confidence": max_conf
                                }
            
            # 如果以上格式都不匹配，嘗試找到任何非零值
            if len(pred_shape) >= 2:
                flat_pred = predictions.flatten()
                max_val = float(np.max(flat_pred))
                if max_val > 0.1:  # 降低閾值
                    max_idx = np.argmax(flat_pred)
                    # 嘗試從索引推斷位置
                    if len(pred_shape) == 2:
                        y_idx = max_idx // pred_shape[1]
                        x_idx = max_idx % pred_shape[1]
                        x = int((x_idx / pred_shape[1]) * orig_w)
                        y = int((y_idx / pred_shape[0]) * orig_h)
                        ball_size = min(orig_w, orig_h) * 0.02
                        w = h = int(ball_size)
                        
                        return {
                            "center": [x, y],
                            "bbox": [max(0, x - w//2), max(0, y - h//2), 
                                    min(orig_w, x + w//2), min(orig_h, y + h//2)],
                            "confidence": max_val
                        }
            
            # 如果無法識別格式，返回 None（靜默失敗，不影響其他檢測）
            return None
            
        except Exception as e:
            # 靜默處理錯誤，避免中斷整個分析流程
            # print(f"球檢測後處理錯誤: {e}")  # 取消註釋以調試
            return None
    
    def track_players(self, players):
        # players = [{bbox:..., confidence:...}]
        norfair_dets = []
        for d in players:
            # 確保座標是 Python 標量
            bbox = d['bbox']
            cx = (float(bbox[0]) + float(bbox[2])) / 2.0
            cy = (float(bbox[1]) + float(bbox[3])) / 2.0
            conf = float(d['confidence'])
            norfair_dets.append(norfair.Detection(points=np.array([cx, cy]), scores=np.array([conf])))
        tracked = self.tracker.update(norfair_dets)
        output = []
        for t in tracked:
            # 預設20*20 bbox，實際可根據模型微調判斷
            est = t.estimate
            
            # 確保轉換為 Python 標量 - 先將整個 est 轉為數組再取元素
            try:
                est_arr = np.asarray(est).flatten()
                if len(est_arr) >= 2:
                    est_x = float(est_arr[0])
                    est_y = float(est_arr[1])
                elif len(est_arr) == 1:
                    est_x = float(est_arr[0])
                    est_y = 0.0
                else:
                    est_x = est_y = 0.0
            except (AttributeError, TypeError, ValueError):
                # 如果轉換失敗，嘗試直接訪問
                try:
                    est_x = float(est[0]) if hasattr(est, '__getitem__') else 0.0
                    est_y = float(est[1]) if hasattr(est, '__getitem__') and len(est) > 1 else 0.0
                except (IndexError, TypeError, ValueError):
                    est_x = est_y = 0.0
            
            # 處理 scores - 確保是標量
            try:
                scores = t.last_detection.scores
                if isinstance(scores, np.ndarray):
                    scores_flat = scores.flatten()
                    max_score = float(np.max(scores_flat)) if len(scores_flat) > 0 else 0.0
                elif hasattr(scores, '__len__') and len(scores) > 0:
                    max_score = float(max(scores))
                else:
                    max_score = float(scores) if not hasattr(scores, '__len__') else 0.0
            except (AttributeError, TypeError, ValueError):
                max_score = 0.0
            
            output.append({
                'id': int(t.id),
                'bbox': [float(est_x-20), float(est_y-20), float(est_x+20), float(est_y+20)],
                'confidence': max_score
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
        if not tracked_players:
            return None
        
        max_iou, player_id = 0, None
        for p in tracked_players:
            if 'bbox' not in p or not p['bbox']:
                continue
            iou = self._iou(action_bbox, p['bbox'])
            if iou > max_iou:
                max_iou, player_id = iou, p['id']
        
        # 降低 IOU 閾值到 0.05，並考慮距離（如果 IOU 低但距離近也匹配）
        if max_iou > 0.05:
            return player_id
        
        # 如果 IOU 低，嘗試基於中心點距離匹配
        action_center = [(action_bbox[0] + action_bbox[2]) / 2, (action_bbox[1] + action_bbox[3]) / 2]
        min_distance = float('inf')
        closest_player_id = None
        
        for p in tracked_players:
            if 'bbox' not in p or not p['bbox']:
                continue
            player_bbox = p['bbox']
            player_center = [(player_bbox[0] + player_bbox[2]) / 2, (player_bbox[1] + player_bbox[3]) / 2]
            distance = ((action_center[0] - player_center[0])**2 + (action_center[1] - player_center[1])**2)**0.5
            
            # 計算動作框的對角線長度作為參考距離
            action_diagonal = ((action_bbox[2] - action_bbox[0])**2 + (action_bbox[3] - action_bbox[1])**2)**0.5
            
            # 如果距離小於動作框對角線的1.5倍，認為是匹配的
            if distance < action_diagonal * 1.5 and distance < min_distance:
                min_distance = distance
                closest_player_id = p['id']
        
        return closest_player_id

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
        
        # 獲取影片信息（確保轉換為 Python 標量）
        fps_raw = cap.get(cv2.CAP_PROP_FPS)
        fps = float(fps_raw) if not isinstance(fps_raw, (list, tuple, np.ndarray)) else float(fps_raw[0] if len(fps_raw) > 0 else 30.0)
        if fps <= 0:
            fps = 30.0  # 默認 FPS
        
        total_frames_raw = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        total_frames = int(float(total_frames_raw) if not isinstance(total_frames_raw, (list, tuple, np.ndarray)) else float(total_frames_raw[0] if len(total_frames_raw) > 0 else 0))
        
        width_raw = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        width = int(float(width_raw) if not isinstance(width_raw, (list, tuple, np.ndarray)) else float(width_raw[0] if len(width_raw) > 0 else 640))
        
        height_raw = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        height = int(float(height_raw) if not isinstance(height_raw, (list, tuple, np.ndarray)) else float(height_raw[0] if len(height_raw) > 0 else 360))
        
        print(f"📊 影片信息: {width}x{height}, {fps:.2f} FPS, {total_frames} 幀")
        
        # 初始化結果
        results = {
            "video_info": {
                "width": int(width),
                "height": int(height),
                "fps": float(fps),
                "total_frames": int(total_frames),
                "duration": float(total_frames) / float(fps)
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
            "players_tracking": [],  # 球員追蹤數據
            "scores": [],
            "game_states": [],  # 遊戲狀態（Play/No-Play/Timeout等）
            "analysis_time": time.time()
        }
        
        frame_count = 0
        start_time = time.time()
        
        # 動作合併追蹤：{(player_id, action_type): current_action_data}
        # current_action_data = {
        #   "start_frame": int,
        #   "end_frame": int,
        #   "start_timestamp": float,
        #   "end_timestamp": float,
        #   "bbox": [x1, y1, x2, y2],
        #   "max_confidence": float,
        #   "frame_count": int,  # 連續檢測到的幀數
        #   "last_seen_frame": int  # 最後一次檢測到的幀
        # }
        active_actions: Dict[Tuple[int, str], Dict] = {}
        
        # 動作合併參數
        MIN_ACTION_FRAMES = 3  # 最小動作持續時間（幀數）
        MAX_GAP_FRAMES = 5  # 最大間隔幀數（超過此幀數認為動作結束）
        
        def finalize_action(key: Tuple[int, str], current_frame: int, current_timestamp: float):
            """完成並保存一個動作"""
            if key not in active_actions:
                return
            
            action_data = active_actions[key]
            player_id, action_type = key
            
            # 只有當動作持續時間足夠長時才保存
            if action_data["frame_count"] >= MIN_ACTION_FRAMES:
                final_action = {
                    "frame": action_data["start_frame"],  # 使用開始幀
                    "timestamp": action_data["start_timestamp"],  # 使用開始時間
                    "end_frame": action_data["end_frame"],
                    "end_timestamp": action_data["end_timestamp"],
                    "bbox": action_data["bbox"],
                    "confidence": action_data["max_confidence"],  # 使用最大置信度
                    "action": action_type,
                    "player_id": player_id if player_id is not None else None,
                    "duration": action_data["end_timestamp"] - action_data["start_timestamp"]  # 動作持續時間
                }
                results["action_recognition"]["actions"].append(final_action)
                
                # 統計動作數量
                if action_type not in results["action_recognition"]["action_counts"]:
                    results["action_recognition"]["action_counts"][action_type] = 0
                results["action_recognition"]["action_counts"][action_type] += 1
                
                # 若此action=得分，可加score event
                if action_type in ["score", "spike_score", "attack_score"]:
                    results["scores"].append({
                        "player_id": player_id,
                        "frame": action_data["start_frame"],
                        "timestamp": action_data["start_timestamp"],
                        "score_type": action_type
                    })
            
            del active_actions[key]
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 確保 fps 是標量（在循環開始時計算一次）
                fps_scalar = float(fps)
                timestamp = float(frame_count) / fps_scalar
                
                # ----- 球員偵測 + 追蹤 -----
                players = self.detect_players(frame)
                tracked_players = self.track_players(players)
                if tracked_players:
                    results["players_tracking"].append({
                        "frame": int(frame_count),
                        "timestamp": timestamp,
                        "players": tracked_players
                    })
                    results["player_detection"]["total_players_detected"] += len(tracked_players)

                # ----- 球偵測 -----
                ball_info = self.detect_ball(frame)
                if ball_info:
                    results["ball_tracking"]["trajectory"].append({
                        "frame": int(frame_count),
                        "timestamp": timestamp,
                        "center": ball_info["center"],
                        "bbox": ball_info["bbox"],
                        "confidence": ball_info["confidence"]
                    })
                    results["ball_tracking"]["detected_frames"] += 1
                
                # ----- 動作偵測並關聯球員id，合併連續動作 -----
                actions = self.detect_actions(frame)
                detected_action_keys = set()
                
                for action in actions:
                    pid = self.assign_action_to_player(action["bbox"], tracked_players)
                    player_id = int(pid) if pid is not None else None
                    action_type = action["action"]
                    key = (player_id, action_type)
                    detected_action_keys.add(key)
                    
                    if key in active_actions:
                        # 更新現有動作：延長結束時間
                        active_actions[key]["end_frame"] = int(frame_count)
                        active_actions[key]["end_timestamp"] = timestamp
                        active_actions[key]["frame_count"] += 1
                        active_actions[key]["last_seen_frame"] = int(frame_count)
                        # 更新最大置信度和bbox（使用最新的）
                        if action["confidence"] > active_actions[key]["max_confidence"]:
                            active_actions[key]["max_confidence"] = action["confidence"]
                            active_actions[key]["bbox"] = action["bbox"]
                    else:
                        # 開始新動作
                        active_actions[key] = {
                            "start_frame": int(frame_count),
                            "end_frame": int(frame_count),
                            "start_timestamp": timestamp,
                            "end_timestamp": timestamp,
                            "bbox": action["bbox"],
                            "max_confidence": action["confidence"],
                            "frame_count": 1,
                            "last_seen_frame": int(frame_count)
                        }
                
                # 檢查並完成中斷的動作（超過最大間隔幀數沒有檢測到）
                keys_to_finalize = []
                for key in active_actions:
                    if key not in detected_action_keys:
                        gap = frame_count - active_actions[key]["last_seen_frame"]
                        if gap > MAX_GAP_FRAMES:
                            keys_to_finalize.append(key)
                
                for key in keys_to_finalize:
                    finalize_action(key, frame_count, timestamp)
                
                # ----- 簡單的遊戲狀態判斷：有動作時為Play，否則為No-Play -----
                # 這是一個簡化實現，實際可以根據動作類型、球位置等更精確判斷
                has_action = len(actions) > 0 or ball_info is not None
                current_state = "Play" if has_action else "No-Play"
                
                # 更新遊戲狀態（簡單邏輯：如果狀態改變，記錄新狀態段）
                if not results["game_states"] or results["game_states"][-1]["state"] != current_state:
                    results["game_states"].append({
                        "state": current_state,
                        "start_frame": int(frame_count),
                        "end_frame": int(frame_count),  # 將在下次狀態改變時更新
                        "start_timestamp": timestamp,
                        "end_timestamp": timestamp
                    })
                else:
                    # 更新當前狀態段的結束時間
                    results["game_states"][-1]["end_frame"] = int(frame_count)
                    results["game_states"][-1]["end_timestamp"] = timestamp
                
                # 進度顯示
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    elapsed = time.time() - start_time
                    print(f"⏳ 進度: {progress:.1f}% ({frame_count}/{total_frames}) - {elapsed:.1f}s")
            
            # 視頻處理完成，完成所有未完成的動作
            final_timestamp = float(frame_count) / fps_scalar if frame_count > 0 else 0.0
            for key in list(active_actions.keys()):
                finalize_action(key, frame_count, final_timestamp)
        
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

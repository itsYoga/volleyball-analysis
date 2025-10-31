#!/usr/bin/env python3
"""
排球分析系統 - 離線測試腳本
用於驗證模型串接流程
"""

import os
import sys
import argparse
import json
from pathlib import Path

# 添加AI核心到路徑
sys.path.append(str(Path(__file__).parent / "ai_core"))

from processor import VolleyballAnalyzer

def main():
    parser = argparse.ArgumentParser(description="排球分析系統離線測試")
    parser.add_argument("--video", required=True, help="輸入影片路徑")
    parser.add_argument("--ball-model", help="球追蹤模型路徑")
    parser.add_argument("--action-model", help="動作識別模型路徑")
    parser.add_argument("--player-model", help="球員偵測模型路徑")
    parser.add_argument("--output", help="輸出結果路徑")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"], help="運行設備")
    
    args = parser.parse_args()
    
    # 檢查輸入文件
    if not os.path.exists(args.video):
        print(f"❌ 影片文件不存在: {args.video}")
        return 1
    
    # 設置模型路徑
    ball_model_path = args.ball_model
    action_model_path = args.action_model
    player_model_path = args.player_model
    
    # 如果沒有指定模型路徑，嘗試從項目中找
    if not ball_model_path:
        # 嘗試從現有項目中複製模型
        source_ball_model = Path("../Volley_Vision/volleyball_capstone/models/VballNetV1_seq9_grayscale_148_h288_w512.onnx")
        if source_ball_model.exists():
            ball_model_path = str(source_ball_model)
            print(f"🔍 找到球追蹤模型: {ball_model_path}")
        else:
            print("⚠️  未找到球追蹤模型，將跳過球追蹤功能")
    
    if not action_model_path:
        # 嘗試從現有項目中複製模型
        source_action_model = Path("../Volley_Vision/volleyball_capstone/models/action_recognition_yv11m.pt")
        if source_action_model.exists():
            action_model_path = str(source_action_model)
            print(f"🔍 找到動作識別模型: {action_model_path}")
        else:
            print("⚠️  未找到動作識別模型，將跳過動作識別功能")

    if not player_model_path:
        source_player_model = Path("../Volley_Vision/volleyball_capstone/models/player_detection_yv8.pt")
        if source_player_model.exists():
            player_model_path = str(source_player_model)
            print(f"🔍 找到球員偵測模型: {player_model_path}")
        else:
            print("⚠️  未找到球員偵測模型，將跳過球員偵測功能")
    
    # 創建分析器
    print("🚀 初始化分析器...")
    analyzer = VolleyballAnalyzer(
        ball_model_path=ball_model_path,
        action_model_path=action_model_path,
        player_model_path=player_model_path,
        device=args.device
    )
    
    # 設置輸出路徑
    if args.output:
        output_path = args.output
    else:
        video_name = Path(args.video).stem
        output_path = f"{video_name}_analysis_results.json"
    
    # 執行分析
    print(f"🎬 開始分析影片: {args.video}")
    try:
        results = analyzer.analyze_video(args.video, output_path)
        
        print("\n" + "="*50)
        print("📊 分析結果摘要")
        print("="*50)
        
        # 影片信息
        video_info = results["video_info"]
        print(f"📹 影片信息:")
        print(f"   - 解析度: {video_info['width']}x{video_info['height']}")
        print(f"   - 幀率: {video_info['fps']:.2f} FPS")
        print(f"   - 總幀數: {video_info['total_frames']}")
        print(f"   - 時長: {video_info['duration']:.2f} 秒")
        
        # 球追蹤結果
        ball_tracking = results["ball_tracking"]
        print(f"\n⚽ 球追蹤結果:")
        print(f"   - 檢測幀數: {ball_tracking['detected_frames']}/{ball_tracking['total_frames']}")
        print(f"   - 檢測率: {ball_tracking['detected_frames']/ball_tracking['total_frames']*100:.1f}%")
        print(f"   - 軌跡點數: {len(ball_tracking['trajectory'])}")
        
        # 動作識別結果
        action_recognition = results["action_recognition"]
        print(f"\n🏐 動作識別結果:")
        print(f"   - 總動作數: {action_recognition['total_actions']}")
        print(f"   - 動作分布:")
        for action, count in action_recognition['action_counts'].items():
            print(f"     * {action}: {count} 次")
        
        # 性能信息
        print(f"\n⏱️  性能信息:")
        print(f"   - 分析耗時: {results['analysis_time']:.2f} 秒")
        print(f"   - 平均幀率: {video_info['total_frames']/results['analysis_time']:.2f} FPS")
        
        print(f"\n✅ 分析完成! 結果已保存到: {output_path}")
        return 0
        
    except Exception as e:
        print(f"❌ 分析失敗: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

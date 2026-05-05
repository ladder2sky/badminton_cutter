import sys
import os
# Set NUMBA_CACHE_DIR to a local directory to avoid permission errors in system directories
os.environ['NUMBA_CACHE_DIR'] = os.path.join(os.getcwd(), '.numba_cache')
if not os.path.exists(os.environ['NUMBA_CACHE_DIR']):
    os.makedirs(os.environ['NUMBA_CACHE_DIR'])

# Set ULTRALYTICS_CONFIG_DIR to local directory to avoid permission errors
local_config_dir = os.path.join(os.getcwd(), '.ultralytics_config')
os.environ['ULTRALYTICS_CONFIG_DIR'] = local_config_dir
os.environ['YOLO_CONFIG_DIR'] = local_config_dir
os.environ['ULTRALYTICS_NO_AUTOINSTALL'] = '1' # Disable auto-install of dependencies
if not os.path.exists(local_config_dir):
    os.makedirs(local_config_dir)

import cv2
import time
from typing import List, Dict

# 添加项目根目录到 python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.ai_engine.player_detector import PlayerDetector
from src.ai_engine.tracknet import ShuttlecockTracker
from src.ai_engine.tracknetv3 import ShuttlecockTrackerV3 # 导入 V3 类
from src.ai_engine.audio_analyzer import AudioAnalyzer
from src.decision.rally_analyzer import RallyAnalyzer
from src.input.video_processor import VideoProcessor
from src.output.video_cutter import VideoCutter
from src.utils.static_filter import StaticFilter
import argparse
import collections

class BadmintonCutterEngine:
    def __init__(self, video_path, output_dir, use_gpu=False, max_frames=None, start_time=0, end_time=None,
                  camera_pos='center', skip_frames=0, generate_video=False, save_screenshots=False,
                  model_version=1):
        self.video_path = video_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.max_frames = max_frames
        self.start_time = start_time
        self.end_time = end_time
        self.camera_pos = camera_pos
        self.skip_frames = skip_frames
        self.generate_video = generate_video
        self.save_screenshots = save_screenshots
        self.model_version = model_version
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        if self.save_screenshots:
            self.screenshot_dir = os.path.join(output_dir, "debug_screenshots")
            if not os.path.exists(self.screenshot_dir):
                os.makedirs(self.screenshot_dir)

        # 配置参数
        self.inference_size = (640, 360) # 推理分辨率
        self.frame_interval = 3 if not use_gpu else 1 # 跳帧策略
        
        # 0. 初始化模块
        device_str = 'cuda' if use_gpu else 'cpu'
        print(f"[Engine] Initializing modules on {device_str} using TrackNet V{model_version}...")
        self.processor = VideoProcessor(video_path, resize_dim=self.inference_size, 
                                        start_time=self.start_time, end_time=self.end_time)
        self.detector = PlayerDetector(device=device_str)
        
        # 根据模型版本选择权重和加载器
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if model_version == 3:
            tracknet_weights = os.path.join(base_dir, "weights", "TrackNet_best.pt")
            self.tracker = ShuttlecockTrackerV3(model_path=tracknet_weights, video_path=video_path, device=device_str)
            print(f"[Engine] TrackNetV3 loaded with weights: {tracknet_weights}")
        else:
            tracknet_weights = os.path.join(base_dir, "weights", "track.pt")
            self.tracker = ShuttlecockTracker(model_path=tracknet_weights, device=device_str)
            print(f"[Engine] TrackNetV1 loaded with weights: {tracknet_weights}")

        if not os.path.exists(tracknet_weights):
            print(f"[Warning] Weights not found at {tracknet_weights}!")
            
        self.audio_analyzer = AudioAnalyzer(video_path) 
        self.rally_analyzer = RallyAnalyzer(config={})
        self.static_filter = StaticFilter()
        self.frame_interval = 3  # 每3帧检测一次运动员
        
    def run(self):
        start_time_total = time.time()
        
        # 1. 音频分析 (可选)
        hit_events, cheer_events = [], []
        # 注意：BadmintonCutterEngine 没有 self.config 属性，应直接使用 self.model_version 判断
        if self.model_version == 1:
            print("[Engine] Analyzing audio events for V1...")
            hit_events = self.audio_analyzer.detect_hits()
            cheer_events = self.audio_analyzer.detect_cheers()
            
        self.rally_analyzer.initialize_debug_writer("debug_frames.csv", hit_events)
        
        # 2. 视频分析 (使用已有的 frame_gen 逻辑)
        frame_gen = self.processor.get_frame_generator()
        fps = self.processor.fps
        
        # 初始化预览视频写入
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.output_dir, "debug_preview.mp4")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, self.inference_size)
        
        video_events = [] 
        processed_count = 0

        print(f"[Engine] Starting video processing...")
        try:
            # 这里的 frame_gen 已经处理了跳帧和起始时间[cite: 6, 7]
            for original_frame, small_frame, frame_id in frame_gen:
                if self.max_frames and processed_count >= self.max_frames:
                    break

                # --- 核心推理逻辑 ---
                # 注意：tracker 是 self.tracker，不是外部变量
                if self.model_version == 3:
                    ball_pos, ball_conf, ball_speed = self.tracker.predict(small_frame)
                else:
                    # V1 只有 2 个返回值
                    ball_pos, ball_conf = self.tracker.predict(small_frame)
                    ball_speed = 0.0 # 手动补 0
                
                # 球员检测 (复用你原本的 self.detector)
                raw_detections = self.detector.detect(small_frame)
                player_count = len(raw_detections)
                
                # 构造数据
                event_data = {
                    "frame_id": frame_id,
                    "time": frame_id / fps,
                    "ball_pos": ball_pos,
                    "ball_conf": ball_conf,
                    "ball_speed": ball_speed,
                    "player_count": player_count
                }
                video_events.append(event_data)
                # --- 新增：将这一帧的数据实时写入 CSV ---
                self.rally_analyzer.write_debug_frame(event_data) 
                
                # 绘制预览图 (可选)
                debug_frame = self.detector.draw_detections(small_frame, raw_detections)
                if ball_pos:
                    cv2.circle(debug_frame, ball_pos, 5, (0, 0, 255), -1)
                out_writer.write(debug_frame)

                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} frames...")

        except Exception as e:
            print(f"Error during processing: {e}")
        
        # --- 释放资源 (这就是你 159, 160 行该待的地方) ---
        out_writer.release()
        self.processor.release()
        self.rally_analyzer.close_debug_writer()
        
        # 3. 综合决策
        print(f"[Engine] Analyzing rallies using V{self.model_version} logic...")

        if self.model_version == 3:
            # V3 逻辑：以视觉状态机为主，音频作为打分或边界辅助 (暂不作为状态切换核心)
            rallies = self.rally_analyzer.analyze_v3(video_events, fps=fps)
        else:
            # V1 逻辑：保留你原始的 analyze 方法，该方法强依赖 hit_events
            rallies = self.rally_analyzer.analyze(hit_events, video_events, cheer_events)
        
        print(f"\n[Result] Self-inspection (TrackNet V{self.model_version}):")
        for i, rally in enumerate(rallies):
            print(f"Rally #{i+1} | {rally.start_time:.2f}s - {rally.end_time:.2f}s | Score: {rally.score:.2f}")

        # 4. 生成视频
        if args.generate_video and rallies:
            clips_data = [(r.start_time, r.end_time) for r in rallies]
            cutter = VideoCutter(self.video_path, self.output_dir)
            cutter.cut_and_merge(clips_data, "highlights.mp4")
            
        total_time = time.time() - start_time_total
        print(f"Complete. Time: {total_time:.2f}s. Preview: {out_path}")

    def process_results(self, video_events, hit_events, cheer_events, start_time_total, preview_path):
        print(f"[Engine] Analysis phase...")
        
        # 根据模型版本选择分析算法
        if self.model_version == 3:
            rallies = self.rally_analyzer.analyze_v3(video_events)
        else:
            rallies = self.rally_analyzer.analyze(hit_events, video_events, cheer_events)
        
        # 后续裁剪逻辑 (VideoCutter)
        if rallies and self.generate_video:
            cutter = VideoCutter(self.video_path, self.output_dir)
            clips_data = [(r.start_time, r.end_time) for r in rallies]
            cutter.cut_and_merge(clips_data, "highlights.mp4")
            
        print(f"Complete. Total Time: {time.time() - start_time_total:.2f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Badminton Video Auto-Cutter")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--model-version", type=int, default=1, choices=[1, 3], help="TrackNet version (1 or 3)")
    parser.add_argument("--preview", action="store_true", default=True, help="Generate debug preview video")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process")
    parser.add_argument("--skip-frames", type=int, default=0, help="Skip frames at start")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds (e.g., --start-time 60.0 to skip first 60 seconds)")
    parser.add_argument("--end-time", type=float, default=None, help="End time in seconds (e.g., --end-time 300.0 to stop at 5 minutes). Default is video end.")
    parser.add_argument("--camera-pos", type=str, default="center", choices=["left", "center", "right"])
    parser.add_argument("--generate-video", action="store_true", help="Generate final highlights video")
    parser.add_argument("--save-screenshots", action="store_true", help="Save debug screenshots")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
    else:
        engine = BadmintonCutterEngine(
            args.video_path, 
            output_dir=args.output, 
            use_gpu=args.gpu, 
            max_frames=args.max_frames, 
            start_time=args.start_time,
            end_time=args.end_time,
            camera_pos=args.camera_pos,
            skip_frames=args.skip_frames,
            generate_video=args.generate_video,
            save_screenshots=args.save_screenshots,
            model_version=args.model_version
        )
        engine.run()
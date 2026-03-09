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
from src.ai_engine.audio_analyzer import AudioAnalyzer
from src.decision.rally_analyzer import RallyAnalyzer
from src.input.video_processor import VideoProcessor
from src.output.video_cutter import VideoCutter
from src.utils.static_filter import StaticFilter
import argparse
import collections

class BadmintonCutterEngine:
    def __init__(self, video_path, output_dir, use_gpu=False, max_frames=None, start_time=0, camera_pos='center', skip_frames=0, generate_video=False, save_screenshots=False):
        self.video_path = video_path
        self.output_dir = output_dir
        self.use_gpu = use_gpu
        self.max_frames = max_frames
        self.start_time = start_time
        self.camera_pos = camera_pos
        self.skip_frames = skip_frames
        self.generate_video = generate_video
        self.save_screenshots = save_screenshots
        
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
        print(f"[Engine] Initializing modules on {'gpu' if use_gpu else 'cpu'}...")
        self.processor = VideoProcessor(video_path, resize_dim=self.inference_size)
        self.detector = PlayerDetector(device='cuda' if use_gpu else 'cpu')
        
        # Load TrackNet weights
        tracknet_weights = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "weights", "track.pt")
        if not os.path.exists(tracknet_weights):
            print(f"[Warning] TrackNet weights not found at {tracknet_weights}. Using random weights!")
            tracknet_weights = None
            
        self.tracker = ShuttlecockTracker(model_path=tracknet_weights, device='cuda' if use_gpu else 'cpu')
        self.audio_analyzer = AudioAnalyzer(video_path) # 直接尝试从视频读取音频
        self.rally_analyzer = RallyAnalyzer(config={})
        
        # 初始化静态过滤器
        self.static_filter = StaticFilter()
        
        # 优化策略参数
        self.frame_interval = 3  # 每3帧检测一次运动员
        
    def run(self):
        start_time = time.time()
        
        # 1. 音频分析 (预处理)
        print("[Engine] Analyzing audio events...")
        hit_events = self.audio_analyzer.detect_hits()
        cheer_events = self.audio_analyzer.detect_cheers()
        print(f"[Engine] Detected {len(hit_events)} hits and {len(cheer_events)} cheer events.")
        
        # Initialize Debug CSV writer
        self.rally_analyzer.initialize_debug_writer("debug_frames.csv", hit_events)
        
        # 2. 视频分析 (逐帧)
        frame_gen = self.processor.get_frame_generator()
        last_frame = None
        
        # 结果可视化视频写入
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path = os.path.join(self.output_dir, "debug_preview.mp4")
        fps = self.processor.fps
        out_writer = cv2.VideoWriter(
            out_path, 
            fourcc, 
            fps, 
            self.inference_size
        )
        
        processed_count = 0
        total_frames = self.processor.total_frames
        
        # Calculate start frame based on start_time or skip_frames
        start_frame_idx = 0
        if hasattr(self, 'skip_frames') and self.skip_frames > 0:
            start_frame_idx = self.skip_frames
        if self.start_time > 0:
            start_frame_idx = max(start_frame_idx, int(self.start_time * fps))
            
        if start_frame_idx > 0:
            print(f"[Engine] Skipping first {start_frame_idx} frames...")
            # Consume generator to skip frames efficiently
            for _ in range(start_frame_idx):
                next(frame_gen, None)
            
        video_events = [] # 存储每帧的事件数据
        
        print(f"[Engine] Total frames to process: {total_frames}")
        
        last_detections = [] # Initialize last_detections
        
        try:
            for original_frame, small_frame, frame_id in frame_gen:
                
                # Check max frames
                if self.max_frames and processed_count >= self.max_frames:
                    print(f"[Engine] Reached max frames limit ({self.max_frames}). Stopping.")
                    break

                # 遮挡检测
                if self.processor.is_occluded(small_frame, last_frame):
                    print(f"Warning: Occlusion detected at frame {frame_id}")
                
                # detections = [] # Don't reset to empty, keep last detections
                ball_pos = None
                
                # 运动员检测 (根据策略跳帧)
                if frame_id % self.frame_interval == 0:
                    raw_detections = self.detector.detect(small_frame)
                    # 应用过滤逻辑 (Camera Pos + ROI + Size)
                    h, w = small_frame.shape[:2]
                    detections = self.detector.filter_players(raw_detections, self.camera_pos, w, h)
                    last_detections = detections
                else:
                    detections = last_detections # Reuse last detections
                
                # 羽毛球追踪 (必须逐帧调用以维护buffer)
                # TrackNet requires 3 consecutive frames. Skipping frames will break the temporal context.
                raw_pos, ball_conf = self.tracker.predict(small_frame)
                
                # 应用静态过滤器
                if raw_pos and hasattr(self, 'static_filter') and self.static_filter.is_static(raw_pos):
                    # print(f"Frame {frame_id}: Filtered static object at {raw_pos}")
                    raw_pos = None
                    ball_conf = 0.0 # 强制置信度归零
                
                # Check threshold
                ball_pos = None
                if raw_pos is not None and ball_conf > self.tracker.confidence_threshold:
                    ball_pos = raw_pos
                
                # if ball_pos:
                #     print(f"[Debug] Frame {frame_id}: Ball at {ball_pos}")
                
                # 收集事件数据
                event_data = {
                    "frame_id": frame_id,
                    "time": frame_id / fps,
                    "ball_pos": ball_pos,
                    "ball_conf": ball_conf,
                    "player_count": len(detections)
                }
                video_events.append(event_data)
                self.rally_analyzer.write_debug_frame(event_data)
                
                # 绘制结果 (仅用于调试)
                debug_frame = self.detector.draw_detections(small_frame, detections)
                
                if ball_pos:
                    cv2.circle(debug_frame, ball_pos, 5, (0, 0, 255), -1) # 画红点
                    cv2.putText(debug_frame, "Ball", (ball_pos[0]+5, ball_pos[1]), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                # 写入预览视频
                out_writer.write(debug_frame)
                
                # 截图保存逻辑
                if self.save_screenshots:
                    # Logic 1: Save high confidence frames (ball detected)
                    if ball_pos is not None and ball_conf > 0.5:
                         # Limit total screenshots to avoid spamming
                         current_screenshots = len(os.listdir(self.screenshot_dir))
                         if current_screenshots < 50: # Max 50 screenshots
                            screenshot_path = os.path.join(self.screenshot_dir, f"frame_{frame_id}_conf{ball_conf:.2f}.jpg")
                            cv2.imwrite(screenshot_path, debug_frame)
                            print(f"[Debug] Saved detection screenshot: {screenshot_path}")

                    # Logic 2: Save periodic screenshots for context (every 20 frames)
                    if processed_count % 20 == 0:
                         current_screenshots = len(os.listdir(self.screenshot_dir))
                         if current_screenshots < 50:
                            screenshot_path = os.path.join(self.screenshot_dir, f"frame_{frame_id}_periodic.jpg")
                            cv2.imwrite(screenshot_path, debug_frame)
                            print(f"[Debug] Saved periodic screenshot: {screenshot_path}")

                last_frame = small_frame
                processed_count += 1
                
                if processed_count % 100 == 0:
                    progress = (processed_count / total_frames) * 100 if total_frames > 0 else 0
                    print(f"Processed {processed_count}/{total_frames} frames ({progress:.1f}%)...")
        
        except KeyboardInterrupt:
            print("\n[Engine] Processing interrupted by user. Proceeding with collected data...")
            
        out_writer.release()
        self.processor.release()
        
        # 3. 综合决策
        print("[Engine] Analyzing rallies...")
        rallies = self.rally_analyzer.analyze(hit_events, video_events, cheer_events)
        
        print(f"\n[Result] Found {len(rallies)} rallies:")
        print("-" * 50)
        print("SELF-INSPECTION REPORT")
        print("-" * 50)
        
        for i, rally in enumerate(rallies):
            print(f"Rally #{i+1}")
            print(f"  Start: {rally.start_time:.2f}s")
            print(f"  End:   {rally.end_time:.2f}s")
            print(f"  Duration: {rally.end_time - rally.start_time:.2f}s")
            print(f"  Score: {rally.score:.2f}")
            
            # Check player count at start
            start_frames = [e for e in video_events if rally.start_time <= e['time'] <= rally.start_time + 1.0]
            start_players = sum(e['player_count'] for e in start_frames) / len(start_frames) if start_frames else 0
            print(f"  Avg Players (First 1s): {start_players:.2f}")
            
            # Check ball density
            rally_frames = [e for e in video_events if rally.start_time <= e['time'] <= rally.end_time]
            frames_with_ball = sum(1 for e in rally_frames if e['ball_pos'] is not None)
            density = frames_with_ball / len(rally_frames) if rally_frames else 0
            print(f"  Ball Density: {density:.2f}")
            print("-" * 30)

        # Wait for manual check? No, user said output results and wait for check.
        # But I cannot pause here. I will just print it and skip generation if user wants?
        # User said: "I hope you output self-inspection results for me to see, wait for my manual check, then do next step."
        # This implies stopping execution here or asking user.
        # But in this environment, I can't pause. I'll just stop here and NOT generate video if I find suspicious rallies?
        # Or I just print it and let the user decide in next turn.
        # To be safe and follow instructions: I will NOT generate video automatically this time, or generate it but warn user.
        # Actually, user said "wait for my manual check... then do next step". 
        # So I should PROBABLY STOP after printing report.
        
        print("[Engine] Self-inspection report generated.")
        print("[Engine] Please review the above report. If it looks correct, run with --generate-video to produce highlights.")
        print("[Engine] Stopping as requested for manual review.")
        
        # 4. 生成最终视频 (Optional / Conditional)
        clips_data = [] # Initialize clips_data
        if args.generate_video and rallies:
            clips_data = [(r.start_time, r.end_time) for r in rallies]
            print(f"[Engine] Generating highlights video with {len(clips_data)} clips...")
            cutter = VideoCutter(self.video_path, self.output_dir)
            cutter.cut_and_merge(clips_data, "highlights.mp4")
            print(f"Highlights video saved to: {os.path.join(self.output_dir, 'highlights.mp4')}")
        elif rallies:
             clips_data = [(r.start_time, r.end_time) for r in rallies]
             # Save clips data even if not generating video
             with open(os.path.join(self.output_dir, "clips.txt"), "w") as f:
                 for i, rally in enumerate(rallies):
                     f.write(f"{rally.start_time},{rally.end_time},{rally.score}\n")
             print("[Engine] Video generation skipped. Use --generate-video to create output.")
        else:
            print("[Engine] No rallies found.")
        
        total_time = time.time() - start_time
        print(f"Processing complete. Time taken: {total_time:.2f}s")
        print(f"Debug video saved to: {out_path}")
        if args.generate_video and clips_data:
            print(f"Highlights video saved to: {os.path.join(self.output_dir, 'highlights.mp4')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Badminton Video Auto-Cutter")
    parser.add_argument("video_path", type=str, help="Path to the input video file")
    parser.add_argument("--output", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--gpu", action="store_true", help="Enable GPU acceleration")
    parser.add_argument("--preview", action="store_true", default=True, help="Generate debug preview video")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (for debugging)")
    parser.add_argument("--skip-frames", type=int, default=0, help="Number of frames to skip at start")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--camera-pos", type=str, default="center", choices=["left", "center", "right"], help="Camera position relative to court (left/center/right)")
    parser.add_argument("--generate-video", action="store_true", help="Generate final highlights video (after inspection)")
    parser.add_argument("--save-screenshots", action="store_true", help="Save debug screenshots during processing")
    
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
            camera_pos=args.camera_pos,
            skip_frames=args.skip_frames,
            generate_video=args.generate_video,
            save_screenshots=args.save_screenshots
        )
        engine.run()

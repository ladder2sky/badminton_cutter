import os
import subprocess
from typing import List, Tuple
try:
    # 兼容 MoviePy 1.x
    from moviepy import VideoFileClip, concatenate_videoclips
except ImportError:
    # 兼容 MoviePy 2.x 及以上版本
    from moviepy.video.io.VideoFileClip import VideoFileClip
    from moviepy.video.compositing.concatenate import concatenate_videoclips

class VideoCutter:
    def __init__(self, video_path, output_dir="output"):
        self.video_path = video_path
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def cut_and_merge(self, segments: List[Tuple[float, float]], output_filename="highlights.mp4"):
        """
        根据给定的时间段剪辑视频并合并。
        优先尝试使用 ffmpeg 命令行进行无损剪辑 (Stream Copy)，如果失败则回退到 moviepy (重新编码)。
        """
        if not segments:
            print("[VideoCutter] No segments provided.")
            return

        final_output_path = os.path.join(self.output_dir, output_filename)
        
        # 尝试使用 ffmpeg 命令行 (无损且快速)
        if self._check_ffmpeg():
            print("[VideoCutter] Using ffmpeg for lossless cutting...")
            try:
                self._cut_with_ffmpeg(segments, final_output_path)
                print(f"[VideoCutter] Successfully saved to {final_output_path}")
                return
            except Exception as e:
                print(f"[VideoCutter] FFmpeg failed: {e}. Falling back to moviepy...")
        
        # 回退到 moviepy (较慢，会重新编码)
        print("[VideoCutter] Using moviepy (re-encoding)...")
        self._cut_with_moviepy(segments, final_output_path)
        print(f"[VideoCutter] Successfully saved to {final_output_path}")

    def _check_ffmpeg(self):
        """检查系统是否安装了 ffmpeg"""
        try:
            subprocess.run(["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _cut_with_ffmpeg(self, segments, output_path):
        """
        使用 ffmpeg 生成 concat 列表文件并合并。
        由于我们需要从原始视频中提取多个片段，最快的方法是：
        1. 对每个片段使用 -ss -to -c copy 提取临时文件
        2. 使用 concat demuxer 合并临时文件
        3. 删除临时文件
        """
        temp_files = []
        try:
            # 1. 提取片段
            for i, (start, end) in enumerate(segments):
                temp_filename = os.path.join(self.output_dir, f"temp_clip_{i:03d}.mp4")
                # 注意: -ss 放在 -i 前面以利用关键帧索引 (虽不精确但快)，或者放在后面 (精确但慢)
                # 为了精确剪辑，通常建议重新编码。Stream copy 只能在关键帧处切割，可能导致几秒的误差。
                # 鉴于我们需要精确到击球点，stream copy 可能不够精确。
                # 但为了速度，我们先尝试 stream copy，如果用户反馈不准，再改为重新编码。
                # 这里为了精确度，我们使用重新编码 (但只针对片段)，或者使用 smart encoding (复杂)。
                # 妥协方案：使用 -ss (放在 -i 前) + re-encoding (确保精确)
                # 或者使用 -ss (放在 -i 后) + copy (非常慢但精确)
                
                # 既然是高光集锦，稍微重新编码是可以接受的，且片段较短。
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", str(start),
                    "-i", self.video_path,
                    "-to", str(end - start), # -ss 后 -to 表示时长
                    "-c:v", "libx264", "-preset", "fast", "-crf", "22",
                    "-c:a", "aac",
                    temp_filename
                ]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                temp_files.append(temp_filename)
                print(f"[VideoCutter] Processed clip {i+1}/{len(segments)}")

            # 2. 创建 concat list
            list_path = os.path.join(self.output_dir, "concat_list.txt")
            with open(list_path, "w") as f:
                for tf in temp_files:
                    # ffmpeg concat list format: file 'path'
                    # Windows path compatibility
                    tf_abs = os.path.abspath(tf).replace("\\", "/")
                    f.write(f"file '{tf_abs}'\n")

            # 3. 合并
            cmd_concat = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-c", "copy", # 已经编码过的片段可以直接 copy
                output_path
            ]
            subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        finally:
            # 清理临时文件
            for tf in temp_files:
                if os.path.exists(tf):
                    os.remove(tf)
            if os.path.exists(list_path):
                os.remove(list_path)

    def _cut_with_moviepy(self, segments, output_path):
        """使用 moviepy 进行剪辑"""
        clips = []
        try:
            # 只需要加载一次源视频对象
            video = VideoFileClip(self.video_path)
            
            for start, end in segments:
                # subclip 可能会很慢，因为它需要解码
                clip = video.subclip(start, end)
                clips.append(clip)
            
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")
            
        finally:
            if 'video' in locals():
                video.close()
            for clip in clips:
                clip.close()

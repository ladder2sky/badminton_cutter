import cv2
import numpy as np

class VideoProcessor:
    """
    负责视频文件的读取、预处理（缩放、ROI裁剪）及基本信息获取。
    """
    def __init__(self, video_path, resize_dim=None, start_time=0.0, end_time=None):
        """
        初始化视频处理器。
        :param video_path: 视频文件路径
        :param resize_dim: (width, height) 如果设置，输出帧将缩放到此尺寸
        :param start_time: 开始处理的时间（秒），默认从视频开头开始
        :param end_time: 结束处理的时间（秒），默认为视频末尾
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        # 获取视频元数据
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        self.resize_dim = resize_dim
        self.start_time = start_time
        self.end_time = end_time if end_time is not None else self.duration
         
        print(f"[VideoProcessor] Video loaded: {self.width}x{self.height} @ {self.fps:.2f}fps, Duration: {self.duration:.2f}s")
        print(f"[VideoProcessor] Processing range: {self.start_time:.2f}s - {self.end_time:.2f}s")

    def get_frame_generator(self):
        """
        生成器：逐帧返回图像。
        :yield: (original_frame, processed_frame, frame_id)
                original_frame: 原始分辨率图像
                processed_frame: 缩放后的图像 (用于推理)
                frame_id: 当前帧号（相对于视频开头的绝对帧号）
        """
        # 计算开始和结束对应的帧号
        start_frame = int(self.start_time * self.fps)
        end_frame = int(self.end_time * self.fps)
        
        # 跳转到起始帧
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_id = start_frame
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 如果超过了结束帧，停止处理
            if frame_id >= end_frame:
                break
            
            processed_frame = frame
            if self.resize_dim:
                processed_frame = cv2.resize(frame, self.resize_dim)
                
            yield frame, processed_frame, frame_id
            frame_id += 1
            
    def is_occluded(self, current_frame, last_frame, threshold=50):
        """
        简单的遮挡检测：计算帧间像素差异。
        :param current_frame: 当前帧
        :param last_frame: 上一帧
        :param threshold: 差异阈值
        :return: Boolean 是否遮挡
        """
        if last_frame is None:
            return False
            
        # 转灰度加速计算
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差
        diff = cv2.absdiff(curr_gray, last_gray)
        mean_diff = np.mean(diff)
        
        # 如果平均像素差异过大，可能意味着镜头被遮挡或场景剧烈切换
        return mean_diff > threshold

    def release(self):
        self.cap.release()

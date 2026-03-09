import librosa
import numpy as np
import scipy.signal

class AudioAnalyzer:
    """
    负责从视频中提取音频并识别关键事件（击球声、欢呼声）。
    """
    def __init__(self, audio_path, sample_rate=16000):
        """
        初始化音频分析器。
        :param audio_path: 音频文件路径 (通常是提取出的 wav)
        :param sample_rate: 采样率，默认16kHz
        """
        self.sample_rate = sample_rate
        try:
            # 加载音频 (mono=True 强制单声道)
            self.y, self.sr = librosa.load(audio_path, sr=sample_rate, mono=True)
            self.duration = librosa.get_duration(y=self.y, sr=self.sr)
            print(f"[AudioAnalyzer] Loaded audio: {self.duration:.2f}s")
        except Exception as e:
            print(f"[AudioAnalyzer] Error loading audio: {e}")
            self.y = None
            self.sr = sample_rate

    def detect_hits(self, threshold_energy=0.04, min_interval=0.3):
        """
        检测击球声 (Hit Detection)。
        基于短时能量和频谱特征。
        :param threshold_energy: 能量阈值
        :param min_interval: 最小击球间隔 (秒)
        :return: 击球时间戳列表 [t1, t2, ...]
        """
        if self.y is None:
            return []

        # 计算短时能量 (RMSE)
        hop_length = 512
        frame_length = 1024
        rms = librosa.feature.rms(y=self.y, frame_length=frame_length, hop_length=hop_length)[0]
        
        # 获取时间轴
        times = librosa.times_like(rms, sr=self.sr, hop_length=hop_length)
        
        # 寻找峰值
        peaks, _ = scipy.signal.find_peaks(rms, height=threshold_energy, distance=int(min_interval * self.sr / hop_length))
        
        hit_times = times[peaks]
        
        # 进一步过滤：击球声通常具有高频分量
        # 这里可以使用 spectral_centroid 或 zero_crossing_rate 进行二次确认
        # 简化版仅使用 RMS
        
        return hit_times.tolist()

    def detect_cheers(self, threshold_energy=0.05, min_duration=1.0):
        """
        检测欢呼声 (Cheers Detection)。
        持续的高能量片段。
        :return: 欢呼片段列表 [(start, end), ...]
        """
        if self.y is None:
            return []
            
        hop_length = 512
        rms = librosa.feature.rms(y=self.y, frame_length=1024, hop_length=hop_length)[0]
        times = librosa.times_like(rms, sr=self.sr, hop_length=hop_length)
        
        # 简单的阈值判定
        is_loud = rms > threshold_energy
        
        # 寻找连续的 True 片段
        cheer_segments = []
        start_idx = None
        
        for i, val in enumerate(is_loud):
            if val and start_idx is None:
                start_idx = i
            elif not val and start_idx is not None:
                duration = times[i] - times[start_idx]
                if duration >= min_duration:
                    cheer_segments.append((times[start_idx], times[i]))
                start_idx = None
                
        return cheer_segments

    def get_energy_at_time(self, time_sec):
        """
        获取指定时间点的音频能量。
        """
        if self.y is None:
            return 0.0
        
        idx = int(time_sec * self.sr)
        if idx < 0 or idx >= len(self.y):
            return 0.0
            
        # 取周围一小段的 RMS
        start = max(0, idx - 512)
        end = min(len(self.y), idx + 512)
        segment = self.y[start:end]
        return np.sqrt(np.mean(segment**2))

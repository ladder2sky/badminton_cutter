import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import collections
import torch
import torch.nn as nn


# ---------------------------------------------------------
# 1. 模型定义 (务必从 GitHub 仓库复制完整的 TrackNetV3 类)
# ---------------------------------------------------------
class Conv2DBlock(nn.Module):
    """ Conv2D + BN + ReLU """
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv2DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding='same', bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Double2DConv(nn.Module):
    """ Conv2DBlock x 2 """
    def __init__(self, in_dim, out_dim):
        super(Double2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x
    
class Triple2DConv(nn.Module):
    """ Conv2DBlock x 3 """
    def __init__(self, in_dim, out_dim):
        super(Triple2DConv, self).__init__()
        self.conv_1 = Conv2DBlock(in_dim, out_dim)
        self.conv_2 = Conv2DBlock(out_dim, out_dim)
        self.conv_3 = Conv2DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        return x

class TrackNetV3(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TrackNetV3, self).__init__()
        self.down_block_1 = Double2DConv(in_dim, 64)
        self.down_block_2 = Double2DConv(64, 128)
        self.down_block_3 = Triple2DConv(128, 256)
        self.bottleneck = Triple2DConv(256, 512)
        self.up_block_1 = Triple2DConv(768, 256)
        self.up_block_2 = Double2DConv(384, 128)
        self.up_block_3 = Double2DConv(192, 64)
        self.predictor = nn.Conv2d(64, out_dim, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_block_1(x)                                       # (N,   64,  288,   512)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x1)                     # (N,   64,  144,   256)
        x2 = self.down_block_2(x)                                       # (N,  128,  144,   256)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x2)                     # (N,  128,   72,   128)
        x3 = self.down_block_3(x)                                       # (N,  256,   72,   128)
        x = nn.MaxPool2d((2, 2), stride=(2, 2))(x3)                     # (N,  256,   36,    64)
        x = self.bottleneck(x)                                          # (N,  512,   36,    64)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x3], dim=1)      # (N,  768,   72,   128)
        x = self.up_block_1(x)                                          # (N,  256,   72,   128)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x2], dim=1)      # (N,  384,  144,   256)
        x = self.up_block_2(x)                                          # (N,  128,  144,   256)
        x = torch.cat([nn.Upsample(scale_factor=2)(x), x1], dim=1)      # (N,  192,  288,   512)
        x = self.up_block_3(x)                                          # (N,   64,  288,   512)
        x = self.predictor(x)                                           # (N,    3,  288,   512)
        x = self.sigmoid(x)                                             # (N,    3,  288,   512)
        return x

    
class Conv1DBlock(nn.Module):
    """ Conv1D + LeakyReLU"""
    def __init__(self, in_dim, out_dim, **kwargs):
        super(Conv1DBlock, self).__init__(**kwargs)
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size=3, padding='same', bias=True)
        self.relu = nn.LeakyReLU()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class Double1DConv(nn.Module):
    """ Conv1DBlock x 2"""
    def __init__(self, in_dim, out_dim):
        super(Double1DConv, self).__init__()
        self.conv_1 = Conv1DBlock(in_dim, out_dim)
        self.conv_2 = Conv1DBlock(out_dim, out_dim)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x

class InpaintNet(nn.Module):
    def __init__(self):
        super(InpaintNet, self).__init__()
        self.down_1 = Conv1DBlock(3, 32)
        self.down_2 = Conv1DBlock(32, 64)
        self.down_3 = Conv1DBlock(64, 128)
        self.buttleneck = Double1DConv(128, 256)
        self.up_1 = Conv1DBlock(384, 128)
        self.up_2 = Conv1DBlock(192, 64)
        self.up_3 = Conv1DBlock(96, 32)
        self.predictor = nn.Conv1d(32, 2, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, m):
        x = torch.cat([x, m], dim=2)                                   # (N,   L,   3)
        x = x.permute(0, 2, 1)                                         # (N,   3,   L)
        x1 = self.down_1(x)                                            # (N,  16,   L)
        x2 = self.down_2(x1)                                           # (N,  32,   L)
        x3 = self.down_3(x2)                                           # (N,  64,   L)
        x = self.buttleneck(x3)                                        # (N,  256,  L)
        x = torch.cat([x, x3], dim=1)                                  # (N,  384,  L)
        x = self.up_1(x)                                               # (N,  128,  L)
        x = torch.cat([x, x2], dim=1)                                  # (N,  192,  L)
        x = self.up_2(x)                                               # (N,   64,  L)
        x = torch.cat([x, x1], dim=1)                                  # (N,   96,  L)
        x = self.up_3(x)                                               # (N,   32,  L)
        x = self.predictor(x)                                          # (N,   2,   L)
        x = self.sigmoid(x)                                            # (N,   2,   L)
        x = x.permute(0, 2, 1)                                         # (N,   L,   2)
        return x

# ---------------------------------------------------------
# 2. 背景提取器 (解决边线丢球的关键)[cite: 2, 3]
# --------------------------------------------------------- 
class BackgroundExtractor:
    """
    计算视频的中值背景，通过差分去除静止的白色边线[cite: 3]
    """
    def __init__(self, video_path):
        self.video_path = video_path
        self.background = self._generate_background()

    def _generate_background(self):
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 均匀采样 100-150 帧，足以计算出稳定的背景
        sample_indices = np.linspace(0, total_frames - 1, 100, dtype=int)
        frames = []
        for idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(cv2.resize(frame, (512, 288)))
        cap.release()
        
        # 取中值：人会消失，边线被保留[cite: 3]
        return np.median(frames, axis=0).astype(np.uint8)

    def get_residual(self, frame):
        # 将当前帧减去背景，白色边线会变黑，移动的白球会变亮
        resized = cv2.resize(frame, (512, 288))
        diff = cv2.absdiff(resized, self.background)
        return diff 
    
class ShuttlecockTrackerV3:
    def __init__(self, model_path, video_path, device='cuda', confidence_threshold=0.6):
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.pos_history = collections.deque(maxlen=10) # 存储最近5帧的真实坐标
        # 修正：参数名改为 in_dim 以匹配你的 TrackNetV3 类定义
        self.model = TrackNetV3(in_dim=27, out_dim=8).to(self.device)
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 核心修正：根据你的权重结构提取 'model'
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            # 这里的备选方案是为了兼容其他可能的权重格式
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        # 自动初始化背景提取
        print("正在提取视频背景...")
        bg_extractor = BackgroundExtractor(video_path)
        self.background = bg_extractor.background
        
        self.frame_buffer = [] 

    def predict(self, frame):
        """
        [纯净版] 适配 27通道输入推理，去除了物理过滤逻辑，用于观察真实噪点数据。
        """
        if frame is None:
            return None, 0.0, 0.0

        orig_h, orig_w = frame.shape[:2]
        
        # 1. 预处理：缩放到模型标准尺寸 (512, 288)
        resized = cv2.resize(frame, (512, 288))
        
        # 2. 归一化 (0-255 -> 0-1)
        img_norm = resized.astype(np.float32) / 255.0
        
        # 3. 维护 9 帧滑动窗口 (27通道要求 9 帧 RGB)
        self.frame_buffer.append(img_norm)
        if len(self.frame_buffer) > 9:
            self.frame_buffer.pop(0)
            
        # 如果缓冲区没满 9 帧，无法进行推理
        if len(self.frame_buffer) < 9:
            return None, 0.0, 0.0

        # 4. 构造 27 通道输入
        combined_input = np.concatenate(self.frame_buffer, axis=-1)
        input_data = np.transpose(combined_input, (2, 0, 1)) 
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).to(self.device)

        # 5. 推理与解析
        with torch.no_grad():
            output = self.model(input_tensor) 
            heatmap = torch.max(output[0], dim=0)[0].cpu().numpy() 
            
            y_idx, x_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
            confidence = float(heatmap[y_idx, x_idx])
            
            # 将坐标从 512x288 映射回视频原始尺寸
            cx = int(x_idx * orig_w / 512)
            cy = int(y_idx * orig_h / 288)

            speed = 0.0
            current_pos = (cx, cy)

            # --- 修改说明：移除物理拦截逻辑 ---
            # 保留了基础置信度阈值判定，防止 CSV 被海量无意义的背景噪点填满。
            # 如果你希望观察连低置信度的噪点也记录下来，可以将 threshold 改为 0.0
            if confidence >= self.confidence_threshold:
                
                # 计算与上一帧检测的距离（仅用于观察 speed 输出）
                if len(self.pos_history) > 0:
                    prev_pos = self.pos_history[-1]
                    speed = np.sqrt((cx - prev_pos[0])**2 + (cy - prev_pos[1])**2)
                
                # 无论 speed 多大，都强制记录坐标并更新历史
                self.pos_history.append(current_pos)
                return current_pos, confidence, speed

            # 置信度不足时
            return None, 0.0, 0.0
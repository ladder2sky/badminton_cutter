
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )
        
    def forward(self, x):
        return self.block(x)

class TrackNet(nn.Module):
    """
    TrackNet V1 Architecture (Unofficial PyTorch)
    Matches weights from yastrebksv/TrackNet
    """
    def __init__(self, in_channels=9, out_channels=256):
        super(TrackNet, self).__init__()
        
        # Encoder Block 1
        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 2
        self.conv3 = ConvBlock(64, 128)
        self.conv4 = ConvBlock(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Encoder Block 3
        self.conv5 = ConvBlock(128, 256)
        self.conv6 = ConvBlock(256, 256)
        self.conv7 = ConvBlock(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # Center Block
        self.conv8 = ConvBlock(256, 512)
        self.conv9 = ConvBlock(512, 512)
        self.conv10 = ConvBlock(512, 512)
        
        # Decoder Block 1
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv11 = ConvBlock(512, 256)
        self.conv12 = ConvBlock(256, 256)
        self.conv13 = ConvBlock(256, 256)
        
        # Decoder Block 2
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv14 = ConvBlock(256, 128)
        self.conv15 = ConvBlock(128, 128)
        
        # Decoder Block 3
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv16 = ConvBlock(128, 64)
        self.conv17 = ConvBlock(64, 64)
        
        # Output Layer
        self.conv18 = ConvBlock(64, out_channels)
        
    def forward(self, x):
        # Encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        
        # Center
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        
        # Decoder
        x = self.upsample1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        
        x = self.upsample2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        
        x = self.upsample3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        
        x = self.conv18(x)
        return x

class ShuttlecockTracker:
    def __init__(self, model_path=None, device='cpu', confidence_threshold=0.2):
        self.device = torch.device(device)
        self.model = TrackNet().to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
        if model_path:
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict) # Strict=True by default
                print(f"[TrackNet] Loaded weights from {model_path}")
            except Exception as e:
                print(f"[TrackNet] Error loading weights: {e}")
                # Don't fail silently anymore, let the user know or raise error if critical
                # For now, we print error. The verify script will catch this.
        
        self.frame_buffer = []

    def predict(self, frame):
        """
        输入单帧图像，返回预测的羽毛球坐标 (x, y)。
        需要累积 3 帧连续图像才能执行一次有效预测。
        """
        if frame is None:
            return None, 0.0

        # 1. 记录原始尺寸和模型目标尺寸
        orig_h, orig_w = frame.shape[:2]
        target_w, target_h = 512, 288
        
        # 2. 图像预处理
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 缩放到 TrackNet 标准输入尺寸 (512x288)
        resized = cv2.resize(frame_rgb, (target_w, target_h))
        
        # 归一化到 [0, 1]
        processed = resized.astype(np.float32) / 255.0
        
        # HWC -> CHW (3, 288, 512)
        processed = np.transpose(processed, (2, 0, 1))
        
        # 3. 维护 3 帧滑动窗口缓冲区
        self.frame_buffer.append(processed)
        
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
            
        if len(self.frame_buffer) < 3:
            # 缓冲区未满时返回空结果
            return None, 0.0
            
        # 4. 模型推理
        # 将 3 帧沿通道维度合并 -> (9, 288, 512)
        input_tensor = np.concatenate(self.frame_buffer, axis=0)
        # 添加 Batch 维度并发送到设备 -> (1, 9, 288, 512)
        input_tensor_torch = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor_torch) # (1, 256, 288, 512)
            
        # --- 修复逻辑开始 ---
        # 1. 使用 Softmax 将输出转为概率分布 (沿通道维度)
        probs = torch.softmax(output, dim=1) # (1, 256, 288, 512)
        
        # 2. 我们关注代表“球”的特征层。
        # 在该模型中，通常索引越高代表越接近球心。
        # 我们取概率加权后的均值，或者直接取除背景外(索引>0)的最大概率映射
        # 这里采用一种更鲁棒的方法：提取背景层(index 0)的相反面
        ball_map = 1.0 - probs[0, 0, :, :].cpu().numpy() # (288, 512)
        
        # 3. 在这个 ball_map 中找最大值
        y_idx, x_idx = np.unravel_index(ball_map.argmax(), ball_map.shape)
        max_prob = ball_map[y_idx, x_idx]
        
        # 这里的 max_prob 才是真正的置信度 (0.0 - 1.0)
        confidence = float(max_prob)
        
        # 4. 坐标映射
        cx = int(x_idx * orig_w / target_w)
        cy = int(y_idx * orig_h / target_h)
        # --- 修复逻辑结束 ---

        return (cx, cy), confidence
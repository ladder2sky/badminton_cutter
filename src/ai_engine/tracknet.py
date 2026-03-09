
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
    def __init__(self, model_path=None, device='cpu', confidence_threshold=0.3):
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
        Input single frame, returns predicted shuttlecock coordinates (x, y).
        Requires 3 accumulated frames to perform one prediction.
        """
        # Preprocessing: Use input frame size directly
        # Ensure dimensions are divisible by 32 (TrackNet has 3 pools => 8, but let's be safe for 32)
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to 512x288 (Standard TrackNet Input)
        resized = cv2.resize(frame_rgb, (512, 288))
        
        # Normalize (Standard ImageNet normalization)
        processed = resized.astype(np.float32) / 255.0
        # mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        # std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # processed = (processed - mean) / std
        
        # HWC -> CHW (3, 288, 512)
        processed = np.transpose(processed, (2, 0, 1))
        
        self.frame_buffer.append(processed)
        
        if len(self.frame_buffer) > 3:
            self.frame_buffer.pop(0)
            
        if len(self.frame_buffer) < 3:
                    return None, 0.0
            
        # Concatenate 3 frames along channel dimension -> (9, 288, 512)
        input_tensor = np.concatenate(self.frame_buffer, axis=0)
        # Add batch dimension -> (1, 9, 288, 512)
        input_tensor_torch = torch.from_numpy(input_tensor).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor_torch) # (1, 256, 288, 512)
            
        # Output is (1, 256, H, W). We want the class with highest probability per pixel.
        # Since it's likely a regression-via-classification (0-255), 
        # the 'class index' IS the heatmap intensity.
        # So we take argmax along dim 1.
        heatmap_class = torch.argmax(output, dim=1).float() # (1, 288, 512)
        heatmap = heatmap_class[0].cpu().numpy() # (288, 512), values 0-255
        
        # Post-processing
        # Find position of max value in the heatmap
        # Use unravel_index to find the 2D index of the flattened max
        y_idx, x_idx = np.unravel_index(heatmap.argmax(), heatmap.shape)
        max_val = heatmap[y_idx, x_idx]
        
        # Normalize confidence to 0-1
        confidence = max_val / 255.0
        
        # Debug print every 100 frames or if confidence is high
        # if confidence > 0.01:
        #      print(f"[TrackNet Debug] Max val: {max_val}, Conf: {confidence:.4f}, Pos: ({x_idx}, {y_idx})")
        
        # Map back to original image coordinates
        orig_h, orig_w = frame.shape[:2]
        cx = int(x_idx * orig_w / w)
        cy = int(y_idx * orig_h / h)
        
        return (cx, cy), confidence

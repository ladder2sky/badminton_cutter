import cv2
import torch
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    """
    使用 YOLOv8 进行运动员检测的模块。
    支持自动切换 GPU/CPU，并针对 'person' 类别进行过滤。
    """
    def __init__(self, model_path='yolov8n.pt', conf_thres=0.4, device=None):
        """
        初始化检测器。
        :param model_path: 模型路径，默认使用 YOLOv8n
        :param conf_thres: 置信度阈值
        :param device: 指定设备 ('cuda', 'cpu')，None则自动检测
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[PlayerDetector] Loading model on device: {self.device}")
        
        # 加载模型
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"[PlayerDetector] Error loading model: {e}")
            raise e
            
        self.conf_thres = conf_thres
        # COCO 数据集中 'person' 类别 ID 为 0
        self.target_class_id = 0 

    def detect(self, frame):
        """
        对单帧图像进行推理。
        :param frame: 输入图像 (BGR格式, numpy array)
        :return: 包含检测结果的列表，每个元素为 {'bbox': [x1, y1, x2, y2], 'conf': float}
        """
        if frame is None:
            return []

        # 执行推理，仅检测 'person' 类 (classes=[0])
        # verbose=False 关闭详细日志输出
        results = self.model.predict(
            source=frame, 
            conf=self.conf_thres, 
            classes=[self.target_class_id], 
            device=self.device,
            verbose=False
        )

        detections = []
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # 获取边界框坐标 (xyxy format)
                coords = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                
                detections.append({
                    'bbox': [int(c) for c in coords],
                    'conf': round(conf, 3)
                })
                
        return detections

    def filter_players(self, detections, camera_pos, frame_width, frame_height):
        """
        根据摄像机位置和检测框大小过滤球员。
        :param detections: 原始检测结果列表
        :param camera_pos: 摄像机位置 ('left', 'center', 'right')
        :param frame_width: 画面宽度
        :param frame_height: 画面高度
        :return: 过滤后的检测结果列表
        """
        filtered_detections = []
        
        # 1. 设定高度阈值 (画面高度的比例)
        min_height_ratio = 0.05  # 最小高度 5% (过滤远处观众)
        max_height_ratio = 0.8   # 最大高度 80% (过滤贴脸干扰)
        
        min_h = frame_height * min_height_ratio
        max_h = frame_height * max_height_ratio
        
        # 2. 设定 ROI 剔除区域 (Exclusion Zones)
        # x_min_exclude: 左侧剔除边界 (x < x_min_exclude 被剔除)
        # x_max_exclude: 右侧剔除边界 (x > x_max_exclude 被剔除)
        x_min_exclude = 0
        x_max_exclude = frame_width
        
        if camera_pos == 'left':
            # 左侧机位：剔除右侧 1/4 (干扰源在右)
            x_max_exclude = frame_width * 0.75
        elif camera_pos == 'right':
            # 右侧机位：剔除左侧 1/4 (干扰源在左)
            x_min_exclude = frame_width * 0.25
        elif camera_pos == 'center':
            # 正后方机位：剔除左右边缘各 10%
            x_min_exclude = frame_width * 0.1
            x_max_exclude = frame_width * 0.9
            
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            w = x2 - x1
            h = y2 - y1
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # 过滤1: 高度检查
            if h < min_h or h > max_h:
                continue
                
            # 过滤2: 区域检查 (基于中心点 cx)
            if cx < x_min_exclude or cx > x_max_exclude:
                continue
                
            filtered_detections.append(det)
            
        # 过滤3: 面积排序，只保留最大的 N 个 (例如 Top 6，避免过多误检)
        # 按面积降序排列
        filtered_detections.sort(key=lambda d: (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]), reverse=True)
        
        return filtered_detections[:6] # 最多保留6个 (双打4人+可能的裁判)

    def draw_detections(self, frame, detections):
        """
        在图像上绘制检测框 (用于调试/可视化)。
        """
        annotated_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            conf = det['conf']
            
            # 绘制矩形框
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 绘制标签
            label = f"Player: {conf}"
            cv2.putText(annotated_frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return annotated_frame

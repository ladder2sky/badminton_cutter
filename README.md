# Badminton Match Auto-Cutter (羽毛球比赛视频自动剪辑系统)

这是一个基于计算机视觉和音频分析的自动化视频剪辑系统，专门用于处理固定支架手机录制的羽毛球比赛视频。系统能够自动识别比赛中的精彩回合（Rally），并将其剪辑成连贯的高光时刻视频。

## 功能特性
*   **自动回合识别**: 结合视觉（羽毛球/运动员追踪）和听觉（击球声/欢呼声）特征，精准定位回合。
*   **智能剪辑**: 自动生成 `clips.txt` 剪辑列表，并合成 `highlights.mp4` 高光集锦。
*   **双模运行**: 支持 GPU 加速 (CUDA) 高性能模式，同时也提供 CPU 兼容模式。
*   **鲁棒性设计**: 内置抗干扰机制，有效处理路人遮挡、远端背景干扰及球飞出屏幕等复杂情况。

## 快速开始

### 1. 环境准备
确保已安装 Python 3.8+ 和 ffmpeg (推荐)。
```bash
pip install -r requirements.txt
```

### 2. 下载模型权重
本系统依赖预训练模型，请下载后放置在 `weights/` 目录下（如无该目录请自行创建）：

*   **YOLOv8 (自动下载)**: 首次运行会自动下载 `yolov8n.pt`。
*   **TrackNet (需手动下载)**: 
    *   推荐使用 [TrackNetV2-pytorch](https://github.com/ChgygLin/TrackNetV2-pytorch) 的权重。
    *   或者下载 [TrackNet (Tennis)](https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view?usp=sharing) 权重重命名为 `tracknet.pth` 放入 `weights/` 目录。
    *   *注意：如果未检测到权重文件，系统将使用随机权重运行（仅用于调试流程，无实际效果）。*

### 3. 运行参数说明

系统支持多种参数以适应不同的运行需求：

**基本参数**
*   `video_path`: (必填) 输入视频文件的路径。
*   `--output`: 输出目录路径。默认为 `output`。
*   `--gpu`: 启用 GPU (CUDA) 加速。强烈建议在支持的设备上开启。
*   `--generate-video`: 在分析完成后自动合成最终的 `highlights.mp4` 视频。如果不加此参数，仅生成分析报告和剪辑列表。

**调试与优化参数**
*   `--camera-pos`: 摄像机位置设置。可选值：`center` (默认), `left`, `right`。用于辅助过滤相邻场地的干扰（例如：选 `center` 会优先关注画面中央区域的运动）。
*   `--save-screenshots`: 开启后，会在输出目录的 `debug_screenshots` 文件夹中保存检测到羽毛球的高置信度帧截图，用于调试模型效果。
*   `--max-frames`: 仅处理前 N 帧。用于快速测试代码逻辑或模型效果。
*   `--start-time`: 从视频的第 N 秒开始处理。
*   `--skip-frames`: 跳过视频开头的前 N 帧。

### 4. 常见运行示例

**标准运行 (生成集锦)**
使用 GPU 加速并生成最终视频：
```bash
python src/core/main.py video.mp4 --gpu --generate-video
```

**快速测试 (调试模式)**
仅处理前 1000 帧，保存调试截图，不生成最终视频：
```bash
python src/core/main.py video.mp4 --max-frames 1000 --save-screenshots
```

**抗干扰测试**
指定摄像机位置为中间，输出到特定目录：
```bash
python src/core/main.py test_video.mp4 --output output_test --camera-pos center
```

**分段处理**
从第 60 秒开始处理：
```bash
python src/core/main.py video.mp4 --start-time 60 --generate-video
```

## 输出文件
运行完成后，输出目录将包含：
*   `highlights.mp4`: 自动剪辑合成的精彩集锦视频 (需加 `--generate-video` 参数)。
*   `debug_preview.mp4`: 带有可视化标记（红点追踪球、检测框）的预览视频。
*   `clips.txt`: 剪辑时间段及评分列表。
*   `debug_frames.csv`: 详细的逐帧分析数据（球坐标、置信度等）。
*   `debug_screenshots/`: (可选) 包含检测到球的关键帧截图。

## 目录结构
*   `src/ai_engine`: 包含 YOLO (球员检测), TrackNet (羽毛球追踪), AudioAnalyzer (音频分析)。
*   `src/decision`: 包含 RallyAnalyzer (回合决策逻辑)。
*   `src/input`: 视频预处理与加载。
*   `src/output`: 视频剪辑与合成。
*   `src/core`: 主程序入口。
*   `docs`: 设计文档。

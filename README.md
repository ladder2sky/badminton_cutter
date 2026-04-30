# Badminton Match Auto-Cutter (羽毛球比赛视频自动剪辑系统)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

这是一个基于计算机视觉和音频分析的自动化视频剪辑系统，专门用于处理固定支架录制的羽毛球比赛视频。系统能够自动识别比赛中的精彩回合（Rally），并将其剪辑成连贯的高光时刻视频。

## ✨ 功能特性

*   **多模态回合识别**: 融合视觉（羽毛球/运动员追踪）和听觉（击球声/欢呼声）特征，精准定位比赛回合
*   **智能评分系统**: 基于回合时长、击球频率、音频能量等多维度特征自动评分
*   **自动化剪辑**: 生成 `clips.txt` 剪辑清单，并自动合成 `highlights.mp4` 高光集锦视频
*   **双模运行**: 支持 GPU 加速 (CUDA) 高性能模式，也提供 CPU 兼容模式
*   **鲁棒性设计**: 内置抗干扰机制，有效处理路人遮挡、背景干扰、球出界等复杂场景
*   **调试友好**: 提供可视化预览视频、逐帧数据分析、关键帧截图等调试工具

## 🏗️ 系统架构

系统采用模块化设计，包含四个核心模块：

```
┌─────────────────────────────────────────────────────────────┐
│                    视频输入模块 (input)                       │
│  视频读取 → 元数据提取 → 预处理 (降噪/去抖动/ROI裁切)          │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   AI 分析引擎 (ai_engine)                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ PlayerDetect │  │Shuttlecock   │  │ AudioAnalyzer│       │
│  │  (YOLOv8n)  │  │ Tracker      │  │  击球/欢呼声  │       │
│  │  运动员检测  │  │ (TrackNetV2) │  │  音频分析    │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  剪辑决策模块 (decision)                       │
│  RallyAnalyzer: 事件识别 → 精彩评分 → 时间轴生成              │
│  评分公式: Score = w1·Duration + w2·HitCount + w3·Audio      │
└──────────────────────┬──────────────────────────────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                   输出渲染模块 (output)                        │
│  片段提取 → 转场效果 → 慢动作回放(可选) → 1080p 视频合成        │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 环境准备

确保已安装 **Python 3.8+** 和 **ffmpeg**（推荐）。

```bash
# 克隆项目
git clone https://github.com/ladder2sky/badminton_cutter.git
cd badminton_cutter

# 安装依赖
pip install -r requirements.txt
```

### 2. 下载模型权重

运行一键下载脚本，自动获取所需模型权重到 `weights/` 目录：

```bash
python setup_models.py
```

| 模型 | 文件路径 | 用途 | 来源 |
|------|---------|------|------|
| YOLOv8-Nano | `weights/yolov8n.pt` | 运动员检测 | 首次运行自动下载 |
| TrackNet V2 | `weights/track.pt` | 羽毛球轨迹追踪 | [TrackNetV2-pytorch](https://github.com/ChgygLin/TrackNetV2-pytorch) |

### 3. 基本运行

```bash
# 使用 GPU 加速并生成高光集锦视频
python src/core/main.py <视频路径> --gpu --generate-video

# CPU 模式快速测试（仅处理前1000帧）
python src/core/main.py <视频路径> --max-frames 1000 --save-screenshots
```

## ⚙️ 运行参数

### 基本参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `video_path` | 位置参数 | 输入视频文件路径 | (必填) |
| `--output` | 可选 | 输出目录路径 | `output` |
| `--gpu` | 标志 | 启用 GPU (CUDA) 加速 | 关闭 |
| `--generate-video` | 标志 | 分析完成后自动合成高光视频 | 关闭 |

### 高级参数

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `--camera-pos` | 可选 | 摄像机位置: `center` / `left` / `right` | `center` |
| `--save-screenshots` | 标志 | 保存羽毛球检测截图到 `debug_screenshots/` | 关闭 |
| `--max-frames` | int | 仅处理前 N 帧（调试用） | 全部 |
| `--start-time` | float | 从第 N 秒开始处理 | `0` |
| `--skip-frames` | int | 跳过开头前 N 帧 | `0` |

## 📋 运行示例

```bash
# 标准运行: GPU加速 + 生成集锦
python src/core/main.py video.mp4 --gpu --generate-video

# 快速测试: 仅处理前1000帧 + 保存调试截图
python src/core/main.py video.mp4 --max-frames 1000 --save-screenshots

# 抗干扰测试: 指定摄像机位置
python src/core/main.py test_video.mp4 --output output_test --camera-pos center

# 分段处理: 从第60秒开始
python src/core/main.py video.mp4 --start-time 60 --generate-video

# 完整功能: GPU + 自定义输出 + 截图调试
python src/core/main.py match.mp4 --gpu --generate-video --output my_result --save-screenshots
```

## 📁 输出文件

运行完成后，输出目录将包含：

| 文件 | 说明 |
|------|------|
| `highlights.mp4` | 精彩集锦视频 (需 `--generate-video`) |
| `debug_preview.mp4` | 带可视化标记的预览视频（红点追踪球、检测框） |
| `clips.txt` | 剪辑时间段及评分列表 |
| `debug_frames.csv` | 逐帧分析数据（球坐标、置信度等） |
| `debug_screenshots/` | 高置信度检测帧截图 (需 `--save-screenshots`) |

## 📂 目录结构

```
badminton_cutter/
├── src/
│   ├── ai_engine/         # AI 分析引擎
│   │   ├── player_detector.py   # YOLOv8n 运动员检测
│   │   ├── tracknet.py          # TrackNetV2 羽毛球追踪
│   │   └── audio_analyzer.py    # 音频分析 (击球声/欢呼声)
│   ├── decision/          # 剪辑决策模块
│   │   └── rally_analyzer.py    # 回合分析与评分
│   ├── input/             # 视频输入与预处理
│   ├── output/            # 视频输出与渲染
│   ├── utils/             # 工具函数
│   └── core/
│       └── main.py            # 主程序入口
├── docs/                  # 设计文档
│   ├── design_scheme.md   # 系统设计方案
│   └── troubleshooting_tracknet.md
├── input/                 # 输入视频目录
├── weights/               # 模型权重目录
├── requirements.txt       # Python 依赖
├── setup_models.py        # 模型下载脚本
└── README.md
```

## 💻 硬件要求

| 组件 | 无 GPU 轻量模式 (最低) | GPU 加速模式 (推荐) |
|------|------------------------|---------------------|
| **CPU** | Intel i5 (8代+) / Ryzen 5 (6核+) | Intel i7 (10代+) / Ryzen 7 |
| **GPU** | 集成显卡 | NVIDIA RTX 3060 (6GB+) |
| **RAM** | 8GB | 16GB |
| **处理速度** | ~0.5~1.0x (1小时视频需1-2小时) | >3.0x (1小时视频<20分钟) |

## 🔧 技术细节

### 核心算法

*   **运动员检测**: YOLOv8-Nano，通过 ROI 过滤观众区域
*   **羽毛球追踪**: TrackNet V2，输入3帧输出球坐标，卡尔曼滤波平滑轨迹
*   **音频分析**: 短时能量 + 过零率，识别击球声（高频短时能量）
*   **回合评分**: `Score = w1·Duration + w2·HitCount + w3·AudioEnergy`

### 抗干扰机制

*   **球出界/遮挡**: 超时判定 (丢失>2秒无击球音 → 死球) + 运动员行为分析
*   **背景干扰**: ROI 区域过滤 + 主球员锁定追踪
*   **镜头遮挡**: 帧间变化率检测 → 自动暂停 → 恢复后全局重检测

## 📖 文档

*   [系统设计方案](docs/design_scheme.md) - 详细架构设计与技术实现路径
*   [TrackNet 故障排查](docs/troubleshooting_tracknet.md) - 羽毛球追踪模型常见问题

## 📄 License

MIT License

## 🙏 致谢

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 目标检测模型
*   [TrackNetV2-pytorch](https://github.com/ChgygLin/TrackNetV2-pytorch) - 羽毛球追踪模型
*   [OpenCV](https://opencv.org/) - 计算机视觉库
*   [PyTorch](https://pytorch.org/) - 深度学习框架
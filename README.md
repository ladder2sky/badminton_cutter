# Badminton Match Auto-Cutter (羽毛球比赛视频自动剪辑系统)

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyTorch 2.10](https://img.shields.io/badge/PyTorch-2.10-EE4C2C.svg)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4-3480J4.svg)](https://github.com/ultralytics/ultralytics)

基于计算机视觉和音频分析的自动化视频剪辑系统，用于处理固定支架录制的羽毛球比赛视频。系统能够自动识别比赛中的精彩回合（Rally），并将其剪辑成连贯的高光时刻视频。

## ✨ 功能特性

*   **多模态回合识别**: 融合视觉（羽毛球/运动员追踪）和听觉（击球声/欢呼声）特征，精准定位比赛回合
*   **智能评分系统**: 基于回合时长、击球频率、音频能量等多维度特征自动评分
*   **自动化剪辑**: 生成 `clips.txt` 剪辑清单，并自动合成 `highlights.mp4` 高光集锦视频
*   **双模运行**: 支持 GPU 加速 (CUDA) 高性能模式，也提供 CPU 兼容模式
*   **鲁棒性设计**: 内置抗干扰机制，有效处理路人遮挡、背景干扰、球出界等复杂场景
*   **调试友好**: 提供可视化预览视频、逐帧数据分析、关键帧截图等调试工具
*   **自检报告**: 分析完成后自动生成自检报告，供用户手动确认后再进行视频合成

## 🏗️ 系统架构

```
视频输入 → AI分析引擎(YOLOv8n+TrackNetV2+Audio) → 剪辑决策 → 视频输出
```

### 核心模块

| 模块 | 技术 | 功能 |
|------|------|------|
| **运动员检测** | YOLOv8-Nano | 实时检测场地上运动员位置和数量 |
| **羽毛球追踪** | TrackNet V2 | 追踪羽毛球飞行轨迹，输出球坐标和置信度 |
| **音频分析** | Librosa + 自定义算法 | 检测击球声和欢呼声事件 |
| **回合决策** | 多特征融合算法 | 综合视觉和音频特征，识别精彩回合 |
| **视频剪辑** | MoviePy + OpenCV | 生成调试预览和高光集锦视频 |

## 🚀 快速开始

### 1. 环境准备

确保已安装 **Python 3.10+** 和 **ffmpeg**（推荐）。

```bash
git clone https://github.com/ladder2sky/badminton_cutter.git
cd badminton_cutter
pip install -r requirements.txt
```

### 2. 下载模型权重

```bash
python setup_models.py
```

| 模型 | 文件路径 | 用途 |
|------|---------|------|
| YOLOv8-Nano | `weights/yolov8n.pt` | 运动员检测 |
| TrackNet V2 | `weights/track.pt` | 羽毛球轨迹追踪 |

### 3. 基本运行

```bash
# 第一步: 分析视频，生成自检报告
python src/core/main.py <视频路径>

# 第二步: 确认报告无误后，生成高光集锦视频
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
| `--preview` | 标志 | 生成调试预览视频（带可视化标记） | 开启 |

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
# 标准流程: 先分析检查
python src/core/main.py video.mp4

# 确认无误后: GPU加速 + 生成集锦
python src/core/main.py video.mp4 --gpu --generate-video

# 快速测试: 仅处理前1000帧 + 保存调试截图
python src/core/main.py video.mp4 --max-frames 1000 --save-screenshots

# 分段处理: 从第60秒开始
python src/core/main.py video.mp4 --start-time 60 --generate-video

# 完整功能: GPU + 自定义输出 + 截图调试 + 生成视频
python src/core/main.py match.mp4 --gpu --generate-video --output my_result --save-screenshots
```

## 📁 输出文件

| 文件 | 说明 |
|------|------|
| `highlights.mp4` | 精彩集锦视频 (需 `--generate-video`) |
| `debug_preview.mp4` | 带可视化标记的预览视频（红点追踪球、检测框） |
| `clips.txt` | 剪辑时间段及评分列表 |
| `debug_frames.csv` | 逐帧分析数据（球坐标、置信度等） |
| `debug_screenshots/` | 高置信度检测帧截图 (需 `--save-screenshots`) |

### 自检报告说明

分析完成后，控制台将输出自检报告：

```
SELF-INSPECTION REPORT
--------------------------------------------------
Rally #1
  Start: 123.45s
  End:   130.78s
  Duration: 7.33s
  Score: 0.85
  Avg Players (First 1s): 4.20
  Ball Density: 0.65
--------------------------------------------------
```

| 指标 | 说明 | 正常范围 |
|------|------|----------|
| `Duration` | 回合持续时间 | 2s ~ 30s |
| `Score` | 精彩评分 | 越高越精彩 |
| `Avg Players` | 开头1秒平均在场人数 | 4~6人（双打） |
| `Ball Density` | 羽毛球检测密度 | >0.3 表示检测良好 |

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
│   │   └── video_processor.py   # 视频帧读取与处理
│   ├── output/            # 视频输出与渲染
│   │   └── video_cutter.py      # 视频剪辑与合成
│   ├── utils/             # 工具函数
│   │   └── static_filter.py     # 静态物体过滤
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

| 组件 | CPU模式(最低) | GPU模式(推荐) |
|------|--------------|---------------|
| **CPU** | Intel i5 / Ryzen 5 | Intel i7 / Ryzen 7 |
| **GPU** | 无需独立显卡 | NVIDIA RTX 3060+ |
| **RAM** | 8 GB | 16 GB |
| **存储** | 500 MB (模型) | 500 MB (模型) |

## ⚠️ 常见问题

1. **TrackNet 检测效果差**: 可能是视频分辨率或拍摄角度不匹配，请参考 [docs/troubleshooting_tracknet.md](docs/troubleshooting_tracknet.md)
2. **内存不足**: 使用 `--max-frames` 限制处理帧数，或关闭 `--preview`
3. **视频合成失败**: 确保已安装 ffmpeg，检查 `ffmpeg -version` 是否可用
4. **GPU 模式报错**: 确认 NVIDIA 驱动正常，CUDA 版本与 PyTorch 匹配
5. **静态物体误检测**: 系统内置静态过滤器，如仍有问题可调整 `static_filter.py` 中的阈值参数
6. **音频检测不准确**: 确保视频包含音频轨道，可检查 `debug_frames.csv` 中的音频事件数据

## 🔧 开发指南

### 本地开发环境

```bash
# 安装依赖
pip install -r requirements.txt

# 下载模型
python setup_models.py

# 运行测试
python src/core/main.py test_video.mp4 --max-frames 100
```

### 调试技巧

- 使用 `--save-screenshots` 保存检测帧，可视化查看检测效果
- 查看 `debug_frames.csv` 分析逐帧数据
- 使用 `--max-frames` 限制处理帧数进行快速测试
- 查看 `debug_preview.mp4` 中的可视化标记

## 📝 更新日志

### v1.0.0 (2026-04)
- 初始版本发布
- 支持 YOLOv8n 运动员检测
- 支持 TrackNet V2 羽毛球追踪
- 支持音频事件检测
- 支持 GPU/CPU 双模式运行
- 内置自检报告功能

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

## 📜 许可证

[MIT License](LICENSE)

## 🙏 致谢

*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - 运动员检测
*   [TrackNetV2](https://github.com/ChgygLin/TrackNetV2-pytorch) - 羽毛球追踪网络
*   [MoviePy](https://github.com/Zulko/moviepy) - 视频处理库
*   [Librosa](https://librosa.org/) - 音频分析库
*   [OpenCV](https://opencv.org/) - 计算机视觉库
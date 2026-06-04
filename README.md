# AI-Blind-Assistant: 端云协同智能导盲眼镜系统

![UI Preview](https://via.placeholder.com/1000x600.png?text=Please+Upload+Your+GUI+Screenshot+Here)

## 📖 项目简介
本项目是一款专为视障人群设计的端云协同智能辅助感知系统。系统融合了**边缘计算硬件（ESP32-S3）**、**轻量化目标检测算法（YOLOv8s）**、**低照度图像深度增强（DE-Retinex）**以及**多模态大模型智能语音交互**，致力于解决视障人群在复杂光照环境下的动态避障与静态寻物难题。

## ✨ 核心特性
- **端云协同架构**：采用 ESP32-S3 Sense 作为边缘采集端，通过 WebSocket 实现低延迟高帧率的视频流上行，结合 HTTP Chunked 实现流式语音下发。
- **极暗环境自适应增强**：内置基于通道注意力机制的 DE-Retinex 深度学习模型，在重度低照度场景下将检测精度（mAP50）提升 30% 以上。
- **轻量化高精度识别**：定制并微调了针对街道场景的 YOLOv8s 模型，引入 VFL 与 CIoU 损失函数，模型权重仅 21MB，单帧推理耗时极低。
- **多模态智能语音闭环**：首创“离线热词唤醒 + 视觉快照抓取 + 视觉大模型推理 + 实体文件 TTS 映射”的异步非阻塞多模态交互管线。
- **可视化监控大盘**：提供基于 PySide6 开发的现代化 PC 控制台，支持实时调整 Gamma 增强、监控端云网络延迟及大模型意图状态。

## 🛠️ 系统架构
1. **感知采集层 (Edge)**：ESP32-S3 + OV2640 + PDM 麦克风阵列
2. **核心处理层 (Cloud)**：FastAPI 异步网关 + YOLOv8 推理引擎 + ASR/NLP 意图状态机
3. **交互展示层 (Client)**：PySide6 监控大盘 + 端侧 I2S 功放语音反馈

## 🚀 快速开始

### 1. 环境依赖
```bash
git clone https://github.com/Aaronlu07/aiglass.git
cd aiglass
pip install -r requirements.txt
```

### 2. 下载预训练权重
由于模型文件较大，请下载以下权重文件并放置于 `weights/` 目录下：
- YOLOv8s 街道微调模型: `best.pt` (请在此处提供您的网盘或 Releases 链接)

### 3. 启动系统
```bash
python app.py
```
启动后，您可以在弹出的 UI 界面中配置 ESP32 的局域网 IP、调整图传分辨率，并一键开启端云协同监控。

## 📄 许可证
本项目采用 MIT License 开源许可证。

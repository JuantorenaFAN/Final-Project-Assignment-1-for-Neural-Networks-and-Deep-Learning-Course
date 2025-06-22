# TensoRF: 基于张量分解的神经辐射场

本项目实现了基于TensoRF的物体重建和新视图合成系统。TensoRF是NeRF的一个高效变体，通过低秩张量分解大幅提升了训练和渲染速度。

## 🎯 项目概述

### 主要功能
- **物体重建**: 从多角度图像重建3D物体
- **新视图合成**: 生成任意新视角的高质量图像
- **视频渲染**: 自动生成环绕物体的流畅视频
- **定量评估**: 提供PSNR、SSIM等多种评估指标

### 技术特点
- **高效训练**: 相比原版NeRF训练速度提升10-100倍
- **优质重建**: 保持高质量的3D重建效果
- **内存友好**: 显存使用优化，支持大分辨率图像
- **易于使用**: 完整的训练和推理流程

## 🚀 环境配置

### 系统要求
- **操作系统**: Ubuntu 20.04+ / macOS 10.15+ / Windows 10+
- **Python版本**: 3.8+
- **GPU**: NVIDIA GPU (推荐RTX 3080或更高)
- **显存**: 至少8GB (推荐12GB+)

### 安装步骤

```bash
pip install -r requirements.txt
```

## 📁 数据准备

### 数据集结构
将你的多角度图像放置在`data/`目录下：
```
data/
├── IMG_5082.jpg
├── IMG_5083.jpg
├── ...
└── IMG_5140.jpg
```

### 拍摄要求
为了获得最佳重建效果，请遵循以下拍摄指南：

1. **拍摄角度**: 围绕物体拍摄20-100张不同角度的图像
2. **重叠度**: 相邻图像应有60-80%的重叠
3. **光照**: 保持一致的光照条件
4. **背景**: 使用简单、一致的背景
5. **焦距**: 保持相同的焦距和相机设置

### 数据预处理
系统会自动进行以下预处理：
- 图像缩放到指定分辨率
- 相机参数估计
- 数据集分割（训练/验证/测试）
- 射线生成和缓存

## 🏋️ 训练模型

### 基础训练

使用默认参数开始训练：
```bash
python train.py --data_dir ./data --expname my_object
```

### 自定义训练参数

```bash
python train.py \
    --data_dir ./data \
    --expname my_object \
    --img_wh 800 800 \
    --n_iters 30000 \
    --batch_size 4096 \
    --lr_init 0.02 \
    --lr_basis 0.001
```

### 重要训练参数

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `--data_dir` | `./data` | 数据目录路径 |
| `--expname` | `dvgo_hotdog` | 实验名称 |
| `--img_wh` | `800 800` | 图像宽度和高度 |
| `--n_iters` | `30000` | 训练总步数 |
| `--batch_size` | `4096` | 射线批大小 |
| `--lr_init` | `0.02` | 初始学习率 |
| `--n_lamb_sigma` | `16 16 16` | 密度张量rank |
| `--n_lamb_sh` | `48 48 48` | 颜色张量rank |

### 训练监控

训练过程中可以通过以下方式监控：

1. **终端输出**: 实时显示损失和PSNR
2. **Tensorboard**: 可视化训练曲线
```bash
tensorboard --logdir logs/my_object/logs
```
3. **检查点**: 定期保存在`logs/my_object/checkpoints/`

### 典型训练时间
- **CPU**: Intel i7 + RTX 3080: ~30分钟
- **内存使用**: ~8-12GB显存
- **最终PSNR**: 通常达到25-30dB

## 🎨 测试和渲染

### 加载训练好的模型

渲染测试图像：
```bash
python render.py \
    --ckpt logs/my_object/checkpoints/step_030000.pth \
    --data_dir ./data \
    --expname my_object \
    --render_test 1 \
    --render_path 1
```

### 渲染选项

| 参数 | 说明 |
|------|------|
| `--render_test` | 渲染测试集图像并计算PSNR |
| `--render_path` | 生成环绕视频 |
| `--video_fps` | 视频帧率 (默认30) |
| `--video_format` | 视频格式 (默认mp4) |

### 输出文件

渲染完成后，结果保存在`logs/my_object/rendered/`：
```
rendered/
├── test_images/           # 测试图像对比
│   ├── test_000_psnr_28.5.png
│   └── ...
├── path_images/           # 路径渲染图像
│   ├── frame_000.png
│   └── ...
├── spiral_video.mp4       # 环绕视频
└── spiral_loop_video.mp4  # 循环视频
```

## 📁 项目结构

```
project/
├── config.py              # 配置参数定义
├── dataset.py             # 数据加载和预处理
├── model.py               # TensoRF模型实现
├── utils.py               # 工具函数
├── train.py               # 训练脚本
├── render.py              # 渲染脚本
├── requirements.txt       # 依赖包列表
├── README.md              # 项目说明
├── 实验报告.md             # 详细实验报告
├── data/                  # 原始图像数据
│   ├── IMG_5082.jpg
│   └── ...
├── cache/                 # 数据预处理缓存
├── logs/                  # 训练日志和模型
│   └── my_object/
│       ├── checkpoints/   # 模型检查点
│       ├── logs/          # Tensorboard日志
│       ├── results/       # 中间结果
│       └── rendered/      # 渲染输出
└── outputs/               # 最终输出
```

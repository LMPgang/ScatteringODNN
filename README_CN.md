# Real-time Imaging Through Moving Scattering Media Enabled by Fixed Optical Modulations

[English](README.md) | 简体中文

[![arXiv](https://img.shields.io/badge/arXiv-2508.15286-b31b1b.svg)](https://arxiv.org/pdf/2508.15286)
[![Paper](https://img.shields.io/badge/Paper-Photonics%20Research-blue.svg)](https://doi.org/10.1364/PRJ.581675)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://fishwowater.github.io/ScatteringODNN)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-yellow.svg)](https://drive.google.com/file/d/1wfkLWg8jOsXjXztB9hAKdhlGxgg7Grmy/view?usp=share_link)
[![License](https://img.shields.io/badge/License-Apache2.0-red.svg)](LICENSE)

**[Yuegang Li](https://scholar.google.com/citations?user=Wsn0Jx0AAAAJ&hl=en), [Junjie Wang](https://scholar.google.com/citations?user=lkjG-IAAAAAJ&hl=en), [Tailong Xiao](https://scholar.google.com/citations?hl=en&user=qo67_eUAAAAJ), [Ze Zheng](https://scholar.google.com/citations?user=8tH2pRwAAAAJ&hl=en), [Jingzheng Huang](https://scholar.google.com/citations?user=LnMMo7QAAAAJ&hl=en), [Ming He](https://scholar.google.com/citations?user=bWIHRvoAAAAJ&hl=en), [Jianping Fan](https://scholar.google.com/citations?user=-YsOqQcAAAAJ&hl=en), [Guihua Zeng](https://scholar.google.com/citations?user=LOADXDsAAAAJ&hl=en)**

*Photonics Research, 2026*

![teaser](./assets/teaser.jpg)

## 概述

本工作提出了一种新颖的成像策略，通过固定光学调制模块来对抗时变散射扰动。通过在物体和散射介质的模拟数据集上训练光学衍射神经网络（ODNNs），我们实现了在去相关时间 < 1 ms 的动态散射介质中以 80 Hz 进行实时成像。

**主要特点：**
- 实现穿透运动散射介质的实时成像
- 80 Hz 成像速度，光速处理
- 对散斑去相关免疫
- 在模拟数据集上训练，泛化至真实场景
- 适用于大视场和非相干照明

## 仓库结构

```
odnn/
├── training/          # 训练代码（100% Python）
│   ├── DynaDiffuser_2layer_v2_SLM_git.py  # 主训练程序
│   ├── dataset.py     # 物体图像数据加载器（MNIST）
│   ├── diffuser.py    # 模拟散射介质加载器
│   ├── run_SLM.py     # 训练参数配置
│   ├── run_SLM.slurm  # 作业提交脚本
│   ├── diffuser_data/ # 散射介质数据集位置
│   └── mnist_data/    # 物体数据集位置
├── inference/         # 推理代码（100% MATLAB）
│   └── Imaging_demo/  # 测试训练好的衍射网络
│       ├── Main.m     # 主程序（静态/动态/旋转演示）
│       ├── diffuser/  # 散射介质文件
│       ├── object/    # 物体文件
│       └── ODNN/      # 训练好的衍射层
├── assets/           # 图片和视频
└── requirements.txt  # Python 依赖项
```

## 安装

### 前置要求
- Python 3.7
- PyTorch 1.8
- MATLAB R2019b 或更高版本（用于推理）
- CUDA（可选，用于 GPU 加速）

### 环境配置

1. 克隆仓库：
```bash
git clone https://github.com/FishWoWater/ScatteringODNN odnn
cd odnn
```

2. 创建虚拟环境（推荐）：
```bash
conda create -n odnn python=3.8
conda activate odnn
```

3. 安装 Python 依赖项：
```bash
pip install -r requirements.txt
```

4. 下载数据集：
从 [Google Drive](https://drive.google.com/file/d/1wfkLWg8jOsXjXztB9hAKdhlGxgg7Grmy/view?usp=share_link) 下载数据集并解压到 `training/` 目录。

## 使用方法

### 训练

训练过程使用模拟的物体和散射介质来训练光学衍射神经网络。

**步骤 1：** 在 `DynaDiffuser_2layer_v2_SLM_git.py` 中配置网络参数

**步骤 2：** 在 `run_SLM.py` 中设置训练超参数：
```bash
cd training
python run_SLM.py
```

**可选 - 用于高性能计算集群：**
```bash
sbatch run_SLM.slurm
```

**可选演示：**

如需演示物体旋转，在 `DynaDiffuser_2layer_v2_SLM_git.py` 中取消注释旋转代码：
```python
# Object rotate 90 degree
train_object_image = train_object_image.flip(1).flip(2).contiguous()
train_object_image = train_object_image.permute(0, 2, 1).flip(2).contiguous()
```

如需演示物体缩放，取消注释缩放代码：
```python
# Object scaling
train_object_image = train_object_image.unsqueeze(1)  
train_object_image_interp = F.interpolate(train_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
train_object_image_padded = F.pad(train_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
train_object_image = train_object_image_padded.squeeze(1)
```

训练结果（包括图像、训练好的网络和参数）将保存在 `results/` 文件夹中。

### 推理

推理过程使用 MATLAB 来测试训练好的衍射网络。

1. 打开 MATLAB 并导航到推理目录：
```matlab
cd inference/Imaging_demo
```

2. 运行主脚本：
```matlab
Main.m
```

主程序包括两个演示：
- **演示 1：** 在静态/动态散射介质下的成像
- **演示 2：** 物体旋转下的成像

训练好的衍射层位于 `ODNN/` 文件夹中（包括正立和旋转90度版本）。

## 结果

我们的方法实现了：
- 80 Hz 穿透动态散射介质的实时成像
- 在 1-2 个输运平均自由程内的有效重建
- 对散斑去相关时间 < 1 ms 的鲁棒性
- 从模拟场景泛化到真实散射场景

请访问我们的[项目页面](https://fishwowater.github.io/ScatteringODNN)观看视频演示。

## 许可证

本项目采用 Apache2.0 许可证 - 详见 [LICENSE](LICENSE) 文件。

## 引用

如果您发现这项工作对您的研究有用，请考虑引用：

```bibtex
@article{li2026real,
  title={Real-time imaging through moving scattering media enabled by fixed optical modulations},
  author={Yuegang Li and Junjie Wang and Tailong Xiao and Ze Zheng and Jingzheng Huang and Ming He and Jianping Fan and Guihua Zeng},
  journal={Photonics Research},
  url={https://doi.org/10.1364/PRJ.581675},
  doi={10.1364/PRJ.581675},
  year={2026}
}
```

# Real-time Imaging Through Moving Scattering Media Enabled by Fixed Optical Modulations

[![arXiv](https://img.shields.io/badge/arXiv-2508.15286-b31b1b.svg)](https://arxiv.org/pdf/2508.15286)
[![Paper](https://img.shields.io/badge/Paper-Photonics%20Research-blue.svg)](https://doi.org/10.1364/PRJ.581675)
[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://fishwowater.github.io/ScatteringODNN)
[![Dataset](https://img.shields.io/badge/Dataset-Google%20Drive-yellow.svg)](https://drive.google.com/file/d/1wfkLWg8jOsXjXztB9hAKdhlGxgg7Grmy/view?usp=share_link)
[![License](https://img.shields.io/badge/License-Apache2.0-red.svg)](LICENSE)

**[Yuegang Li](https://scholar.google.com/citations?user=Wsn0Jx0AAAAJ&hl=en), [Junjie Wang](https://scholar.google.com/citations?user=lkjG-IAAAAAJ&hl=en), [Tailong Xiao](https://scholar.google.com/citations?hl=en&user=qo67_eUAAAAJ), Ze Zheng, [Jingzheng Huang](https://scholar.google.com/citations?user=LnMMo7QAAAAJ&hl=en), [Ming He](https://scholar.google.com/citations?user=bWIHRvoAAAAJ&hl=en), [Jianping Fan](https://scholar.google.com/citations?user=-YsOqQcAAAAJ&hl=en), [Guihua Zeng](https://scholar.google.com/citations?user=LOADXDsAAAAJ&hl=en)**

*Photonics Research, 2026*

![teaser](./assets/teaser.jpg)

## Overview

This work proposes a novel imaging strategy that counteracts time-dependent scattering perturbations through fixed optical modulation modules. By training optical diffraction neural networks (ODNNs) on simulated datasets of objects and scattering media, we achieve real-time imaging through dynamic scattering media with decorrelation times < 1 ms at 80 Hz.

**Key Features:**
- Real-time imaging through moving scattering media
- 80 Hz imaging speed with light-speed processing
- Immune to speckle decorrelation
- Trained on simulated datasets, generalizes to real-world scenarios
- Applicable to large-field-of-view and incoherent illumination

## Repository Structure

```
odnn/
├── training/          # Training code (100% Python)
│   ├── DynaDiffuser_2layer_v2_SLM_git.py  # Main training program
│   ├── dataset.py     # Object image data loader (MNIST)
│   ├── diffuser.py    # Simulated scattering media loader
│   ├── run_SLM.py     # Training parameter configuration
│   ├── run_SLM.slurm  # Job submission script
│   ├── diffuser_data/ # Scattering media dataset location
│   └── mnist_data/    # Object dataset location
├── inference/         # Inference code (100% MATLAB)
│   └── Imaging_demo/  # Testing trained diffraction networks
│       ├── Main.m     # Main program (static/dynamic/rotation demos)
│       ├── diffuser/  # Scattering media files
│       ├── object/    # Object files
│       └── ODNN/      # Trained diffraction layers
├── assets/           # Figures and videos
└── requirements.txt  # Python dependencies
```

## Installation

### Prerequisites
- Python 3.7
- PyTorch 1.8
- MATLAB R2019b or later (for inference)
- CUDA (optional, for GPU acceleration)

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/FishWoWater/ScatteringODNN odnn
cd odnn
```

2. Create a virtual environment (recommended):
```bash
conda create -n odnn python=3.8
conda activate odnn
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Download the dataset:
Download the dataset from [Google Drive](https://drive.google.com/file/d/1wfkLWg8jOsXjXztB9hAKdhlGxgg7Grmy/view?usp=share_link) and extract it to the `training/` directory.

## Usage

### Training

The training process uses simulated objects and scattering media to train optical diffraction neural networks.

**Step 1:** Configure network parameters in `DynaDiffuser_2layer_v2_SLM_git.py`

**Step 2:** Set training hyperparameters in `run_SLM.py`:
```bash
cd training
python run_SLM.py
```

**Optional - For HPC clusters:**
```bash
sbatch run_SLM.slurm
```

**Optional Demonstrations:**

To demonstrate object rotation, uncomment the rotation code in `DynaDiffuser_2layer_v2_SLM_git.py`:
```python
# Object rotate 90 degree
train_object_image = train_object_image.flip(1).flip(2).contiguous()
train_object_image = train_object_image.permute(0, 2, 1).flip(2).contiguous()
```

To demonstrate object scaling, uncomment the scaling code:
```python
# Object scaling
train_object_image = train_object_image.unsqueeze(1)  
train_object_image_interp = F.interpolate(train_object_image, size=(128, 128), mode='bilinear', align_corners=False) 
train_object_image_padded = F.pad(train_object_image_interp, (64, 64, 64, 64), mode='constant', value=0) 
train_object_image = train_object_image_padded.squeeze(1)
```

Training results (including images, trained networks, and parameters) will be saved in the `results/` folder.

### Inference

The inference process uses MATLAB to test trained diffraction networks.

1. Open MATLAB and navigate to the inference directory:
```matlab
cd inference/Imaging_demo
```

2. Run the main script:
```matlab
Main.m
```

The main program includes two demonstrations:
- **Demo 1:** Imaging under static/dynamic scattering media
- **Demo 2:** Imaging under object rotation

The trained diffraction layers are located in the `ODNN/` folder (both upright and rotate90 versions).

## Results

Our method achieves:
- 80 Hz real-time imaging through dynamic scattering media
- Effective reconstruction within 1-2 transport mean free paths
- Robustness to speckle decorrelation times < 1 ms
- Generalization from simulated to real-world scattering scenarios

See our [project page](https://fishwowater.github.io/ScatteringODNN) for video demonstrations.

## License

This project is licensed under the Apache2.0 License - see the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful in your research, please consider citing:

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
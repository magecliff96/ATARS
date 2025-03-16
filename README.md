# ATARS: Aerial Traffic Atomic Activity Recognition and Segmentation Dataset

## Overview
The **Aerial Traffic Atomic Activity Recognition and Segmentation (ATARS) Dataset** is a novel dataset designed for **multi-label atomic activity analysis** in traffic scenes. Captured from a **drone perspective**, ATARS is the first dataset to provide **frame-level annotations** of atomic traffic activities, making it ideal for **multi-label temporal atomic activity segmentation and recognition**.

This repository contains the dataset and source code for benchmarking **state-of-the-art models** in atomic activity recognition and segmentation tasks.

## Features
- **Top-down UAV perspective**: Unlike traditional egocentric datasets, ATARS captures full-scene traffic dynamics.
- **Frame-wise atomic activity annotations**: Provides fine-grained **spatial-temporal** labeling.
- **Multi-label traffic activity segmentation**: Introduces a novel **Multi-label Temporal Atomic Activity (M-TAA) Segmentation** task.
- **Comprehensive benchmarking**: Evaluates various **action recognition** and **temporal segmentation** models.

## Dataset
ATARS consists of **39 Full-HD untrimmed videos** from **4-way intersections**, recorded at **30 FPS**. Each video is manually labeled with **atomic activity annotations**, including:
- **Movement direction** (entry & exit roadways)
- **Traffic participant types** (vehicles, pedestrians, and grouped participants)
- **Chronological activity segmentation** for untrimmed videos

| Dataset Split | Videos |
|--------------|--------|
| Train        | 27     |
| Validation   | 6      |
| Test        | 6      |

## Installation
### Prerequisites
- Python 3.x
- PyTorch
- OpenCV
- CUDA-enabled GPU (for training models)

### Setup
```bash
# Clone the repository
git clone https://github.com/magecliff96/ATARS.git
cd ATARS

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training Action-Slot Model
To train the **Action-Slot** model for multi-label **atomic activity recognition**, run:
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_rus.py \
  --arch action_slot \
  --batch_size 2
```

### Predicting with ASFormer
For **multi-label temporal atomic activity segmentation**, use the **ASFormer** model:
```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --action=predict --arch asformer
```
For training:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --arch asformer
```

### Running PointTAD
For **temporal atomic activity segmentation** using **PointTAD**, execute:
```bash
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset carom
```
For distributed training with multiple GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,3 torch.distributed.run --nproc_per_node=2 python3 main.py --dataset carom
```
For evaluation:
```bash
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset carom --eval
```

## Benchmarked Models
| Model         | Task                           | Performance |
|--------------|-------------------------------|------------|
| Action-Slot  | Atomic Activity Recognition   | State-of-the-art on egocentric data |
| ASFormer    | Temporal Activity Segmentation | Best results for long-tail activities |
| PointTAD    | Multi-label Temporal Segmentation | Struggles with small pedestrian detection |

## Citation
If you use **ATARS** in your research, please cite our paper:
```bibtex
@article{chen2024atars,
  title={ATARS: An Aerial Traffic Atomic Activity Recognition and Temporal Segmentation Dataset},
  author={Chen, Zihao and Wu, Hsuanyu and Kung, Chi-Hsi and Chen, Yi-Ting and Peng, Yan-Tsung},
  journal={Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```

## License
This project is licensed under the **MIT License**.

---
For more details, visit our **[GitHub repository](https://github.com/magecliff96/ATARS)** or refer to our **ATARS dataset paper**.


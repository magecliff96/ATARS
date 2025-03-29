# ATARS: Aerial Traffic Atomic Activity Recognition and Segmentation Dataset

![ATARS Introduction](https://raw.githubusercontent.com/magecliff96/ATARS/main/images/intro.png)


## Overview
The **Aerial Traffic Atomic Activity Recognition and Segmentation (ATARS) Dataset** is a novel dataset designed for **multi-label atomic activity analysis** in traffic scenes. Captured from a **drone perspective**, ATARS is the first dataset to provide **frame-level annotations** of atomic traffic activities, making it ideal for **multi-label temporal atomic activity segmentation and recognition**.

This repository contains the source code for benchmarking **state-of-the-art models** in atomic activity recognition and segmentation tasks.
- Atomic Activity Annotations are in the DatasetLabel folder
- The video dataset can be downloaded here: https://docs.google.com/document/d/1ILzkIccVBLmeqEZhxXrqL1S1Mc7ZNXqR8s5JoJlDTzA/edit?tab=t.0#heading=h.6v1552exvkx
- A video of our presentation can be found here: https://youtu.be/uvjOI7QrPho
- The presentation slides can be found here: https://docs.google.com/presentation/d/1_u2W94oRMJ6jAGGOlLeohwI28nNlfsKG_Dch0_6_gqc/edit?usp=sharing

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

![ATARS Distribution](https://raw.githubusercontent.com/magecliff96/ATARS/main/images/dist.png)

## Installation
### Prerequisites
- CUDA-enabled GPU (for training models)


### Setup
Using Conda (Recommended)

To recreate the exact environment, use the provided environment.yml file:
```bash
# Clone the repository
git clone https://github.com/magecliff96/ATARS.git
cd ATARS

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate atars_env
```

### Dataset Preparation
For Traffic AA Recognition, the dataset can be installed anywhere, but please be sure to edit the address in the parser of train_rus.py or specify it as a command input.
For Temporal AA Segmentation, the dataset should be installed inside TrafficSegmentation/PointTAD-main/data, and data path of ASformer should adjust accordingly. For more details, please see https://github.com/MCG-NJU/PointTAD and https://github.com/ChinaYi/ASFormer.

## Training AA Recognition Model
To train various model for multi-label **atomic activity recognition**, first preprocess video clips using transform_rus.py in TrafficRecognition/video_classification folder.
Then execute below at your TrafficRecognition/video_classification folder:
```bash
CUDA_VISIBLE_DEVICES=1 python3 train_rus.py \
  --arch model_name \
  --batch_size 2
```

## Training AA Segmentation Models
### For ASFormer, execute below at your TrafficSegmentation/ASFormer-test folder:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main.py --arch asformer
```
For evaluation:
```bash
CUDA_VISIBLE_DEVICES=1 python3 main.py --action=predict --arch asformer
```

### For PointTAD,  execute below at your TrafficSegmentation/PointTAD-main folder:
```bash
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset carom
```
For evaluation:
```bash
CUDA_VISIBLE_DEVICES=3 python3 main.py --dataset carom --eval
```

### For **MSTCN** and **MSTCN++**, execute below at your TrafficSegmentation/MS-TCN2 folder:
```bash
python main.py --action=train --dataset=${1} --split=${2} \
                --num_epochs=1000 \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=3 \
                --num_f_maps=64 \
                --lr=0.0001 \
                --bce_pos_weight=1 \
                --bz=8
```
For evaluation:
```bash
CUDA_VISIBLE_DEVICES=1 python main.py --action=predict\
               --num_layers_PG=11 \
               --num_layers_R=10 \
               --num_R=3

```

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


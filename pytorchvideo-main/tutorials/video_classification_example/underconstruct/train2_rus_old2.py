# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
import numpy as np
import cv2
import argparse
import itertools
import logging
import subprocess
import os
import csv
import signal
import time
#import math
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models.resnet
import pytorchvideo.models.x3d
import pytorchvideo.models.r2plus1d
import pytorchvideo.models.csn
import pytorchvideo.models.slowfast
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchmetrics import Accuracy
from sklearn.metrics import precision_score
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import LearningRateMonitor
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    RemoveKey,
    ShortSideScale,
    UniformTemporalSubsample,
)
from slurm import copy_and_run_with_config
from torch.utils.data import DistributedSampler, RandomSampler
from torch.optim import lr_scheduler
#from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
from sklearn.metrics import average_precision_score

from traffic_dataset import *

import x3d_rus as x3d
import csn_rus as csn
import slowfast_rus as slowfast
import i3d_kinetics as i3d_kinetics
import ARG_rus as ARG
import ORN_rus as ORN
#from traffic_dataset_taco import TACO
# def stop_tensorboard():
#     try:
#         subprocess.run(["pkill", "-f", "tensorboard"], check=True)
#         print("TensorBoard stopped.")
#     except subprocess.CalledProcessError:
#         print("No TensorBoard process found.")

# def restart_tensorboard(logdir):
#     stop_tensorboard()
#     time.sleep(2)
#     try:
#         subprocess.run(["tensorboard", "--logdir=" + logdir], check=True)
#         print("TensorBoard restarted.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error restarting TensorBoard: {e}")
#restart_tensorboard(logs_directory)

#logs_directory = r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log'
logs_directory = r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/testland/log'

#writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log')
writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/testland/log')

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):

        self.args = args
        super().__init__()
        self.train_accuracy = Accuracy(task='multilabel', num_labels=39)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=39)

        model_num_class=39

        if self.args.arch == "x3d": #functional
            self.model = x3d.X3D(num_ego_class=0, num_actor_class=model_num_class, args=self.args)
            for t in self.model.model.parameters():
                t.requires_grad=False
            for t in self.model.model.blocks[-1].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-2].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-3].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == "csn":
            self.model = csn.CSN(num_ego_class=0, num_actor_class=model_num_class)
            for t in self.model.model.parameters():
                t.requires_grad=False
            for t in self.model.model.blocks[-1].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-2].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-3].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == "slowfast":
            self.model = slowfast.SlowFast(num_ego_class=0, num_actor_class=model_num_class)
            for t in self.model.parameters():
                t.requires_grad=False
            for t in self.model.model.blocks[-1].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-2].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-3].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == "i3d":
            self.model = i3d_kinetics.I3D_KINETICS(num_ego_class=0, num_actor_class=model_num_class)
            for t in self.model.model.parameters():
                t.requires_grad=False
            for t in self.model.model.blocks[-1].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-2].parameters():
                t.requires_grad=True
            for t in self.model.model.blocks[-3].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == 'ARG' or self.args.arch == 'ORN':
            if self.args.arch == 'ARG':
                self.model = ARG.ARG(args, max_N=63, num_ego_class=0, num_actor_class=model_num_class)
            else:
                self.model = ORN.ORN(args, max_N=63, num_ego_class=0, num_actor_class=model_num_class)
            for t in self.model.parameters():
                t.requires_grad=True
            for t in self.model.resnet.parameters():
                t.requires_grad=False
            for t in self.model.resnet[-1].parameters():
                t.requires_grad=True
            for t in self.model.resnet[-2].parameters():
                t.requires_grad=True

    def on_train_epoch_start(self): #not neck
        epoch = self.trainer.current_epoch
        #if self.trainer.use_ddp:
        if self.trainer.accelerator == "ddp":
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx): #not bottle neck
        x = batch[self.batch_key]
        y_hat = self.model(x)

        pos_lambda = torch.tensor([2],device=y_hat.device)
        loss = 0
        batch_labels_tensor = torch.stack(batch["label"]).t()
        for i in range(39):
           loss = loss + F.binary_cross_entropy_with_logits(y_hat[:, i], batch_labels_tensor[:, i].float(), pos_weight=pos_lambda) 
        loss/=39
        acc = self.train_accuracy(torch.sigmoid(y_hat), batch_labels_tensor.float())

        # Convert tensors to NumPy arrays
        y_hat_array = torch.sigmoid(y_hat).detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()

        ap_scores = []
        for label_index in range(len(batch_labels_array)):
            y_true_label = batch_labels_array[label_index, :]
            y_pred_label = y_hat_array[label_index, :]
            ap = average_precision_score(y_true_label, y_pred_label)
            ap_scores.append(ap)
        # Compute mean average precision
        mAP = sum(ap_scores) / len(ap_scores)

        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.train_dataloader()) + batch_idx
        writer.add_scalar("train/loss", float(loss), iteration)
        writer.add_scalar("train/acc", float(acc), iteration)
        writer.add_scalar("train/mAP", float(mAP), iteration)

        return loss

    def validation_step(self, batch, batch_idx): #not neck
        x = batch[self.batch_key]
        #print(len(x));print(x[0].size());print(x[1].size())
        y_hat = self.model(x)

        # Convert tensors to NumPy arrays
        y_hat_array = torch.sigmoid(y_hat).detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()

        # Save y_hat and batch_labels to CSV files
        # add self.item_names
        headline_list = [
            '12v', '12v+', '12p', '12p+', 
            '13v', '13v+', '14v', '14v+', '14p', '14p+',
            '21v', '21v+', '21p', '21p+', 
            '23v', '23v+', '23p', '23p+', '24v', '24v+', 
            '31v', '31v+', '32v', '32v+', '32p', '32p+',	
            '34v', '34v+', '34p', '34p+', 
            '41v', '41v+', '41p', '41p+',
            '42v', '42v+', '43v', '43v+', '43p+'
            ]

        headline_array = np.array([headline_list])
        headline_array = headline_array.reshape(1, -1)
        y_hat_wh = np.vstack([headline_array, y_hat_array])
        batch_labels_wh = np.vstack([headline_array, batch_labels_array])
        y_hat_wh = y_hat_wh.astype(str)
        batch_labels_wh = batch_labels_wh.astype(str)

        np.savetxt(f'log/y_hat_{batch_idx}.csv', y_hat_wh, delimiter=',', fmt='%s')
        np.savetxt(f'log/batch_labels_{batch_idx}.csv', batch_labels_wh, delimiter=',', fmt='%s')

        pos_lambda = torch.tensor([2],device=y_hat.device)
        loss = 0
        batch_labels_tensor = torch.stack(batch["label"]).t()
        for i in range(39):
            loss = loss + F.binary_cross_entropy_with_logits(y_hat[:, i], batch_labels_tensor[:, i].float(), pos_weight=pos_lambda) 
        loss/=39
        acc = self.val_accuracy(torch.sigmoid(y_hat), batch_labels_tensor.float())
        
        ap_scores = []
        for label_index in range(len(batch_labels_array)):
            y_true_label = batch_labels_array[label_index, :]
            y_pred_label = y_hat_array[label_index, :]
            ap = average_precision_score(y_true_label, y_pred_label)
            ap_scores.append(ap)
        # Compute mean average precision
        mAP = sum(ap_scores) / len(ap_scores)

        #mask_pos_labels = batch_labels_tensor > 0.5
        ##dimensions=mask_pos_labels.size()
        #pos_sum = 0; neg_sum = 0
        #for i in range(dimensions[0]):
        #    for j in range(dimensions[1]):
        #        pos_sum += ((torch.sigmoid(y_hat[i,j]) > 0.5) == 1) and mask_pos_labels[i,j]
        #        neg_sum += ((torch.sigmoid(y_hat[i,j]) > 0.5) == 1) and (not mask_pos_labels[i,j])
        #max_pos = mask_pos_labels.sum()
        #pos_acc = pos_sum/(max_pos + neg_sum)
        y_pred_binary = (torch.sigmoid(y_hat) > 0.5).float().cpu()
        precision_micro = precision_score(batch_labels_array, y_pred_binary, average='micro', zero_division=0)
        #precision_macro = precision_score(batch_labels_array, y_pred_binary, average='macro')

        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.val_dataloader()) + batch_idx
        writer.add_scalar("val/precision_micro", float(precision_micro), iteration)
        #writer.add_scalar("val/precision_macro", float(precision_macro), iteration)
        writer.add_scalar("val/loss", float(loss), iteration)
        writer.add_scalar("val/acc", float(acc), iteration)
        writer.add_scalar("val/mAP", float(mAP), iteration)

        return loss
    
    def configure_optimizers(self): #not neck
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.args.lr,
            #momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, self.args.max_epochs, last_epoch=-1
        )

        return [optimizer], [scheduler]

# def compute_class_weights(targets):
#     # Compute class weights based on the inverse of class frequencies
#     class_counts = torch.bincount(targets.long())
#     class_weights = 1.0 / (class_counts.float() + 1e-5)  # avoid division by zero
#     return class_weights / class_weights.sum()

#class PackPathway(torch.nn.Module):
    """
    Transform for converting video frames as a list of tensors for slowfast.
    """
#    def __init__(self):
        
#        super().__init__()

#    def forward(self, frames: torch.Tensor):
#        alpha = 4
#        fast_pathway = frames
#        # Perform temporal sampling from the fast pathway.
#        slow_pathway = torch.index_select(
#            frames,
#            1,
#            torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // alpha).long(),
#        )
#       frame_list = [slow_pathway, fast_pathway]
#       return frame_list

class TrafficDataModule(pytorch_lightning.LightningDataModule):
    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str): #not neck
        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]
        output = Compose(transform)
        return output

    def _video_transform(self, mode: str): #not neck
        args = self.args
        if self.args.arch == "slowfast":
            num_frames = 32 #it must be 32
            output = ApplyTransformToKey(
                key="video",
                transform=Compose( #already tensor
                    [
                        UniformTemporalSubsample(num_frames),
                        Lambda(lambda x: x/255.0),
                        NormalizeVideo(args.video_means, args.video_stds),
                        ShortSideScale(
                            size=args.video_min_short_side_scale
                        ),
                        CenterCropVideo(args.video_crop_size),
                    ]
                ),
            )
            return output
        else:  ###bottleneck reduce transform
            output = ApplyTransformToKey(
                key="video",
                transform=Compose( #already tensor
                    [
                        UniformTemporalSubsample(args.video_num_subsampled),
                        Normalize(args.video_means, args.video_stds),
                    ]
                    + (
                        [
                            RandomShortSideScale(
                                min_size=args.video_min_short_side_scale,
                                max_size=args.video_max_short_side_scale,
                            ),
                            RandomCrop(args.video_crop_size),
                            RandomHorizontalFlip(p=args.video_horizontal_flip_p),
                        ]
                        if mode == "train"
                        else [
                            ShortSideScale(args.video_min_short_side_scale),
                            CenterCrop(args.video_crop_size),
                        ]
                    )
                ),
            )
            return output

    def train_dataloader(self): #not neck
        #sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        #if self.args.arch!="slowfast":
        #    train_set = TACO(args=args)
        #    dataloader_train = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        #    output = dataloader_train
        #else:
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            Trafficdataloader(
                data_path=os.path.join(self.args.data_path, "train.csv"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        output = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )
        return output


    def val_dataloader(self): #not neck
        #sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        #if self.args.arch!="slowfast":
        #    val_set = TACO(args=args, training=False)                
        #    dataloader_val = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
        #    output = dataloader_val
        #else:
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        val_transform = self._make_transforms(mode="val")
        self.val_dataset = Trafficdataloader(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        output = torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )
        return output


class LimitDataset(torch.utils.data.Dataset): #not neck
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.dataset_iter = itertools.chain.from_iterable(
            itertools.repeat(iter(dataset), 2)
        )


    def __getitem__(self, index):
        return next(self.dataset_iter)

    def __len__(self):
        return self.dataset.num_videos


def main():

    setup_logger()

    pytorch_lightning.trainer.seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_labeling", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Model parameters.
    parser.add_argument("--lr", "--learning-rate", default=0.1, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument(
        "--arch",
        default="x3d",
        choices=["slowfast", "csn", "x3d","i3d","ARG","ORN"],
        type=str,
    )

    # Data parameters.
    parser.add_argument("--data_path", default="/home/magecliff/UAV-benchmark-M", type=str, required=True)
    parser.add_argument("--video_path_prefix", default="/home/magecliff/Traffic_Recognition/frame", type=str)
    parser.add_argument("--workers", default=16, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--clip_duration", default=3, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=16, type=int)
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
    parser.add_argument("--video_horizontal_flip_p", default=0.5, type=float)
    parser.add_argument("--audio_raw_sample_rate", default=44100, type=int)
    parser.add_argument("--audio_resampled_rate", default=16000, type=int)
    parser.add_argument("--audio_mel_window_size", default=32, type=int)
    parser.add_argument("--audio_mel_step_size", default=16, type=int)
    parser.add_argument("--audio_num_mels", default=80, type=int)
    parser.add_argument("--audio_mel_num_subsample", default=128, type=int)
    parser.add_argument("--audio_logmel_mean", default=-7.03, type=float)
    parser.add_argument("--audio_logmel_std", default=4.66, type=float)

    # Trainer parameters.
    parser = pytorch_lightning.Trainer.add_argparse_args(parser)
    parser.set_defaults(
        max_epochs=100,
        callbacks=[LearningRateMonitor()],
        replace_sampler_ddp=False,
        gpus=1,
       # batch_size=16,
       # workers=8,
    )

    # Build trainer, ResNet lightning-module and Kinetics data-module.
    args = parser.parse_args()

    if args.on_cluster:
        copy_and_run_with_config(
            train,
            args,
            args.working_directory,
            job_name=args.job_name,
            time="72:00:00",
            partition=args.partition,
            gpus_per_node=args.gpus,
            ntasks_per_node=args.gpus,
            cpus_per_task=10,
            mem="470GB",
            nodes=args.num_nodes,
            constraint="volta32gb",
        )
    else:  # local
        train(args) #main path

def step_decay(epoch, initial_lr, drop_factor, drop_every):
    return initial_lr * drop_factor**(epoch // drop_every)

def train(args):

    trainer = pytorch_lightning.Trainer.from_argparse_args(args) #nope
    classification_module = VideoClassificationLightningModule(args) #nope
    data_module = TrafficDataModule(args) #nope

    trainer.fit(classification_module, data_module, )
    trainer.save_checkpoint(f'log/model.ckpt')


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
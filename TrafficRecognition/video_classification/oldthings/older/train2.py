# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import cv2
import argparse
import itertools
import logging
import os
#import math
import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models.resnet
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
from torch.utils.tensorboard import SummaryWriter
from pytorch_lightning.callbacks import LearningRateMonitor
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
#from torchaudio.transforms import MelSpectrogram, Resample
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from sklearn.metrics import average_precision_score

# from torchmetrics.detection import MeanAveragePrecision
#from torch.optim import lr_scheduler

writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log')

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True

class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):

        self.args = args
        super().__init__()
        self.train_accuracy = Accuracy(task='multilabel', num_labels=39)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=39)

        if self.args.arch == "video_resnet":
            self.model = pytorchvideo.models.resnet.create_resnet(
                input_channel=3,
                #model_num_class=400,
                model_num_class=39,
            ).to(device="cuda")

            self.batch_key = "video"


    def on_train_epoch_start(self):

        epoch = self.trainer.current_epoch
        #if self.trainer.use_ddp:
        if self.trainer.accelerator == "ddp":
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch[self.batch_key]
        y_hat = self.model(x)


        loss = 0
        batch_labels_tensor = torch.stack(batch["label"]).t()
        for i in range(39):
           loss = loss + F.binary_cross_entropy_with_logits(y_hat[:, i], batch_labels_tensor[:, i].float()) 
        acc = self.train_accuracy(torch.sigmoid(y_hat), batch_labels_tensor.float())


        # Convert tensors to NumPy arrays
        y_hat_array = torch.sigmoid(y_hat).detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()
        
        ap_scores = []
        for label_index in range(39):
            y_true_label = batch_labels_array[:, label_index]
            y_pred_label = y_hat_array[:, label_index]
            
            ap = average_precision_score(y_true_label, y_pred_label)
            ap_scores.append(ap)

        # Compute mean average precision
        mAP = sum(ap_scores) / len(ap_scores)

        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.train_dataloader()) + batch_idx
        writer.add_scalar("train/loss", float(loss), iteration)
        writer.add_scalar("train/acc", float(acc), iteration)
        writer.add_scalar("train/mAP", float(mAP), iteration)
        
        return loss

    def validation_step(self, batch, batch_idx):

        x = batch[self.batch_key]
        y_hat = self.model(x)

        # Convert tensors to NumPy arrays
        y_hat_array = torch.sigmoid(y_hat).detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()

        # Save y_hat and batch_labels to CSV files
        np.savetxt(f'log/y_hat_{batch_idx}.csv', y_hat_array, delimiter=',')
        np.savetxt(f'log/batch_labels_{batch_idx}.csv', batch_labels_array, delimiter=',')

        loss = 0
        batch_labels_tensor = torch.stack(batch["label"]).t()
        for i in range(39):
            loss = loss + F.binary_cross_entropy_with_logits(y_hat[:, i], batch_labels_tensor[:, i].float()) 

        acc = self.val_accuracy(torch.sigmoid(y_hat), batch_labels_tensor.float())



        ap_scores = []
        for label_index in range(39):
            y_true_label = batch_labels_array[:, label_index]
            y_pred_label = y_hat_array[:, label_index]
            
            ap = average_precision_score(y_true_label, y_pred_label)
            ap_scores.append(ap)

        # Compute mean average precision
        mAP = sum(ap_scores) / len(ap_scores)

        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.val_dataloader()) + batch_idx
        writer.add_scalar("val/loss", float(loss), iteration)
        writer.add_scalar("val/acc", float(acc), iteration)
        writer.add_scalar("val/mAP", float(mAP), iteration)



        return loss
    
    def configure_optimizers(self):

        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.lr,
            momentum=self.args.momentum,
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

class KineticsDataModule(pytorch_lightning.LightningDataModule):

    def __init__(self, args):
        self.args = args
        super().__init__()

    def _make_transforms(self, mode: str):

        if self.args.data_type == "video":
            transform = [
                self._video_transform(mode),
                RemoveKey("audio"),
            ]

        return Compose(transform)

    def _video_transform(self, mode: str):

        args = self.args
        return ApplyTransformToKey(
            key="video",
            transform=Compose(
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

    def train_dataloader(self):

        #sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        train_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            pytorchvideo.data.Kinetics(
                data_path=os.path.join(self.args.data_path, "train.csv"),
                clip_sampler=pytorchvideo.data.make_clip_sampler(
                    "random", self.args.clip_duration
                ),
                video_path_prefix=self.args.video_path_prefix,
                transform=train_transform,
                video_sampler=sampler,
            )
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )

    def val_dataloader(self):

        #sampler = DistributedSampler if self.trainer.use_ddp else RandomSampler
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        val_transform = self._make_transforms(mode="val")
        self.val_dataset = pytorchvideo.data.Kinetics(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            clip_sampler=pytorchvideo.data.make_clip_sampler(
                "uniform", self.args.clip_duration
            ),
            video_path_prefix=self.args.video_path_prefix,
            transform=val_transform,
            video_sampler=sampler,
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            num_workers=self.args.workers,
        )


class LimitDataset(torch.utils.data.Dataset):

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
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument(
        "--arch",
        default="video_resnet",
        choices=["video_resnet", "audio_resnet"],
        type=str,
    )

    # Data parameters.
    parser.add_argument("--data_path", default="/home/magecliff/UAV-benchmark-M", type=str, required=True)
    parser.add_argument("--video_path_prefix", default="/home/magecliff/Traffic_Recognition/frame", type=str)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--clip_duration", default=2, type=float)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_num_subsampled", default=8, type=int)
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
        train(args)


def train(args):
    trainer = pytorch_lightning.Trainer.from_argparse_args(args)
    classification_module = VideoClassificationLightningModule(args)
    data_module = KineticsDataModule(args)
    trainer.fit(classification_module, data_module)


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


if __name__ == "__main__":
    main()
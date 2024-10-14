# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from tqdm import tqdm
import random
import warnings
import time
import datetime
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything

from slurm import copy_and_run_with_config
from torch.utils.data import DistributedSampler, RandomSampler
from torch.optim import lr_scheduler

from sklearn.metrics import average_precision_score


from traffic_dataset import *
from collections import OrderedDict
import x3d_rus as x3d
import csn_rus as csn
import slowfast_rus as slowfast
import action_slot as action_slot
import i3d_kinetics as i3d_kinetics
import ARG_rus as ARG
import ORN_rus as ORN
import vivit_rus as vivit
import mvit_rus as mvit
from loss import ActionSlotLoss
import vmae_utils
import modeling_finetune

from timm.models import create_model
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    ToTensor,
)
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

# Initialization
logs_directory = r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log'; writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/log')
#logs_directory = r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/testland/log'; writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/testland/log')
#logs_directory = r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/underconstruction/log'; writer = SummaryWriter(r'/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/underconstruction/log')

torch.backends.cudnn.enable =True
torch.backends.cudnn.benchmark = True


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self, args):
        model_num_class=40
        self.args = args
        super().__init__()
        self.train_accuracy = Accuracy(task='multilabel', num_labels=model_num_class)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=model_num_class)



        if self.args.arch == 'vivit':
            self.model = vivit.ViViT((224, 224), patch_size=16, num_frames=32, num_actor_class=model_num_class)
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == 'mvit':
            self.model = mvit.MViT(num_actor_class=model_num_class, scale=-1.0)
            tune_block_idx = [0,1,-2,-1]
            for t in self.model.parameters():
                t.requires_grad=False
            for t in self.model.head.parameters():
                t.requires_grad=True
                t = nn.init.trunc_normal_(t)
            for t in self.model.model.cls_positional_encoding.parameters():
                t.requires_grad=True
            for t in self.model.model.patch_embed.parameters():
                t.requires_grad=True
            for t in self.model.model.norm_embed.parameters():
                t.requires_grad=True
            # for t in model.custom_posembed.parameters():
            # 	t.requires_grad=True
            for idx in tune_block_idx:
                for t in self.model.model.blocks[idx].parameters():
                    t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == 'videoMAE':
            tune_block_idx = [0,1,-2,-1]
            # print(tune_block_idx)
            self.model = video_mae(scale=-1.0)
            self.model = mvit.VideoMAE_pretrained(video_mae=self.model, num_actor_class=model_num_class)
            # for t in self.model.named_parameters():
            #     print(t[0])
            # raise BaseException
            for t in self.model.parameters():
                t.requires_grad=False
            for t in self.model.head.parameters():
                t.requires_grad=True
                t = nn.init.trunc_normal_(t)
            for t in self.model.model.patch_embed.parameters():
                t.requires_grad=False
            for t in self.model.model.fc_norm.parameters():
                t.requires_grad=True
            for idx in tune_block_idx:
                for t in self.model.model.blocks[idx].parameters():
                    t.requires_grad=False
            self.batch_key = "video"
            self.model.to(device="cuda")

        if self.args.arch == "action_slot":
            self.model = action_slot.ACTION_SLOT(args,num_class=model_num_class)
            for t in self.model.parameters():
                t.requires_grad=False
            for t in self.model.model[-1].parameters():
                t.requires_grad=True
            for t in self.model.model[-2].parameters():
                t.requires_grad=True
            for t in self.model.model[-3].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")
            if hasattr(self.model, 'resolution'):
                attention_res = (self.model.resolution[0]*args.bg_upsample, self.model.resolution[1]*args.bg_upsample)
            else:
                attention_res = None
            self.criterion = ActionSlotLoss(args, model_num_class, attention_res).to(device="cuda")


        if self.args.arch == "x3d": #functional
            self.model = x3d.X3D(num_actor_class=model_num_class, args=self.args)
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
            self.model = csn.CSN(num_actor_class=model_num_class)
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
            self.model = slowfast.SlowFast(num_actor_class=model_num_class)
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
            self.model = i3d_kinetics.I3D_KINETICS(num_actor_class=model_num_class)
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
                self.model = ARG.ARG(args, max_N=64, num_actor_class=model_num_class)
            else:
                self.model = ORN.ORN(args, max_N=64, num_actor_class=model_num_class)
            for t in self.model.parameters():
                t.requires_grad=True
            for t in self.model.resnet.parameters():
                t.requires_grad=False
            for t in self.model.resnet[-1].parameters():
                t.requires_grad=True
            for t in self.model.resnet[-2].parameters():
                t.requires_grad=True
            self.batch_key = "video"
            self.model.to(device="cuda")

    def on_train_epoch_start(self): #not neck
        epoch = self.trainer.current_epoch
        #if self.trainer.use_ddp:
        if self.trainer.accelerator == "ddp":
            self.trainer.datamodule.train_dataset.dataset.video_sampler.set_epoch(epoch)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx): #not bottle neck
        x = batch[self.batch_key]
        batch_labels_tensor = torch.stack(batch["label"]).t()

        ### Label Transform for flipping ###
        # flip = random.choice([True, False])
        # if flip == True:
        #     x = torch.flip(original_input, dims=[-1]) #horizontal flipping
        #     index_transformation = {0: 36, 1: 37, 2: 8, 3: 9, 4: 4, 5: 5, 6: 30, 7: 31, 8: 2, 9: 3, 10: 26, 11: 27, 12: 32, 13: 33, 14: 22, 15: 23, 16: 38, 17: 39, 18: 34, 19: 35, 20: 20, 21: 21, 22: 14, 23: 15, 24: 28, 25: 29, 26: 10, 27: 11, 28: 24, 29: 25, 30: 6, 31: 7, 32: 12, 33: 13, 34: 18, 35: 19, 36: 0, 37: 1, 38: 16, 39: 17}
        #     new_indices = torch.tensor([index_transformation[i] for i in range(40)], device="cuda") #creating index based on transform
        #     batch_labels_tensor = torch.index_select(original_label, 1, new_indices) #applying transform
        # else:
            # x = original_input
            # batch_labels_tensor = original_label
            
        ### Combine Label ###
        # if self.args.arch == "rusnet":
        #     sub_label = torch.zeros((batch_labels_tensor.size(0), batch_labels_tensor.size(1) // 2), device="cuda")
        #     for i in range(0, batch_labels_tensor.size(1), 2):
        #         sub_label[:, i // 2] = torch.max(batch_labels_tensor[:, i], batch_labels_tensor[:, i + 1])
        #     #labels = ['12p','12v','13v','14p','14v','21p','21v','23p','23v','24v','31v','32p','32v','34p','34v','41p','41v','42v','43p','43v']
        #     #         [  0  ,  1  ,  2  ,  3  ,  4  ,  5  ,  6  ,  7  ,  8  ,  9  ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ,  17 ,  18 ,  19 ]
        #     combined_labels = {
        #         0: [0, 3],
        #         1: [1, 2, 4],
        #         2: [5, 7],
        #         3: [6, 8, 9],
        #         4: [11, 13],
        #         5: [10, 12, 14],
        #         6: [15, 18],
        #         7: [16, 17, 19]
        #     }
        #     ternary_label = torch.zeros(batch_labels_tensor.size(0), len(combined_labels), device="cuda")
        #     for n_idx, o_idx in combined_labels.items():
        #         ternary_label[:, n_idx] = (sub_label[:, o_idx].max(dim=1).values > 0).float()


        ### for ARG and ORN with special boxing needs ###
        if self.args.arch == 'ARG' or self.args.arch == 'ORN':
            box_in = batch['bbox']
            if isinstance(box_in,np.ndarray):
                boxes = torch.from_numpy(box_in).to(device="cuda", dtype=torch.float32)
            else:
                boxes = box_in.to(device="cuda", dtype=torch.float32)
            y_hat = torch.sigmoid(self.model(x, boxes))
        elif self.args.arch == "action_slot":
            y_hat_og, attn_masks = self.model(x)
            y_hat = torch.sigmoid(y_hat_og)
            if self.args.bg_attn_weight > 0:
                bg_attn_mask = batch['bg_masks']
                bg_attn_mask = torch.max(bg_attn_mask / 255, dim=1)[0].long()
        else:
            y_hat = torch.sigmoid(self.model(x))

        # elif self.args.arch == "rusnet":
        #     y_hat, sub_y_hat, tri_y_hat = self.model(x)
        #     y_hat = torch.sigmoid(y_hat)
        #     sub_y_hat = torch.sigmoid(sub_y_hat)
        #     tri_y_hat = torch.sigmoid(tri_y_hat)

        ### Loss Calculation ###
        if self.args.arch == "action_slot":
            if self.args.bg_attn_weight == 0:
                main_loss, attention_loss = self.criterion({'actor':y_hat_og.float(),'attn':attn_masks},{'actor':batch_labels_tensor.float()}, False)  # False if mode == 'train' else True
                weighted_bg_loss = 0
            else:
                main_loss, attention_loss = self.criterion({'actor':y_hat_og.float(),'attn':attn_masks},{'actor':batch_labels_tensor.float(), 'bg_seg':bg_attn_mask}, False)  # False if mode == 'train' else True
                weighted_bg_loss = attention_loss['bg_attn_loss'] * self.args.bg_attn_weight
            weighted_attn_loss = attention_loss['attn_loss'] * self.args.action_attn_weight
            loss = main_loss + weighted_attn_loss + weighted_bg_loss
        else:
            pos_lambda = torch.tensor([self.args.non_as_pos_weight],device=y_hat.device) #effect is 1 => +100%
            main_loss = F.binary_cross_entropy(y_hat.float(), batch_labels_tensor.float(), reduction='none')
            weighted_main_loss = (batch_labels_tensor.float() * (pos_lambda) + 1) * main_loss
            loss = weighted_main_loss.mean()
        # if self.args.arch == "rusnet":
        #     sub_loss = F.binary_cross_entropy(sub_y_hat.float(), sub_label.float(), reduction='none')
        #     tri_loss = F.binary_cross_entropy(tri_y_hat.float(), ternary_label.float(), reduction='none')
        #     if self.current_epoch <=10:
        #         loss = (self.current_epoch/30)*weighted_main_loss.mean() + (self.current_epoch/10)*sub_loss.mean() + (1-self.current_epoch/10)*tri_loss.mean()
        #     elif self.current_epoch <=30:
        #         loss = (self.current_epoch/30)*weighted_main_loss.mean() + sub_loss.mean()# + 0.1*tri_loss.mean()
        #     else:
        #         loss = weighted_main_loss.mean()# + 0.1*sub_loss.mean() + 0.1*tri_loss.mean()
        # else:

        # Convert tensors to NumPy arrays
        y_hat_array = y_hat.detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()

        ### AP Scores ###
        ap_scores = []
        for label_index in range(len(batch_labels_array)):
            y_true_label = batch_labels_array[label_index, :]
            y_pred_label = y_hat_array[label_index, :]
            if np.sum(y_true_label) > 0:
                ap = average_precision_score(y_true_label, y_pred_label)
                ap_scores.append(ap)
        # Compute mean average precision
        mAP = np.mean(ap_scores) if ap_scores else 0
        acc = self.train_accuracy(y_hat, batch_labels_tensor.float())
        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.train_dataloader()) + batch_idx
        writer.add_scalar("train/loss", float(loss), iteration)
        writer.add_scalar("train/acc", float(acc), iteration)
        writer.add_scalar("train/mAP", float(mAP), iteration)

        return loss

    def validation_step(self, batch, batch_idx): #not neck
        x = batch[self.batch_key]
        batch_labels_tensor = torch.stack(batch["label"]).t()

        ### Model differences ###
        if self.args.arch == 'ARG' or self.args.arch == 'ORN':
            box_in = batch['bbox']
            if isinstance(box_in,np.ndarray):
                boxes = torch.from_numpy(box_in).to(device="cuda", dtype=torch.float32)
            else:
                boxes = box_in.to(device="cuda", dtype=torch.float32)
            y_hat = torch.sigmoid(self.model(x, boxes))
        elif self.args.arch == "action_slot":
            y_hat_og, attn_masks = self.model(x)
            y_hat = torch.sigmoid(y_hat_og)
            if self.args.bg_attn_weight > 0:
                bg_attn_mask = batch['bg_masks']
                bg_attn_mask = torch.max(bg_attn_mask / 255, dim=1)[0].long()
        else:
            y_hat = torch.sigmoid(self.model(x))
        # elif self.args.arch == "rusnet":
        #     y_hat, sub_y_hat, tri_y_hat = self.model(x)
        #     y_hat = torch.sigmoid(y_hat)
        #     sub_y_hat = torch.sigmoid(sub_y_hat)
        #     tri_y_hat = torch.sigmoid(tri_y_hat)


        ### Results Saving ###
        # Convert tensors to NumPy arrays
        y_hat_array = y_hat.detach().cpu().numpy()
        batch_labels_array = torch.stack(batch["label"]).t().cpu().numpy()
        # Save y_hat and batch_labels to CSV files
        video_names = batch['video_name']
        video_names_array = np.array(video_names).reshape(-1, 1)
        #print(y_hat_array.shape)
        y_hat_with_names = np.hstack([video_names_array, y_hat_array])
        batch_labels_with_names = np.hstack([video_names_array, batch_labels_array])
        headline_list = [ "Video Name",
            '12p', '12p+', '12v', '12v+',
            '13v', '13v+', '14p', '14p+', '14v', '14v+',
            '21p', '21p+', '21v', '21v+',
            '23p', '23p+', '23v', '23v+', '24v', '24v+',
            '31v', '31v+', '32p', '32p+', '32v', '32v+',
            '34p', '34p+', '34v', '34v+',
            '41p', '41p+', '41v', '41v+',
            '42v', '42v+', '43p', '43p+', '43v', '43v+'
            ]
        headline_array = np.array([headline_list])
        headline_array = headline_array.reshape(1, -1)
        y_hat_wh = np.vstack([headline_array, y_hat_with_names])
        batch_labels_wh = np.vstack([headline_array, batch_labels_with_names])
        y_hat_wh = y_hat_wh.astype(str)
        batch_labels_wh = batch_labels_wh.astype(str)
        np.savetxt(f'log/y_hat_{batch_idx}.csv', y_hat_wh, delimiter=',', fmt='%s')
        np.savetxt(f'log/batch_labels_{batch_idx}.csv', batch_labels_wh, delimiter=',', fmt='%s')

        ### Combine Label ###
        # if self.args.arch == "rusnet":
        #     sub_label = torch.zeros((batch_labels_tensor.size(0), batch_labels_tensor.size(1) // 2), device="cuda")
        #     for i in range(0, batch_labels_tensor.size(1), 2):
        #         sub_label[:, i // 2] = torch.max(batch_labels_tensor[:, i], batch_labels_tensor[:, i + 1])
        #     #labels = ['12p','12v','13v','14p','14v','21p','21v','23p','23v','24v','31v','32p','32v','34p','34v','41p','41v','42v','43p','43v']
        #     #         [  0  ,  1  ,  2  ,  3  ,  4  ,  5  ,  6  ,  7  ,  8  ,  9  ,  10 ,  11 ,  12 ,  13 ,  14 ,  15 ,  16 ,  17 ,  18 ,  19 ]
        #     combined_labels = {
        #         0: [0, 3],
        #         1: [1, 2, 4],
        #         2: [5, 7],
        #         3: [6, 8, 9],
        #         4: [11, 13],
        #         5: [10, 12, 14],
        #         6: [15, 18],
        #         7: [16, 17, 19]
        #     }
        #     ternary_label = torch.zeros(batch_labels_tensor.size(0), len(combined_labels), device="cuda")
        #     for n_idx, o_idx in combined_labels.items():
        #         ternary_label[:, n_idx] = (sub_label[:, o_idx].max(dim=1).values > 0).float()
    

        ### Loss Calculation ###
        if self.args.arch == "action_slot":
            if self.args.bg_attn_weight == 0:
                main_loss, attention_loss = self.criterion({'actor':y_hat_og.float(),'attn':attn_masks},{'actor':batch_labels_tensor.float()}, True)  # False if mode == 'train' else True
                weighted_bg_loss = 0
            else:
                main_loss, attention_loss = self.criterion({'actor':y_hat_og.float(),'attn':attn_masks},{'actor':batch_labels_tensor.float(), 'bg_seg':bg_attn_mask}, True)  # False if mode == 'train' else True
                weighted_bg_loss = attention_loss['bg_attn_loss'] * self.args.bg_attn_weight
            weighted_attn_loss = attention_loss['attn_loss'] * self.args.action_attn_weight
            loss = main_loss + weighted_attn_loss + weighted_bg_loss
        else:
            pos_lambda = torch.tensor([self.args.non_as_pos_weight],device=y_hat.device) #effect is 1 => +100%
            main_loss = F.binary_cross_entropy(y_hat.float(), batch_labels_tensor.float(), reduction='none')
            weighted_main_loss = (batch_labels_tensor.float() * (pos_lambda) + 1) * main_loss
            loss = weighted_main_loss.mean()
        # if self.args.arch == "rusnet":
        #     sub_loss = F.binary_cross_entropy(sub_y_hat.float(), sub_label.float(), reduction='none')
        #     tri_loss = F.binary_cross_entropy(tri_y_hat.float(), ternary_label.float(), reduction='none')
        #     if self.current_epoch <=10:
        #         loss = (self.current_epoch/30)*weighted_main_loss.mean() + (self.current_epoch/10)*sub_loss.mean() + (1-self.current_epoch/10)*tri_loss.mean()
        #     elif self.current_epoch <=30:
        #         loss = (self.current_epoch/30)*weighted_main_loss.mean() + sub_loss.mean()# + 0.1*tri_loss.mean()
        #     else:
        #         loss = weighted_main_loss.mean()# + 0.1*sub_loss.mean() + 0.1*tri_loss.mean()
        # else:
        
        ### AP Scores ###
        ap_scores = []
        for label_index in range(len(batch_labels_array)):
            y_true_label = batch_labels_array[label_index, :]
            y_pred_label = y_hat_array[label_index, :]
            if np.sum(y_true_label) > 0:
                ap = average_precision_score(y_true_label, y_pred_label)
                ap_scores.append(ap)
        # Compute mean average precision #precision micro and mAP here differs by rounding
        mAP = np.mean(ap_scores) if ap_scores else 0
        if mAP > getattr(self, 'best_val_mAP', 0):  # This also handles the case if best_val_mAP is not yet set
            self.best_val_mAP = mAP
        self.log("best_val_mAP", self.best_val_mAP, on_step=False, on_epoch=True)
        self.log('val_mAP', mAP, prog_bar=True)

        y_pred_binary = (y_hat > 0.5).float().cpu().numpy()
        precision_micro = precision_score(batch_labels_array, y_pred_binary, average='micro', zero_division=0)
        acc = self.val_accuracy(y_hat, batch_labels_tensor.float())
        iteration = self.trainer.current_epoch * len(self.trainer.datamodule.val_dataloader()) + batch_idx
        writer.add_scalar("val/precision_micro", float(precision_micro), iteration)
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
        if self.args.arch != 'mvit':
            output = ApplyTransformToKey(
                key="video",
                transform=Compose( #already tensor
                    [
                        #UniformTemporalSubsample(16),
                        Lambda(lambda x: x/255.0),
                        Normalize(args.video_means, args.video_stds),
                    ]
                ),
            )
        else:
            output = ApplyTransformToKey(
                key="video",
                transform=Compose( #already tensor
                    [
                        UniformTemporalSubsample(16),
                        Lambda(lambda x: x/255.0),
                        Normalize(args.video_means, args.video_stds),
                    ]
                ),
            )
        return output


    def train_dataloader(self): #not neck
        boxing = False; bg_mask = False
        if self.args.arch == 'ARG' or self.args.arch == 'ORN':
            boxing = True
        if self.args.arch == 'action_slot' and self.args.bg_attn_weight > 0:
            bg_mask = True
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        data_transform = self._make_transforms(mode="train")
        self.train_dataset = LimitDataset(
            Trafficdataloader(
                data_path=os.path.join(self.args.data_path, "train.csv"),
                box = boxing,
                bg_mask = bg_mask,
                transform=data_transform,
                video_path_prefix=self.args.video_path_prefix,
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
        boxing = False; bg_mask = False
        if self.args.arch == 'ARG' or self.args.arch == 'ORN':
            boxing = True
        if self.args.arch == 'action_slot' and self.args.bg_attn_weight > 0:
            bg_mask = True
        sampler = DistributedSampler if self.trainer.accelerator == 'ddp' else RandomSampler
        data_transform = self._make_transforms(mode="val")
        self.val_dataset = Trafficdataloader(
            data_path=os.path.join(self.args.data_path, "val.csv"),
            box = boxing,
            bg_mask = bg_mask,
            transform=data_transform,
            video_path_prefix=self.args.video_path_prefix,
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

class BestMAPCallback(pytorch_lightning.Callback):
    def __init__(self, monitor="val_mAP"):
        super().__init__()
        self.monitor = monitor
        self.best_mAP = 0

    def on_validation_end(self, trainer, pl_module):
        super().on_validation_end(trainer, pl_module)
        if self.monitor in trainer.callback_metrics:
            current_mAP = trainer.callback_metrics[self.monitor]
            if current_mAP > self.best_mAP:
                self.best_mAP = current_mAP
                current_epoch = trainer.current_epoch
                tqdm.write(f"Epoch: {current_epoch}; Best mAP: {self.best_mAP}; Current time: {datetime.datetime.now()}")
            writer.add_scalar("val/epoch_mAP", float(current_mAP), trainer.current_epoch)

def main():

    setup_logger()
    warnings.filterwarnings('ignore')
    seed = seed_everything()
    parser = argparse.ArgumentParser()

    #  Cluster parameters.
    parser.add_argument("--on_cluster", action="store_true")
    parser.add_argument("--job_name", default="ptv_video_labeling", type=str)
    parser.add_argument("--working_directory", default=".", type=str)
    parser.add_argument("--partition", default="dev", type=str)

    # Loss parameters.
    parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-3, type=float)
    parser.add_argument('--non_as_pos_weight', type=float, default=1, help='') #for non-action-slot, should be 1 for other methods (1+1)
    parser.add_argument('--bce_pos_weight', type=float, default=10, help='') #originally 10 for AS (1+0)
    parser.add_argument('--ce_neg_weight', type=float, default=0.05, help='')
    parser.add_argument('--bg_attn_weight', type=float, default=0.5, help='')
    parser.add_argument('--action_attn_weight', type=float, default=1., help='')


    # Model parameters.
    parser.add_argument(
        "--arch",
        default="slowfast",
        choices=["slowfast", "csn", "x3d","i3d","ARG","ORN","vivit","mvit",'videoMAE','action_slot'], #ARG #ORN cannot be used
        type=str,
    )
    parser.add_argument('--backbone', default="x3d-2", type=str)
    parser.add_argument('--channel', type=int, default=256, help='')
    parser.add_argument('--bg_slot', type=bool, default=True)

    # Attention parameters.
    parser.add_argument('--bg_upsample', type=int, default=1, help='')
    parser.add_argument('--mask_every_frame', type=int, default=1, help='')
    parser.add_argument('--vehicle_bg', type=bool, default=True)
    parser.add_argument('--seq_len', type=int, default=32, help='')



    # Data parameters.tnf4_vf2
    parser.add_argument('--dataset', type=str, default='carom', choices=['carom','taco', 'oats', 'nuscenes'])
    parser.add_argument("--data_path", default="/home/magecliff/Traffic_Recognition/Carom2", type=str) #Label
    parser.add_argument("--video_path_prefix", default="/home/magecliff/Traffic_Recognition/Carom2/videos", type=str) #video 
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument(
        "--data_type", default="video", choices=["video", "audio"], type=str
    )
    parser.add_argument("--video_means", default=(0.45, 0.45, 0.45), type=tuple)
    parser.add_argument("--video_stds", default=(0.225, 0.225, 0.225), type=tuple)
    parser.add_argument("--video_crop_size", default=224, type=int)
    parser.add_argument("--video_min_short_side_scale", default=256, type=int)
    parser.add_argument("--video_max_short_side_scale", default=320, type=int)
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
        max_epochs=25,
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
    print(f"Random Seed: {seed}")

def step_decay(epoch, initial_lr, drop_factor, drop_every):
    return initial_lr * drop_factor**(epoch // drop_every)

def train(args):
    #checkpoint_path = os.path.join(logs_directory, "best_model.ckpt")
    checkpoint_callback = ModelCheckpoint(
        dirpath = logs_directory,
        filename='best_model',
        monitor="best_val_mAP",  # Change this to match the monitor used in BestMAPCallback
        mode="max",
        save_top_k=1,
    )
    map_callback = BestMAPCallback(monitor="val_mAP")
    callbacks = [map_callback, checkpoint_callback]
    trainer = pytorch_lightning.Trainer.from_argparse_args(args, callbacks=callbacks) #nope
    classification_module = VideoClassificationLightningModule(args) #nope
    data_module = TrafficDataModule(args) #nope

    trainer.fit(classification_module, data_module, )


def setup_logger():
    ch = logging.StreamHandler()
    formatter = logging.Formatter("\n%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(formatter)
    logger = logging.getLogger("pytorchvideo")
    logger.setLevel(logging.DEBUG)
    logger.addHandler(ch)


def video_mae(scale):
    model_name = "vit_small_patch16_224" # vit_large_patch16_224
    if scale == -1.0:
        input_size = (224,224)
    else:
        input_size = (512//scale,1536//scale)
    window_size, num_frames = (8, 14, 14), 16
    finetune = "/home/magecliff/Traffic_Recognition/pytorchvideo-main/tutorials/video_classification_example/weights/checkpoint_S.pth"
    model_key = "model|module"
    model = create_model( 
        model_name,
        pretrained=False,
        num_classes=400,
        all_frames=32,
        tubelet_size=2,
        fc_drop_rate=0.5,
        drop_rate=0.0,
        drop_path_rate=0.1,
        attn_drop_rate=0.0,
        drop_block_rate=None,
        use_checkpoint=False,
        use_mean_pooling=True,
        init_scale=0.001,
    )
    patch_size = model.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    window_size = (num_frames // 2, input_size[0] // patch_size[0], input_size[1] // patch_size[1])
        # args.patch_size = patch_size

    if finetune:
        checkpoint = torch.load(finetune, map_location='cpu')
        print("Load ckpt from %s" % finetune)
        checkpoint_model = None
        for model_key in model_key.split('|'):
            if model_key in checkpoint:
                checkpoint_model = checkpoint[model_key]
                print("Load state_dict by model_key = %s" % model_key)
                break
        if checkpoint_model is None:
            checkpoint_model = checkpoint
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]

        all_keys = list(checkpoint_model.keys())
        new_dict = OrderedDict()
        for key in all_keys:
            if key.startswith('backbone.'):
                new_dict[key[9:]] = checkpoint_model[key]
            elif key.startswith('encoder.'):
                new_dict[key[8:]] = checkpoint_model[key]
            elif key.startswith('head.'):
                continue
            else:
                new_dict[key] = checkpoint_model[key]
        checkpoint_model = new_dict
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1] # channel dim
            num_patches = model.patch_embed.num_patches # 
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches # 0/1

            # height (== width) for the checkpoint position embedding 
            orig_size = int(((pos_embed_checkpoint.shape[-2] - num_extra_tokens)//(num_frames // model.patch_embed.tubelet_size)) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int((num_patches // (num_frames // model.patch_embed.tubelet_size) )** 0.5)
            # class_token and dist_token are kept unchanged
            if orig_size != new_size:
                print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
                extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                # only the position tokens are interpolated
                pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                # B, L, C -> BT, H, W, C -> BT, C, H, W
                pos_tokens = pos_tokens.reshape(-1, num_frames // model.patch_embed.tubelet_size, orig_size, orig_size, embedding_size)
                pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
                pos_tokens = torch.nn.functional.interpolate(
                    pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
                # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
                pos_tokens = pos_tokens.permute(0, 2, 3, 1).reshape(-1, num_frames // model.patch_embed.tubelet_size, new_size, new_size, embedding_size) 
                pos_tokens = pos_tokens.flatten(1, 3) # B, L, C
                new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                checkpoint_model['pos_embed'] = new_pos_embed

        vmae_utils.load_state_dict(model, checkpoint_model, prefix='')
        return model



if __name__ == "__main__":
    main()



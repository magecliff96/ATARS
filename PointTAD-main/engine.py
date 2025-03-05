# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------
"""
Modified from RTD-Net (https://github.com/MCG-NJU/RTD-Action)

PointTAD Training and Inference functions.

"""

import json
import math
import sys
from typing import Iterable

import torch
from termcolor import colored

import util.misc as utils
from datasets.evaluate import Evaluator
import numpy as np
from sklearn.metrics import average_precision_score
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter

def load_json(file):
    with open(file) as json_file:
        data = json.load(json_file)
        return data


def train_one_epoch(model: torch.nn.Module,
                    criterion: torch.nn.Module,
                    data_loader: Iterable,
                    optimizer: torch.optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    args,
                    postprocessors=None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
   
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1,
                                            fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    max_norm = args.clip_max_norm
    #edit
    all_dense_res = []
    all_dense_gt = []
    #edit
    for vid_name_list, locations, x, targets, num_frames, base in tqdm(data_loader, desc=f"Epoch {epoch}", leave=True):


        x = x.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        outputs = model(x)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys()
                     if k in weight_dict)
        n_parameters = sum(p.numel() for p in model.parameters())
        losses += 0 * n_parameters
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        losses_reduced_scaled = sum(loss_dict_reduced_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print('Loss is {}, stopping training'.format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value,
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])

        #edited
        dense_results = outputs['logits'].sigmoid()

        # Collect dense_res and dense_gt for mAP calculation
        for target, dense_res, base_loc in zip(targets, dense_results, base):
            dense_gt = target['dense_gt']

            # Move dense_res and dense_gt to CPU and convert to NumPy
            dense_res_np = dense_res.detach().cpu().numpy()  # Shape: [64, 40]
            dense_gt_np = dense_gt.detach().cpu().numpy()    # Shape: [64, 40]

            # Collect results for later evaluation
            all_dense_res.append(dense_res_np)
            all_dense_gt.append(dense_gt_np)

        #edited

    metric_logger.synchronize_between_processes()
    #edited
    # Concatenate all dense results and ground truths
    if all_dense_res and all_dense_gt:
        all_dense_res = np.concatenate(all_dense_res, axis=0)  # Shape: [total_frames, 40]
        all_dense_gt = np.concatenate(all_dense_gt, axis=0)    # Shape: [total_frames, 40]

        # Calculate per-class mAP
        num_classes = all_dense_res.shape[1]
        per_class_map = {}
        for class_idx in range(num_classes):
            y_true = all_dense_gt[:, class_idx]  # Ground truths for this class
            y_pred = all_dense_res[:, class_idx]  # Predictions for this class

            # Compute Average Precision for this class
            if np.sum(y_true) > 0:  # Avoid issues with classes having no positive samples
                ap = average_precision_score(y_true, y_pred)
                per_class_map[class_idx] = ap
            else:
                per_class_map[class_idx] = 0.0

        # Calculate overall mAP
        overall_map = np.mean(list(per_class_map.values()))

        # Print results
        print(f"Overall Training mAP: {overall_map:.4f}, Overall Loss: {loss_value:.4f}")

    # writer.add_scalar("Loss/train", loss_value, epoch)
    # writer.add_scalar("mAP/train", overall_map, epoch)
    # writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    # Log per-class mAP
    # for class_idx, ap in per_class_map.items():
    #     writer.add_scalar(f"mAP/Class_{class_idx}", ap, epoch)
    #edit

    return {k: meter.global_avg
            for k, meter in metric_logger.meters.items()}, loss_dict


@torch.no_grad()
# def evaluate(model, criterion, postprocessors, data_loader, device, args):
#     print(colored('evaluate', 'red'))
#     model.eval()
#     criterion.eval()

#     metric_logger = utils.MetricLogger(delimiter='  ')
#     metric_logger.add_meter(
#         'class_error', utils.SmoothedValue(window_size=1,
#                                             fmt='{value:.2f}'))
#     header = 'Test:'

#     evaluator = Evaluator()

#     video_pool = list(load_json(args.annotation_path).keys())
#     video_pool.sort()
#     video_dict = {i: video_pool[i] for i in range(len(video_pool))}
#     for vid_name_list, locations, x, targets, num_frames, base in metric_logger.log_every(
#             data_loader, 10, header):
#         targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
#         x = x.to(device)
#         outputs = model(x)
#         loss_dict = criterion(outputs, targets)
#         weight_dict = criterion.weight_dict

#         loss_dict_reduced = utils.reduce_dict(loss_dict)
#         loss_dict_reduced_scaled = {
#             k: v * weight_dict[k]
#             for k, v in loss_dict_reduced.items() if k in weight_dict
#         }
#         loss_dict_reduced_unscaled = {
#             f'{k}_unscaled': v
#             for k, v in loss_dict_reduced.items()
#         }
#         metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
#                              **loss_dict_reduced_scaled,
#                              **loss_dict_reduced_unscaled)
#         metric_logger.update(class_error=loss_dict_reduced['class_error'])

#         results, dense_results = postprocessors['results'](outputs, num_frames, base)


#         for target, output, dense_res, base_loc in zip(targets, results, dense_results, base):
#             vid = video_dict[target['video_id'].item()]
#             dense_gt = target['dense_gt']
#             if args.dense_result:
#                 # torch.save(dense_res, f'dense_results/{vid}_{base_loc}_dense')
#                 print(dense_res.shape)
#                 print(dense_gt.shape)
#             evaluator.update(vid, output, base_loc)

    # # Gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # evaluator.synchronize_between_processes()
    # print('Averaged stats:', metric_logger)


def evaluate(model, criterion, postprocessors, data_loader, device, args):
    print(colored('evaluate', 'red'))
    model.eval()
    criterion.eval()

    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter(
        'class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}')
    )
    header = 'Test:'

    evaluator = Evaluator()

    video_pool = list(load_json(args.annotation_path).keys())
    video_pool.sort()
    video_dict = {i: video_pool[i] for i in range(len(video_pool))}

    # Collect all dense_res and dense_gt for mAP calculation
    all_dense_res = []
    all_dense_gt = []

    for vid_name_list, locations, x, targets, num_frames, base in tqdm(data_loader, desc=f"Testing", leave=True):
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        x = x.to(device)
        outputs = model(x)
        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict

        loss_dict_reduced = utils.reduce_dict(loss_dict)
        loss_dict_reduced_scaled = {
            k: v * weight_dict[k]
            for k, v in loss_dict_reduced.items() if k in weight_dict
        }
        loss_dict_reduced_unscaled = {
            f'{k}_unscaled': v
            for k, v in loss_dict_reduced.items()
        }
        metric_logger.update(loss=sum(loss_dict_reduced_scaled.values()),
                             **loss_dict_reduced_scaled,
                             **loss_dict_reduced_unscaled)
        metric_logger.update(class_error=loss_dict_reduced['class_error'])

        results, dense_results = postprocessors['results'](outputs, num_frames, base)

        # Collect dense_res and dense_gt for mAP calculation
        for target, output, dense_res, base_loc in zip(targets, results, dense_results, base):
            vid = video_dict[target['video_id'].item()]
            dense_gt = target['dense_gt']

            # Move dense_res and dense_gt to CPU and convert to NumPy
            dense_res_np = dense_res.cpu().numpy()  # Shape: [64, 40]
            dense_gt_np = dense_gt.cpu().numpy()    # Shape: [64, 40]

            # Collect results for later evaluation
            all_dense_res.append(dense_res_np)
            all_dense_gt.append(dense_gt_np)

            evaluator.update(vid, output, base_loc)

    # Concatenate all dense results and ground truths
    if all_dense_res and all_dense_gt:
        dense_list = all_dense_res
        all_dense_res = np.concatenate(all_dense_res, axis=0)  # Shape: [total_frames, 40]
        all_dense_gt = np.concatenate(all_dense_gt, axis=0)    # Shape: [total_frames, 40]

        # Define the class names using headline_list
        headline_list = [
            '12v', '12v+', '13v', '13v+', '14v', '14v+',
            '21v', '21v+', '23v', '23v+', '24v', '24v+',
            '31v', '31v+', '32v', '32v+', '34v', '34v+',
            '41v', '41v+', '42v', '42v+', '43v', '43v+',
            '12p', '14p',
            '21p', '23p',
            '32p', '34p',
            '41p', '43p'
        ]

        # filenames = video_pool; target = "B5_0_0"; amongus=False
        # for k in range(len(filenames)):
        #     print(filenames)
        #     print(len(filenames))
        #     print(len(dense_list))
        #     if filenames[k] == target:
        #         amongus = True
        #         sample = dense_list[k]
        #         file_path = f"outputs/{target}" 
        #         # # print(torch.max(torch.tensor(sample))); print(torch.min(torch.tensor(sample)))
        #         with open(file_path, "w") as file:
        #             for frame_idx in range(sample.shape[0]):
        #                 # Get indices where class is marked as positive
        #                 active_classes = [headline_list[i] for i in range(32) if sample[frame_idx, i] >= 0.5]
        #                 # Join class names and write to file
        #                 file.write(" ".join(active_classes) + "\n")
        # print(amongus)
        # exit()
        # Calculate per-class mAP
        num_classes = all_dense_res.shape[1]
        per_class_map = {}
        for class_idx in range(num_classes):
            y_true = all_dense_gt[:, class_idx]  # Ground truths for this class
            y_pred = all_dense_res[:, class_idx]  # Predictions for this class

            # Compute Average Precision for this class
            if np.sum(y_true) > 0:  # Avoid issues with classes having no positive samples
                ap = average_precision_score(y_true, y_pred)
                per_class_map[class_idx] = ap
            else:
                per_class_map[class_idx] = 0.0

        # Calculate overall mAP
        overall_map = np.mean(list(per_class_map.values()))

        # Print results

        # Print results
        print(f"overall mAP: {overall_map}")
        v_classes = []
        vp_classes = []
        p_classes = []

        for class_id, class_mAP in per_class_map.items():
            class_name = headline_list[class_id]
            print(f"{class_name}: {class_mAP:.4f}")

            # Categorize based on ending character(s)
            if class_name.endswith("v"):
                v_classes.append(class_mAP)
            elif class_name.endswith("v+"):
                vp_classes.append(class_mAP)
            elif class_name.endswith("p"):
                p_classes.append(class_mAP)

        # Compute and print averages
        avg_v = sum(v_classes) / len(v_classes) if v_classes else 0
        avg_vp = sum(vp_classes) / len(vp_classes) if vp_classes else 0
        avg_p = sum(p_classes) / len(p_classes) if p_classes else 0

        print(f"\nAverage mAP for classes ending in 'v': {avg_v:.4f}, 'v+': {avg_vp:.4f}, 'p': {avg_p:.4f}")


    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    evaluator.synchronize_between_processes()
    # print('Averaged stats:', metric_logger)





    return evaluator, loss_dict

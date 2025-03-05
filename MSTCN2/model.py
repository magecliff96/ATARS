#!/usr/bin/python2.7

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger

from sklearn.metrics import average_precision_score
from asformer.model import MyTransformer


class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(out)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x)

        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)

        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, args, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split, device):
        self.arch = args.arch
        if args.arch == 'asformer':
            self.model = MyTransformer(3, args.num_layers, args.r1, args.r2, num_f_maps, dim, num_classes, args.channel_masking_rate)
        else:
            self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes)
        # self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.num_classes = num_classes

        # david
        pos_weight = torch.ones([self.num_classes], device=device)*args.bce_pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='sum', pos_weight=pos_weight)
        self.mse = nn.MSELoss(reduction='none')
        #

        # david
        self.best_val_mAP = 0
        #

        #logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        log_filename = f'logs/bz_{args.bz}_lr_{args.lr}_epoch_{args.num_epochs}_bceposweight_{args.bce_pos_weight}.log'
        logger.add(log_filename)
        
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, batch_gen, batch_gen_val, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            total_valid_elements = 0
            # 初始化用于存储每个类别的预测值和真实标签的字典
            per_class_true = {c: [] for c in range(self.num_classes)}
            per_class_scores = {c: [] for c in range(self.num_classes)}
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()
                if self.arch == "asformer":
                    predictions = self.model(batch_input,mask)
                else:
                    predictions = self.model(batch_input)

                # print(f"mask shape: {mask.shape}")
                # print(f"batch target shape: {batch_target.shape}")

                loss = 0
                for p in predictions:
                    if self.arch == "asformer":
                        loss += self.bce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1, self.num_classes))
                        loss += 0.15 * torch.mean(torch.clamp(
                            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                            max=16) * mask[:, :, 1:])
                    else:
                        # 1. 计算时间平滑的 MSE 损失
                        log_probs_current = F.logsigmoid(p[:, :, 1:])  # 当前时间步的对数概率
                        log_probs_previous = F.logsigmoid(p.detach()[:, :, :-1])  # 前一时间步的对数概率（detach 防止梯度回传）
                        
                        # 计算 MSE 损失
                        mse_loss = self.mse(log_probs_current, log_probs_previous)
                        clamped_mse = torch.clamp(mse_loss, min=0, max=16)
                        masked_mse = clamped_mse * mask[:, :, 1:]  # 应用掩码
                        mean_mse_loss = torch.mean(masked_mse)
                        loss += 0.15 * mean_mse_loss  # 将损失缩放并添加到总损失中

                        # 2. 计算主要的 BCE 损失
                        p_flat = p.permute(0, 2, 1).reshape(-1, self.num_classes)  # (batch_size * seq_len, num_classes)
                        batch_target_flat = batch_target.permute(0, 2, 1).reshape(-1, self.num_classes).float()
                        
                        # 不要对预测和目标应用掩码
                        bce_loss = self.bce(p_flat, batch_target_flat)  # 输出形状：[N, num_classes]
                        
                        # 在损失上应用掩码
                        bce_loss = bce_loss * mask_flat  # 掩码形状：[N, num_classes]
                        
                        # 累加有效位置的损失
                        loss += bce_loss.sum()
                        
                    mask_flat = mask.permute(0, 2, 1).reshape(-1, self.num_classes).float()
                    # 累加有效元素的数量
                    total_valid_elements += mask_flat.sum().item()

                loss.backward()
                optimizer.step()

                # 计算损失
                epoch_loss += loss.item()
                total_valid_elements += mask_flat.sum().item()

                # 收集预测值、真实标签和掩码
                predicted = torch.sigmoid(predictions[-1]).detach().cpu()  # (batch_size, num_classes, seq_len)
                batch_target_cpu = batch_target.detach().cpu()
                mask_cpu = mask.detach().cpu()

                # 对每个类别收集有效的预测值和真实标签
                batch_size, num_classes, seq_len = predicted.shape
                for i in range(batch_size):
                    seq_len_i = int(mask_cpu[i, 0, :].sum().item())
                    for c in range(self.num_classes):
                        y_true = batch_target_cpu[i, c, :seq_len_i].numpy()
                        y_scores = predicted[i, c, :seq_len_i].numpy()
                        if y_true.size > 0:
                            per_class_true[c].extend(y_true.tolist())
                            per_class_scores[c].extend(y_scores.tolist())



            average_epoch_loss = epoch_loss / total_valid_elements if total_valid_elements > 0 else 0
            
            # 在 epoch 结束后计算 mAP
            per_class_ap = []
            for c in range(self.num_classes):
                y_true = np.array(per_class_true[c])
                y_scores = np.array(per_class_scores[c])
                if y_true.size > 0 and np.sum(y_true) > 0:
                    ap = average_precision_score(y_true, y_scores)
                    per_class_ap.append(ap)
            mAP = np.mean(per_class_ap) if per_class_ap else 0

            batch_gen.reset()
            # david
            # 计算验证集的 mAP
            val_mAP = self.validate(batch_gen_val, device)
            logger.info("[epoch %d]: training loss = %f, training mAP = %f, validation mAP = %f" % (
                        epoch + 1, average_epoch_loss, mAP, val_mAP))
            
            # 检查当前的 val_mAP 是否是最好的
            if val_mAP > self.best_val_mAP:
                self.best_val_mAP = val_mAP
                # 如果需要，可以在这里保存最佳模型
                torch.save(self.model.state_dict(), save_dir + "/best.model")
                torch.save(optimizer.state_dict(), save_dir + "/best.opt")
                logger.info(f"New best validation mAP: {self.best_val_mAP:.4f}")
            # else:
            #     logger.info(f"Current validation mAP: {val_mAP:.4f} (Best: {self.best_val_mAP:.4f})")
            
            # torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
            # torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            # logger.info("[epoch %d]: epoch loss = %f,   mAP = %f" % (
            #     epoch + 1, epoch_loss / len(batch_gen.list_of_examples), mAP))
            print("[epoch %d]: training loss = %f, training mAP = %f, validation mAP = %f" % (
                        epoch + 1, average_epoch_loss, mAP, val_mAP))
            print(f"New best validation mAP: {self.best_val_mAP:.4f}")
        
    def validate(self, batch_gen, device):
        self.model.eval()
        per_class_true = {c: [] for c in range(self.num_classes)}
        per_class_scores = {c: [] for c in range(self.num_classes)}
        with torch.no_grad():
            while batch_gen.has_next():
                batch_input, batch_target, mask = batch_gen.next_batch(1)
                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                if self.arch == "asformer":
                    predictions = self.model(batch_input,mask)
                else:
                    predictions = self.model(batch_input)

                # 收集预测值、真实标签和掩码
                predicted = torch.sigmoid(predictions[-1]).cpu()
                batch_target_cpu = batch_target.cpu()
                mask_cpu = mask.cpu()
                batch_size_curr, num_classes_curr, seq_len = predicted.shape
                for i in range(batch_size_curr):
                    seq_len_i = int(mask_cpu[i, 0, :].sum().item())
                    for c in range(self.num_classes):
                        y_true = batch_target_cpu[i, c, :seq_len_i].numpy()
                        y_scores = predicted[i, c, :seq_len_i].numpy()
                        if y_true.size > 0:
                            per_class_true[c].extend(y_true.tolist())
                            per_class_scores[c].extend(y_scores.tolist())
            batch_gen.reset()

        # 计算验证集的 mAP
        per_class_ap = []
        for c in range(self.num_classes):
            y_true = np.array(per_class_true[c])
            y_scores = np.array(per_class_scores[c])
            if y_true.size > 0 and np.sum(y_true) > 0:
                ap = average_precision_score(y_true, y_scores)
                per_class_ap.append(ap)
        mAP = np.mean(per_class_ap) if per_class_ap else 0

        self.model.train()
        return mAP
    #

    # def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
    #     self.model.eval()
    #     with torch.no_grad():
    #         self.model.to(device)
    #         self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
    #         file_ptr = open(vid_list_file, 'r')
    #         list_of_vids = file_ptr.read().split('\n')[:-1]
    #         file_ptr.close()
    #         idx_to_action = {v: k for k, v in actions_dict.items()}
    #         for vid in list_of_vids:
    #             features = np.load(features_path + vid.split('.')[0] + '.npy')
    #             features = features[:, ::sample_rate]
    #             input_x = torch.tensor(features, dtype=torch.float)
    #             input_x.unsqueeze_(0)
    #             input_x = input_x.to(device)
    #             predictions = self.model(input_x)
    #             predicted = (torch.sigmoid(predictions[-1]) > 0.5).float()
    #             predicted = predicted.squeeze(0)  # Remove batch dimension

    #             recognition = []
    #             for t in range(predicted.shape[1]):
    #                 frame_labels = []
    #                 for c in range(self.num_classes):
    #                     if predicted[c, t] == 1:
    #                         frame_labels.append(idx_to_action[c])
    #                 recognition.append(' '.join(frame_labels))

    #             f_name = vid.split('/')[-1].split('.')[0]
    #             f_ptr = open(results_dir + "/" + f_name, "w")
    #             f_ptr.write("### Frame level recognition: ###\n")
    #             f_ptr.write('\n'.join(recognition))
    #             f_ptr.close()
    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict,
            device, sample_rate, gt_path, mapping_file):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()

            # 加载类别映射
            file_ptr = open(mapping_file, 'r')
            actions = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            idx_to_action = {}
            for a in actions:
                idx, action_name = a.split()
                idx_to_action[int(idx)] = action_name

            # 初始化存储预测得分和真实标签的字典
            per_class_true = {c: [] for c in range(self.num_classes)}
            per_class_scores = {c: [] for c in range(self.num_classes)}

            for vid in list_of_vids:
                # 加载特征
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x)
                predicted = torch.sigmoid(predictions[-1]).cpu().squeeze(0)  # (num_classes, seq_len)

                # 加载真实标签
                file_ptr = open(gt_path + vid, 'r')
                content = file_ptr.read().split('\n')[:-1]
                file_ptr.close()
                num_frames = min(predicted.shape[1], len(content))
                gt_labels = np.zeros((self.num_classes, num_frames))
                for t in range(num_frames):
                    labels_list = list(map(int, content[t].split()))
                    gt_labels[:, t] = labels_list

                # 收集每个类别的预测得分和真实标签
                for c in range(self.num_classes):
                    y_true = gt_labels[c, :]
                    y_scores = predicted[c, :num_frames].numpy()
                    per_class_true[c].extend(y_true.tolist())
                    per_class_scores[c].extend(y_scores.tolist())

                # 生成识别结果文件
                recognition = []
                for t in range(predicted.shape[1]):
                    frame_labels = []
                    for c in range(self.num_classes):
                        if predicted[c, t] >= 0.5:
                            frame_labels.append(idx_to_action[c])
                    recognition.append(' '.join(frame_labels))

                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write('\n'.join(recognition))
                f_ptr.close()

            # 计算每个类别的 mAP 和 Recall
            from sklearn.metrics import average_precision_score, recall_score

            per_class_ap = {}
            per_class_recall = {}
            for c in range(self.num_classes):
                y_true = np.array(per_class_true[c])
                y_scores = np.array(per_class_scores[c])
                if np.sum(y_true) > 0:
                    ap = average_precision_score(y_true, y_scores)
                    per_class_ap[idx_to_action[c]] = ap

                    # 将预测得分二值化，阈值为 0.5
                    y_pred_binary = (y_scores >= 0.5).astype(int)
                    recall = recall_score(y_true, y_pred_binary)
                    per_class_recall[idx_to_action[c]] = recall

            # 将结果保存到 txt 文件
            results_file = results_dir + "/per_class_metrics.txt"
            with open(results_file, 'w') as f:
                # f.write("Per-class mAP and Recall:\n")
                # f.write("Class\tmAP\tRecall\n")
                f.write("Per-class mAP:\n")
                f.write("Class\tmAP\n")
                for action in idx_to_action.values():
                    ap = per_class_ap.get(action, 0)
                    recall = per_class_recall.get(action, 0)
                    # f.write(f"{action}\t{ap:.4f}\t{recall:.4f}\n")
                    f.write(f"{action}\t{ap:.4f}\n")


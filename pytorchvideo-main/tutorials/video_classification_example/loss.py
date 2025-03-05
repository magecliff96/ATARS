import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from utils import inter_and_union

class ActionSlotLoss(nn.Module):
    def __init__(self, args, num_actor_class, attention_res=None):
        super(ActionSlotLoss, self).__init__()
        self.args = args
        self.num_actor_class = num_actor_class
        self.attention_res = attention_res
        self.ego_ce = nn.CrossEntropyLoss(reduction='mean')
        self.actor_loss_type =  self._parse_actor_loss(args)
        self.attn_loss_type = self._parse_attn_loss(args)

    def _parse_actor_loss(self,args):
        pos_weight = torch.ones([self.num_actor_class])*args.bce_pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=pos_weight)

    def _parse_attn_loss(self,args):
        flag = 0
        if not args.bg_attn_weight>0. and args.action_attn_weight>0:
            flag = 2
        elif args.bg_attn_weight>0. and args.action_attn_weight>0.:
            flag = 3
        if flag >0:
            self.obj_bce = nn.BCELoss()
        
        return flag
    
    def actor_loss(self, pred, label):
        actor_loss = self.bce(pred, label)
        return actor_loss

    def attn_loss(self, attn, label, actor, validate):
        attn_loss = None
        bg_attn_loss = None
        action_attn = None
        bg_attn = None

        if self.args.bg_attn_weight>0:
            bg_seg = label['bg_seg'].to(device="cuda", dtype=torch.float32)
            bg_attn = attn[:, ::self.args.mask_every_frame, :, :]

        if self.attn_loss_type == 2:
            b, l, n, h, w = attn.shape
            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)
            action_attn = attn[:, :, :self.num_actor_class, :, :]

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = class_idx.permute((0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32).cuda()
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

        elif self.attn_loss_type == 3:
            b, l, n, h, w = attn.shape

            if self.args.bg_upsample != 1:
                attn = attn.reshape(-1, 1, h, w)
                attn = F.interpolate(attn, size=self.attention_res, mode='bilinear')
                _, _, h, w = attn.shape
                attn = attn.reshape(b, l, n, h, w)

            action_attn = attn[:, :, :self.num_actor_class, :, :]

            # if self.args.vehicle_bg:
            #     bg_attn, _ = torch.max(attn[:, ::self.args.mask_every_frame, :24, :, :], dim=2)
            #     #bg_attn = torch.sum(attn[:, ::self.args.mask_every_frame, :24, :, :], dim=2)
            # else:
            bg_attn = attn[:, ::self.args.mask_every_frame, -1, :, :].reshape(b, -1, h, w) 

            class_idx = label['actor'] == 0.0
            class_idx = class_idx.view(b, self.num_actor_class, 1, 1, 1).repeat(1, 1, l, h, w)
            class_idx = class_idx.permute((0, 2, 1, 3, 4))

            attn_gt = torch.zeros([b, l, self.num_actor_class, h, w], dtype=torch.float32).cuda()
            attn_loss = self.obj_bce(action_attn[class_idx], attn_gt[class_idx])

            bg_attn_loss = self.obj_bce(bg_attn, bg_seg)
            # attn_loss = self.args.action_attn_weight*action_attn_loss + self.args.bg_attn_weight*bg_attn_loss
            
        loss = {'attn_loss':attn_loss,'bg_attn_loss':bg_attn_loss}
        if validate:
            loss['action_inter'] = None
            loss['action_union'] = None
            loss['bg_inter'] = None
            loss['bg_union'] = None

            if action_attn is not None:
                action_attn_pred = action_attn[class_idx] > 0.5
                inter, union = inter_and_union(action_attn_pred.reshape(-1, h, w), attn_gt[class_idx].reshape(-1, h, w), 1, 0)
                loss['action_inter'] = inter
                loss['action_union'] = union
            if bg_attn is not None:
                bg_attn_pred = bg_attn > 0.5
                inter, union = inter_and_union(bg_attn_pred, bg_seg, 1, 1)
                loss['bg_inter'] = inter
                loss['bg_union'] = union

        return loss

    def forward(self, pred, label, validate=False):
        actor_loss = self.actor_loss(pred['actor'],label['actor'])
        attention_loss = self.attn_loss(pred['attn'], label, pred['actor'], validate)

        return actor_loss, attention_loss


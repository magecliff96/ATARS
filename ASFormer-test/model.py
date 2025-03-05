import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from sklearn.metrics import precision_recall_curve, average_precision_score

import einops
import copy
import numpy as np
import math

from eval import segment_bars_with_confidence

import warnings

from UVAST import uvast_model
from UVAST_parts.losses import AttentionLoss, DurAttnCALoss, FrameWiseLoss, SegmentLossAction

# Suppress the specific sklearn warning about no positive class
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

class AttentionHelper(nn.Module):
    def __init__(self):
        super(AttentionHelper, self).__init__()
        self.softmax = nn.Softmax(dim=-1)


    def scalar_dot_att(self, proj_query, proj_key, proj_val, padding_mask):
        '''
        scalar dot attention.
        :param proj_query: shape of (B, C, L) => (Batch_Size, Feature_Dimension, Length)
        :param proj_key: shape of (B, C, L)
        :param proj_val: shape of (B, C, L)
        :param padding_mask: shape of (B, C, L)
        :return: attention value of shape (B, C, L)
        '''
        m, c1, l1 = proj_query.shape
        m, c2, l2 = proj_key.shape
        
        assert c1 == c2
        
        energy = torch.bmm(proj_query.permute(0, 2, 1), proj_key)  # out of shape (B, L1, L2)
        attention = energy / np.sqrt(c1)
        attention = attention + torch.log(padding_mask + 1e-6) # mask the zero paddings. log(1e-6) for zero paddings
        attention = self.softmax(attention) 
        attention = attention * padding_mask
        attention = attention.permute(0,2,1)
        out = torch.bmm(proj_val, attention)
        return out, attention

class AttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type): # r1 = r2
        super(AttLayer, self).__init__()
        
        self.query_conv = nn.Conv1d(in_channels=q_dim, out_channels=q_dim // r1, kernel_size=1)
        self.key_conv = nn.Conv1d(in_channels=k_dim, out_channels=k_dim // r2, kernel_size=1)
        self.value_conv = nn.Conv1d(in_channels=v_dim, out_channels=v_dim // r3, kernel_size=1)
        
        self.conv_out = nn.Conv1d(in_channels=v_dim // r3, out_channels=v_dim, kernel_size=1)

        self.bl = bl
        self.stage = stage
        self.att_type = att_type
        assert self.att_type in ['normal_att', 'block_att', 'sliding_att']
        assert self.stage in ['encoder','decoder']
        
        self.att_helper = AttentionHelper()
        self.window_mask = self.construct_window_mask()
        
    
    def construct_window_mask(self):
        '''
            construct window mask of shape (1, l, l + l//2 + l//2), used for sliding window self attention
        '''
        window_mask = torch.zeros((1, self.bl, self.bl + 2* (self.bl //2)))
        for i in range(self.bl):
            window_mask[:, :, i:i+self.bl] = 1
        return window_mask.to(device)
    
    def forward(self, x1, x2, mask):
        # x1 from the encoder
        # x2 from the decoder
        
        query = self.query_conv(x1)
        key = self.key_conv(x1)
         
        if self.stage == 'decoder':
            assert x2 is not None
            value = self.value_conv(x2)
        else:
            value = self.value_conv(x1)
            
        if self.att_type == 'normal_att':
            return self._normal_self_att(query, key, value, mask)
        elif self.att_type == 'block_att':
            return self._block_wise_self_att(query, key, value, mask)
        elif self.att_type == 'sliding_att':
            return self._sliding_window_self_att(query, key, value, mask)

    
    def _normal_self_att(self,q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        padding_mask = torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:]
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
        
    def _block_wise_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _,c2,L = k.size()
        _,c3,L = v.size()
        
        nb = L // self.bl
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1

        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)

        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        padding_mask = padding_mask.reshape(m_batchsize, 1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb,1, self.bl)
        k = k.reshape(m_batchsize, c2, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c2, self.bl)
        v = v.reshape(m_batchsize, c3, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c3, self.bl)
        
        output, attentions = self.att_helper.scalar_dot_att(q, k, v, padding_mask)
        output = self.conv_out(F.relu(output))
        
        output = output.reshape(m_batchsize, nb, c3, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, c3, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]  
    
    def _sliding_window_self_att(self, q,k,v, mask):
        m_batchsize, c1, L = q.size()
        _, c2, _ = k.size()
        _, c3, _ = v.size()
        
        
        # assert m_batchsize == 1  # currently, we only accept input with batch size 1
        # padding zeros for the last segment
        nb = L // self.bl 
        if L % self.bl != 0:
            q = torch.cat([q, torch.zeros((m_batchsize, c1, self.bl - L % self.bl)).to(device)], dim=-1)
            k = torch.cat([k, torch.zeros((m_batchsize, c2, self.bl - L % self.bl)).to(device)], dim=-1)
            v = torch.cat([v, torch.zeros((m_batchsize, c3, self.bl - L % self.bl)).to(device)], dim=-1)
            nb += 1
        padding_mask = torch.cat([torch.ones((m_batchsize, 1, L)).to(device) * mask[:,0:1,:], torch.zeros((m_batchsize, 1, self.bl * nb - L)).to(device)],dim=-1)
        
        # sliding window approach, by splitting query_proj and key_proj into shape (c1, l) x (c1, 2l)
        # sliding window for query_proj: reshape
        q = q.reshape(m_batchsize, c1, nb, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize * nb, c1, self.bl)
        
        # sliding window approach for key_proj
        # 1. add paddings at the start and end
        k = torch.cat([torch.zeros(m_batchsize, c2, self.bl // 2).to(device), k, torch.zeros(m_batchsize, c2, self.bl // 2).to(device)], dim=-1)
        v = torch.cat([torch.zeros(m_batchsize, c3, self.bl // 2).to(device), v, torch.zeros(m_batchsize, c3, self.bl // 2).to(device)], dim=-1)
        padding_mask = torch.cat([torch.zeros(m_batchsize, 1, self.bl // 2).to(device), padding_mask, torch.zeros(m_batchsize, 1, self.bl // 2).to(device)], dim=-1)
        
        # 2. reshape key_proj of shape (m_batchsize*nb, c1, 2*self.bl)
        k = torch.cat([k[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # special case when self.bl = 1
        v = torch.cat([v[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) 
        # 3. construct window mask of shape (1, l, 2l), and use it to generate final mask
        padding_mask = torch.cat([padding_mask[:,:, i*self.bl:(i+1)*self.bl+(self.bl//2)*2] for i in range(nb)], dim=0) # of shape (m*nb, 1, 2l)
        final_mask = self.window_mask.repeat(m_batchsize * nb, 1, 1) * padding_mask 
        
        output, attention = self.att_helper.scalar_dot_att(q, k, v, final_mask)
        output = self.conv_out(F.relu(output))

        output = output.reshape(m_batchsize, nb, -1, self.bl).permute(0, 2, 1, 3).reshape(m_batchsize, -1, nb * self.bl)
        output = output[:, :, 0:L]
        return output * mask[:, 0:1, :]


class MultiHeadAttLayer(nn.Module):
    def __init__(self, q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type, num_head):
        super(MultiHeadAttLayer, self).__init__()
#         assert v_dim % num_head == 0
        self.conv_out = nn.Conv1d(v_dim * num_head, v_dim, 1)
        self.layers = nn.ModuleList(
            [copy.deepcopy(AttLayer(q_dim, k_dim, v_dim, r1, r2, r3, bl, stage, att_type)) for i in range(num_head)])
        # self.dropout = nn.Dropout(p=0.1)#originally 0.5
        
    def forward(self, x1, x2, mask):
        out = torch.cat([layer(x1, x2, mask) for layer in self.layers], dim=1)
        # out = self.conv_out(self.dropout(out)) #removed
        out = self.conv_out(out)
        return out
            

class ConvFeedForward(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(ConvFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FCFeedForward(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FCFeedForward, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),  # conv1d equals fc
            nn.ReLU(),
            # nn.Dropout(), #removed, default 0.5
            nn.Conv1d(out_channels, out_channels, 1)
        )
        
    def forward(self, x):
        return self.layer(x)
    

class AttModule(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, r1, r2, att_type, stage, alpha):
        super(AttModule, self).__init__()
        self.feed_forward = ConvFeedForward(dilation, in_channels, out_channels)
        self.instance_norm = nn.InstanceNorm1d(in_channels, track_running_stats=False)
        self.att_layer = AttLayer(in_channels, in_channels, out_channels, r1, r1, r2, dilation, att_type=att_type, stage=stage) # dilation
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        # self.dropout = nn.Dropout()
        self.alpha = alpha
        
    def forward(self, x, f, mask):
        out = self.feed_forward(x)
        out = self.alpha * self.att_layer(self.instance_norm(out), f, mask) + out
        out = self.conv_1x1(out)
        # out = self.dropout(out) #removed
        return (x + out) * mask[:, 0:1, :]


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, max_len=10000):
        super(PositionalEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).permute(0,2,1) # of shape (1, d_model, l)
        self.pe = nn.Parameter(pe, requires_grad=True)
#         self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :, 0:x.shape[2]]

class Encoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type, alpha):
        super(Encoder, self).__init__()
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1) # fc layer
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'encoder', alpha) for i in # 2**i
             range(num_layers)])
        
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)
        # self.dropout = nn.Dropout2d(p=channel_masking_rate)
        self.channel_masking_rate = channel_masking_rate

    def forward(self, x, mask):
        '''
        :param x: (N, C, L)
        :param mask:
        :return:
        '''

        if self.channel_masking_rate > 0:
            x = x.unsqueeze(2)
            # x = self.dropout(x) #removed
            x = x.squeeze(2)

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, None, mask)
        
        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature


class Decoder(nn.Module):
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, att_type, alpha):
        super(Decoder, self).__init__()#         self.position_en = PositionalEncoding(d_model=num_f_maps)
        self.conv_1x1 = nn.Conv1d(input_dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, num_f_maps, num_f_maps, r1, r2, att_type, 'decoder', alpha) for i in # 2 ** i
             range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, fencoder, mask):

        feature = self.conv_1x1(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)

        out = self.conv_out(feature) * mask[:, 0:1, :]

        return out, feature
    
class MyTransformer(nn.Module):
    def __init__(self, num_decoders, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate):
        super(MyTransformer, self).__init__()
        self.encoder = Encoder(num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, att_type='sliding_att', alpha=1)
        self.decoders = nn.ModuleList([copy.deepcopy(Decoder(num_layers, r1, r2, num_f_maps, num_classes, num_classes, att_type='sliding_att', alpha=exponential_descrease(s))) for s in range(num_decoders)]) # num_decoders
        
        
    def forward(self, x, mask):
        out, feature = self.encoder(x, mask)
        outputs = out.unsqueeze(0)
        
        for decoder in self.decoders:
            out, feature = decoder(F.softmax(out, dim=1) * mask[:, 0:1, :], feature* mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
 
        return outputs

    
class Trainer:
    def __init__(self, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate, args):
        self.device = args.device
        if args.arch == 'asformer':
            self.model = MyTransformer(3, num_layers, r1, r2, num_f_maps, input_dim, num_classes, channel_masking_rate)
        elif args.arch == 'UVAST':
            self.model = uvast_model(args)
            self.frame_wise_loss = FrameWiseLoss(args)
            self.segment_wise_loss = SegmentLossAction(args)
            self.attn_action_loss = AttentionLoss(args)
            self.attn_dur_loss = DurAttnCALoss(args)

        pos_weight = torch.tensor([2]).to(device)
        self.ce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        print('Model Size: ', sum(p.numel() for p in self.model.parameters()))
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def load_model(self, checkpoint_path):
        """Load the model weights from a checkpoint"""
        self.model.to(device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)  # Ensure it loads to the correct device
        self.model.load_state_dict(checkpoint)  # Load state_dict properly
        print("Model loaded successfully from", checkpoint_path)

    def calculate_mAP(self, predictions, ground_truth):
        num_classes = predictions.shape[1]
        average_precisions = []

        for i in range(num_classes):
            # Compute average precision (AP) for each class
            ap = average_precision_score(ground_truth[:, i], predictions[:, i])
            average_precisions.append(ap)

        # Calculate the mean of the average precision values (mAP)
        mAP = sum(average_precisions) / num_classes
        return mAP

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, args, batch_gen_tst=None, writer=None):
        self.model.train()
        if args.arch == "UVAST":
            self.frame_wise_loss.reset()
            self.segment_wise_loss.reset()
            self.attn_action_loss.reset()
            self.attn_dur_loss.reset()


        self.model.to(device)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        if args.load_model == True:
            optimizer.load_state_dict(torch.load(args.pretrain + ".opt"))
        print('LR:{}'.format(learning_rate))
        
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
        for epoch in range(num_epochs):
            epoch_loss = 0
            all_predictions = []; all_ground_truths = []


            while batch_gen.has_next():
                data = batch_gen.next_batch(batch_size, False)
                batch_input = data["input"]; batch_target = data["target"]
                mask = data["mask"]; vids = data["batch"]

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)
                optimizer.zero_grad()

                if args.arch == 'asformer':
                    ps = self.model(batch_input, mask)

                    loss = 0
                    for p in ps: # p => [B, C, T]   target => [B, T, C]
                            loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1, self.num_classes))
                            loss += 0.15 * torch.mean(torch.clamp(
                                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
                                max=16) * mask[:, :, 1:])

                # elif args.arch == 'UVAST':
                #     if args.extra_loss == True:
                #         seg_gt, seg_dur = data['seg_gt'], data['seg_dur']#[b,class, data]

                #         seg_dur_norm = []
                #         for b in range(len(seg_dur)):
                #             seg_dur_norm.append([])
                #             for i in range(self.num_classes): #range of class
                #                 temp = seg_dur[b][i] / data['len_seq_seg'][b][i][0]
                #                 seg_dur_norm[b].append(temp.clone().detach())

                #         seg_dur_norm = [torch.stack(batch) for batch in seg_dur_norm]

                #         seg_gt_act = torch.stack(seg_gt).to(device)
                #         seg_dur_normalize = torch.stack(seg_dur_norm).to(device)
                        
                #         # to ignore the <eos> token we change it to -1 (in act class) and zero (in duration)
                #         seg_gt_act_train = seg_gt_act.clone().detach()
                #         seg_dur_normalize_train = seg_dur_normalize.clone().detach()
                #         mask_train = mask.clone().detach()
                #         seg_gt_act_train[seg_gt_act == 1] = -1
                #         seg_dur_normalize_train[seg_gt_act == 1] = 0
                #         seg_data = (seg_gt_act_train, seg_dur_normalize_train)


                #         seg_gt_no_split = [
                #             [tensor.to(device) for tensor in sublist]
                #             for sublist in data['seg_gt_no_split']
                #         ]
                #         seg_dur_no_split = [
                #             [tensor.to(device) for tensor in sublist]
                #             for sublist in data['seg_dur_no_split']
                #         ]
                #         predictions_framewise, pred_transcript, pred_crossattn, frames_to_segment_assignment = self.model(batch_input, mask, seg_data, no_split_data=(seg_gt_no_split, seg_dur_no_split))
                #         seg_gt_act = einops.rearrange(seg_gt_act, 'B C E -> C B E')
                #     else:
                #         predictions_framewise = self.model(batch_input, mask)

                #     # mask_c [1 , 40 , t]
                #     # seg_data_c [2, 1 , 40, 100] => (seg_gt_act_train, seg_dur_normalize_train)
                #     # seg_gt_no_split_c [1, 40, random]
                #     # seg_dur_no_split_c [1, 40, random]
                #     #seg_dur_no_split => [B C Random] 
                #     # #predictions_framewise => [1 C B T] 
                #     # #pred_transcript => [2 C B 40+2, E] 
                #     # #pred_crossattn => [2 C B E T]

                #     batch_target = einops.rearrange(batch_target, 'B E C -> C B E')
                #     mask = einops.rearrange(mask, 'B C E -> C B E')


                #     framewise_losses = 0; segwise_losses=0; attn_action_losses=0; attn_duration_losses=0; 
                #     for c in range(self.num_classes):
                #         predictions_framewise_class = torch.stack([inner_list[c] for inner_list in predictions_framewise])
                #         if args.do_framewise_loss or args.do_framewise_loss_g:
                #             framewise_losses += self.frame_wise_loss(predictions=predictions_framewise_class, batch_target=batch_target[c], mask=mask[c], epoch=epoch)

                #             #additional UVAST LOSS
                #         if args.extra_loss == True:
                #             seg_dur_no_split_class = torch.stack([inner_list[c] for inner_list in seg_dur_no_split])
                #             pred_transcript_class = torch.stack([inner_list[c] for inner_list in pred_transcript])
                #             pred_crossattn_class = torch.stack([inner_list[c] for inner_list in pred_crossattn])


                #             attn_mask_gt = torch.zeros(seg_dur_no_split_class.shape[0], seg_dur_no_split_class.shape[1], int(seg_dur_no_split_class.sum().item())) 
                #             seg_cumsum = torch.cumsum(seg_dur_no_split_class, dim=1)
                #             for i in range(seg_dur_no_split_class.shape[1]):
                #                 if i > 0:
                #                     attn_mask_gt[0, i, int(seg_cumsum[0, i - 1].item()):int(seg_cumsum[0, i].item())] = 1
                #                 else:
                #                     attn_mask_gt[0, i, :int(seg_cumsum[0, i].item())] = 1
                #             attn_mask_gt_dur = attn_mask_gt.to(batch_input.device)

                #             if args.do_segwise_loss or args.do_segwise_loss_g:
                #                 seg_gt_act_loss = F.pad(seg_gt_act[c].clone().detach()[:, 1:], pad=(0, 1), mode='constant', value=-1)
                #                 segwise_losses += self.segment_wise_loss(pred_transcript_class, seg_gt_act_loss, batch_input.shape[-1], epoch) #what shape is -1

                #             # if args.do_crossattention_action_loss_nll:    
                #             #     attn_action_losses += self.attn_action_loss(pred_crossattn_class, attn_mask_gt, batch_target[c])
                                        
                #             if args.use_alignment_dec and args.do_crossattention_dur_loss_ce:
                #                 attn_duration_losses += self.attn_dur_loss(frames_to_segment_assignment, attn_mask_gt_dur) 

                #     loss = (framewise_losses + segwise_losses + attn_action_losses + attn_duration_losses)/self.num_classes
                #     p = torch.stack(predictions_framewise)
                #     p  = einops.rearrange(p, 'l B C T -> (l B) C T') #target:  [B, C, T]
                #     batch_target  = einops.rearrange(batch_target, 'C B T -> B T C') #target:  [B, T, C]

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()
                all_predictions.append(torch.sigmoid(p.transpose(2, 1).contiguous().view(-1, self.num_classes)).cpu().detach().numpy())
                all_ground_truths.append(batch_target.view(-1, self.num_classes).cpu().detach().numpy())


            all_predictions = np.concatenate(all_predictions, axis=0)
            all_ground_truths = np.concatenate(all_ground_truths, axis=0)
            mAP = self.calculate_mAP(all_predictions, all_ground_truths)
            
            scheduler.step(epoch_loss)
            batch_gen.reset()
            if writer is not None:
                writer.add_scalar('Loss/epoch_loss', epoch_loss / len(batch_gen.list_of_examples), epoch)
                writer.add_scalar('Metrics/train_mAP', mAP, epoch)
            print("[epoch %d]: epoch loss = %f, batch mAP = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples), mAP))

            if (epoch + 1) % 5 == 0 and batch_gen_tst is not None:
                self.test(batch_gen_tst, args, epoch, writer=writer)
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")

    def test(self, batch_gen_tst, args, epoch=0, writer=None):
        self.model.eval()
        all_predictions = []; all_ground_truths = []
        if_warp = False  # When testing, always false
        with torch.no_grad():
            while batch_gen_tst.has_next():

                data = batch_gen_tst.next_batch(1, if_warp)
                batch_input = data["input"]; batch_target = data["target"]
                mask = data["mask"]; vids = data["batch"]

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)


                if args.arch == 'asformer':
                    p = self.model(batch_input, mask)


                elif args.arch == 'UVAST':
                    if args.extra_loss == True:
                        seg_gt, seg_dur = data['seg_gt'], data['seg_dur']#[b,class, data]

                        seg_dur_norm = []
                        for b in range(len(seg_dur)):
                            seg_dur_norm.append([])
                            for i in range(self.num_classes): #range of class
                                temp = seg_dur[b][i] / data['len_seq_seg'][b][i][0]
                                seg_dur_norm[b].append(temp.clone().detach())

                        seg_dur_norm = [torch.stack(batch) for batch in seg_dur_norm]
                        seg_gt_act = torch.stack(seg_gt).to(device)
                        seg_dur_normalize = torch.stack(seg_dur_norm).to(device)
                        seg_gt_act_train = seg_gt_act.clone().detach()
                        seg_dur_normalize_train = seg_dur_normalize.clone().detach()
                        seg_gt_act_train[seg_gt_act == 1] = -1
                        seg_dur_normalize_train[seg_gt_act == 1] = 0
                        seg_data = (seg_gt_act_train, seg_dur_normalize_train)
                        seg_gt_no_split = [
                            [tensor.to(device) for tensor in sublist]
                            for sublist in data['seg_gt_no_split']
                        ]
                        seg_dur_no_split = [
                            [tensor.to(device) for tensor in sublist]
                            for sublist in data['seg_dur_no_split']
                        ]

                        predictions_framewise, pred_transcript, pred_crossattn, frames_to_segment_assignment = self.model(batch_input, mask, seg_data, no_split_data=(seg_gt_no_split, seg_dur_no_split))
                    else:
                        predictions_framewise = self.model(batch_input, mask)


                    p = torch.stack(predictions_framewise)
                    p  = einops.rearrange(p, 'l B C T -> (l B) C T') #target:  [B, C, T]
                    batch_target  = einops.rearrange(batch_target, 'C B T -> B T C') #target:  [B, T, C]

                if args.arch == 'asformer':
                    all_predictions.append(torch.sigmoid(p[-1].transpose(2, 1).contiguous().view(-1, self.num_classes)).cpu().detach().numpy())
                    all_ground_truths.append(batch_target.view(-1, self.num_classes).cpu().detach().numpy())
                else:
                    all_predictions.append(torch.sigmoid(p.transpose(2, 1).contiguous().view(-1, self.num_classes)).cpu().detach().numpy())
                    all_ground_truths.append(batch_target.view(-1, self.num_classes).cpu().detach().numpy())



        all_predictions = np.concatenate(all_predictions, axis=0)
        all_ground_truths = np.concatenate(all_ground_truths, axis=0)
        mAP = self.calculate_mAP(all_predictions, all_ground_truths)
        if writer is not None:
            writer.add_scalar('Metrics/test_mAP', mAP, epoch)
        print("---[epoch %d]---: mAP = %f" % (epoch + 1, mAP))

        self.model.train()
        batch_gen_tst.reset()

    def predict(self, args, model_dir, results_dir, features_path, batch_gen_tst, epoch, actions_dict, sample_rate):
#         self.model.eval()
#         with torch.no_grad():
#             self.model.to(device)
#             self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

#             batch_gen_tst.reset()
#             import time
            
#             time_start = time.time()
#             while batch_gen_tst.has_next():
#                 batch_input, batch_target, mask, vids = batch_gen_tst.next_batch(1)
#                 vid = vids[0]
# #                 print(vid)
#                 features = np.load(features_path + vid.split('.')[0] + '.npy')
#                 features = features[:, ::sample_rate]

#                 input_x = torch.tensor(features, dtype=torch.float)
#                 input_x.unsqueeze_(0)
#                 input_x = input_x.to(device)
#                 predictions = self.model(input_x, torch.ones(input_x.size(), device=device))

#                 for i in range(len(predictions)):
#                     confidence, predicted = torch.max(F.softmax(predictions[i], dim=1).data, 1)
#                     confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
#                     batch_target = batch_target.squeeze()
#                     confidence, predicted = confidence.squeeze(), predicted.squeeze()
 
#                     segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
#                                                  confidence.tolist(),
#                                                  batch_target.tolist(), predicted.tolist())

#                 recognition = []
#                 for i in range(len(predicted)):
#                     recognition = np.concatenate((recognition, [list(actions_dict.keys())[
#                                                                     list(actions_dict.values()).index(
#                                                                         predicted[i].item())]] * sample_rate))
#                 f_name = vid.split('/')[-1].split('.')[0]
#                 f_ptr = open(results_dir + "/" + f_name, "w")
#                 f_ptr.write("### Frame level recognition: ###\n")
#                 f_ptr.write(' '.join(recognition))
#                 f_ptr.close()

        self.model.eval()
        all_predictions = []; all_ground_truths = []
        if_warp = False  # When testing, always false
        with torch.no_grad():
            # self.model.to(device)
            # self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(epoch) + ".model"))

            batch_gen_tst.reset()
            import time
            
            time_start = time.time()
            while batch_gen_tst.has_next():

                data = batch_gen_tst.next_batch(1, if_warp)
                batch_input = data["input"]; batch_target = data["target"]
                mask = data["mask"]; vids = data["batch"]

                batch_input, batch_target, mask = batch_input.to(device), batch_target.to(device), mask.to(device)


                if args.arch == 'asformer':
                    p = self.model(batch_input, mask)


                elif args.arch == 'UVAST':
                    if args.extra_loss == True:
                        seg_gt, seg_dur = data['seg_gt'], data['seg_dur']#[b,class, data]

                        seg_dur_norm = []
                        for b in range(len(seg_dur)):
                            seg_dur_norm.append([])
                            for i in range(self.num_classes): #range of class
                                temp = seg_dur[b][i] / data['len_seq_seg'][b][i][0]
                                seg_dur_norm[b].append(temp.clone().detach())

                        seg_dur_norm = [torch.stack(batch) for batch in seg_dur_norm]
                        seg_gt_act = torch.stack(seg_gt).to(device)
                        seg_dur_normalize = torch.stack(seg_dur_norm).to(device)
                        seg_gt_act_train = seg_gt_act.clone().detach()
                        seg_dur_normalize_train = seg_dur_normalize.clone().detach()
                        seg_gt_act_train[seg_gt_act == 1] = -1
                        seg_dur_normalize_train[seg_gt_act == 1] = 0
                        seg_data = (seg_gt_act_train, seg_dur_normalize_train)
                        seg_gt_no_split = [
                            [tensor.to(device) for tensor in sublist]
                            for sublist in data['seg_gt_no_split']
                        ]
                        seg_dur_no_split = [
                            [tensor.to(device) for tensor in sublist]
                            for sublist in data['seg_dur_no_split']
                        ]

                        predictions_framewise, pred_transcript, pred_crossattn, frames_to_segment_assignment = self.model(batch_input, mask, seg_data, no_split_data=(seg_gt_no_split, seg_dur_no_split))
                    else:
                        predictions_framewise = self.model(batch_input, mask)


                    p = torch.stack(predictions_framewise)
                    p  = einops.rearrange(p, 'l B C T -> (l B) C T') #target:  [B, C, T]
                    batch_target  = einops.rearrange(batch_target, 'C B T -> B T C') #target:  [B, T, C]

                if args.arch == 'asformer':
                    all_predictions.append(torch.sigmoid(p[-1].transpose(2, 1).contiguous().view(-1, self.num_classes)).cpu().detach().numpy())
                    all_ground_truths.append(batch_target.view(-1, self.num_classes).cpu().detach().numpy())
                else:
                    all_predictions.append(torch.sigmoid(p.transpose(2, 1).contiguous().view(-1, self.num_classes)).cpu().detach().numpy())
                    all_ground_truths.append(batch_target.view(-1, self.num_classes).cpu().detach().numpy())


        #perclass mAP
        # class_names = [
        #     '12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', 
        #     '23v', '23v+', '24v', '24v+', '31v', '31v+', '32v', '32v+', 
        #     '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+', 
        #     '12p', '12p+', '14p', '14p+', '21p', '21p+', '23p', '23p+', 
        #     '32p', '32p+', '34p', '34p+', '41p', '41p+', '43p', '43p+'
        # ]
        class_names = [
            '12v', '12v+', '13v', '13v+', '14v', '14v+', '21v', '21v+', 
            '23v', '23v+', '24v', '24v+', '31v', '31v+', '32v', '32v+', 
            '34v', '34v+', '41v', '41v+', '42v', '42v+', '43v', '43v+', 
            '12p', '14p', '21p', '23p', '32p', '34p', '41p', '43p'
        ]
        filenames = data['filename']; target = "B5_0_0.txt"
        for k in range(len(filenames)):
            if filenames[k] == target:
                sample = all_predictions[k]
                file_path = f"results/{target}" 
                # print(torch.max(torch.tensor(sample))); print(torch.min(torch.tensor(sample)))
                with open(file_path, "w") as file:
                    for frame_idx in range(sample.shape[0]):
                        # Get indices where class is marked as positive
                        active_classes = [class_names[i] for i in range(32) if sample[frame_idx, i] >= 0.5]
                        # Join class names and write to file
                        file.write(" ".join(active_classes) + "\n")


        all_predictions = np.concatenate(all_predictions, axis=0)
        all_ground_truths = np.concatenate(all_ground_truths, axis=0)
        mAP = self.calculate_mAP(all_predictions, all_ground_truths)

        # Calculate per-class mAP
        per_class_mAP = {}
        for i, class_name in enumerate(class_names):
            y_true = all_ground_truths[:, i]
            y_pred = all_predictions[:, i]
            if np.sum(y_true) > 0:  # Only calculate if there are positive samples
                per_class_mAP[class_name] = average_precision_score(y_true, y_pred)
            else:
                per_class_mAP[class_name] = 0.0  # No positive samples for this class


        # Print results
        print("---[epoch %d]---: mAP = %f" % (epoch + 1, mAP))
        print("Per-class mAP:")
        v_classes = []
        vp_classes = []
        p_classes = []

        for class_name, class_mAP in per_class_mAP.items():
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

        print(f"\nAverage mAP for classes ending in 'v': {avg_v:.4f}")
        print(f"Average mAP for classes ending in 'v+': {avg_vp:.4f}")
        print(f"Average mAP for classes ending in 'p': {avg_p:.4f}")

            
            

if __name__ == '__main__':
    pass

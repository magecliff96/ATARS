import torch

# B, C, T, H, W = 2, 3, 8, 224, 224
# input_tensor1 = torch.zeros(B, C, T, H, W)
# B, C, T, H, W = 2, 3, 32, 224, 224
# input_tensor2 = torch.zeros(B, C, T, H, W)
# input_tensor = [input_tensor1, input_tensor2]
# output = model(input_tensor)

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, in_channel, num_class, emb_channels=1000):
        super(Head, self).__init__()
        self.emb_channels = emb_channels
        self.fc_tri_to_sub = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(8, num_class//2)
            )
        self.fc_sub_to_pri = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(num_class//2, num_class)
            )
        self.fc_tri = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel,  8)
            )
        self.fc_sub = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel,  num_class//2)
            )
        self.fc_pri = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(in_channel, num_class)
            )
    def forward(self, x):
        #tri-sub-pri initiation
        y_tri = self.fc_tri(x)
        y_sub = self.fc_sub(x)
        y_pri = self.fc_pri(x)
        #sub
        sub_weight = torch.sigmoid(self.fc_tri_to_sub(y_tri))
        y_sub = y_sub * sub_weight
        #pri
        pri_weight = torch.sigmoid(self.fc_sub_to_pri(y_sub))
        y_pri = y_pri * pri_weight
        return y_pri, y_sub, y_tri

class SlowFast_rus(nn.Module):
    def __init__(self, num_class):
        super(SlowFast_rus, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)

        self.head = Head(2304, num_class)
        self.model.blocks[-1] = nn.Sequential(
						        	nn.Dropout(p=0.3, inplace=False),
						        	# nn.Linear(in_features=2304, out_features=400, bias=True),
						        	self.head,
						        	)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)

    def forward(self, x):
        seq_len = x[0].shape[1]
        batch_size = len(x)
        height, width = x[0].shape[2], x[0].shape[3]
        slow_x = x[:, :, ::4, :, :]
        #print("amongus"); print(len(slow_x)); print(slow_x.size()); print(slow_x.size()) #[3,3,]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            slow_x = torch.stack(slow_x, dim=0)
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            slow_x = torch.permute(slow_x, (1,2,0,3,4))
        num_block = len(self.model.blocks)

        x = [slow_x, x]
        for i in range(num_block-1):
            x = self.model.blocks[i](x)
        
        #x_sub = x
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))

        x, x_sub, x_tri = self.model.blocks[-1](x)
        return x, x_sub, x_tri

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
from classifier import Head

class RusNet(nn.Module):
    def __init__(self, num_actor_class):
        super(RusNet, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.optic_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)

        self.head = Head(2304, num_actor_class)
        self.model.blocks[-1] = nn.Sequential(
						        	nn.Dropout(p=0.3, inplace=False), #originally .5
						        	# nn.Linear(in_features=2304, out_features=400, bias=True),
						        	self.head,
						        	)
        self.optic_head = Head(2304, num_actor_class)
        self.optic_model.blocks[-1] = nn.Sequential(
						        	nn.Dropout(p=0.3, inplace=False), #originally .5
						        	# nn.Linear(in_features=2304, out_features=400, bias=True),
						        	self.optic_head,
						        	)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)


        self.layer_shape_slow = [[80, 8, 56, 56],[320, 8, 56, 56],[640, 8, 28, 28],[1280, 8, 14, 14],[2048, 8, 7, 7]]
        self.layer_shape_fast = [[8, 32, 56, 56],[32, 32, 56, 56],[64, 32, 28, 28],[128, 32, 14, 14],[256, 32, 7, 7]]

        self.in_channels_slow_list = [80, 320, 640, 1280, 2048] 
        self.in_channels_fast_list = [8, 32, 64, 128, 256] 
        self.reduction_ratio = 8

        self.layernorms_slow_optic = nn.ModuleList(
            [nn.LayerNorm(shape) for shape in self.layer_shape_slow]
        )
        self.layernorms_fast_optic = nn.ModuleList(
            [nn.LayerNorm(shape) for shape in self.layer_shape_fast]
        )

        self.layernorms_slow = nn.ModuleList(
            [nn.LayerNorm(shape) for shape in self.layer_shape_slow]
        )
        self.layernorms_fast = nn.ModuleList(
            [nn.LayerNorm(shape) for shape in self.layer_shape_fast]
        )

        self.CA_optic_fast = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(in_channels_fast, in_channels_fast // self.reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels_fast // self.reduction_ratio, in_channels_fast, bias=False),
                nn.Sigmoid()
            ) for in_channels_fast in self.in_channels_fast_list
        ])

        self.ICA_fast = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(in_channels_fast, in_channels_fast // self.reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels_fast // self.reduction_ratio, in_channels_fast, bias=False),
                nn.Sigmoid()
            ) for in_channels_fast in self.in_channels_fast_list
        ])

        self.CA_optic_slow = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(in_channels_slow, in_channels_slow // self.reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels_slow // self.reduction_ratio, in_channels_slow, bias=False),
                nn.Sigmoid()
            ) for in_channels_slow in self.in_channels_slow_list
        ])

        self.ICA_slow = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(in_channels_slow, in_channels_slow // self.reduction_ratio, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels_slow // self.reduction_ratio, in_channels_slow, bias=False),
                nn.Sigmoid()
            ) for in_channels_slow in self.in_channels_slow_list


        ])# Reshape to (batch_size, channels, 1, 1, 1): y = self.model(x).view(b, c, 1, 1, 1)

        self.alpha_slow = nn.Parameter(torch.ones(len(self.in_channels_slow_list)))
        self.alpha_fast = nn.Parameter(torch.ones(len(self.in_channels_fast_list)))

        self.beta_slow = nn.Parameter(torch.ones(len(self.in_channels_slow_list)))
        self.beta_fast = nn.Parameter(torch.ones(len(self.in_channels_fast_list)))

        # current shape at 0 block: torch.Size([8, 80, 8, 56, 56]), torch.Size([8, 8, 32, 56, 56])
        # current shape at 1 block: torch.Size([8, 320, 8, 56, 56]), torch.Size([8, 32, 32, 56, 56])
        # current shape at 2 block: torch.Size([8, 640, 8, 28, 28]), torch.Size([8, 64, 32, 28, 28])
        # current shape at 3 block: torch.Size([8, 1280, 8, 14, 14]), torch.Size([8, 128, 32, 14, 14])
        # current shape at 4 block: torch.Size([8, 2048, 8, 7, 7]), torch.Size([8, 256, 32, 7, 7])
        # current shape at 5 block: torch.Size([2304, 1, 1, 1]), torch.Size([2304, 1, 1, 1])


    def forward(self, x, optic):
        seq_len = x[0].shape[1]
        batch_size = len(x)
        height, width = x[0].shape[2], x[0].shape[3]
        slow_x = x[:, :, ::4, :, :]

        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            slow_x = torch.stack(slow_x, dim=0)
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
            slow_x = torch.permute(slow_x, (1,2,0,3,4))
        num_block = len(self.model.blocks)

        x = [slow_x, x]

        if [optic[0].shape[2], optic[0].shape[3]]!= [height,width]:
            optic = optic.permute(0, 4, 1, 2, 3) #[8, 3, 32, 224, 224]
        # seq_len = optic[0].shape[1]
        # batch_size = len(optic)
        # height, width = optic[0].shape[2], optic[0].shape[3]
        
        slow_optic = optic[:, :, ::4, :, :]
        if isinstance(optic, list):
            optic = torch.stack(optic, dim=0) #[v, b, 2048, h, w]
            slow_optic = torch.stack(slow_optic, dim=0)
            # l, b, c, h, w
            optic = torch.permute(optic, (1,2,0,3,4)) #[b, v, 2048, h, w]
            slow_optic = torch.permute(slow_optic, (1,2,0,3,4))
        num_block = len(self.model.blocks)

        optic = [slow_optic, optic]

        for i in range(num_block-1):
            optic = self.optic_model.blocks[i](optic)
            x = self.model.blocks[i](x)
            # print(f"current shape at {i} block: {x[0].shape}, {x[1].shape}")
            if i!=5:
                b0, c0, t0, h0, w0 = x[0].size()#slow
                b1, c1, t1, h1, w1 = x[1].size()#fast

                optic_auxW_slow= self.CA_optic_slow[i](optic[0]).view(b0, c0, 1, 1, 1)
                optic_auxW_fast= self.CA_optic_fast[i](optic[1]).view(b1, c1, 1, 1, 1)
                # optic[0] = self.layernorms_slow_optic[i](optic[0] + self.alpha_slow[i] * optic[0] * optic_auxW_slow)
                # optic[1] = self.layernorms_fast_optic[i](optic[1] + self.alpha_fast[i] * optic[1] * optic_auxW_fast)
                optic[0] = self.layernorms_slow_optic[i]((optic[0] + optic[0] * optic_auxW_slow))
                optic[1] = self.layernorms_fast_optic[i]((optic[1] + optic[1] * optic_auxW_fast))

                x_auxW_slow = self.ICA_slow[i](x[0]).view(b0, c0, 1, 1, 1) + optic_auxW_slow
                x_auxW_fast = self.ICA_fast[i](x[1]).view(b1, c1, 1, 1, 1) + optic_auxW_fast
                # x[0] = self.layernorms_slow[i](x[0] + self.beta_slow[i] * x[0] * x_auxW_slow)
                # x[1] = self.layernorms_fast[i](x[1] + self.beta_fast[i] * x[1] * x_auxW_fast)
                x[0] = self.layernorms_slow[i]((x[0] + x[0] * x_auxW_slow))
                x[1] = self.layernorms_fast[i]((x[1] + x[1] * x_auxW_fast))

        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))
        x = self.model.blocks[-1](x) #[8,2304]

        optic = self.pool(optic)
        optic = torch.reshape(optic, (batch_size, -1))
        optic = self.optic_model.blocks[-1](optic) #problem [8,2304]
        ## work left to do => incorperate pathway exchange
        # x = (x + optic)/2
        return x, optic

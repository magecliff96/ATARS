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

class SlowFast(nn.Module):
    def __init__(self, num_actor_class):
        super(SlowFast, self).__init__()
        self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)

        self.head = Head(2304, num_actor_class)
        self.model.blocks[-1] = nn.Sequential(
						        	nn.Dropout(p=0.3, inplace=False), #originally .5
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

        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))

        x = self.model.blocks[-1](x)
        return x

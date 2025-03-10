import pytorchvideo.models as models

import torch
import torch.nn as nn
# import torchvision.models as models
import torch.nn.functional as F
from classifier import Head

from pytorchvideo.models.hub import csn_r101

# model = models.create_csn()


# model.load_state_dict(torch.load(PATH))
#         # self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# for i, b in enumerate(csn.blocks):
#     print(i)
#     print(b)


class CSN(nn.Module):
    def __init__(self, num_actor_class):
        super(CSN, self).__init__()
        self.model = csn_r101(True)

        self.head = Head(2048, num_actor_class)
        self.model.blocks[-1] = nn.Sequential(
						        	nn.Dropout(p=0.5, inplace=False),
						        	# nn.Linear(in_features=2304, out_features=400, bias=True),
						        	self.head,
						        	)
        self.pool = nn.AdaptiveAvgPool3d(output_size=1)


    def forward(self, x):
        seq_len = x[0].shape[1]
        batch_size = len(x)
        height, width = x[0].shape[2], x[0].shape[3]
        if isinstance(x, list):
            x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = torch.permute(x, (1,2,0,3,4)) #[b, v, 2048, h, w]
        num_block = len(self.model.blocks)
        for i in range(num_block-1):
            x = self.model.blocks[i](x)
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, 2048))

        x = self.model.blocks[-1](x)
        return x

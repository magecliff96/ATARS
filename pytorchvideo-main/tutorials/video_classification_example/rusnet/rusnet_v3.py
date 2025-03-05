
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head

class ReparameterizedConv(nn.Module):
    def __init__(self, main_conv, aux_conv):
        super(ReparameterizedConv, self).__init__()
        self.main_weight = nn.Parameter(main_conv.weight.clone())
        self.aux_weight = nn.Parameter(aux_conv.weight.clone())
        self.lambda_ = nn.Parameter(torch.tensor(0.0))  # Learnable scaling factor

        # Clone biases if present
        if main_conv.bias is not None:
            self.main_bias = nn.Parameter(main_conv.bias.clone())
            self.aux_bias = nn.Parameter(aux_conv.bias.clone())
        else:
            self.main_bias = None
            self.aux_bias = None

    def forward(self, x):
        # Combine weights
        combined_weight = self.main_weight + self.lambda_ * self.aux_weight

        # Combine biases if present
        if self.main_bias is not None:
            combined_bias = self.main_bias + self.lambda_ * self.aux_bias
        else:
            combined_bias = None

        # Perform convolution
        return nn.functional.conv3d(x, combined_weight, combined_bias, stride=1, padding=1)

def reparameterize_slowfast(slowfast_model, optic_model):
        for name, layer in slowfast_model.named_modules():
            # Replace Conv3D layers with ReparameterizedConv
            if isinstance(layer, nn.Conv3d):
                # Find the corresponding layer in the optic model
                aux_layer = dict(optic_model.named_modules())[name]

                # Replace the layer
                setattr(
                    slowfast_model,
                    name,
                    ReparameterizedConv(layer, aux_layer)
                )
        return slowfast_model

class RusNet(nn.Module):
    def __init__(self, num_actor_class):
        super(RusNet, self).__init__()
        self.slowfast_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        self.optic_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
        # for i, b in enumerate(self.model.blocks):
        #     print(i)
        #     print(b)

        self.head = Head(2304, num_actor_class)
        self.slowfast_model.blocks[-1] = nn.Sequential(
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
        self.model = reparameterize_slowfast(self.slowfast_model, self.optic_model)


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


        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1))
        x = self.model.blocks[-1](x) #[8,2304]

        optic = self.pool(optic)
        optic = torch.reshape(optic, (batch_size, -1))
        optic = self.optic_model.blocks[-1](optic) #problem [8,2304]

        return x, optic

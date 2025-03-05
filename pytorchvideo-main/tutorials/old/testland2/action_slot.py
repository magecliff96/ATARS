import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from classifier import Head, Allocated_Head


class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, num_actor_class=40, eps=1e-8, input_dim=64, resolution=[16, 8, 24], allocated_slot=True):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        self.num_actor_class = num_actor_class
        self.allocated_slot = allocated_slot
        self.eps = eps
        self.scale = dim ** -0.5
        self.resolution = resolution
        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim)).cuda()
        self.slots_sigma = torch.randn(1, 1, dim).cuda()
        self.slots_sigma = nn.Parameter(self.slots_sigma.absolute())


        self.FC1 = nn.Linear(dim, dim)
        self.FC2 = nn.Linear(dim, dim)
        self.LN = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.gru = nn.GRUCell(dim, dim)
        
        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

        mu = self.slots_mu.expand(1, self.num_slots, -1)
        sigma = self.slots_sigma.expand(1, self.num_slots, -1)
        slots = torch.normal(mu, sigma)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.pe = SoftPositionEmbed3D(dim, [resolution[0], resolution[1], resolution[2]])

        slots = slots.contiguous()
        self.register_buffer("slots", slots)
    def extend_slots(self):
        mu = self.slots_mu.expand(1, 29, -1)
        sigma = self.slots_sigma.expand(1, 29, -1)
        slots = torch.normal(mu, sigma)
        slots = slots.contiguous()

        slots = torch.cat((self.slots[:, :-1, :], slots[:, :, :], torch.reshape(self.slots[:, -1, :], (1, 1, -1))), 1)
        self.register_buffer("slots", slots)

    def extract_slots_for_oats(self):

        oats_slot_idx = [
            13, 12, 50, 6, 3,
            55, 1, 0, 5, 10,
            8, 51, 9, 53, 2,
            4, 48, 59, 52, 61,
            63, 49, 60, 7, 30, 
            11, 57, 22, 62, 58,
            18, 54, 29, 17, 25,
            64
            ]
        slots = tuple([torch.reshape(self.slots[:, idx, :], (1, 1, -1)) for idx in oats_slot_idx])
        slots = torch.cat(slots, 1)
        self.register_buffer("slots", slots)

    # def extract_slots_for_nuscenes(self):
    #     slots = torch.cat((self.slots[:, :24, :], slots[:, 33:34, :], self.slots[:, -17:, :]), 1)
        self.register_buffer("slots", slots)
    def get_3d_slot(self, slots, inputs):
        b, l, h, w, d = inputs.shape
        inputs = self.pe(inputs)
        inputs = torch.reshape(inputs, (b, -1, d))

        inputs = self.LN(inputs)
        inputs = self.FC1(inputs)
        inputs = F.relu(inputs)
        inputs = self.FC2(inputs)

        slots_prev = slots

        b, n, d = inputs.shape
        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn_ori = dots.softmax(dim=1) + self.eps
        attn = attn_ori / attn_ori.sum(dim=-1, keepdim=True)
        slots = torch.einsum('bjd,bij->bid', v, attn)

        slots = slots.reshape(b, -1, d)
        if self.allocated_slot:
            slots = slots[:, :self.num_actor_class, :]
        else:
            slots = slots[:, :self.num_slots, :]
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots, attn_ori

    def forward(self, inputs, num_slots = None):
        b, nf, h, w, d = inputs.shape
        slots = self.slots.expand(b,-1,-1)
        slots_out, attns = self.get_3d_slot(slots, inputs)
        # b, n, c
        return slots_out, attns


def build_3d_grid(resolution):
    ranges = [torch.linspace(0.0, 1.0, steps=res) for res in resolution]
    grid = torch.meshgrid(*ranges)
    grid = torch.stack(grid, dim=-1)
    grid = torch.reshape(grid, [resolution[0], resolution[1], resolution[2], -1])
    grid = grid.unsqueeze(0)
    return torch.cat([grid, 1.0 - grid], dim=-1)


class SoftPositionEmbed3D(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(6, hidden_size, bias=True)
        self.register_buffer("grid", build_3d_grid(resolution))
    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


class ACTION_SLOT(nn.Module):
    def __init__(self, args, num_class, num_slots=40):
        super(ACTION_SLOT, self).__init__()
        self.args = args
        if args.backbone == 'slowfast':
            self.model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
            self.model = self.model.blocks[:-2]
            self.in_c = 2304

            self.path_pool = nn.AdaptiveAvgPool3d((8, 7, 7))
            if args.dataset == 'oats':
                self.resolution = (7, 7)
                self.resolution3d = (4, 7, 7)
            else:
                self.resolution = (7, 7) 
                self.resolution3d = (8, 7, 7)
        else:
            self.model = torch.hub.load('facebookresearch/pytorchvideo:main', 'x3d_m', pretrained=True)
            self.model = self.model.blocks[:-1] #edited originally -1
            self.in_c = 192 #original 192

            self.resolution = (28, 28)#originally 7x7
            self.resolution3d = (32, 28, 28)

        self.pool = nn.AdaptiveAvgPool3d(output_size=1)
        self.hidden_dim = args.channel
        self.slot_dim = args.channel

        self.num_slots = num_slots

        if args.bg_slot:
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots+1,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim,
                resolution=self.resolution3d,
                num_actor_class = num_class
                ) 
        else:
            self.slot_attention = SlotAttention(
                num_slots=self.num_slots,
                dim=self.slot_dim,
                eps = 1e-8,
                input_dim=self.hidden_dim,
                resolution=self.resolution3d,
                num_actor_class = num_class
                ) 

        self.head = Allocated_Head(self.slot_dim, num_class)

        self.conv3d = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm3d(self.in_c),
                nn.Conv3d(self.in_c, self.hidden_dim, (1, 1, 1), stride=1),
                nn.ReLU(),)
        self.drop = nn.Dropout(p=0.5)         

    def forward(self, x): #starts at [b,c,t,h,w]
        x = x.permute(2, 0, 1, 3, 4)
        seq_len = len(x)
        batch_size = x[0].shape[0]
        height, width = x[0].shape[2], x[0].shape[3]

        # seq_len = x[0].shape[1]
        # batch_size = len(x)
        # height, width = x[0].shape[2], x[0].shape[3]

        #[t,b,c,h,w]
        if self.args.backbone == 'slowfast':
            slow_x = []
            for i in range(0, seq_len, 4):
                slow_x.append(x[i])

            if isinstance(x, list):
                x = torch.stack(x, dim=0) #[v, b, 2048, h, w]
            # l, b, c, h, w
            x = x.permute((1,2,0,3,4)) #[b, v, 2048, h, w]
            slow_x = torch.stack(slow_x, dim=0)
            slow_x = slow_x.permute((1,2,0,3,4))
            x = [slow_x, x]

            for i in range(len(self.model)):
                x = self.model[i](x)
            x[1] = self.path_pool(x[1])
            x = torch.cat((x[0], x[1]), dim=1)
        else:
            if isinstance(x, list):
                x = torch.stack(x, dim=0) #[T, b, C, h, w]
                # l, b, c, h, w
            x = x.permute((1,2,0,3,4)) #[b, C, T, h, w]
            for i in range(len(self.model)):
                x = self.model[i](x)

        # b,c,t,h,w
        x = self.drop(x)
        new_seq_len = x.shape[2]
        new_h, new_w = x.shape[3], x.shape[4]

        # # [b, c, n , w, h]
        x = self.conv3d(x)
        x = x.permute((0, 2, 3, 4, 1))

        # [bs, n, w, h, c]
        x = torch.reshape(x, (batch_size, new_seq_len, new_h, new_w, -1))

        #edited
        x = x.permute(0, 4, 1, 2, 3)  # [batch_size, c, n, w, h]
        # Step 2: Reshape to combine c and n into one dimension, making the tensor 4D
        batch_size, c, n, w, h = x.shape
        x = torch.reshape(x, (batch_size, c * n, w, h))  # [batch_size, c*n, w, h]
        # Step 3: Apply bilinear interpolation on the spatial dimensions
        x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=True)
        # Step 4: Reshape back to the 5D shape
        new_w, new_h = x.shape[2], x.shape[3]  # New interpolated width and height
        x = torch.reshape(x, (batch_size, c, n, new_w, new_h))  # [batch_size, c, n, new_w, new_h]
        # Step 5: Permute back to original shape [batch_size, n, new_w, new_h, c]
        x = x.permute(0, 2, 3, 4, 1) 
        #edited

        x, attn_masks = self.slot_attention(x)

        # no pool, 3d slot
        b, n, thw = attn_masks.shape
        attn_masks = attn_masks.reshape(b, n, -1)
        attn_masks = attn_masks.view(b, n, new_seq_len, self.resolution[0], self.resolution[1])
        attn_masks = attn_masks.unsqueeze(-1)
        # b*s, n, 4, h, w, 1
        attn_masks = attn_masks.reshape(b, n, -1)
        # b*s, n, 4*h*w
        attn_masks = attn_masks.view(b, n, new_seq_len, self.resolution[0], self.resolution[1])
        # b*s, n, 4, h, w
        attn_masks = attn_masks.unsqueeze(-1)
        # b*s, n, 4, h, w, 1
        attn_masks = attn_masks.view(b*n, 1, new_seq_len, attn_masks.shape[3], attn_masks.shape[4])
        # b, n, t, h, w
        if seq_len > new_seq_len:
            attn_masks = F.interpolate(attn_masks, size=(seq_len, new_h, new_w), mode='trilinear')
        # b, l, n, h, w
        attn_masks = torch.reshape(attn_masks, (b, n, seq_len, new_h, new_w))
        attn_masks = attn_masks.permute((0, 2, 1, 3, 4))


        x = self.drop(x) #
        x = self.head(x) #
        return x, attn_masks #

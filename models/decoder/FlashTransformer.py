# ------------------------------------------------------------------
"""
Decoder FlashTransformer: Flash-Attention backbone

Implementation of RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
https://arxiv.org/abs/2505.22535

Built upon:
- Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer (Apache-2.0 license)
- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
https://github.com/Dao-AILab/flash-attention (BSD 3-Clause License)

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

try:
    from timm.layers import DropPath, trunc_normal_
except:
    from timm.models.layers import DropPath, trunc_normal_

from functools import reduce
from operator import mul
import math
from flash_attn import flash_attn_qkvpacked_func
#from flash_attn.modules.mha import MHA
from models.mha import MHA
from models.loan import LOAN

# - - - - - - - - - - - - - - - - - - - - - - - - - -

class MLP(nn.Module):
    """ Multilayer perceptron."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 use_checkpoint=False, use_reentrant=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

    def forward(self, x):
        if self.use_checkpoint:
            x = self.drop(self.act(checkpoint.checkpoint(self.fc1, x, use_reentrant=self.use_reentrant)))
            x = checkpoint.checkpoint(self.fc2, x, use_reentrant=self.use_reentrant)
        else:
            x = self.drop(self.act(self.fc1(x)))
            x = self.fc2(x)
        return x


def serialization(x, curve):
    """
    B, T, P, C = x.shape
    B, P = curve.shape
    """
    x_curve = []
    for b in range(len(curve)):
        x_curve.append(x[b, :, curve[b], :])

    x_curve = torch.stack(x_curve, dim=0)

    return x_curve


def window_partition(x, grouping_size, order='spatial_first'):
    """
    Args:
        x: (B, T, P, C)
        grouping_size (tuple[int]): grouping_size
        order (string): 'temporal_first', 'spatial_first', 'temporal', 'spatial'
    Returns:
        windows: (B*num_groups, grouping_size[0]*grouping_size[1], C)
    """
    B, T, P, C = x.shape
    if order == 'temporal_first':
        x = x.view(B, T // grouping_size[0], grouping_size[0], P // grouping_size[1], grouping_size[1], C)
        windows = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, reduce(mul, grouping_size), C)
    elif order == 'spatial_first':
        x = x.view(B, T // grouping_size[0], grouping_size[0], P // grouping_size[1], grouping_size[1], C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, reduce(mul, grouping_size), C)
    elif order == 'temporal':
        windows = x.permute(0, 2, 1, 3).contiguous().view(-1, T, C)#.contiguous()
    elif order == 'spatial':
        windows = x.contiguous().contiguous().view(-1, P, C)#.contiguous()
    return windows


def window_reverse(windows, grouping_size, B, T, P, order='spatial_first'):
    """
    Args:
        windows: (B*num_groups, grouping_size[0]*grouping_size[1], C)
        grouping_size (tuple[int]): grouping_size
        B (int): batch size
        T (int): temporal resolution
        W (int): number of the points
        order (string): 'temporal_first', 'spatial_first', 'temporal', 'spatial'
    Returns:
        x: (B, T, P, C)
    """
    if order == 'temporal_first':
        x = windows.view(B, T // grouping_size[0], P // grouping_size[1], grouping_size[0], grouping_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, T, P, -1)
    elif order == 'spatial_first':
        x = windows.view(B, P // grouping_size[1], T // grouping_size[0], grouping_size[1], grouping_size[0], -1)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, T, P, -1)
    elif order == 'temporal':
        x = windows.view(B, P, T, -1).permute(0, 2, 1, 3).contiguous()
    elif order == 'spatial':
        x = windows.view(B, T, P, -1).contiguous()
    return x


# From https://github.com/ofirpress/attention_with_linear_biases/blob/4b92f28a005ead2567abe2359f633e73e08f3833/fairseq/models/transformer.py#L742
# see https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/modules/mha.py
def get_alibi_slopes(nheads):
    def get_slopes_power_of_2(nheads):
        start = 2 ** (-(2 ** -(math.log2(nheads) - 3)))
        ratio = start
        return [start * ratio**i for i in range(nheads)]

    if math.log2(nheads).is_integer():
        return get_slopes_power_of_2(nheads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(nheads))
        return (
            get_slopes_power_of_2(closest_power_of_2)
            + get_alibi_slopes(2 * closest_power_of_2)[0::2][: nheads - closest_power_of_2]
        )

class TransformerBlock(nn.Module):
    """ RiverMamba Transformer Block """
    def __init__(self,
                 dim,
                 dim_static,
                 in_hres,
                 embed_hres,
                 n_heads,
                 grouping_size=(3, 10),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 is_causal=False,
                 is_alibi_slopes=False,
                 att_window_size=(-1, -1),
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 curve_id=0,
                 curve_order='spatial_first',
                 use_checkpoint=False,
                 use_reentrant=True,
                 ):

        super().__init__()
        self.dim = dim
        self.dim_static = dim_static
        self.n_heads = n_heads
        self.grouping_size = grouping_size
        self.mlp_ratio = mlp_ratio
        self.curve_id = curve_id
        self.curve_order = curve_order
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        self.softmax_scale = qk_scale
        self.is_causal = is_causal
        self.proj_drop = proj_drop
        self.attn_drop = attn_drop

        self.is_alibi_slopes = is_alibi_slopes
        #alibi_slopes = torch.tensor(get_alibi_slopes(n_heads), device='cuda') if is_alibi_slopes else None

        self.norm1 = norm_layer(dim + embed_hres, elementwise_affine=False)
        self.loan1 = LOAN(dim + embed_hres, dim_static)

        self.attn = MHA(embed_dim=dim + embed_hres, num_heads=n_heads, num_heads_kv=None, cross_attn=False,
                        qkv_proj_bias=qkv_bias, out_proj_bias=False, dropout=attn_drop, softmax_scale=qk_scale,
                        causal=False, layer_idx=None, dwconv=False, rotary_emb_dim=0, rotary_emb_base=10000.0,
                        rotary_emb_scale_base=None, rotary_emb_interleaved=False, use_alibi=is_alibi_slopes,
                        window_size=att_window_size, fused_bias_fc=False, use_flash_attn=True, return_residual=False,
                        checkpointing=use_checkpoint, device='cuda', dtype=None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim, elementwise_affine=False)
        self.loan2 = LOAN(dim, dim_static)

        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                       use_checkpoint=use_checkpoint, use_reentrant=use_reentrant)

        self.point_embed_hres = InputEmbed(in_hres, embed_hres)
        self.out_proj = nn.Linear(dim + embed_hres, dim, bias=False)

    def forward_part1(self, x, x_static, x_hres):
        """ Forward part1 function """

        if self.use_checkpoint:
            x_hres = checkpoint.checkpoint(self.point_embed_hres, x_hres, use_reentrant=self.use_reentrant)
        else:
            x_hres = self.point_embed_hres(x_hres)

        x = torch.cat((x, x_hres), dim=-1)

        B, T, P, C = x.shape

        x = self.norm1(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.loan1, x_static, use_reentrant=self.use_reentrant)
        else:
            x = x + self.loan1(x_static)

        # padding
        # pad feature maps to multiples of group size
        pad_t = pad_p = 0
        pad_tt = (self.grouping_size[0] - T % self.grouping_size[0]) % self.grouping_size[0]
        pad_pp = (self.grouping_size[1] - P % self.grouping_size[1]) % self.grouping_size[1]
        x = F.pad(x, (0, 0, pad_p, pad_pp, pad_t, pad_tt))
        _, Tp, Pp, _ = x.shape

        # partition groups
        x = window_partition(x, self.grouping_size, order=self.curve_order)  # B*nW, Wd*Wh*Ww, C
        # W-MSA/SW-MSA
        x = self.attn(x)  # B*nW, Wd*Wh*Ww, C

        # merge groups
        x = x.view(-1, *(self.grouping_size + (C,)))
        x = window_reverse(x, self.grouping_size, B, Tp, Pp, order=self.curve_order)  # B D' H' W' C

        if pad_tt > 0 or pad_pp > 0:
            x = x[:, :T, :P, :].contiguous()

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.out_proj, x, use_reentrant=self.use_reentrant)
        else:
            x = self.out_proj(x)

        return x

    def forward_part2(self, x, x_static):
        """ Forward part2 function """

        x = self.norm2(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.loan2, x_static, use_reentrant=self.use_reentrant)
        else:
            x = x + self.loan2(x_static)

        x = self.mlp(x)

        return x

    def forward(self, x, x_static, x_hres, x_curves):
        """ Forward function """

        # form the curves
        x = serialization(x, x_curves[:, self.curve_id, 0, :])
        x_static = serialization(x_static, x_curves[:, self.curve_id, 0, :])
        x_hres = serialization(x_hres, x_curves[:, self.curve_id, 0, :])

        # first forward function
        x = x + self.drop_path(self.forward_part1(x, x_static, x_hres))
        # second forward function
        x = x + self.drop_path(self.forward_part2(x, x_static))

        # rearrange the point cloud
        x = serialization(x, x_curves[:, self.curve_id, 1, :])

        return x


class BasicLayer(nn.Module):
    """ A basic RiverMamba Transformer layer """

    def __init__(self,
                 in_hres,
                 in_features,
                 in_dim,
                 embed_hres,
                 dim,
                 dim_static,
                 depth,
                 n_heads,
                 att_window_size,
                 grouping_size=(3, 10),
                 mlp_ratio=4.,
                 qkv_bias=False,
                 drop=0.,
                 attn_drop=0.,
                 proj_drop=0.,
                 qk_scale=None,
                 is_causal=False,
                 is_alibi_slopes=False,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=None,
                 embed_norm=None,
                 curve_ids=None,
                 curve_order='spatial_first',
                 use_checkpoint=False,
                 use_reentrant=False
                 ):

        super().__init__()

        self.grouping_size = grouping_size
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant
        self.dim = dim
        self.dim_static = dim_static
        self.in_dim = in_dim
        self.curve_ids = curve_ids
        self.curve_order = curve_order
        self.in_hres = in_hres
        self.in_features = in_features
        self.embed_hres = embed_hres

        # build embedding
        if embed_layer:
            if in_dim != dim:
                self.point_embed = embed_layer(in_chans=in_dim, embed_dim=dim, norm_layer=embed_norm)
            else:
                self.point_embed = None

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                dim_static=dim_static,
                in_hres=in_hres,
                embed_hres=embed_hres,
                n_heads=n_heads,
                grouping_size=grouping_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                is_causal=is_causal,
                is_alibi_slopes=is_alibi_slopes,
                att_window_size=att_window_size,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                curve_id=curve_ids[i],
                curve_order=curve_order,
                use_checkpoint=use_checkpoint,
                use_reentrant=use_reentrant
            )
            for i in range(depth)])

    def forward(self, x, x_hres, x_static, x_curves):
        """ Forward function """

        if self.point_embed is not None:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.point_embed, x, use_reentrant=self.use_reentrant)
            else:
                x = self.point_embed(x)

        for blk in self.blocks:
            x = blk(x, x_static, x_hres, x_curves)

        return x


class PointEmbed(nn.Module):
    """
    Embedding for the SSM layers
    """
    def __init__(self, in_chans=28, embed_dim=32, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_chans, embed_dim, bias=False)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):

        x = self.proj(x)

        if self.norm is not None:
            x = self.norm(x)
        return x


class InputEmbed(nn.Module):
    """
    Embedding for the input data
    """
    def __init__(self, in_chans=28, embed_dim=32):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.embed = nn.Linear(in_chans, embed_dim, bias=True)
        self.act = nn.Tanh()

    def forward(self, x):
        return self.act(self.embed(x))


class FlashTransformer(nn.Module):
    """
    Decoder RiverMamba: FlashTransformer backbone

    Implementation of RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
    https://arxiv.org/abs/2505.22535
    """
    def __init__(self,
                 in_hres=7,
                 in_features=256,
                 in_static=10,
                 n_curves=6,
                 curve_order='spatial',
                 embed_hres=32,
                 embed_norm=True,
                 embed_dim=None,
                 n_heads=None,
                 depths=None,
                 grouping_size=None,
                 mlp_ratio=1,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 proj_drop=0.,
                 drop_path_rate=0.,
                 qkv_bias=False,
                 qk_scale=None,
                 is_causal=False,
                 is_alibi_slopes=False,
                 att_window_size=None,
                 use_checkpoint=False,
                 use_reentrant=True,
                 ):
        """
            Args:
                in_hres (int, optional): number of input ECMWF-HRES meteorological forecast. Defaults to 7
                in_features (int, optional): number of input features. Defaults to 256
                in_static (int, optional): number of input static variables. Defaults to 10
                n_curves (int, optional): number of curves. Defaults to 6
                curve_order (str, optional): the order of curves. Defaults to 'spatial'
                embed_hres (int, optional): number of embedding dimension for ECMWF-HRES. Defaults to 32
                embed_norm (bool, optional): whether to use normalization after embedding. Defaults to True
                embed_dim (list, optional): embedding dimension per layer. Defaults to [32, 32, 32]
                depths (list, optional): the number of blocks in each layer. Defaults to [1, 1, 1]
                grouping_size (list, optional): the group size in each layer. Defaults to [(4, 15), (2, 15), (1, 15)]
                mlp_ratio (int, optional): ratio of mlp hidden dim to embedding dim. Defaults to 1
                drop_rate (float, optional): dropout rate. Defaults to 0.
                drop_path_rate (float, optional): dropout rate. Defaults to 0.
                n_heads(list, optional): number of heads in each layer. Defaults to 2
                attn_drop_rate (float, optional): dropout rate for attention module. Defaults to 0.
                proj_drop (float, optional): dropout rate for projection module. Defaults to 0.
                qkv_bias (bool, optional): whether to add a learnable bias to query, key, value. Defaults to False
                qk_scale (float, optional): override default qk scale of head_dim ** -0.5 if set. Defaults to None
                is_causal (bool, optional): whether to use causal attention. Defaults to False
                is_alibi_slopes (bool, optional): whether to use alibi encoding. Defaults to False
                att_window_size (list, optional): attention window size in each layer. Defaults to (-1, -1)
                use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False
                use_reentrant (bool, optional): whether to use reentrant. Defaults to True
        """
        super(FlashTransformer, self).__init__()

        self.in_hres = in_hres
        self.in_features = in_features
        self.in_static = in_static
        self.n_curves = n_curves
        self.curve_order = curve_order
        assert curve_order in ['spatial_first', 'temporal_first', 'spatial', 'temporal']

        self.embed_hres = embed_hres
        self.embed_norm = embed_norm

        self.in_channels = embed_hres + in_features

        self.embed_dim = embed_dim if embed_dim is not None else [256]
        self.embed_dim = self.embed_dim if len(self.embed_dim) == in_hres else self.embed_dim * in_hres

        self.num_layers = len(self.embed_dim)

        self.depths = depths if depths is not None else [2]
        self.depths = self.depths if len(self.depths) == in_hres else self.depths * in_hres

        self.grouping_size = grouping_size if grouping_size is not None else [(1, 10)]
        self.grouping_size = self.grouping_size if len(self.grouping_size) == in_hres else self.grouping_size * in_hres

        self.att_window_size = att_window_size if att_window_size is not None else [(-1, -1)]
        self.att_window_size = self.att_window_size if len(self.att_window_size) == in_hres else self.att_window_size * in_hres

        self.n_heads = n_heads if n_heads is not None else [4]
        self.n_heads = self.n_heads if len(self.n_heads) == in_hres else self.n_heads * in_hres

        norm_layer = nn.LayerNorm  # fixed norm layer

        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.proj_drop = proj_drop
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.is_causal = is_causal
        self.is_alibi_slopes = is_alibi_slopes
        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        self.curve_ids = []
        c_1 = np.arange(self.n_curves).tolist()
        c_2 = np.roll(c_1, -1).tolist()

        for i in range(self.num_layers):
            self.curve_ids.append(c_1[:self.depths[i]] if i % 2 == 0 else c_2[:self.depths[i]])

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):

            layer = BasicLayer(
                in_features=self.in_features,
                in_hres=self.in_hres,
                in_dim=self.embed_dim[i_layer - 1] if i_layer != 0 else in_features,  # + embed_hres,
                dim=self.embed_dim[i_layer],
                dim_static=self.in_static,
                embed_hres=self.embed_hres,
                depth=self.depths[i_layer],
                n_heads=self.n_heads[i_layer],
                grouping_size=self.grouping_size[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                proj_drop=proj_drop,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                qk_scale=self.qk_scale,
                is_causal=self.is_causal,
                is_alibi_slopes=self.is_alibi_slopes,
                att_window_size=self.att_window_size[i_layer],
                norm_layer=norm_layer,
                embed_layer=PointEmbed,
                embed_norm=norm_layer if self.embed_norm else None,  # and i_layer == 0 else None,#
                curve_ids=self.curve_ids[i_layer], #[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                curve_order='spatial',# if i_layer != 0 else 'temporal',
                use_checkpoint=self.use_checkpoint,
                use_reentrant=self.use_reentrant
            )

            self.layers.append(layer)

        self.init_weights()

    def init_weights(self):
        """Initialize the weights in backbone"""

        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)

        for l in range(len(self.layers)):
            layer = self.layers[l]
            for block in range(self.depths[l]):
                layer.blocks[block].loan1.init_weights()
                layer.blocks[block].loan2.init_weights()

    def forward(self, x, x_hres, x_static, x_curves):
        """
            Forward function
            Args:
                x (torch.tensor): input encoded features [B, 1, P, K]
                x_hres (torch.tensor): input dynamic variables [B, lead time, P, V]
                x_static (torch.tensor): input static variables [B, P, V]
                x_curves (torch.tensor): input curves indices  [B, Number of Curves, 2, P]
            Returns:
                x_heads (torch.tensor): output features [B, 1, P, K] x lead time
        """

        # extend static features by T=1
        x_static = x_static[:, None, :, :]

        x_heads = []
        for t in range(self.in_hres):
            x = self.layers[t](x, x_hres[:, t:t+1, :, :], x_static, x_curves)
            x_heads.append(x)

        return x_heads


if __name__ == '__main__':

    """
    data_hres      shape:  (B, lead time, P, V)
    data_features  shape:  (B, 1, P, K)
    data_static    shape:  (B, P, V)
    curves         shape:  (B, Number of Curves, 2, P)
    """

    data_hres = torch.randn((1, 7, 82804, 7)).cuda()
    data_features = torch.randn((1, 1, 82804, 256)).cuda()
    data_static = torch.randn((1, 82804, 10)).cuda()
    curves = torch.arange(0, 82804)
    curves = curves[None, None, None, :].repeat(1, 6, 2, 1).cuda()

    model = FlashTransformer().cuda()
    print(model)
    n_parameters = sum(p.numel() for p in model.parameters())
    print(f"number of parameters: {n_parameters}")

    import time
    from torch import autocast

    with torch.no_grad():
        for i in range(10):
            time_start = time.time()
            with autocast(device_type='cuda', dtype=torch.float16):
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                test = model(data_features, data_hres, data_static, curves)
            # print(test.shape)
            print(time.time() - time_start)
            ##print(prof)

# ------------------------------------------------------------------
"""
Encoder FlashTransformer: Flash-Attention backbone

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

        self.norm1 = norm_layer(dim, elementwise_affine=False)
        self.loan1 = LOAN(dim, dim_static)

        self.attn = MHA(embed_dim=dim, num_heads=n_heads, num_heads_kv=None, cross_attn=False,
                        qkv_proj_bias=qkv_bias, out_proj_bias=False, dropout=attn_drop, softmax_scale=qk_scale,
                        causal=False, layer_idx=None, dwconv=False, rotary_emb_dim=0, rotary_emb_base=10000.0,
                        rotary_emb_scale_base=None, rotary_emb_interleaved=False, use_alibi=is_alibi_slopes,
                        window_size=att_window_size, fused_bias_fc=False, use_flash_attn=True, return_residual=False,
                        checkpointing=use_checkpoint, device='cuda', dtype=None)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim, elementwise_affine=False)
        self.loan2 = LOAN(dim, dim_static)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                       use_checkpoint=False, use_reentrant=True)

    def forward_part1(self, x, x_static):
        """ Forward part1 function """

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

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.attn, x, use_reentrant=self.use_reentrant)
        else:
            x = self.attn(x)  # B*nW, Wd*Wh*Ww, C

        # merge groups
        x = x.view(-1, *(self.grouping_size + (C,)))
        x = window_reverse(x, self.grouping_size, B, Tp, Pp, order=self.curve_order)  # B D' H' W' C

        if pad_tt > 0 or pad_pp > 0:
            x = x[:, :T, :P, :].contiguous()

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

    def forward(self, x, x_static, x_curves):
        """ Forward function """

        # form the curves
        x = serialization(x, x_curves[:, self.curve_id, 0, :])
        x_static = serialization(x_static, x_curves[:, self.curve_id, 0, :])

        # first forward function
        x = x + self.drop_path(self.forward_part1(x, x_static))
        # second forward function
        x = x + self.drop_path(self.forward_part2(x, x_static))

        # rearrange the point cloud
        x = serialization(x, x_curves[:, self.curve_id, 1, :])

        return x


class BasicLayer(nn.Module):
    """ A basic RiverMamba Transformer layer """

    def __init__(self,
                 in_dim,
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

        # build embedding
        if embed_layer:
        #if in_dim != dim:
            self.point_embed = embed_layer(in_chans=in_dim, embed_dim=dim, norm_layer=embed_norm)
        else:
            self.point_embed = None

        # build blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                dim_static=dim_static,
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

    def forward(self, x, x_static, x_curves):
        """ Forward function """

        if self.point_embed is not None:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(self.point_embed, x, use_reentrant=self.use_reentrant)
            else:
                x = self.point_embed(x)

        for blk in self.blocks:
            x = blk(x, x_static, x_curves)

        return x


class PointEmbed(nn.Module):
    """
    Embedding for the SSM layers
    Downsample the input by 2 along the T dimension
    """
    def __init__(self, in_chans=28, embed_dim=32, norm_layer=None):
        super().__init__()

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Linear(in_chans * 2, embed_dim, bias=False)

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim, elementwise_affine=False)
        else:
            self.norm = None

    def forward(self, x):

        B, T, P, C = x.shape

        pad_T = (T % 2 == 1) and T != 1

        if pad_T:
            x = F.pad(x, (0, 0, 0, 0, 0, T % 2))

        if T != 1:
            x0 = x[:, 0::2, :, :]  # B D H/2 W/2 C
            x1 = x[:, 1::2, :, :]  # B D H/2 W/2 C
        else:
            x0 = x[:, 0::2, :, :]  # B D/2 H/2 W/2 C
            x1 = x[:, 1::2, :, :]  # B D/2 H/2 W/2 C

        x = torch.cat([x0, x1], -1)  # B T/2 P 4*C

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
    Encoder RiverMamba: FlashTransformer backbone

    Implementation of RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
    https://arxiv.org/abs/2505.22535
    """
    def __init__(self,
                 in_glofas=4,
                 in_era5=24,
                 in_static=10,
                 n_curves=6,
                 curve_order='spatial',
                 embed_glofas=8,
                 embed_era5=16,
                 embed_cpc=8,
                 embed_norm=True,
                 embed_dim=None,
                 n_heads=None,
                 depths=None,
                 grouping_size=None,
                 mlp_ratio=4,
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
                 use_reentrant=True
                 ):

        """
            Args:
                in_glofas (int, optional): number of input GloFAS reanalysis variables. Defaults to 4
                in_era5 (int, optional): number of input ERA5-Land reanalysis variables. Defaults to 24
                in_static (int, optional): number of input static variables. Defaults to 10
                n_curves (int, optional): number of curves. Defaults to 6
                curve_order (str, optional): the order of curves. Defaults to 'spatial_first'
                embed_glofas (int, optional): number of embedding dimension for GloFAS. Defaults to 8
                embed_era5 (int, optional): number of embedding dimension for ERA5-Land. Defaults to 16
                embed_cpc (int, optional): number of embedding dimension for CPC. Defaults to 8
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

        self.in_glofas = in_glofas
        self.in_era5 = in_era5
        self.in_static = in_static

        self.n_curves = n_curves
        self.curve_order = curve_order
        assert curve_order in ['spatial_first', 'temporal_first', 'spatial', 'temporal']

        self.embed_glofas = embed_glofas
        self.embed_era5 = embed_era5
        self.embed_cpc = embed_cpc

        self.embed_norm = embed_norm
        self.embed_dim = embed_dim if embed_dim is not None else [32, 32, 32]
        self.num_layers = len(self.embed_dim)
        self.depths = depths if depths is not None else [1 for _ in range(self.num_layers)]
        self.in_channels = embed_glofas + embed_era5 + embed_cpc

        self.grouping_size = grouping_size if grouping_size is not None else [(4, 15), (2, 15), (1, 15)]
        self.att_window_size = att_window_size if att_window_size is not None else [(-1, -1) for _ in range(self.num_layers)]

        norm_layer = nn.LayerNorm  # fixed norm layer

        self.n_heads = n_heads if n_heads is not None else [2 for _ in range(self.num_layers)]
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
        for i in range(sum(self.depths)):
            self.curve_ids.append(i - i // self.n_curves * self.n_curves)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        self.layers = nn.ModuleList()

        for i_layer in range(self.num_layers):

            layer = BasicLayer(
                in_dim=self.embed_dim[i_layer - 1] if i_layer > 0 else self.in_channels,
                dim=self.embed_dim[i_layer],
                dim_static=self.in_static,
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
                embed_layer=PointEmbed if i_layer > 0 else None,
                embed_norm=norm_layer if self.embed_norm else None,  # and i_layer == 0 else None,#
                curve_ids=self.curve_ids[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                curve_order=curve_order,# if i_layer != 0 else 'temporal',
                use_checkpoint=self.use_checkpoint,
                use_reentrant=self.use_reentrant
            )

            self.layers.append(layer)

        self.point_embed_glofas = InputEmbed(in_glofas, self.embed_glofas)
        self.point_embed_cpc = InputEmbed(1, self.embed_cpc)
        self.point_embed_era5 = InputEmbed(in_era5, self.embed_era5)

        self.norm_input_embedding = nn.LayerNorm(self.in_channels, elementwise_affine=True)

        self.time_pos_embed = nn.Parameter(torch.zeros(1, 4, 1, self.in_channels))

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

    def forward(self, x_glofas, x_era5, x_cpc, x_static, x_curves):
        """
            Forward function
            Args:
                x_glofas (torch.tensor): input dynamic variables [B, T, P, V]
                x_era5 (torch.tensor): input dynamic variables [B, T, P, V]
                x_cpc (torch.tensor): input dynamic variables [B, T, P, 1]
                x_static (torch.tensor): input static variables [B, P, V]
                x_curves (torch.tensor): input curves indices  [B, Number of Curves, 2, P]
            Returns:
                x (torch.tensor): output dynamic features [B, 1, P, K_embedding]
        """

        # extend static features by T=1
        x_static = x_static[:, None, :, :]

        # embed the input with multi-modality
        if self.use_checkpoint:
            x_glofas = checkpoint.checkpoint(self.point_embed_glofas, x_glofas, use_reentrant=self.use_reentrant)
            x_era5 = checkpoint.checkpoint(self.point_embed_era5, x_era5, use_reentrant=self.use_reentrant)
            x_cpc = checkpoint.checkpoint(self.point_embed_cpc, x_cpc, use_reentrant=self.use_reentrant)
        else:
            x_glofas = self.point_embed_glofas(x_glofas)
            x_era5 = self.point_embed_era5(x_era5)
            x_cpc = self.point_embed_cpc(x_cpc)

        # concatenate the input
        x = torch.cat((x_glofas, x_era5, x_cpc), dim=-1) + self.time_pos_embed
        x = self.norm_input_embedding(x)

        # Encoder/Hindcast layers
        for layer in self.layers:
            x = layer(x, x_static, x_curves)

        return x


if __name__ == '__main__':


    """
    data_glofas    shape:  (B, T, P, V)
    data_era5_land shape:  (B, T, P, V)
    data_cpc       shape:  (B, T, P, 1)
    data_static    shape:  (B, P, V)
    curves         shape:  (B, Number of Curves, 2, P)
    """

    data_glofas = torch.randn((1, 4, 82804, 4)).cuda()
    data_era5_land = torch.randn((1, 4, 82804, 24)).cuda()
    data_static = torch.randn((1, 82804, 10)).cuda()
    data_cpc = torch.randn((1, 4, 82804, 1)).cuda()

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
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                # with torch.autograd.profiler.profile(use_cuda=True) as prof:
                test = model(data_glofas, data_era5_land, data_cpc, data_static, curves)
            # print(test.shape)
            print(time.time() - time_start)
            #print(prof)


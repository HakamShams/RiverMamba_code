# ------------------------------------------------------------------
"""
Encoder RiverMamba: Mamba backbone

Implementation of RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
https://arxiv.org/abs/2505.22535

Built upon:
- Video Swin Transformer
https://github.com/SwinTransformer/Video-Swin-Transformer (Apache-2.0 license)
- Mamba: Linear-Time Sequence Modeling with Selective State Spaces
https://github.com/state-spaces/mamba (Apache-2.0 license)

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

try:
    from timm.layers import DropPath, trunc_normal_, lecun_normal_
except:
    from timm.models.layers import DropPath, trunc_normal_, lecun_normal_

from functools import reduce, partial
from operator import mul
from mamba_ssm import Mamba as Mamba_v1
from models.loan import LOAN
import math

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
        windows = x.permute(0, 2, 1, 3).contiguous().view(-1, T, C)  # .contiguous()
    elif order == 'spatial':
        windows = x.contiguous().contiguous().view(-1, P, C)  # .contiguous()

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


class MambaBlock(nn.Module):
    """ RiverMamba Block """

    def __init__(self,
                 dim,
                 dim_static,
                 grouping_size=(3, 10),
                 mlp_ratio=1.,
                 d_state=1,
                 expand=1,
                 d_conv=3,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 curve_id=0,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 is_divide_out=True,
                 out_norm=False,
                 ):

        super().__init__()

        self.dim = dim
        self.dim_static = dim_static
        self.grouping_size = grouping_size

        self.mlp_ratio = mlp_ratio
        self.d_state = d_state
        self.expand = expand
        self.d_conv = d_conv
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.dt_rank = dt_rank
        self.dt_init = dt_init
        self.dt_scale = dt_scale
        self.dt_init_floor = dt_init_floor
        self.conv_bias = conv_bias
        self.proj_bias = proj_bias

        self.curve_id = curve_id
        self.curve_order = curve_order
        self.bi_ssm = bi_ssm

        self.use_checkpoint = use_checkpoint
        self.use_reentrant = use_reentrant

        self.is_divide_out = is_divide_out

        self.norm1 = norm_layer(dim * expand, elementwise_affine=False)
        self.loan1 = LOAN(dim, dim_static)

        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.loan2 = LOAN(int(expand * dim), dim_static)

        self.in_proj = nn.Linear(dim, int(expand * dim) * 2, bias=proj_bias)

        self.ssm = Mamba_v1(
            d_model=dim,
            d_state=d_state,
            expand=expand,
            d_conv=d_conv,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_rank=dt_rank,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            conv_bias=conv_bias,
            bias=proj_bias,
        )

        if bi_ssm:
            self.ssm_b = Mamba_v1(
                d_model=dim,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_rank=dt_rank,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                bias=proj_bias,
            )

        self.out_proj = nn.Linear(int(expand * dim), dim, bias=proj_bias)

        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop,
                       use_checkpoint=use_checkpoint, use_reentrant=use_reentrant)

        self.drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if out_norm:
            self.out_norm = norm_layer(int(expand * dim), elementwise_affine=False)
        else:
            self.out_norm = None

    def forward_part1(self, x, x_static):
        """ Forward part1 function """

        B, T, P, C = x.shape

        x = self.norm1(x)

        if self.use_checkpoint:
            x = x + checkpoint.checkpoint(self.loan1, x_static, use_reentrant=self.use_reentrant)
        else:
            x = x + self.loan1(x_static)

        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.in_proj, x, use_reentrant=self.use_reentrant)
        else:
            x = self.in_proj(x)

        # padding
        # pad feature maps to multiples of group size
        pad_t = pad_p = 0
        pad_tt = (self.grouping_size[0] - T % self.grouping_size[0]) % self.grouping_size[0]
        pad_pp = (self.grouping_size[1] - P % self.grouping_size[1]) % self.grouping_size[1]
        x = F.pad(x, (0, 0, pad_p, pad_pp, pad_t, pad_tt))
        _, Tp, Pp, _ = x.shape

        # partition groups
        if self.bi_ssm:
            x_b = window_partition(x.flip([-2]), self.grouping_size, order=self.curve_order)  # B*nW, Wd*Wh*Ww, C
        x = window_partition(x, self.grouping_size, order=self.curve_order)  # B*nW, Wd*Wh*Ww, C

        # SSM
        if self.use_checkpoint:
            x = checkpoint.checkpoint(self.ssm, x.permute(0, 2, 1), use_reentrant=self.use_reentrant)
        else:
            x = self.ssm(x.permute(0, 2, 1))

        # if bidirectional SSM
        if self.bi_ssm:
            if self.use_checkpoint:
                x_b = checkpoint.checkpoint(self.ssm_b, x_b.permute(0, 2, 1), use_reentrant=self.use_reentrant)
            else:
                x_b = self.ssm_b(x_b.permute(0, 2, 1))

        # merge groups
        x = x.permute(0, 2, 1).contiguous().view(-1, *(self.grouping_size + (C,)))
        x = window_reverse(x, self.grouping_size, B, Tp, Pp, order=self.curve_order)  # B D' H' W' C

        if self.bi_ssm:
            x_b = x_b.permute(0, 2, 1).reshape(-1, *(self.grouping_size + (C,)))
            x_b = window_reverse(x_b, self.grouping_size, B, Tp, Pp, order=self.curve_order)  # B D' H' W' C

        if self.bi_ssm:
            x = (x + x_b.flip([-2])) / 2 if self.is_divide_out else x + x_b.flip([-2])

        if pad_tt > 0 or pad_pp > 0:
            x = x[:, :T, :P, :].contiguous()

        if self.out_norm:
            x = self.out_norm(x)

        #x = self.drop(x)

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
    """ A basic RiverMamba layer """

    def __init__(self,
                 in_dim,
                 dim,
                 dim_static,
                 depth,
                 grouping_size=(3, 10),
                 mlp_ratio=4.,
                 d_state=1,
                 expand=1,
                 d_conv=3,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 embed_layer=None,
                 embed_norm=None,
                 curve_ids=None,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 is_divide_out=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 out_norm=False
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
        self.bi_ssm = bi_ssm
        self.is_divide_out = is_divide_out

        # build embedding
        if embed_layer:
            # if in_dim != dim:
            self.point_embed = embed_layer(in_chans=in_dim, embed_dim=dim, norm_layer=embed_norm)
        else:
            self.point_embed = None

        # build blocks
        self.blocks = nn.ModuleList([
            MambaBlock(
                dim=dim,
                dim_static=dim_static,
                grouping_size=grouping_size,
                mlp_ratio=mlp_ratio,
                d_state=d_state,
                expand=expand,
                d_conv=d_conv,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_rank=dt_rank,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                proj_bias=proj_bias,
                drop=drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer,
                curve_id=curve_ids[i],
                curve_order=curve_order,
                bi_ssm=bi_ssm,  # 0 if (i % 2 == 0) else 1,
                is_divide_out=is_divide_out,
                use_checkpoint=use_checkpoint,
                use_reentrant=self.use_reentrant,
                out_norm=out_norm
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

        #self.drop = nn.Dropout(0.3)

    def forward(self, x):
        return self.act(self.embed(x))
        #return self.drop(self.act(self.embed(x)))


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights_m(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=2,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class Mamba(nn.Module):
    """
    Encoder RiverMamba: Mamba backbone

    Implementation of RiverMamba: A State Space Model for Global River Discharge and Flood Forecasting
    https://arxiv.org/abs/2505.22535
    """
    def __init__(self,
                 in_glofas=4,
                 in_era5=24,
                 in_static=10,
                 n_curves=6,
                 curve_order='spatial_first',
                 bi_ssm=True,
                 embed_glofas=8,
                 embed_era5=16,
                 embed_cpc=8,
                 embed_norm=True,
                 embed_dim=None,
                 depths=None,
                 grouping_size=None,
                 mlp_ratio=1,
                 drop_rate=0.,
                 drop_path_rate=0.,
                 d_state=None,
                 expand=None,
                 d_conv=None,
                 dt_min=0.01,
                 dt_max=0.1,
                 dt_rank="auto",
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4,
                 conv_bias=True,
                 proj_bias=False,
                 is_divide_out=True,
                 use_checkpoint=False,
                 use_reentrant=True,
                 out_norm=False
                 ):

        """
            Args:
                in_glofas (int, optional): number of input GloFAS reanalysis variables. Defaults to 4
                in_era5 (int, optional): number of input ERA5-Land reanalysis variables. Defaults to 24
                in_static (int, optional): number of input static variables. Defaults to 10
                n_curves (int, optional): number of curves. Defaults to 6
                curve_order (str, optional): the order of curves. Defaults to 'spatial_first'
                bi_ssm (bool, optional): whether to use bidirectional SSM. Defaults to True
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
                d_state (list, optional): SSM state expansion factor per layer. Defaults to [1, 1, 1]
                d_conv (list, optional): SSM local convolution width per layer. Defaults to [3, 3, 3]
                expand (list, optional): SSM d_inner expansion factor per layer. Defaults to [1, 1, 1]
                dt_min (float, optional): SSM dt_min. Defaults to 0.01
                dt_max (float, optional): SSM dt_max. Defaults to 0.1
                dt_rank (str, optional): SSM dt_rank. Defaults to 'auto'
                dt_init (str, optional): SSM dt_init. Defaults to 'random'
                dt_scale (float, optional): SSM dt_scale. Defaults to 1.0
                dt_init_floor (float, optional): SSM dt_init_floor. Defaults to 1e-4
                conv_bias (bool, optional): whether to use bias in the convolution of SSM. Defaults to True
                proj_bias (bool, optional): whether to use bias in the projection of SSM. Defaults to False
                is_divide_out (bool, optional): whether to divide bidirectional SSM by 2 before the output. Defaults to True
                out_norm (bool, optional): whether to use normalization after at the end of SSM. Defaults to False
                use_checkpoint (bool, optional): whether to use checkpoint. Defaults to False
                use_reentrant (bool, optional): whether to use reentrant. Defaults to True
                ngroups (list, optional): the number of groups in each layer. Defaults to [1, 1, 1]
                headdim (list, optional): the head dimension in each layer. Defaults to [16, 16, 16]
                chunk_size(list, optional): the chunk size. Defaults to 256
        """

        super(Mamba, self).__init__()

        self.in_glofas = in_glofas
        self.in_era5 = in_era5
        self.in_static = in_static

        self.n_curves = n_curves
        self.curve_order = curve_order
        assert curve_order in ['spatial_first', 'temporal_first', 'temporal', 'spatial']

        self.bi_ssm = bi_ssm
        self.is_divide_out = is_divide_out

        self.embed_glofas = embed_glofas
        self.embed_era5 = embed_era5
        self.embed_cpc = embed_cpc

        self.embed_norm = embed_norm
        self.embed_dim = embed_dim if embed_dim is not None else [32, 32, 32]
        self.num_layers = len(self.embed_dim)
        self.depths = depths if depths is not None else [1 for _ in range(self.num_layers)]
        self.in_channels = embed_glofas + embed_era5 + embed_cpc

        self.grouping_size = grouping_size if grouping_size is not None else [(4, 15), (2, 15), (1, 15)]

        self.d_state = d_state if d_state is not None else [1 for _ in range(self.num_layers)]
        self.expand = expand if expand is not None else [1 for _ in range(self.num_layers)]
        self.d_conv = d_conv if d_conv is not None else [3 for _ in range(self.num_layers)]

        norm_layer = nn.LayerNorm  # fixed norm layer

        self.mlp_ratio = mlp_ratio
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
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
                grouping_size=self.grouping_size[i_layer],
                mlp_ratio=mlp_ratio,
                d_state=self.d_state[i_layer],
                expand=self.expand[i_layer],
                d_conv=self.d_conv[i_layer],
                dt_min=dt_min,
                dt_max=dt_max,
                dt_rank=dt_rank,
                dt_init=dt_init,
                dt_scale=dt_scale,
                dt_init_floor=dt_init_floor,
                conv_bias=conv_bias,
                proj_bias=proj_bias,
                drop=self.drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=norm_layer,
                embed_layer=PointEmbed if i_layer > 0 else None,
                embed_norm=norm_layer if self.embed_norm else None,  # and i_layer == 0 else None,#
                curve_ids=self.curve_ids[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                curve_order=curve_order,  # if i_layer != 0 else 'temporal',
                bi_ssm=bi_ssm,  # if i_layer != 0 else False,
                is_divide_out=is_divide_out,
                use_checkpoint=self.use_checkpoint,
                use_reentrant=self.use_reentrant,
                out_norm=out_norm
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
                # torch.nn.init.xavier_uniform(m.weight)
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)

        #self.apply(_init_weights)

        self.point_embed_glofas.apply(_init_weights)
        self.point_embed_era5.apply(_init_weights)
        self.point_embed_cpc.apply(_init_weights)
        self.norm_input_embedding.apply(_init_weights)
        trunc_normal_(self.time_pos_embed, std=.02)

        for l in range(len(self.layers)):
            layer = self.layers[l]

            if layer.point_embed is not None:
                layer.point_embed.apply(_init_weights)

            for block in range(self.depths[l]):

                layer.blocks[block].apply(
                                        partial(
                                            _init_weights_m,
                                            n_layer=sum(self.depths),
                                            #**(initializer_cfg if initializer_cfg is not None else {}),
                                        )
                                    )
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

    model = Mamba().cuda()
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


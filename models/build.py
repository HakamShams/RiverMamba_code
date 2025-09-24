# ------------------------------------------------------------
"""
This script includes the main class to import and build the RiverMamba

Contact Person: Mohamad Hakam Shams Eddin <shams@iai.uni-bonn.de>
Computer Vision Group - Institute of Computer Science III - University of Bonn
"""
# ------------------------------------------------------------

import torch
import torch.nn as nn
import importlib

try:
    from timm.layers import trunc_normal_
except:
    from timm.models.layers import trunc_normal_

from models.loss import MSE, L1

# ------------------------------------------------------------


def import_class(m, name):
    module = importlib.import_module("models." + m + '.' + name)
    return getattr(module, name)


class Model(nn.Module):
    """ Main RiverMamba model including the encoder, decoder and regression heads """
    def __init__(self, config):
        super(Model, self).__init__()

        """
        Parameters
        ----------
        config : argparse
            configuration file from config.py
        """

        if config.encoder == 'FlashTransformer':
            self.encoder = import_class('encoder', config.encoder)(
                in_glofas=config.in_glofas,
                in_era5=config.in_era5,
                in_static=config.in_static,
                n_points=config.n_points,
                n_curves=len(config.curves),
                curve_order=config.en_curve_order,
                embed_glofas=config.en_embed_glofas,
                embed_era5=config.en_embed_era5,
                embed_cpc=config.en_embed_cpc,
                embed_norm=config.en_embed_norm,
                embed_dim=config.en_embed_dim,
                n_heads=config.en_n_heads,
                depths=config.en_depths,
                grouping_size=config.en_grouping_size,
                mlp_ratio=config.en_mlp_ratio,
                drop_rate=config.en_drop_rate,
                attn_drop_rate=config.en_attn_drop_rate,
                drop_path_rate=config.en_drop_path_rate,
                qkv_bias=config.en_qkv_bias,
                use_checkpoint=config.en_use_checkpoint,
                use_reentrant=config.en_use_reentrant,
                proj_drop=config.en_proj_drop,
                qk_scale=config.en_qk_scale,
                is_causal=config.en_is_causal,
                is_alibi_slopes=config.en_is_alibi_slopes,
                att_window_size=config.en_att_window_size,
            )
        elif config.encoder == 'Mamba':
            self.encoder = import_class('encoder', config.encoder)(
                in_glofas=config.in_glofas,
                in_era5=config.in_era5,
                in_static=config.in_static,
                n_curves=len(config.curves),
                curve_order=config.en_curve_order,
                bi_ssm=config.en_bi_ssm,
                embed_glofas=config.en_embed_glofas,
                embed_era5=config.en_embed_era5,
                embed_cpc=config.en_embed_cpc,
                embed_norm=config.en_embed_norm,
                embed_dim=config.en_embed_dim,
                depths=config.en_depths,
                grouping_size=config.en_grouping_size,
                mlp_ratio=config.en_mlp_ratio,
                drop_rate=config.en_drop_rate,
                drop_path_rate=config.en_drop_path_rate,
                d_state=config.en_d_state,
                expand=config.en_expand,
                d_conv=config.en_d_conv,
                dt_min=config.en_dt_min,
                dt_max=config.en_dt_max,
                dt_rank=config.en_dt_rank,
                dt_init=config.en_dt_init,
                dt_scale=config.en_dt_scale,
                dt_init_floor=config.en_dt_init_floor,
                conv_bias=config.en_conv_bias,
                proj_bias=config.en_proj_bias,
                is_divide_out=config.en_is_divide_out,
                use_checkpoint=config.en_use_checkpoint,
                use_reentrant=config.en_use_reentrant,
                out_norm=config.en_is_out_norm
            )
        elif config.encoder == 'Mamba2':
            self.encoder = import_class('encoder', config.encoder)(
                in_glofas=config.in_glofas,
                in_era5=config.in_era5,
                in_static=config.in_static,
                n_curves=len(config.curves),
                curve_order=config.en_curve_order,
                bi_ssm=config.en_bi_ssm,
                embed_glofas=config.en_embed_glofas,
                embed_era5=config.en_embed_era5,
                embed_cpc=config.en_embed_cpc,
                embed_norm=config.en_embed_norm,
                embed_dim=config.en_embed_dim,
                depths=config.en_depths,
                grouping_size=config.en_grouping_size,
                mlp_ratio=config.en_mlp_ratio,
                drop_rate=config.en_drop_rate,
                drop_path_rate=config.en_drop_path_rate,
                d_state=config.en_d_state,
                expand=config.en_expand,
                d_conv=config.en_d_conv,
                dt_min=config.en_dt_min,
                dt_max=config.en_dt_max,
                dt_rank=config.en_dt_rank,
                dt_init=config.en_dt_init,
                dt_scale=config.en_dt_scale,
                dt_init_floor=config.en_dt_init_floor,
                conv_bias=config.en_conv_bias,
                proj_bias=config.en_proj_bias,
                is_divide_out=config.en_is_divide_out,
                use_checkpoint=config.en_use_checkpoint,
                use_reentrant=config.en_use_reentrant,
                ngroups=config.en_ngroups,
                headdim=config.en_headdim,
                chunk_size=config.en_chunk_size,
                )
        else:
            raise NotImplementedError(f"Encoder {config.encoder} not implemented")

        if config.decoder == 'FlashTransformer':
            self.decoder = import_class('decoder', config.decoder)(
                in_features=config.en_embed_dim[-1],
                in_hres=config.in_hres,
                in_static=config.in_static,
                n_curves=len(config.curves),
                curve_order=config.de_curve_order,
                embed_hres=config.de_embed_hres,
                embed_norm=config.de_embed_norm,
                embed_dim=config.de_embed_dim,
                n_heads=config.de_n_heads,
                depths=config.de_depths,
                grouping_size=config.de_grouping_size,
                mlp_ratio=config.de_mlp_ratio,
                drop_rate=config.de_drop_rate,
                attn_drop_rate=config.de_attn_drop_rate,
                drop_path_rate=config.de_drop_path_rate,
                qkv_bias=config.en_qkv_bias,
                proj_drop=config.de_proj_drop,
                qk_scale=config.de_qk_scale,
                is_causal=config.de_is_causal,
                is_alibi_slopes=config.de_is_alibi_slopes,
                att_window_size=config.de_att_window_size,
                use_checkpoint=config.de_use_checkpoint,
                use_reentrant=config.de_use_reentrant,
            )
        elif config.decoder == 'Mamba':
            self.decoder = import_class('decoder', config.decoder)(
                in_features=config.en_embed_dim[-1],
                in_hres=config.in_hres,
                in_static=config.in_static,
                n_curves=len(config.curves),
                curve_order=config.de_curve_order,
                bi_ssm=config.de_bi_ssm,
                embed_hres=config.de_embed_hres,
                embed_dim=config.de_embed_dim,
                depths=config.de_depths,
                grouping_size=config.de_grouping_size,
                mlp_ratio=config.de_mlp_ratio,
                drop_rate=config.de_drop_rate,
                drop_path_rate=config.de_drop_path_rate,
                d_state=config.de_d_state,
                expand=config.de_expand,
                d_conv=config.de_d_conv,
                dt_min=config.de_dt_min,
                dt_max=config.de_dt_max,
                dt_rank=config.de_dt_rank,
                dt_init=config.de_dt_init,
                dt_scale=config.de_dt_scale,
                dt_init_floor=config.de_dt_init_floor,
                conv_bias=config.de_conv_bias,
                proj_bias=config.de_proj_bias,
                is_divide_out=config.de_is_divide_out,
                use_checkpoint=config.de_use_checkpoint,
                use_reentrant=config.de_use_reentrant,
                out_norm=config.de_is_out_norm
            )
        elif config.decoder == 'Mamba2':
            self.decoder = import_class('decoder', config.decoder)(
                in_features=config.en_embed_dim[-1],
                in_hres=config.in_hres,
                in_static=config.in_static,
                n_curves=len(config.curves),
                curve_order=config.de_curve_order,
                bi_ssm=config.de_bi_ssm,
                embed_hres=config.de_embed_hres,
                embed_dim=config.de_embed_dim,
                depths=config.de_depths,
                grouping_size=config.de_grouping_size,
                mlp_ratio=config.de_mlp_ratio,
                drop_rate=config.de_drop_rate,
                drop_path_rate=config.de_drop_path_rate,
                d_state=config.de_d_state,
                expand=config.de_expand,
                d_conv=config.de_d_conv,
                dt_min=config.de_dt_min,
                dt_max=config.de_dt_max,
                dt_rank=config.de_dt_rank,
                dt_init=config.de_dt_init,
                dt_scale=config.de_dt_scale,
                dt_init_floor=config.de_dt_init_floor,
                conv_bias=config.de_conv_bias,
                proj_bias=config.de_proj_bias,
                is_divide_out=config.de_is_divide_out,
                use_checkpoint=config.de_use_checkpoint,
                use_reentrant=config.de_use_reentrant,
                ngroups=config.de_ngroups,
                headdim=config.de_headdim,
                chunk_size=config.de_chunk_size
            )
        else:
            raise NotImplementedError(f"Decoder {config.decoder} not implemented")

        # define the regression heads with lead time delta_t_f
        self.delta_t_f = config.delta_t_f
        self.heads = nn.ModuleList()

        for t in range(self.delta_t_f):
            if config.head == 'MLP':
                index_list = self.decoder.embed_dim.copy()
                del index_list[t]
                head = import_class('head', config.head)(embed_dim=self.decoder.embed_dim[t],
                                                         embed_dim_all=sum(index_list),
                                                         hidden_dim=config.head_hidden_dim,
                                                         out_dim=config.head_out_dim,
                                                         drop_rate=config.head_drop_rate,
                                                         use_checkpoint=config.head_use_checkpoint,
                                                         use_reentrant=config.head_use_reentrant
                                                         )
            else:
                raise NotImplementedError(f"Head {config.head} not implemented")

            self.heads.append(head)

        # define the loss
        #self.loss = import_class('loss', config.loss)()
        self.loss = MSE()

        self.pretrained_model = config.pretrained_model
        self.init_weights()

        self.index_list = [j for j in range(self.delta_t_f)]

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
                trunc_normal_(m.weight, std=.01)
                # torch.nn.init.xavier_uniform(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv2d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)
                if isinstance(m, nn.Conv3d) and m.bias is not None:
                    nn.init.constant_(m.bias, 0.)

            elif isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                nn.init.constant_(m.bias, 0.)
                nn.init.constant_(m.weight, 1.0)

        # self.apply(_init_weights)

        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights()

        if self.pretrained_model:
            print('initialize weights from pretrained model {} ...'.format(self.pretrained_model))
            # loc = f"cuda:{self.device}"
            checkpoint = torch.load(self.pretrained_model)   #, weights_only=True)
            state_dict = checkpoint['model']
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.load_state_dict(state_dict, strict=True)
            del state_dict, checkpoint
            torch.cuda.empty_cache()

    def forward(self, x_hres, x_glofas, x_era5, x_cpc, x_static, x_curves,
                y_glofas=None, y_obs=None, y_weight=None):
        """
            Forward function
            Args:
                x_hres (torch.tensor): input dynamic variables [B, lead time, P, V]
                x_glofas (torch.tensor): input dynamic variables [B, T, P, V]
                x_era5 (torch.tensor): input dynamic variables [B, T, P, V]
                x_cpc (torch.tensor): input dynamic variables [B, T, P, 1]
                x_static (torch.tensor): input static variables [B, P, V]
                x_curves (torch.tensor): input curves indices  [B, Number of Curves, 2, P]
                y_glofas (torch.tensor): output variables [B, lead time, P, out_dim]
                y_obs (torch.tensor): output observed variables [B, lead time, P, out_dim]
                y_weight (torch.tensor): weight for the loss [B, lead time, P, out_dim]
            Returns:
                x (torch.tensor): output variable [B, lead time, P, out_dim]
                loss (torch.tensor): loss if y_glofas or y_obs are not None
        """

        # Encoder
        x_features = self.encoder(x_glofas, x_era5, x_cpc, x_static, x_curves)
        # x_features (torch.tensor): output dynamic features [B, 1, P, K_embedding]

        # Decoder
        x_heads = self.decoder(x_features, x_hres, x_static, x_curves)
        #x_heads (torch.tensor): output features [B, 1, P, K] x lead time

        # Regression heads
        x = []
        for t in range(self.delta_t_f):
            index_list = self.index_list.copy()
            index_list.remove(t)
            x.append(self.heads[t](x_heads[t], [x_heads[i] for i in index_list]))

        x = torch.cat(x, dim=1)
        # x (torch.tensor): output variable [B, lead time, P, out_dim]

        # compute loss w.r.t. GloFAS reanalysis
        if y_glofas is not None:
            loss = self.loss(x, y_glofas, y_weight)
            return x, loss

        # compute loss w.r.t. observational river discharge
        elif y_obs is not None:
            mask = ~torch.isnan(y_obs)
            loss = self.loss(x[mask], y_obs[mask], y_weight[mask])
            return x, loss

        return x


if __name__ == '__main__':

    import config
    import os

    config_file = config.read_arguments(train=True, print=False, save=False)

    if config_file.gpu_id != -1:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(config_file.gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'

    """
       data_hres      shape:  (B, lead time, P, V)
       data_glofas    shape:  (B, T, P, V)
       data_era5_land shape:  (B, T, P, V)
       data_cpc       shape:  (B, T, P, 1)
       data_static    shape:  (B, P, V)
       curves         shape:  (B, Number of Curves, 2, P)
       y_glofas       shape:  (B, lead time, P, out_dim)
       y_obs          shape:  (B, lead time, P, out_dim)
       y_weight       shape:  (B, lead time, P, out_dim)
       
    """

    model = Model(config_file).cuda()
    print(model)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of parameters: {n_parameters}")

    # Creates model and optimizer in default precision
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    # Creates a GradScaler once at the beginning of training.
    from torch.cuda.amp import GradScaler
    scaler = GradScaler()

    from torch import autocast
    import time

    print('test training...')

    for epoch in range(10):
        optimizer.zero_grad()

        data_glofas = torch.randn((1, 4, 82804, 4)).cuda()
        data_hres = torch.randn((1, 7, 82804, 7)).cuda()
        data_era5_land = torch.randn((1, 4, 82804, 35 - 3)).cuda()
        data_cpc = torch.randn((1, 4, 82804, 1)).cuda()
        data_static = torch.randn((1, 82804, 96 + 3)).cuda()
        curves = torch.arange(0, 82804)
        curves = curves[None, None, None, :].repeat(1, 6, 2, 1).cuda()
        y_glofas = torch.randn((1, 7, 82804, 1)).cuda()
        y_obs = torch.randn((1, 7, 82804, 1)).cuda()
        y_weight = torch.randn((1, 7, 82804, 1)).cuda()

        # Runs the forward pass with autocasting
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output, loss = model(data_hres, data_glofas, data_era5_land, data_cpc, data_static, curves, y_glofas, y_obs, y_weight)

        print(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        scaler.step(optimizer)
        scaler.update()

    print('inference...')

    model.eval()
    with torch.no_grad():
        for epoch in range(10):
            time_start = time.time()
            # Runs the forward pass with autocasting
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                output = model(data_hres, data_glofas, data_era5_land, data_cpc, data_static, curves)
            print(time.time() - time_start)


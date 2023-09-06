## Adapted from: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch

from functools import partial
from typing import Sequence, Tuple

import torch
import torch.nn as nn
from einops import rearrange

from .modules.blocks import (
    Downsample,
    RandomOrLearnedSinusoidalPosEmb,
    Residual,
    ResnetBlock,
    RMSNorm,
    SinusoidalPosEmb,
    Upsample,
)
from .modules.transformer import Attention1D, Attention2D, LinearAttention
from .modules.utils import cast_tuple, default, divisible_by


class FancyUnet2D(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=(False, False, False, True),
        flash_attn=False,
    ):
        """UNet with optional self-attention and sinusoidal positional embeddings

        Args:
            dim (int): channel dim for Unet in features
        """
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features
            )
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # attention

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)

        assert len(full_attn) == len(dim_mults)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(zip(in_out, full_attn, attn_heads, attn_dim_head)):
            is_last = ind >= (num_resolutions - 1)

            attn_klass = Attention2D if layer_full_attn else LinearAttention

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        attn_klass(
                            dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads
                        ),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Attention2D(
            mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1]
        )
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (
            (dim_in, dim_out),
            layer_full_attn,
            layer_attn_heads,
            layer_attn_dim_head,
        ) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)

            attn_klass = Attention2D if layer_full_attn else LinearAttention

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        attn_klass(
                            dim_out,
                            dim_head=layer_attn_dim_head,
                            heads=layer_attn_heads,
                        ),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all(
            [divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]
        ), f"your input dimensions {x.shape[-2:]} need to be divisible by {self.downsample_factor}, given the unet"

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x) + x
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)


class SimpleConditionalUnet2D(nn.Module):
    def __init__(
        self,
        dim: int,
        init_dim: int = None,
        out_channels: int = None,
        block_channels: Sequence = (16, 32, 64, 128, 64, 32, 16),
        channels: int = 1,
        input_conditioning_dims: int = None,
        is_self_conditioned: bool = False,
        resnet_block_groups: int = 8,
        learned_variance: bool = False,
        dropout=None,
        is_time_conditioned: bool = True,
        learned_sinusoidal_cond: bool = False,
        random_fourier_features: bool = False,
        learned_sinusoidal_dim: int = 16,
    ) -> None:
        """Unet1D
            Notation:
                Tensor Shape: [..., B, C, D]
                    B: Batch size
                    C: Channels
                    D: Dims
        Args:
            dim (int): input dims (D)
            init_dim (int, optional): init dim TODO. Defaults to None.
            out_channels (int, optional): output dim (D) . Defaults to None.
            dim_mults (Sequence, optional): Dimension multiplier per Residual block,
                Length of the sequence is the number of Residual Blocks. Defaults to (1, 2, 4, 8).

            channels (int, optional): input channels (C) [..., C, D]. Defaults to 3.
            input_conditioning_dims (int, optional): conditioning latent dims (D), If conditioning with an input, . Defaults to None.
            is_self_conditioned (bool, optional): enable self conditioning. Defaults to False.
                From: Generating discrete data using Diffusion Models with self-conditioning
                    https://arxiv.org/abs/2208.04202

            is_time_conditioned (bool, optional): Defaults to True.
                        True, if conditioned with time, when using Unet as a denoiser net in DDMs .
                        False if no time conditioning, i.e. normal Unet without Diffusion.

            resnet_block_groups (int, optional): Groups is residual blocks. Defaults to 8.
            learned_variance (bool, optional): Learned Variance. Defaults to False.
            learned_sinusoidal_cond (bool, optional): Learned Sinusoidal embeddings. Defaults to False.
            random_fourier_features (bool, optional): Random fourier projection. Defaults to False.
            learned_sinusoidal_dim (int, optional): Learned sinusoidal embedding dims. Defaults to 16.
        """
        super().__init__()

        # determine dimensions

        self.channels = channels

        self.is_self_conditioned = is_self_conditioned
        input_channels = channels * (2 if is_self_conditioned else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = (dim,) + block_channels
        in_out = list(zip(dims[:-1], dims[1:]))

        self.in_features = dim
        self.out_features = dim

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        self.dropout = (
            nn.Dropout(p=dropout, inplace=True) if dropout is not None else None
        )

        # Time and Input embedding

        emb_dim = dim * 4
        self.emb_dim = emb_dim

        self.random_or_learned_sinusoidal_cond = (
            learned_sinusoidal_cond or random_fourier_features
        )
        if is_time_conditioned:
            self.is_time_conditioned = True
            if self.random_or_learned_sinusoidal_cond:
                sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                    learned_sinusoidal_dim, random_fourier_features
                )
                fourier_dim = learned_sinusoidal_dim + 1
            else:
                sinu_pos_emb = SinusoidalPosEmb(dim)
                fourier_dim = dim

            self.time_mlp = nn.Sequential(
                sinu_pos_emb,
                nn.Linear(fourier_dim, emb_dim),
                nn.GELU(),
                nn.Linear(emb_dim, emb_dim),
            )
        else:
            self.is_time_conditioned = False
            self.time_mlp = None

        if input_conditioning_dims is not None:
            self.is_input_conditioned = True

            # TODO: Add linear layer at the end
            self.input_emb_layers = nn.Sequential(
                nn.Conv2d(input_conditioning_dims, emb_dim), nn.SiLU()
            )
        else:
            self.is_input_conditioned = False
            self.input_emb_layers = None

        # ResBlock layers
        self.blocks = nn.ModuleList([])

        for _, (dim_in, dim_out) in enumerate(in_out):
            module_list = nn.ModuleList(
                [
                    block_klass(dim_in, dim_in, time_emb_dim=emb_dim),
                    block_klass(dim_in, dim_in, time_emb_dim=emb_dim),
                    Residual(LinearAttention(dim_in)),
                    nn.Conv2d(dim_in, dim_out, 3, padding=1),
                ]
            )
            self.blocks.append(module_list)

        default_out_channels = channels * (1 if not learned_variance else 2)
        self.out_channels = default(out_channels, default_out_channels)

        self.final_res_block = block_klass(dims[-1], dims[-1], time_emb_dim=emb_dim)
        self.final_conv = nn.Conv2d(dims[-1], self.out_channels, 1)

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor = None,
        z_cond: torch.Tensor = None,
        x_self_cond: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): input
            time (torch.Tensor): timestep for diffusion
                Note: Set to None, when using the architecture outside diffusion.
                    i.e. self.is_time_conditioned = False
            z_cond (torch.Tensor, optional): conditioning latent. Defaults to None.
            x_self_cond (torch.Tensor, optional): self conditioning vector. Defaults to None.

        Returns:
            torch.Tensor: output
        """
        if self.is_self_conditioned:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        # r = x.clone()

        # Time embedding for diffusion, None for non-diffusion
        if self.is_time_conditioned and self.time_mlp is not None:
            assert time is not None
            latent_emb = self.time_mlp(time)
        else:
            latent_emb = None

        # Add input embedding if inupt conditioned
        if self.is_input_conditioned:
            input_emb = self.input_emb_layers(z_cond)
            if input_emb.ndim != 2 and input_emb.ndim == 3:
                latent_emb = latent_emb.unsqueeze(-2).repeat([1, input_emb.shape[1], 1])
            elif input_emb.ndim == 2:
                pass
            else:
                raise NotImplementedError
            latent_emb = latent_emb + input_emb if latent_emb is not None else input_emb

        for block1, block2, attn, updownsample in self.blocks:
            x = block1(x, latent_emb)

            x = block2(x, latent_emb)
            x = attn(x)

            x = updownsample(x)
            if self.dropout:
                x = self.dropout(x)

        x = self.final_res_block(x, latent_emb)
        return self.final_conv(x)

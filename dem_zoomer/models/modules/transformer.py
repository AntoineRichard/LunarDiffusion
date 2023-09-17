## Adapted from: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch

import torch
from einops import rearrange
from torch import einsum, nn

from .blocks import FeedForward, RMSNorm
from .utils import l2norm


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim=1)
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1), RMSNorm(dim, normalize_dim=1)
        )

    def forward(self, x):
        residual = x

        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale

        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)

        return self.to_out(out) + residual


class Attention1D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, scale=8, dropout=0.0):
        super().__init__()
        self.scale = scale
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim=2)

        self.attn_dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, hidden_dim * 3, bias=False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        q, k = map(l2norm, (q, k))

        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * self.scale

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Attention2D(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, dropout=0.0):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads

        self.norm = RMSNorm(dim, normalize_dim=1)
        self.attn_dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape

        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h (x y) c", h=self.heads), qkv
        )

        scale = q.shape[-1] ** -0.5

        # similarity

        sim = einsum(f"b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum(f"b h i j, b h j d -> b h i d", attn, v)

        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        time_cond_dim,
        depth,
        dim_head=32,
        heads=4,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention1D(
                            dim=dim, dim_head=dim_head, heads=heads, dropout=dropout
                        ),
                        FeedForward(
                            dim=dim,
                            mult=ff_mult,
                            cond_dim=time_cond_dim,
                            dropout=dropout,
                        ),
                    ]
                )
            )

    def forward(self, x, t):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x, t) + x

        return x


# model

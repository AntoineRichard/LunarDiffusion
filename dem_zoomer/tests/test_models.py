import os
import sys

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from dem_zoomer.models.modules.blocks import ResnetBlock
from dem_zoomer.models.unet2d import FancyUnet2D, SimpleConditionalUnet2D
from dem_zoomer.models.uvit import UViT

# TODO: UViT does not work
# TODO: Fancy UNet2D and UViT do not have FiLM conditioning. Check if self cond concat op is good enough


def timed(fn, *args, **kwargs):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn(*args, **kwargs)
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def test_simple_unet(device="cuda:0", check_scripting=False, is_timed=True):
    model = SimpleConditionalUnet2D(
        dim=16,
    )
    model = model.to(device=device)

    # Dummy input
    x = torch.randn(1, 1, 256, 256).to(device=device)

    # Random time step
    time = torch.randint(0, 1000, (1,)).long().to(device=device)

    # Random conditioning input
    z_cond = torch.randn(1, 1, 256, 256).to(device=device)

    if is_timed:
        # Time the forward pass
        output, time_forward = timed(model, x, time=time, z_cond=z_cond)

        print(
            f"{model.__class__.__name__} Model ({count_parameters(model)}M) : Forward pass took {time_forward} seconds on {device}"
        )
    else:
        output = model(x, time=time, x_self_cond=z_cond)

    # Just for fun, check that scripting works
    if check_scripting:
        from einops._torch_specific import allow_ops_in_compiled_graph

        allow_ops_in_compiled_graph()
        traced_model = torch.jit.trace(
            model, example_kwarg_inputs={"x": x, "time": time, "z_cond": z_cond}
        )

    assert output.shape == (1, 1, 256, 256)


def test_fancy_unet(device="cuda:0", is_timed=True, use_compile=True):
    # Test Fany2D
    model = FancyUnet2D(
        dim=16,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
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
    )
    model = model.to(device=device)

    # Dummy input
    x = torch.randn(1, 1, 256, 256).to(device=device)

    # Random time step
    time = torch.randint(0, 1000, (1,)).long().to(device=device)

    # Random conditioning input
    z_cond = torch.randn(1, 1, 256, 256).to(device=device)

    # Time the forward pass
    if is_timed:
        output, time_forward = timed(model, x, time=time, x_self_cond=z_cond)

        print(
            f"{model.__class__.__name__} Model ({count_parameters(model)}M) : Forward pass took {time_forward} seconds on {device}"
        )
    else:
        output = model(x, time=time, x_self_cond=z_cond)

    assert output.shape == (1, 1, 256, 256)

    return


def test_uvit(device="cuda:0"):
    # Test 2D
    test = UViT(
        dim=16,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        downsample_factor=2,
        channels=3,
        vit_depth=6,
        vit_dropout=0.2,
        attn_dim_head=32,
        attn_heads=4,
        ff_mult=4,
        resnet_block_groups=8,
        learned_sinusoidal_dim=16,
        init_img_transform=None,
        final_img_itransform=None,
        patch_size=1,
        dual_patchnorm=False,
    )
    x = torch.randn(1, 1, 256, 256)
    time = torch.randn(1, 1)
    x_self_cond = torch.randn(1, 1, 256, 256)
    output = test(x, time, x_self_cond)
    assert output.shape == (1, 1, 256, 256)

    return


def test_diffusion_model(denoiser_model):
    pass


if __name__ == "__main__":
    # Get cuda runtime running

    test_fancy_unet(device="cuda:0", is_timed=False)

    # clear cache
    torch.cuda.empty_cache()

    device = "cuda:0"

    test_simple_unet(device=device, check_scripting=False)
    test_fancy_unet(device=device)

    device = "cpu"
    test_simple_unet(device=device)
    test_fancy_unet(device=device)

import os
import sys

import numpy as np
import torch
import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Experimental: Einops fix for torch.compile graph
from einops._torch_specific import allow_ops_in_compiled_graph

from dem_zoomer.models import MODEL_REGISTRY

allow_ops_in_compiled_graph()


# TODO: UViT does not work
# TODO: Fancy UNet2D and UViT do not have FiLM conditioning. Check if self cond concat op is good enough


def timed(fn, *args, **kwargs):
    times = []
    for _ in tqdm.tqdm(range(32)):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()

        times.append(start.elapsed_time(end) / 1000)

    return result, (np.median(times[1:]), np.std(times[1:]))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6


def get_denoiser_model(model_type, device="cuda:0", is_compiled=False):
    assert isinstance(model_type, MODEL_REGISTRY), "What are you doing"
    if model_type == MODEL_REGISTRY.SIMPLE_UNET:
        kwargs = dict(dim=16)
    elif model_type == MODEL_REGISTRY.FANCY_UNET:
        kwargs = dict(
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
    elif model_type == MODEL_REGISTRY.VIT:
        kwargs = dict(
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

    else:
        raise NotImplementedError

    model = model_type.build(**kwargs)
    model = model.to(device).eval()
    model = torch.compile(model) if is_compiled else model
    return model


def get_diffuser_model(model_type, device="cuda:0", is_compiled=False):
    if model_type == MODEL_REGISTRY.DEM_DIFFUSER:
        denoiser_model = get_denoiser_model(
            MODEL_REGISTRY.SIMPLE_UNET, device=device, is_compiled=is_compiled
        )
        kwargs = dict(
            model=denoiser_model,
            in_channels=1,
            in_dims=(256, 256),
            diffusion_timesteps=1000,
            diffusion_loss="l2",
            beta_schedule="linear",
            noise_scheduler_type="ddim",
            is_conditioned=True,
            variance_type="fixed_small",
            beta_start=5e-5,
            beta_end=5e-2,
        )
    else:
        raise NotImplementedError

    model = model_type.build(**kwargs)
    model = model.to(device).eval()
    model = torch.compile(model) if is_compiled else model
    return model


def get_dummy_inputs(batch_size, use_cond=True, device="cuda:0"):
    # Dummy input
    x = torch.randn(batch_size, 1, 256, 256).to(device=device)

    # Random time step
    time = torch.randint(0, 1000, (batch_size,)).long().to(device=device)

    # Random conditioning input
    z_cond = (
        torch.randn(batch_size, 1, 256, 256).to(device=device) if use_cond else None
    )

    return x, time, z_cond


@torch.no_grad()
def test_denoiser(
    model_type,
    batch_size=16,
    device="cuda:0",
    is_compiled=True,
    is_timed=True,
):
    # Get model
    model = get_denoiser_model(model_type, device=device, is_compiled=is_compiled)

    # Get dummy inputs
    x, time, z_cond = get_dummy_inputs(batch_size)

    if is_timed:
        # Time the forward pass
        output, fw_time = timed(model, x, time=time, z_cond=z_cond)

        # Out
        test_string = (
            f"\n{model.__class__.__name__} Model ({count_parameters(model):.2f}M)"
        )
        test_string += "[Compiled]" if is_compiled else ""
        test_string += f": Forward pass took {fw_time[0]:.2f}+/- {fw_time[1]:.2f} seconds on {device}"

        print(test_string)
    else:
        # Run the forward pass
        output = model(x, time=time, x_self_cond=z_cond)

    # Check output shape
    assert output.shape == (batch_size, 1, 256, 256)

    del model, x, time, z_cond, output
    return


def test_diffuser(
    model_type,
    batch_size=1,
    device="cuda:0",
    is_compiled=True,
    is_timed=True,
):
    # Get model
    dem_diffuser = get_diffuser_model(
        model_type=model_type, device=device, is_compiled=is_compiled
    )

    if dem_diffuser.diffusion_model._noise_scheduler_type == "ddim":
        dem_diffuser.diffusion_model.set_inference_timesteps(100)

    # Get dummy inputs
    _, _, z_cond = get_dummy_inputs(batch_size)

    if is_timed:
        import time

        t0 = time.time()
        dem_diffuser.generate_samples(num_samples=batch_size, z_cond=z_cond)

        print(
            f"\nReverse diffusion sampling with {type(dem_diffuser.diffusion_model.noise_scheduler)}"
            " for (batch num samples={batch_size}) took {time.time() - t0:.2f} seconds"
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    ## GPU runs
    device = "cuda:0"

    print("\n Testing diffusion model reverse sampling \n")
    test_diffuser(
        model_type=MODEL_REGISTRY.DEM_DIFFUSER,
        device="cuda:0",
        is_timed=True,
        is_compiled=False,
    )

    # Time the models
    print("\n Testing denoiser model forward pass \n")
    test_denoiser(
        model_type=MODEL_REGISTRY.SIMPLE_UNET,
        device="cuda:0",
        is_timed=True,
        is_compiled=False,
    )

    print("\n Testing denoiser model forward pass with torch.compile() \n")
    test_denoiser(
        model_type=MODEL_REGISTRY.SIMPLE_UNET,
        device="cuda:0",
        is_timed=True,
        is_compiled=True,
    )

    ## CPU runs
    # device = "cpu"
    # test_denoiser(
    #     model_type=MODEL_REGISTRY.SIMPLE_UNET,
    #     device="cpu",
    #     is_timed=True,
    #     is_compiled=False,
    # )

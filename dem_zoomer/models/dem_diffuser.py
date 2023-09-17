import torch
from torch import nn

from .diffusion.gaussian_diffusion import GaussianDiffusion2D


class DEMDiffuser(nn.Module):
    def __init__(
        self,
        model,
        in_channels,
        in_dims,
        diffusion_timesteps,
        diffusion_loss,
        beta_schedule="linear",
        noise_scheduler_type: str = "ddpm",
        is_conditioned=True,
        denoising_loss_weight=1,
        variance_type="fixed_small",
        beta_start=5e-5,
        beta_end=5e-2,
    ) -> None:
        """DEM Diffusion Model

        Args:
            model (nn.Module): denoiser model
                denoiser model should have forward argument structure:
                    ```
                        def forward (self, x, *, t=t, z_cond=z_cond):
                    ```
                    x: Input tensor [B,C,H,W]
                    t: Batched timestep tensor (long) [B,1]
                    z_cond: conditioning [B, ...]

            latent_in_features (int): input data dims (D)
            diffusion_timesteps (int): Number of diffusion timesteps
            diffusion_loss (str): Diffusion loss type
            beta_schedule (str, optional): beta noise schedule. Defaults to "linear".
                    Valid: [linear, scaled_linear, or squaredcos_cap_v2]
            noise_scheduler_type (str, optional): Noise scheduler  type.
                    Valid:  ["ddpm", "ddim"]
            is_conditioned (bool, optional): Whether the diffusion model is conditioned. Defaults to True.
            variance_type (str, optional): Type of variance used to add noise. Defaults to "fixed_small".
                    Valid:  [fixed_small, fixed_small_log, fixed_large, fixed_large_log, learned or learned_range
            beta_start (float, optional): Starting beta value. Defaults to 5e-5.
            beta_end (float, optional): Ending beta value. Defaults to 5e-2.
        """
        super().__init__()
        self.diffusion_model = GaussianDiffusion2D(
            model=model,
            in_channels=in_channels,
            in_dims=in_dims,
            num_steps=diffusion_timesteps,
            loss_type=diffusion_loss,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            noise_scheduler_type=noise_scheduler_type,
            variance_type=variance_type,
        )

        self.is_conditioned = is_conditioned

        self.loss_weight = denoising_loss_weight

    @property
    def scheduler_type(self):
        """Get Diffusion noise scheduler type"""
        return self.diffusion_model._noise_scheduler_type

    def set_inference_timesteps(self, num_inference_steps):
        """Set the number of inference steps for reverse diffusion sampler

        Args:
            num_inference_steps (int): Number of inference steps
        """
        self.diffusion_model.set_inference_timesteps(num_inference_steps)
        return

    def forward(self, x, z_cond):
        """Training Forward Pass: Computes loss for batched pc and grasps

        Args:
            x (torch.Tensor): Input image tensor [B,C,H,W]
            z_cond (torch.Tensor): Conditioning tensor [B, Cn, H, W]

        Returns:
            dict: Dictionary of losses

        """
        denoising_loss = self.diffusion_model(
            x, z_cond=z_cond, is_conditioned=self.is_conditioned
        )
        loss_dict = {"denoising_loss": denoising_loss}

        return loss_dict

    @torch.no_grad()
    def generate_samples(
        self,
        *,
        z_cond=None,
        num_samples=4,
        return_intermediate=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    ):
        """Generation/Sampling : Generates samples

        Args:
            xyz (torch.Tensor): Input point cloud of shape (batch_size, num_points, 3)
            num_grasps (int, optional): Number of grasps to generate per point cloud. Defaults to 10.
            return_intermediate (bool, optional): Whether to return intermediate outputs. Defaults to False.

        Returns:
            torch.Tensor: Generated grasps of shape (batch_size*num_grasps, 6/7)
            torch.Tensor: Intermediate outputs of shape (batch_size, num_grasps, num_steps, latent_dim)
                            or empty list [] if return_intermediate is False
        """

        # TODO: High batch size will cause OOM
        out, all_outs = self.diffusion_model.sample(
            z_cond=z_cond,
            batch_size=num_samples,
            return_all=return_intermediate,
            device=z_cond.device if z_cond is not None else device,
        )

        if not return_intermediate:
            return (out, [])

        return out, all_outs

    def print_params_info(self):
        """Prints model parameters information"""

        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        non_trainable_params = sum(
            p.numel() for p in self.parameters() if not p.requires_grad
        )

        print("------------------------------------------------")
        print("Model Trainable Parameters: ", trainable_params)
        print("Model Non-Trainable Parameters: ", non_trainable_params)
        print("------------------------------------------------")

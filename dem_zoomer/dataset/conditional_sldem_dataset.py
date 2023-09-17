import cv2
import einops
import torch
from torchvision import transforms

from .base_dataset import SLDEMDataset


class CenterResolveSLDEMDatset(SLDEMDataset):
    def __init__(
        self,
        data_root,
        data_split="train",
        out_shape=...,
        prefix="MoonORTO2DEM",
        num_repeat_dataset=2,
        is_debug=False,
        norm_config=dict(
            mean=0.5,
            scale=0.5,
        ),
    ) -> None:
        super().__init__(
            data_root, data_split, out_shape, prefix, num_repeat_dataset, is_debug
        )
        assert (
            "mean" in norm_config and "scale" in norm_config
        ), "Norm config must contain mean and scale as floats"

        self.input_norm_mean = norm_config["mean"]
        self.input_norm_scale = norm_config["scale"]

        # TODO: Temporary. Unify into a transforms pipeline specified in config
        self.resize = transforms.Resize(
            (self.out_height, self.out_width), antialias=False
        )
        self.center_crop = transforms.CenterCrop((self.out_height, self.out_width))
        self.normalize = transforms.Normalize(
            mean=[self.input_norm_mean], std=[self.input_norm_scale]
        )

    def __getitem__(self, index: int) -> dict:
        normalized_dem = super().__getitem__(index)

        # Visualize, if debug
        if self.is_debug:
            self.visualize(normalized_dem)

        # Prepare tensor: resize and adjust batch/channel dim
        normalized_dem = einops.rearrange(
            normalized_dem,
            f"h w -> {self.out_channels} h w",
        )

        # Pre-process
        center_cropped = self.center_crop(normalized_dem)
        cond_dem = self.resize(normalized_dem)

        return dict(
            img=self.normalize(center_cropped),
            cond=self.normalize(cond_dem),
            metas=dict(
                crop=dict(
                    center=(self.out_height // 2, self.out_width // 2),
                    size=(self.out_height, self.out_width),
                )
            ),
        )

    def get_crop_visualization(self, dem, metas):
        crop_center = metas["crop"]["center"]
        crop_size = metas["crop"]["size"]

        # Draw a rectangle on the center crop
        dem = dem.cpu().numpy()

        # Draw a rectangle on crop using cv2
        dem = cv2.cvtColor(dem, cv2.COLOR_GRAY2RGB)
        dem = cv2.rectangle(
            dem,
            (crop_center[1] - crop_size[1] // 2, crop_center[0] - crop_size[0] // 2),
            (crop_center[1] + crop_size[1] // 2, crop_center[0] + crop_size[0] // 2),
            (0, 255, 0),
            2,
        )

        return dem

from typing import Any

import einops
import torchvision.transforms.functional as tvF

from ..utils.data_utils import load_pickle, min_max_normalize
from .base_dataset import SLDEMDataset


class LunarSLDEMDataset(SLDEMDataset):
    def __init__(
        self,
        data_root,
        data_split="train",
        out_shape=(1, 256, 256),
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

    def __getitem__(self, index: int) -> dict:
        norm_dem_tensor = super().__getitem__(index)

        # Visualize, if debug
        if self.is_debug:
            self.visualize(norm_dem_tensor)

        # Prepare tensor: resize and adjust batch/channel dim

        out_dem_tensor = einops.rearrange(
            norm_dem_tensor,
            f"h w -> {self.out_channels} h w",
        )
        out_dem_tensor = tvF.resize(
            out_dem_tensor, (self.out_height, self.out_width), antialias=False
        )

        out_dem_tensor = tvF.normalize(
            out_dem_tensor, mean=[self.input_norm_mean], std=[self.input_norm_scale]
        )

        return dict(img=out_dem_tensor, cond=[], metas=[])

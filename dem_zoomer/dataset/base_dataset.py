import os.path as osp

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.data_utils import load_pickle, min_max_normalize


class SLDEMDataset(Dataset):
    def __init__(
        self,
        data_root,
        data_split="train",
        out_shape=(1, 256, 256),
        prefix="MoonORTO2DEM",
        num_repeat_dataset=2,
        is_debug=False,
    ) -> None:
        super().__init__()

        self._h5 = h5py.File(
            osp.join(data_root, f"{prefix}.hdf5"),
            "r",
        )
        self._split_dict = load_pickle(
            osp.join(data_root, f"{prefix}_{data_split}.pkl")
        )

        self.data_ids = list(self._split_dict)
        self.num_repeat_dataset = num_repeat_dataset
        self.is_debug = is_debug

        # Output tensor shape
        self.out_channels, self.out_height, self.out_width = out_shape

    def __len__(self):
        """Get length of dataset"""
        return self.num_repeat_dataset * len(self.data_ids)

    def _map_index_to_dataset_id(self, index: int) -> int:
        """Map index to data_id

        Args:
            index (int): index of dataitem in the repeated dataset

        Returns:
            int: index of dataitem in the original dataset

        """
        return index % len(self.data_ids)

    def __getitem__(self, index: int) -> dict:
        """Get a dataitem from the dataset

        Args:
            index (int): index of dataitem

        Returns:
            torch.Tensor: normalized DEM tensor [1, 1, H, W]

        """
        # Map to true index in repeated dataset
        true_index = self._map_index_to_dataset_id(index)

        # Fetch dataitem
        data_id = self.data_ids[true_index]
        raw_dem = self.get_data(data_id)

        # Normalize DEM
        norm_dem = min_max_normalize(raw_dem)

        # Convert to float tensor with batch dim
        norm_dem_tensor = torch.from_numpy(norm_dem).to(dtype=torch.float32)

        return norm_dem_tensor

    def get_data(self, data_id: str) -> np.ndarray:
        """Get data from h5 file

        Args:
            data_id (str): id of dataitem

        Returns:
            np.ndarray: raw DEM array [H, W]
        """
        img_id = self._split_dict[data_id][0]
        return self._h5[img_id][:].copy()

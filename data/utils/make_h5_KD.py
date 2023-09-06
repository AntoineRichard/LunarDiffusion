import argparse
import os
import pickle

import cv2
import h5py
import numpy as np

tile_size = 1000
tile_offset = 500

coord = [
    "N0-60_W000-120",
    "N0-60_W120-240",
    "N0-60_W240-360",
    "S0-60_W000-120",
    "S0-60_W120-240",
    "S0-60_W240-360",
]
DEM_tiles = {
    "N0-60_W000-120": "sldem2015_256_0n_60n_000_120_float.img",
    "N0-60_W120-240": "sldem2015_256_0n_60n_120_240_float.img",
    "N0-60_W240-360": "sldem2015_256_0n_60n_240_360_float.img",
    "S0-60_W000-120": "sldem2015_256_60s_0s_000_120_float.img",
    "S0-60_W120-240": "sldem2015_256_60s_0s_120_240_float.img",
    "S0-60_W240-360": "sldem2015_256_60s_0s_240_360_float.img",
}


def loadDEM(path):
    return np.fromfile(path, dtype=np.float32).reshape([15360, -1])


def tile(dem, key, h5, dct, tile_size, tile_offset):
    h, w = dem.shape
    htiles = int(h / tile_offset)
    wtiles = int(w / tile_offset)
    wrem = w - wtiles * tile_offset
    for i in range(htiles):
        for j in range(wtiles):
            dem_tile = dem[
                tile_offset * i : tile_offset * i + tile_size,
                tile_offset * j : tile_offset * j + tile_size,
            ]
            dem_tile = (
                (dem_tile - dem_tile.min())
                / (dem_tile.max() - dem_tile.min())
                * (2**16)
            )
            dem_tile = dem_tile.astype(np.uint16)
            if dem_tile.shape[1] != tile_size:
                break
            if dem_tile.shape[0] != tile_size:
                break
            dem_lbl = key + "-dem-" + str(i * tile_offset) + "-" + str(j * tile_offset)
            h5[dem_lbl] = dem_tile
            dct[key + "-" + str(i) + "-" + str(j)] = [dem_lbl]
    return h5, dct


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--output_path", type=str, default=".")
    return parser.parse_args()


args = parse()

h5 = h5py.File(os.path.join(args.output_path, "MoonORTO2DEM.hdf5"), "w")
dct = {}

for key in coord:
    dem = loadDEM(args.data_path, key, DEM_tiles)
    h5, dct = tile(dem, key, h5, dct, tile_size, tile_offset)
h5.close
num_samples = len(dct.keys())
keys = list(dct.keys())
idx = np.random.choice(range(num_samples - 100), size=50, replace=False)
idx = np.concatenate([idx + i for i in range(20)])

train_dct = {}
val_dct = {}
for i in range(num_samples):
    if i in idx:
        val_dct[keys[i]] = dct[keys[i]]
    else:
        train_dct[keys[i]] = dct[keys[i]]

with open(os.path.join(args.output_path, "MoonORTO2DEM_train.pkl"), "wb") as f:
    pickle.dump(train_dct, f)

with open(os.path.join(args.output_path, "MoonORTO2DEM_val.pkl"), "wb") as f:
    pickle.dump(val_dct, f)

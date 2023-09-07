import json
import os
import pathlib
import shutil
import sys
import warnings

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse

from dem_zoomer.trainer.dem_trainer import DEMDiffusionTrainer
from dem_zoomer.utils.config import Config


def parse_args():
    parser = argparse.ArgumentParser(description="Runner for Training Grasp Samplers")
    parser.add_argument("--config", "-c", help="Path to config file")
    parser.add_argument("--root-dir", "-d", help="Root directory")
    parser.add_argument("--num-gpus", "-g", type=int, help="Number of GPUs to use")
    parser.add_argument("--batch-size", "-b", type=int, help="Batch size per device")
    parser.add_argument(
        "-debug",
        action="store_true",
        default=False,
        help="Setting this will disable wandb logger and ... TODO",
    )
    return parser.parse_args()


def main(args):
    ## -- Config --
    config = Config.fromfile(args.config)

    # Overwrite config with args
    if args.num_gpus:
        config.trainer.devices = args.num_gpus
        config.trainer.num_workers = args.num_gpus * config.num_workers_per_gpu
    if args.batch_size:
        config.trainer.batch_size = args.batch_size
    if args.root_dir:
        for split in config.data:
            config.data[split].args.data_root_dir = args.root_dir

    ## -- Trainer --
    trainer = DEMDiffusionTrainer(config)
    trainer.run()


if __name__ == "__main__":
    args = parse_args()
    main(args)

import os
from argparse import ArgumentParser

import torch

from dem_zoomer.models import MODEL_REGISTRY
from dem_zoomer.trainer.experiment import Experiment
from dem_zoomer.utils.config import Config


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/boilerplate.yaml")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)


def get_model(model_config, ckpt_path, use_compile=False):
    model = MODEL_REGISTRY.get(model_config.model["type"], **model_config.model["args"])

    # Load state dict
    model.load_state_dict(torch.load(ckpt_path))

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    if use_compile:
        from einops._torch_specific import allow_ops_in_compiled_graph

        allow_ops_in_compiled_graph()

        model.compile(model)
    return model


def visualize(sample_batch):
    pass


def generate(args, ckpt_path=None):
    exp_config_path = args.config
    num_samples = args.num_samples
    batch_size = args.batch_size

    # Default Checkpoint path, if none
    if ckpt_path is None:
        experiment = Experiment(exp_config_path)
        ckpt_path = experiment.last_checkpoint

    # Config
    config = Config.fromfile(exp_config_path)

    # Model
    model = get_model(model_config=config.model, ckpt_path=ckpt_path, use_compile=True)

    # Generate
    for idx in range(0, num_samples, batch_size):
        batch_size = (
            num_samples - idx if (idx + batch_size) > num_samples else batch_size
        )
        outs = model.generate_samples(num_samples=4, return_intermediate=True)

    if args.visualize:
        visualize(outs)


if __name__ == "__main__":
    args = parse_args()
    generate(args)

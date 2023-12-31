import sys

import numpy as np
import torch

# Experimental: Einops fix for torch.compile graph
from einops._torch_specific import allow_ops_in_compiled_graph
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger

from dem_zoomer.models import MODEL_REGISTRY
from dem_zoomer.utils.config import Config

from ..dataset import DATASET_REGISTRY
from ..models import MODEL_REGISTRY
from ..utils.config import Config
from .experiment import Experiment
from .trainer import LightningTrainer

# medium/high. Helps on some tensor core GPUs
torch.set_float32_matmul_precision("medium")

# If compile is enabled, makes sure einops ops in graph are compiled properly
allow_ops_in_compiled_graph()

try:
    import wandb
except:
    pass


class DEMDiffusionTrainer(LightningTrainer):
    def __init__(self, config: Config):
        model_config = config.model
        data_config = config.data
        trainer_config = config.trainer

        # Experiment and config
        self._config = config
        self._experiment = Experiment(config.filename)

        # Checkpointing
        self._checkpointing_freq = (
            trainer_config.checkpointing_freq
            if hasattr(trainer_config, "checkpointing_freq")
            else 1000
        )
        trainer_config.default_root_dir = self._experiment.ckpt_dir

        # Initialize parent trainer class
        super().__init__(model_config, data_config, trainer_config)

        self.resume_from_checkpoint = self._experiment.default_resume_checkpoint

    def _build_dataset(self, data_config, split):
        data_split_config = data_config[split]
        return DATASET_REGISTRY.get(data_split_config.type, **data_split_config.args)

    def _build_model(self, model_config):
        model = MODEL_REGISTRY.get(model_config.type, **model_config.args)

        if "use_compile" in self.trainer_config:
            if self.trainer_config.use_compile:
                print("Using `torch.compile`. Model compilation may take some time.")
                model = torch.compile(model)
        return model

    def _get_callbacks(self) -> list:
        """Custom callbacks to be used by the trainer."""

        checkpoint_callback1 = ModelCheckpoint(
            save_top_k=3,
            monitor="loss",
            mode="min",
            dirpath=self._experiment.ckpt_dir,
            filename="epoch_{epoch:02d}-step_{step}-loss_{loss:.2f}",
            save_last=True,
            every_n_train_steps=self._checkpointing_freq,
        )

        lr_monitor_callback = LearningRateMonitor(logging_interval="step")

        callbacks = [checkpoint_callback1, lr_monitor_callback]

        return callbacks

    def _get_logger(self) -> Logger:
        """Custom logger to be used by the trainer."""
        if hasattr(self.trainer_config, "logger"):
            logger_config = self.trainer_config.logger

            if logger_config.type == "WandbLogger":
                assert hasattr(
                    logger_config, "project"
                ), "WandbLogger requires a project name to be specified in the config."

                logger = WandbLogger(
                    name=self._experiment.name,
                    project=logger_config.project,
                    save_dir=self._experiment.log_dir,
                    config=self._config,
                )
            elif logger_config.type == "TensorBoardLogger":
                logger = TensorBoardLogger(
                    save_dir=self._experiment.log_dir,
                    name=self._experiment.name,
                )
            elif logger_config.type == "CSVLogger":
                logger = CSVLogger(
                    save_dir=self._experiment.log_dir,
                    name=self._experiment.name,
                )
            else:
                raise NotImplementedError(f"Logger - `{logger.type}` not supported")
        else:
            logger = CSVLogger(
                save_dir=self._experiment.log_dir,
                name=self._experiment.name,
            )
        return logger

    def _compute_metrics(self):
        # TODO: Add metrics, if any
        pass

    def training_step(self, batch, batch_idx):
        # Data item
        img = batch["img"]
        img_cond = batch["cond"] if batch["cond"] else None

        # Forward pass
        loss_dict = self.model(img, img_cond)

        # Log
        self.log_dict(loss_dict, sync_dist=True)
        loss = loss_dict["denoising_loss"]
        self.log("loss", loss, sync_dist=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print("Validation loop - Generate demo samples")
            self.model.set_inference_timesteps(100)
            samples, _ = self.model.generate_samples(num_samples=2, device=self.device)

            # Unnormalize TODO: Make available in config
            samples = samples * 0.5 + 0.5
            samples = samples.permute(0, 2, 3, 1).cpu().numpy()

            # Log demo images to wandb if it is imported
            if hasattr(self.logger, "experiment") and "wandb" in sys.modules:
                self.logger.experiment.log(
                    {
                        "Generation samples": [
                            wandb.Image(sample, caption=f"Sample {i}")
                            for i, sample in enumerate(samples)
                        ],
                    },
                )
        # TODO: Metric ? FID?
        return

    def get_dummy_dataset(self):
        from torch.utils.data import Dataset

        class DummyDataset(Dataset):
            def __init__(self, shape=(1, 256, 256), length=1000):
                self.shape = shape
                self.length = length

            def __len__(self):
                return self.length

            def __getitem__(self, index):
                return torch.randn(*self.shape), torch.randn(*self.shape)

        return DummyDataset()


# Conditional trainer
class ConditionalDEMDiffusionTrainer(DEMDiffusionTrainer):
    def __init__(self, config: Config):
        super().__init__(config)

    def validation_step(self, batch, batch_idx):
        if batch_idx % 100 == 0:
            print("Validation loop - Generate demo samples")

            # Data item
            img = batch["img"]
            img_cond = batch["cond"] if batch["cond"] else None
            metas = batch["metas"]

            # Conditional generation
            # TODO: Check batching in z_cond in generation
            self.model.set_inference_timesteps(100)
            gen_samples, _ = self.model.generate_samples(
                num_samples=2, z_cond=img_cond, device=self.device
            )

            # Unnormalize TODO: Make available in config
            gen_samples = gen_samples * 0.5 + 0.5
            gen_samples = gen_samples.permute(0, 2, 3, 1).cpu().numpy()

            cond_imgs = img_cond * 0.5 + 0.5
            cond_imgs = cond_imgs.permute(0, 2, 3, 1).cpu().numpy()

            # Make a crop vis TODO: batched output vis
            crop_vis_img = self.dataset.get_crop_visualization(gen_samples, metas)

            # Stack arrays horizontally
            samples = np.hstack([cond_imgs, crop_vis_img])

            # Log demo images to wandb if it is imported
            if hasattr(self.logger, "experiment") and "wandb" in sys.modules:
                self.logger.experiment.log(
                    {
                        "Generation samples": [
                            wandb.Image(sample, caption=f"Sample {i}")
                            for i, sample in enumerate(samples)
                        ],
                    },
                )

        return

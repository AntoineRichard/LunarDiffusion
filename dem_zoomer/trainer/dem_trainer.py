import torch

# Experimental: Einops fix for torch.compile graph
from einops._torch_specific import allow_ops_in_compiled_graph
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, Logger, TensorBoardLogger, WandbLogger

from dem_zoomer.models import MODEL_REGISTRY

from ..dataset import DATASET_REGISTRY
from ..models import MODEL_REGISTRY
from ..utils.config import Config
from .experiment import Experiment
from .trainer import LightningTrainer

# medium/high. Helps on some tensor core GPUs
torch.set_float32_matmul_precision("medium")

# If compile is enabled, makes sure einops ops in graph are compiled properly
allow_ops_in_compiled_graph()


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
        # TODO: Change to a real dataset
        return self.get_dummy_dataset()

    def _build_model(self, model_config):
        model = MODEL_REGISTRY.get(model_config.type, **model_config.args)

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

        checkpoint_callback2 = ModelCheckpoint(
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            dirpath=self._experiment.ckpt_dir,
            filename="best",
            save_last=True,
            every_n_train_steps=1000,
        )

        lr_monitor_callback = LearningRateMonitor(logging_interval="step")

        callbacks = [checkpoint_callback1, checkpoint_callback2, lr_monitor_callback]

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
        img, img_cond = batch

        loss_dict = self.model(img, img_cond)
        self.log_dict(loss_dict)

        loss = loss_dict["denoising_loss"]
        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        # TODO: Add cond generation pipeline here

        # Val dataloader -> z_cond
        img, img_cond = batch

        loss_dict = self.model(img, img_cond)
        self.log_dict(loss_dict)

        loss = loss_dict["denoising_loss"]
        self.log("loss", loss)

        return loss

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

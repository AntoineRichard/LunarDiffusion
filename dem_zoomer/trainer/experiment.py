import glob
import os
import shutil
import warnings


class Experiment:
    def __init__(
        self, config_path, resume_from="last", out_dir="output/", ckpt_format="ckpt"
    ) -> None:
        # Checkpoint format
        self._ckpt_format = ckpt_format

        # Experiment naming
        self.name = config_path.split("/")[-1].split(".")[0]
        self.category = os.path.join(
            *(config_path.split("/")[-3:-1])
        )  # max path hierarchy = 2 # category/subcat/name.py

        # Experiment directories
        self.out_dir = out_dir
        self.exp_dir = os.path.join(os.path.abspath(out_dir), self.category, self.name)
        self.ckpt_dir = os.path.join(self.exp_dir, "checkpoints")
        self.log_dir = os.path.join(self.exp_dir, "logs")
        self._make_dirs()

        # Make a copy of the config file when training
        self.src_config_path = config_path
        self.dst_config_path = os.path.join(self.exp_dir, f"{self.name}.py")

        # Maintain a single config in exp dir. Warn if exists and over-write
        if os.path.isfile(self.dst_config_path):
            warnings.warn(
                f"Existing config file will be over-written: {self.dst_config_path}"
            )
        shutil.copy(self.src_config_path, self.dst_config_path)

        # Resume from checkpoint
        self.resume_from = resume_from

    @property
    def all_checkpoints(self):
        return glob.glob(os.path.join(self.ckpt_dir, f"*.{self._ckpt_format}"))

    @property
    def exists(self):
        return os.path.isdir(self.exp_dir)

    @property
    def last_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"last.{self._ckpt_format}")
        return ckpt_path if os.path.exists(ckpt_path) else None

    @property
    def best_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, f"best.{self._ckpt_format}")
        return ckpt_path if os.path.exists(ckpt_path) else None

    @property
    def default_resume_checkpoint(self):
        _default_checkpoint = self.last_checkpoint

        if self.resume_from in ("best", "last"):
            ckpt_path = (
                self.last_checkpoint
                if self.resume_from == "last"
                else self.best_checkpoint
            )
        else:
            ckpt_path = self.resume_from

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            _default_checkpoint = ckpt_path
        else:
            # Do nothing and start from scratch
            pass

            # warnings.warn(f"Could not find checkpoint: {ckpt_path}")
            # if _default_checkpoint is None:
            #     warnings.warn(
            #         f"Default checkpoint {_default_checkpoint} also not found."
            #     )
        return _default_checkpoint

    def _make_dirs(self):
        # Warn existing checkpoint directory
        if os.path.exists(self.ckpt_dir):
            warnings.warn(
                f"Experiment Checkpoint directory exists: {self.ckpt_dir} \nCheckpoints may be auto-overwritten by the trainer."
            )
        else:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        return

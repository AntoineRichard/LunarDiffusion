import enum

from .lunar_sldem_dataset import LunarSLDEMDataset

__fake_registry__ = {"LunarSLDEMDataset": LunarSLDEMDataset}


class DATASET_REGISTRY(enum.Enum):
    LUNAR_SLDEM = "LunarSLDEMDataset"

    def __repr__(self):
        return __fake_registry__[self.value]

    def build(self, *args, **kwargs):
        return __fake_registry__[self.value](*args, **kwargs)

    def get(dataset_type, *args, **kwargs):
        return __fake_registry__[dataset_type](*args, **kwargs)

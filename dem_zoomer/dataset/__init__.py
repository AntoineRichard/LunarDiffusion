import enum

__fake_registry__ = {}


class DATASET_REGISTRY(enum.Enum):
    # TODO: Add Datasets
    LOLA_DATASET_COARSE = "LolaDatasetCoarse"

    def __repr__(self):
        return __fake_registry__[self.value]

    def build(self, *args, **kwargs):
        return __fake_registry__[self.value](*args, **kwargs)

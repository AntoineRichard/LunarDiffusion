import enum

from .dem_diffuser import DEMDiffuser
from .unet2d import FancyUnet2D, SimpleConditionalUnet2D
from .uvit import UViT

__fake_registry__ = {
    "SimpleConditionalUnet2D": SimpleConditionalUnet2D,
    "FancyUnet2D": FancyUnet2D,
    "UViT": UViT,
    "DEMDiffuser": DEMDiffuser,
}


class MODEL_REGISTRY(enum.Enum):
    SIMPLE_UNET = "SimpleConditionalUnet2D"
    FANCY_UNET = "FancyUnet2D"
    VIT = "UViT"
    DEM_DIFFUSER = "DEMDiffuser"

    def __repr__(self):
        return __fake_registry__[self.value]

    def build(self, *args, **kwargs):
        return __fake_registry__[self.value](*args, **kwargs)

    def get(model_type, *args, **kwargs):
        assert (
            model_type in __fake_registry__.keys()
            and model_type in MODEL_REGISTRY.__members__.keys()
        ), f"Model type {model_type} not found in registry."

        return MODEL_REGISTRY(model_type)(*args, **kwargs)

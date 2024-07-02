import math

import torch

from ksuit.distributed import is_rank0, barrier
from ksuit.initializers.base import InitializerBase


class VimPretrainedInitializer(InitializerBase):
    """ initialize with weights from a state_dict loaded via torchhub """

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def _get_model_kwargs(self):
        model = self.model.lower().replace("_", "")
        if "tiny" in model:
            return dict(
                patch_size=16,
                dim=192,
                depth=24,
            )
        if "small" in model:
            return dict(
                patch_size=16,
                dim=384,
                depth=24,
            )
        raise NotImplementedError(f"get_model_kwargs of '{self.model}' is not implemented")

    def init_weights(self, model):
        self.logger.info(f"loading Vim weights of model '{self.model}'")
        sd = torch.load(self.path_provider.model_path / self.model)
        sd = sd["model"]
        sd = {f"model.{key}": value for key, value in sd.items()}
        model.load_state_dict(sd)

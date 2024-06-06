from pathlib import Path

import einops
import torch

from ksuit.initializers.base import InitializerBase
from ksuit.distributed import is_rank0, barrier

class TorchhubInitializer(InitializerBase):
    """ initialize with weights from a state_dict loaded via torchhub """

    def __init__(self, repo, model, source="github", missing_keys=None, unexpected_keys=None, **kwargs):
        super().__init__(**kwargs)
        self.repo = repo
        self.model = model
        self.source = source
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []

    def init_weights(self, model):
        self.logger.info(f"loading weights from '{self.source}:{self.repo}/{self.model}'")
        if is_rank0():
            sd = torch.hub.load(repo_or_dir=self.repo, model=self.model, source=self.source).state_dict()
            barrier()
        else:
            # wait for rank0 to download
            barrier()
            # load cached model
            sd = torch.hub.load(repo_or_dir=self.repo, model=self.model, source=self.source).state_dict()
        strict = len(self.missing_keys) == 0 and len(self.unexpected_keys) == 0
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=strict)
        if not strict:
            assert missing_keys == self.missing_keys, f"{missing_keys} != {self.missing_keys}"
            assert unexpected_keys == self.unexpected_keys, f"{unexpected_keys} != {self.unexpected_keys}"

from pathlib import Path

import torch

from ksuit.initializers.base import InitializerBase


class StateDictInitializer(InitializerBase):
    """ initialize with weights from a state_dict loaded from the disk """

    def __init__(
            self,
            filename,
            base_path=None,
            weights_key=None,
            ctor_kwargs_key="ctor_kwargs",
            key_mapping=None,
            missing_keys=None,
            unexpected_keys=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.filename = filename
        if base_path is None:
            self.uri = Path(self.path_provider.model_path / filename).expanduser()
        else:
            self.uri = Path(base_path).expanduser() / filename
        assert self.uri.exists() and self.uri.is_file(), self.uri.as_posix()
        self.key_mapping = key_mapping
        self.weights_key = weights_key
        self.ctor_kwargs_key = ctor_kwargs_key
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []

    def _get_model_kwargs(self):
        if self.ctor_kwargs_key is None:
            raise NotImplementedError
        self.logger.info(f"loading ctor_kwargs from '{self.uri.as_posix()}' with key '{self.ctor_kwargs_key}'")
        sd = torch.load(self.uri, map_location="cpu")
        ctor_kwargs = sd[self.ctor_kwargs_key]
        return ctor_kwargs

    def init_weights(self, model):
        self.logger.info(f"loading weights from '{self.uri.as_posix()}'")
        sd = torch.load(self.uri, map_location="cpu")
        # select weights
        if self.weights_key is not None:
            assert self.weights_key in sd, sd.keys()
            sd = sd[self.weights_key]
        # remap keys
        if self.key_mapping is not None:
            for old_key, new_key in self.key_mapping.items():
                sd[new_key] = sd.pop(old_key)
        strict = len(self.missing_keys) == 0 and len(self.unexpected_keys) == 0
        missing_keys, unexpected_keys = model.load_state_dict(sd, strict=strict)
        if not strict:
            assert missing_keys == self.missing_keys, f"{missing_keys} != {self.missing_keys}"
            assert unexpected_keys == self.unexpected_keys, f"{unexpected_keys} != {self.unexpected_keys}"

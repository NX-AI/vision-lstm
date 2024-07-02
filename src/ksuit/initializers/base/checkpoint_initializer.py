import torch

from ksuit.models import SingleModel
from ksuit.utils.checkpoint import Checkpoint
from .initializer_base import InitializerBase


class CheckpointInitializer(InitializerBase):
    def __init__(
            self,
            stage_id,
            model_name,
            checkpoint,
            load_optim,
            model_info=None,
            stage_name=None,
            pop_ckpt_kwargs_keys=None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.stage_id = stage_id
        self.model_name = model_name
        self.load_optim = load_optim
        self.model_info = model_info
        self.pop_ckpt_kwargs_keys = pop_ckpt_kwargs_keys or []
        self.stage_name = stage_name or self.path_provider.stage_name

        # checkpoint can be a string (e.g. "best_accuracy" for initializing from a model saved by BestModelLogger)
        # or dictionary with epoch/update/sample values
        if isinstance(checkpoint, str):
            self.checkpoint = checkpoint
        else:
            self.checkpoint = Checkpoint.create(checkpoint)
            assert self.checkpoint.is_minimally_specified or self.checkpoint.is_fully_specified

    def init_weights(self, model):
        raise NotImplementedError

    def _get_model_state_dict(self, model):
        model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, file_type="model")
        sd = torch.load(ckpt_uri, map_location=model.device)
        if "state_dict" in sd:
            sd = sd["state_dict"]
        return sd, model_name, ckpt_uri

    def _get_model_kwargs(self):
        model_name, ckpt_uri = self._get_modelname_and_ckpturi(file_type="model")
        sd = torch.load(ckpt_uri, map_location=torch.device("cpu"))
        kwargs = sd["ctor_kwargs"]
        self.logger.info(f"loaded model kwargs from {ckpt_uri}")
        if len(self.pop_ckpt_kwargs_keys) > 0:
            self.logger.info(f"removing {self.pop_ckpt_kwargs_keys} from ckpt kwargs")
            for key in self.pop_ckpt_kwargs_keys:
                kwargs.pop(key)
        return kwargs

    def init_optim(self, model):
        if not isinstance(model, SingleModel):
            return
        if not self.load_optim:
            return
        assert model.optim is not None
        model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, file_type="optim")
        sd = torch.load(ckpt_uri, map_location=model.device)
        model.optim.load_state_dict(sd)
        self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")

    def _get_modelname_and_ckpturi(self, file_type, model=None, model_name=None):
        model_name = model_name or self.model_name
        if model_name is None:
            assert isinstance(model, SingleModel)
            self.logger.info(f"no model_name provided -> using {model.name}")
            model_name = model.name

        # model_info is e.g. ema=0.99
        model_info_str = "" if self.model_info is None else f" {self.model_info}"
        ckpt_uri = self._get_ckpt_uri(prefix=f"{model_name} cp=", suffix=f"{model_info_str} {file_type}.th")
        assert ckpt_uri.exists(), f"'{ckpt_uri}' doesn't exist"
        return model_name, ckpt_uri

    def _get_ckpt_uri(self, prefix, suffix):
        ckpt_folder = self.path_provider.get_stage_checkpoint_path(
            stage_name=self.stage_name,
            stage_id=self.stage_id,
        )
        # find full checkpoint from minimal specification
        if not isinstance(self.checkpoint, str) and not self.checkpoint.is_fully_specified:
            ckpt = Checkpoint.to_fully_specified_from_fnames(
                ckpt_folder=ckpt_folder,
                ckpt=self.checkpoint,
                prefix=prefix,
                suffix=suffix,
            )
        else:
            ckpt = self.checkpoint
        return ckpt_folder / f"{prefix}{ckpt}{suffix}"

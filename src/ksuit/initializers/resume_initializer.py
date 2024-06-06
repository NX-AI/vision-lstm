import torch

from ksuit.models import CompositeModel, SingleModel
from ksuit.utils.checkpoint import Checkpoint
from .base import CheckpointInitializer


class ResumeInitializer(CheckpointInitializer):
    """
    initializes models/optims from a checkpoint ready for resuming training
    load_optim=True as this is usually used to resume a training run
    stage_name is provided by the trainer as it already knows the correct stage_name
    """

    def __init__(self, load_optim=True, **kwargs):
        super().__init__(load_optim=load_optim, model_name=None, **kwargs)

    def init_weights(self, model):
        self._init_weights(model.name, model)

    def _init_weights(self, name, model):
        if isinstance(model, SingleModel):
            model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="model")
            sd = torch.load(ckpt_uri, map_location=model.device)
            if "state_dict" in sd:
                sd = sd["state_dict"]
            model.load_state_dict(sd)
            self.logger.info(f"loaded weights of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModel):
            for submodel_name, submodel in model.submodels.items():
                self._init_weights(name=f"{name}.{submodel_name}", model=submodel)

    def init_optim(self, model):
        self._init_optim(name=model.name, model=model)

    def _init_optim(self, name, model):
        if isinstance(model, SingleModel):
            if model.optim is None:
                # e.g. EMA target network doesn't have an optimizer
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {name} ({model.name}) "
                    f"(optim is None)"
                )
            elif model.is_frozen:
                self.logger.info(
                    f"skip loading optim from checkpoint '{self.checkpoint}' for {name}  ({model.name}) "
                    f"(is_frozen)"
                )
            else:
                model_name, ckpt_uri = self._get_modelname_and_ckpturi(model=model, model_name=name, file_type="optim")
                sd = torch.load(ckpt_uri, map_location=model.device)
                model.optim.load_state_dict(sd)
                self.logger.info(f"loaded optimizer of {model_name} from {ckpt_uri}")
        if isinstance(model, CompositeModel):
            for submodel_name, submodel in model.submodels.items():
                self._init_optim(name=f"{name}.{submodel_name}", model=submodel)

    def _get_trainer_ckpt_file(self):
        return self._get_ckpt_uri(prefix=f"trainer cp=", suffix=".th")

    def get_start_checkpoint(self):
        if isinstance(self.checkpoint, str):
            trainer_ckpt = torch.load(self._get_trainer_ckpt_file())
            self.logger.info(f"loaded checkpoint from trainer_state_dict: {trainer_ckpt}")
            return Checkpoint(
                epoch=trainer_ckpt["epoch"],
                update=trainer_ckpt["update"],
                sample=trainer_ckpt["sample"],
            )
        else:
            return Checkpoint.to_fully_specified_from_fnames(
                ckpt_folder=self.path_provider.get_stage_checkpoint_path(
                    stage_name=self.stage_name,
                    stage_id=self.stage_id,
                ),
                ckpt=self.checkpoint,
            )

    def init_trainer(self, trainer):
        ckpt_uri = self._get_trainer_ckpt_file()
        trainer.load_state_dict(torch.load(ckpt_uri))
        self.logger.info(f"loaded trainer checkpoint {ckpt_uri}")

    def init_callbacks(self, callbacks, model):
        for callback in callbacks:
            callback.resume_from_checkpoint(
                stage_name=self.stage_name,
                stage_id=self.stage_id,
                model=model,
            )

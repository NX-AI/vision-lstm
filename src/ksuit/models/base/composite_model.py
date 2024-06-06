import torch

from .model_base import ModelBase
from .single_model import SingleModel


class CompositeModel(ModelBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # check that base methods were not overwritten
        assert type(self).after_initializers == CompositeModel.after_initializers

    @property
    def submodels(self):
        raise NotImplementedError

    def clear_buffers(self):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.clear_buffers()

    def set_accumulation_steps(self, accumulation_steps):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.set_accumulation_steps(accumulation_steps)

    def optim_step(self, grad_scaler):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, SingleModel) and submodel.optim is None:
                continue
            submodel.optim_step(grad_scaler)
        # after step (e.g. for EMA)
        self.after_update_step()

    def optim_schedule_step(self):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, SingleModel) and submodel.optim is None:
                continue
            submodel.optim_schedule_step()

    def optim_zero_grad(self, set_to_none=True):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            if isinstance(submodel, SingleModel) and submodel.optim is None:
                continue
            submodel.optim_zero_grad(set_to_none)

    @property
    def is_frozen(self):
        return all(m is None or m.is_frozen for m in self.submodels.values())

    @is_frozen.setter
    def is_frozen(self, value):
        for m in self.submodels.values():
            if submodel is None:
                continue
            m.is_frozen = value

    @property
    def device(self):
        devices = [submodel.device for submodel in self.submodels.values() if submodel is not None]
        assert all(device == devices[0] for device in devices[1:])
        return devices[0]

    @property
    def is_batch_size_dependent(self):
        return any(submodel.is_batch_size_dependent for submodel in self.submodels.values() if submodel is not None)

    def initialize_weights(self):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.initialize_weights()
        return self

    def apply_initializers(self):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.apply_initializers()
        for initializer in self.initializers:
            initializer.init_weights(self)
            initializer.init_optim(self)
        return self

    def after_initializers(self):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.after_initializers()
        self._after_initializers()

    def _after_initializers(self):
        pass

    def initialize_optim(self, lr_scale_factor=None):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.initialize_optim(lr_scale_factor=lr_scale_factor)
        if self.is_frozen:
            self.logger.info(f"{self.name} has only frozen submodels -> put into eval mode")
            self.eval()

    def train(self, mode=True):
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.train(mode=mode)
        # avoid setting mode to train if whole network is frozen
        if self.is_frozen and mode is True:
            return
        return super().train(mode=mode)

    def to(self, device, *args, **kwargs):
        if isinstance(device, str):
            device = torch.device(device)
        assert isinstance(device, torch.device)
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.to(*args, **kwargs, device=device)
        return super().to(*args, **kwargs, device=device)

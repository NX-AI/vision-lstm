from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.models import CompositeModel
from ksuit.utils.model_utils import get_named_models


class LrCallback(PeriodicCallback):
    def should_log_after_update(self, checkpoint):
        if checkpoint.update == 1:
            return True
        return super().should_log_after_update(checkpoint)

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, **_):
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModel) or model.optim is None:
                continue
            for param_group in model.optim.torch_optim.param_groups:
                group_name = f"/{param_group['name']}" if "name" in param_group else ""
                if model.optim.schedule is not None:
                    lr = param_group["lr"]
                    self.writer.add_scalar(f"optim/lr/{model_name}{group_name}", lr)
                if model.optim.weight_decay_schedule is not None:
                    wd = param_group["weight_decay"]
                    self.writer.add_scalar(f"optim/wd/{model_name}{group_name}", wd)

from ksuit.callbacks.base.periodic_callback import PeriodicCallback
from ksuit.models import CompositeModel
from ksuit.utils.model_utils import get_named_models


class FreezerCallback(PeriodicCallback):
    def should_log_after_update(self, checkpoint):
        if checkpoint.update == 1:
            return True
        return super().should_log_after_update(checkpoint)

    # noinspection PyMethodOverriding
    def _periodic_callback(self, model, **_):
        for model_name, model in get_named_models(model).items():
            if isinstance(model, CompositeModel) or model.freezers is None:
                continue
            for freezer in model.freezers:
                if freezer.schedule is None:
                    continue
                self.writer.add_scalar(f"freezers/{model_name}/{freezer}/is_frozen", not freezer.requires_grad)

import numpy as np

from ksuit.callbacks.base.callback_base import CallbackBase
from ksuit.models import CompositeModel
from ksuit.utils.model_utils import get_trainable_param_count, get_frozen_param_count
from ksuit.utils.naming_utils import join_names, snake_type_name


class ParamCountCallback(CallbackBase):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    @staticmethod
    def _get_param_counts(model, trace=None):
        if isinstance(model, CompositeModel):
            result = []
            immediate_children = []
            for name, submodel in model.submodels.items():
                if submodel is None:
                    continue
                subresult = ParamCountCallback._get_param_counts(submodel, trace=join_names(trace, name))
                result += subresult
                immediate_children.append(subresult[0])
            trainable_sum = sum(count for _, count, _ in immediate_children)
            frozen_sum = sum(count for _, _, count in immediate_children)
            return [(trace, trainable_sum, frozen_sum)] + result
        else:
            return [
                (
                    join_names(trace, snake_type_name(model)),
                    get_trainable_param_count(model),
                    get_frozen_param_count(model),
                )
            ]

    def _before_training(self, model, **_):
        param_counts = self._get_param_counts(model)

        _, total_trainable, total_frozen = param_counts[0]
        max_trainable_digits = int(np.log10(total_trainable)) + 1 if total_trainable > 0 else 1
        max_frozen_digits = int(np.log10(total_frozen)) + 1 if total_frozen > 0 else 1
        # add space for thousand seperators
        max_trainable_digits += int(max_trainable_digits / 3)
        max_frozen_digits += int(max_frozen_digits / 3)
        # generate format strings
        tformat = f">{max_trainable_digits},"
        fformat = f">{max_frozen_digits},"

        self.logger.info(f"parameter counts (trainable | frozen)")
        new_summary_entries = {}
        for name, tcount, fcount in param_counts:
            name = name or "total"
            self.logger.info(f"{format(tcount, tformat)} | {format(fcount, fformat)} | {name}")
            new_summary_entries[f"param_count/{name}/trainable"] = tcount
            new_summary_entries[f"param_count/{name}/frozen"] = fcount
        self.summary_provider.update(new_summary_entries)

        # detailed number of params
        if self.verbose:
            self.logger.info("detailed parameters")
            for name, param in model.named_parameters():
                self.logger.info(f"{np.prod(param.shape):>10,} {'train' if param.requires_grad else 'frozen'} {name}")

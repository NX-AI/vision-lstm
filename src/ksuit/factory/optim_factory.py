from copy import deepcopy
from functools import partial

from ksuit.optim import OptimWrapper
from .base import SingleFactory
from .base.reflection import ctor_from_name, all_ctor_kwarg_names


class OptimFactory(SingleFactory):
    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        kwargs = deepcopy(kwargs)

        # extract OptimWrapper kwargs (e.g. clip_grad_value or exclude_bias_from_wd)
        # these should not be passed to the torch optimizer but to the OptimWrapper afterwards
        wrapped_optim_kwargs = {}
        wrapped_optim_kwargs_keys = all_ctor_kwarg_names(OptimWrapper)
        for key in wrapped_optim_kwargs_keys:
            if key in kwargs:
                wrapped_optim_kwargs[key] = kwargs.pop(key)

        module_names = [module_name.format(kind=kind) for module_name in self.module_names]
        torch_optim_ctor = ctor_from_name(
            module_names=module_names,
            type_name=kind,
            **kwargs,
        )
        return partial(
            OptimWrapper,
            torch_optim_ctor=torch_optim_ctor,
            **wrapped_optim_kwargs,
            **(optional_kwargs or {}),
        )

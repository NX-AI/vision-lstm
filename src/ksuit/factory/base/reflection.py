import importlib
import inspect
from functools import partial

import numpy as np


def all_ctor_kwarg_names(cls, result=None):
    if result is None:
        result = set()
    for name in inspect.signature(cls).parameters.keys():
        result.add(name)
    if cls.__base__ is not None:
        all_ctor_kwarg_names(cls.__base__, result)
    return result


class TypeNameNotFoundError(Exception):
    pass


def type_from_name(module_names, type_name):
    """
    tries to import type_name from any of the modules identified by module_names
    e.g. module_names=[loss_functions, torch.nn] type_name=bce_loss will import torch.nn.BCELoss
    """
    for module_name in module_names:
        module_name = module_name.lower()
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            # this also fails if some module could not be imported from within the module to import
            # Example:
            #   module_name="models.resnet" but that module requires torchvision to be installed
            #   if then torchvision is not installed it will raise a ModuleNotFoundError but this
            #   is different from having an invalid module_name -> raise the ModuleNotFoundError
            #   caused by the missing torchvision dependency
            if not module_name.startswith(e.name):
                raise e
            continue

        # try to find typename in module
        type_name_lowercase = type_name.lower().replace("_", "")
        possible_type_names = list(
            filter(
                lambda k: k.lower() in [type_name_lowercase, type_name],
                module.__dict__.keys()
            )
        )
        # in case of ambiguities -> filter out all caps names (e.g. CIFAR10, SPEECHCOMMANDS)
        if len(possible_type_names) > 1:
            possible_type_names = [name for name in possible_type_names if not name.isupper()]
        # prefer names without _ because they are more likely to be classes
        if len(possible_type_names) > 1:
            underscore_counts = [name.count("_") for name in possible_type_names]
            possible_type_names = [possible_type_names[np.argmin(underscore_counts)]]
        assert len(possible_type_names) <= 1, f"error found more than one possible type for {type_name_lowercase}"
        if len(possible_type_names) == 1:
            return getattr(module, possible_type_names[0])

    # raise custom error type in order for MasterFactory to be able to catch only this error
    raise TypeNameNotFoundError(
        f"can't find class {type_name} (searched modules: {module_names}) "
        f"-> check type_name or add modules to factory"
    )


def ctor_from_name(module_names, type_name, **kwargs):
    obj_type = type_from_name(module_names=module_names, type_name=type_name)
    return partial(obj_type, **kwargs)

import importlib
import logging

from ksuit.factory.base.reflection import type_from_name, all_ctor_kwarg_names
from .factory_base import FactoryBase
from .reflection import TypeNameNotFoundError


class SingleFactory(FactoryBase):
    def __init__(self, module_names=None, kind_to_ctor=None, **kwargs):
        super().__init__(**kwargs)
        self.module_names = module_names or []
        # shallow copy to avoid any unwanted side-effects if somewhere in the
        # code a new ctor is registered  via factory.kind_to_ctor[...] = ...
        self.kind_to_ctor = {**(kind_to_ctor or {})}

    def add_module_name(self, module_name, index=None):
        if index is None:
            self.module_names.append(module_name)
        else:
            self.module_names.insert(index, module_name)

    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        if kind in self.kind_to_ctor:
            # custom ctor
            ctor = self.kind_to_ctor[kind]
        else:
            type_name = kind.split(".")[-1]

            # check if kind is a valid module (e.g. kind=ksuit.impl.resnet18_cifar10_94percent
            try:
                custom_module = importlib.import_module(kind)
            except ModuleNotFoundError as e:
                if not kind.startswith(e.name):
                    raise e
                custom_module = None

            if custom_module is None:
                # find ctor in registered module_names
                if len(self.module_names) == 0:
                    raise TypeNameNotFoundError(f"can't find class {type_name} (module_names is empty list)")
                module_names = [module_name.format(kind=kind) for module_name in self.module_names]
            else:
                module_names = [kind]
            # allow relative paths as kind (type_name is always without .)
            # Example:
            #   model.cnn.resnet -> module_names=["model.{kind}"] kind="cnn.resnet"
            #   would be resolved into module_names=["model.cnn.resnet"] type_name="resnet"
            ctor = type_from_name(module_names=module_names, type_name=type_name)

        # prepare optional_kwargs
        # - e.g. pass update_counter to a SchedulableLoss but not to torch.nn.MSELoss
        if optional_kwargs is not None:
            optional_kwargs = {**optional_kwargs}
            # remove all invalid kwargs based on ctor signature
            ctor_kwarg_names = all_ctor_kwarg_names(ctor)
            for key in list(optional_kwargs.keys()):
                if key not in ctor_kwarg_names:
                    optional_kwargs.pop(key)
            # remove all keys that are already contained in kwargs
            for key in kwargs.keys():
                if key in optional_kwargs:
                    optional_kwargs.pop(key)
        else:
            optional_kwargs = {}

        # example create vision transformer
        # - kwargs=dict(patch_size=16, dim=192, depth=12)
        # - optional_kwargs=dict(update_counter=update_counter)
        try:
            return ctor(**kwargs, **optional_kwargs)
        except TypeError as e:
            logging.error(f"error creating object of type {ctor.__name__}: {e}")
            raise

from .empty_factory import EmptyFactory
from .factory_base import FactoryBase
from .reflection import TypeNameNotFoundError
from .single_factory import SingleFactory


class MasterFactory(FactoryBase):
    factories = {}
    optional_kwargs = {}

    @staticmethod
    def add_base_path(base_path):
        for key, factory in MasterFactory.factories.items():
            if isinstance(factory, SingleFactory):
                factory.add_module_name(f"{base_path}.{{kind}}", index=0)
                if key.endswith("s"):
                    # e.g. loss -> losses
                    factory.add_module_name(f"{base_path}.{key}es.{{kind}}", index=0)
                else:
                    # e.g. transform -> transforms
                    factory.add_module_name(f"{base_path}.{key}s.{{kind}}", index=0)

    @staticmethod
    def clear():
        MasterFactory.factories.clear()

    @staticmethod
    def set(key, value):
        MasterFactory.factories[key] = value

    @staticmethod
    def get(key):
        if key in MasterFactory.factories:
            return MasterFactory.factories[key]
        # return empty factory to allow create/create_list/create_dict calls with empty objects
        # Example:
        #   MasterFactory.get("param_group_modifier").create_list(param_group_modifiers)
        #   this call should return an empty list, even if no factory was registered for "param_group_modifier"
        return EmptyFactory(key=key)

    @staticmethod
    def create(obj_or_kwargs, **kwargs):
        return FactoryBase.create(MasterFactory, obj_or_kwargs=obj_or_kwargs, **kwargs)

    @staticmethod
    def create_dict(collection, collate_fn=None, **kwargs):
        return FactoryBase.create_dict(MasterFactory, collection=collection, collate_fn=collate_fn, **kwargs)

    @staticmethod
    def create_list(collection, collate_fn=None, **kwargs):
        return FactoryBase.create_list(MasterFactory, collection=collection, collate_fn=collate_fn, **kwargs)

    @staticmethod
    def instantiate(kind, optional_kwargs=None, **kwargs):
        # allow setting global optional_kwargs (e.g. update_counter or dataset_container)
        if optional_kwargs is None:
            optional_kwargs = {**MasterFactory.optional_kwargs, **(optional_kwargs or {})}
        for factory in MasterFactory.factories.values():
            try:
                obj = factory.instantiate(kind=kind, optional_kwargs=optional_kwargs, **kwargs)
                return obj
            except TypeNameNotFoundError:
                continue
        raise RuntimeError(f"no factory could instantiate object with kind '{kind}'")

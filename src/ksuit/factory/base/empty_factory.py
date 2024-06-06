from .factory_base import FactoryBase


class EmptyFactory(FactoryBase):
    def __init__(self, key):
        super().__init__()
        self.key = key

    def instantiate(self, kind, optional_kwargs=None, **kwargs):
        raise RuntimeError(
            f"Could not create object with kind='{kind}' because no factory is defined for '{self.key}'. "
            f"Register an appropriate factory with MasterFactory.factories['{self.key}'] = MyFactory()."
        )

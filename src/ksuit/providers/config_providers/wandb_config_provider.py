import wandb

from .base.config_provider_base import ConfigProviderBase
from .primitive_config_provider import PrimitiveConfigProvider
from ..path_provider import PathProvider


class WandbConfigProvider(ConfigProviderBase):
    def __init__(self, path_provider: PathProvider):
        super().__init__()
        self.primitive_config_provider = PrimitiveConfigProvider(path_provider=path_provider)

    def update(self, *args, **kwargs):
        wandb.config.update(*args, **kwargs)
        self.primitive_config_provider.update(*args, **kwargs)

    def __setitem__(self, key, value):
        wandb.config[key] = value
        self.primitive_config_provider[key] = value

    def __contains__(self, key):
        return key in self.primitive_config_provider

    def get_config_of_previous_stage(self, stage_name, stage_id):
        return self.primitive_config_provider.get_config_of_previous_stage(stage_name=stage_name, stage_id=stage_id)

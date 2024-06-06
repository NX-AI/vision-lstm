from .base.config_provider_base import ConfigProviderBase


class NoopConfigProvider(ConfigProviderBase):
    def update(self, *args, **kwargs):
        pass

    def __setitem__(self, key, value):
        pass

    def __contains__(self, key):
        return False

    def get_config_of_previous_stage(self, stage_name, stage_id):
        return {}

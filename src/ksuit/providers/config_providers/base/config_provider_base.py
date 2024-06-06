class ConfigProviderBase:
    def update(self, *args, **kwargs):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __contains__(self, key):
        raise NotImplementedError

    def get_config_of_previous_stage(self, stage_name, stage_id):
        raise NotImplementedError

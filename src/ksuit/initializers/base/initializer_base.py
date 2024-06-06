import logging

from ksuit.providers import PathProvider


class InitializerBase:
    def __init__(self, path_provider: PathProvider = None):
        self.logger = logging.getLogger(type(self).__name__)
        self.path_provider = path_provider

        # check if children overwrite the correct method
        assert type(self).get_model_kwargs == InitializerBase.get_model_kwargs

    def init_weights(self, model):
        raise NotImplementedError

    def init_optim(self, model):
        pass

    def get_model_kwargs(self):
        kwargs = self._get_model_kwargs()

        # these properties should not be loaded and are not intended to be part of ctor_kwargs -> warn if present
        def _check(key):
            if key in kwargs:
                self.logger.warning(f"{key} should not be in ctor_kwargs")
                kwargs.pop(key)

        _check("input_shape")
        _check("output_shape")
        _check("optim_ctor")
        _check("is_frozen")
        _check("freezers")
        _check("initializers")
        _check("extractors")

        self.logger.info(f"loaded model kwargs: {kwargs}")
        return kwargs

    def _get_model_kwargs(self):
        return {}

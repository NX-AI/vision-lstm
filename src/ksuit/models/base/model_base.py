import logging

from torch import nn

from ksuit.factory import MasterFactory
from ksuit.providers.path_provider import PathProvider
from ksuit.utils.data_container import DataContainer
from ksuit.utils.naming_utils import snake_type_name
from ksuit.utils.update_counter import UpdateCounter


class ModelBase(nn.Module):
    def __init__(
            self,
            input_shape=None,
            output_shape=None,
            ctor_kwargs=None,
            update_counter: UpdateCounter = None,
            path_provider: PathProvider = None,
            data_container: DataContainer = None,
            initializers=None,
            dynamic_ctx: dict = None,
            static_ctx: dict = None,
    ):
        # non-int types lead to errors with serialization/logging -> force primitive int type
        # non-primitive int types can occour if e.g. types are calculated via np.prod(...) which gives np.int
        if input_shape is not None:
            assert all(dim is None or isinstance(dim, int) for dim in input_shape), \
                f"{input_shape} ({', '.join(type(dim).__name__ for dim in input_shape)},)"
        if output_shape is not None:
            assert all(dim is None or isinstance(dim, int) for dim in output_shape), \
                f"invalid output_shape: {output_shape}"
        super().__init__()
        self.logger = logging.getLogger(type(self).__name__)
        self.name = snake_type_name(self)
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.update_counter = update_counter
        self.path_provider = path_provider
        self.data_container = data_container
        self._optim = None
        self.initializers = MasterFactory.get("initializer").create_list(
            initializers,
            path_provider=self.path_provider,
        )
        # a context allows extractors to store activations for later pooling (e.g. use features from last 4 layers)
        # the context has to be cleared manually after every call (e.g. model.features) to avoid memory leaks
        # "self.outputs = outputs or {}" does not work here as an empty dictionary evaluates to false
        if dynamic_ctx is None:
            self.dynamic_ctx = {}
        else:
            self.dynamic_ctx = dynamic_ctx
        # static_ctx allows composite models to propagate information between them (e.g. patch_size, latent_dim, ...)
        if static_ctx is None:
            self.static_ctx = {}
            if self.input_shape is not None:
                self.static_ctx["input_shape"] = tuple(self.input_shape)
        else:
            self.static_ctx = static_ctx
            if self.input_shape is None and "input_shape" in self.static_ctx:
                self.input_shape = self.static_ctx["input_shape"]

        # store the kwargs that are relevant
        self.ctor_kwargs = ctor_kwargs
        # don't save update_counter in ctor_kwargs
        if self.ctor_kwargs is not None and "update_counter" in self.ctor_kwargs:
            self.ctor_kwargs.pop("update_counter")
        # flag to make sure the model was initialized before wrapping into DDP
        # (parameters/buffers are synced in __init__ of DDP, so if model is not initialized before that,
        # different ranks will have diffefent parameters because the seed is different for every rank)
        # different seeds per rank are needed to avoid stochastic processes being the same across devices
        # (e.g. if seeds are equal, all masks for MAE are the same per batch)
        self.is_initialized = False

    @property
    def submodels(self):
        raise NotImplementedError

    def clear_buffers(self):
        raise NotImplementedError

    def set_accumulation_steps(self, accumulation_steps):
        raise NotImplementedError

    @property
    def is_batch_size_dependent(self):
        raise NotImplementedError

    def initialize(self, lr_scale_factor=None):
        self.initialize_weights()
        self.initialize_optim(lr_scale_factor=lr_scale_factor)
        self.apply_initializers()
        self.after_initializers()
        self.is_initialized = True
        return self

    def initialize_weights(self):
        raise NotImplementedError

    def apply_initializers(self):
        raise NotImplementedError

    def after_initializers(self):
        raise NotImplementedError

    def initialize_optim(self, lr_scale_factor=None):
        raise NotImplementedError

    @property
    def optim(self):
        return self._optim

    def optim_step(self, grad_scaler):
        raise NotImplementedError

    def optim_schedule_step(self):
        raise NotImplementedError

    def optim_zero_grad(self, set_to_none=True):
        raise NotImplementedError

    @property
    def device(self):
        raise NotImplementedError

    def before_accumulation_step(self):
        """ before_accumulation_step hook (e.g. for freezers) """
        for submodel in self.submodels.values():
            if submodel is None:
                continue
            submodel.before_accumulation_step()

    def after_update_step(self):
        """ after_update_step hook (e.g. for EMA) """
        pass

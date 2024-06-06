from ksuit.factory import MasterFactory
from ksuit.utils.select_with_path import select_with_path


class ExtractorBase:
    def __init__(
            self,
            pooling=None,
            finalizer=None,
            raise_exception=False,
            model_path=None,
            hook_kwargs=None,
            static_ctx=None,
            output_path=None,
            add_model_path_to_repr=True,
    ):
        self.pooling = MasterFactory.get("pooling").create(pooling, static_ctx=static_ctx)
        self.finalizer = MasterFactory.get("finalizer").create(finalizer)
        self.raise_exception = raise_exception
        self.model_path = model_path
        self.hook_kwargs = hook_kwargs or {}
        self.static_ctx = static_ctx
        self.add_model_path_to_repr = add_model_path_to_repr
        self.output_path = output_path
        #
        self.hooks = []
        self.registered_hooks = False
        self.outputs = {}

    def __enter__(self):
        self.enable_hooks()

    def __exit__(self, *_, **__):
        self.disable_hooks()

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.add_model_path_to_repr and self.model_path is not None:
            model_path = f"{self.model_path}."
        else:
            model_path = ""
        finalize_str = "" if self.finalizer is None else f".{str(self.finalizer)}"
        if self.pooling is None:
            pooling_str = ""
        else:
            pooling_str = f".{self.pooling}"
        return f"{model_path}{self.to_string()}{pooling_str}{finalize_str}"

    def to_string(self):
        return type(self).__name__

    def register_hooks(self, model):
        assert len(self.hooks) == 0
        assert not self.registered_hooks
        model = select_with_path(obj=model, path=self.model_path)
        assert model is not None, f"model.{self.model_path} is None"
        self._register_hooks(model)
        self.registered_hooks = True
        return self

    def _register_hooks(self, model):
        raise NotImplementedError

    def enable_hooks(self):
        for hook in self.hooks:
            hook.enabled = True

    def disable_hooks(self):
        for hook in self.hooks:
            hook.enabled = False

    def _get_own_outputs(self):
        return [self.outputs[hook.key] for hook in self.hooks]

    def extract(self):
        assert len(self.outputs) > 0, f"nothing was propagated through the module where ForwardHook was registered"
        features = []
        for output in self._get_own_outputs():
            if self.output_path is not None:
                output = select_with_path(obj=output, path=self.output_path)
            if self.pooling is not None:
                output = self.pooling(output, ctx=self.static_ctx)
            features.append(output)
        if self.finalizer is not None:
            features = self.finalizer(features)
        self.outputs.clear()
        return features

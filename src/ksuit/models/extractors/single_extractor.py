from .base import ExtractorBase, ForwardHook


class SingleExtractor(ExtractorBase):
    def _register_hooks(self, model):
        hook = ForwardHook(
            outputs=self.outputs,
            raise_exception=self.raise_exception,
            **self.hook_kwargs,
        )
        model.register_forward_hook(hook)
        self.hooks.append(hook)

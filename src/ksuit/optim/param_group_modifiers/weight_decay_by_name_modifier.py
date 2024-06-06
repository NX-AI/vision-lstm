from .base import ParamGroupModifierBase


class WeightDecayByNameModifier(ParamGroupModifierBase):
    """
    Changes the weight decay value for a single parameter
    Use-cases:
    - ViT exclude CLS token parameters
    - Transformer learned positional embeddings
    - Learnable query tokens for cross attention ("PerceiverPooling")
    """

    def __init__(self, name, value=0.):
        super().__init__()
        self.name = name
        self.value = value
        self.param_was_found = False

    def get_properties(self, model, name, param):
        if name == self.name:
            assert not self.param_was_found, f"found two parameters matching name '{self.name}'"
            self.param_was_found = True
            return dict(weight_decay=self.value)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name})"

    def was_applied_successfully(self):
        return self.param_was_found

from .base import ParamGroupModifierBase


class LrScaleByNameModifier(ParamGroupModifierBase):
    def __init__(self, scale, name):
        super().__init__()
        self.scale = scale
        self.name = name
        self.param_was_found = False

    def get_properties(self, model, name, param):
        if name == self.name:
            assert not self.param_was_found, f"found two parameters matching name '{self.name}'"
            self.param_was_found = True
            return dict(lr_scale=self.scale)
        return {}

    def __str__(self):
        return f"{type(self).__name__}(name={self.name},scale={self.scale})"

    def was_applied_successfully(self):
        return self.param_was_found

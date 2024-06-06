class ParamGroupModifierBase:
    def get_properties(self, model, name, param):
        raise NotImplementedError

    def __repr__(self):
        return str(self)

    def __str__(self):
        raise NotImplementedError

    def was_applied_successfully(self):
        return True

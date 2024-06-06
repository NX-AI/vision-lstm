from ksuit.factory import MasterFactory
from ksuit.models import SingleModel


class LossBase(SingleModel):
    def __init__(self, weight=None, is_frozen=True, **kwargs):
        super().__init__(is_frozen=is_frozen, allow_frozen_train_mode=True, **kwargs)
        self.weight = MasterFactory.get("schedule").create(weight, update_counter=self.update_counter)

    def get_weight(self):
        if self.weight is None:
            return 1.
        return self.weight.get_value()

    def forward(self, *args, **kwargs):
        raise NotImplementedError

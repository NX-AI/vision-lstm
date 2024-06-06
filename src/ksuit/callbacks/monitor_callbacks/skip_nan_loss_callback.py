import torch

from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class SkipNanLossCallback(PeriodicCallback):
    def __init__(self, max_skipped_updates_in_a_row=50, **kwargs):
        super().__init__(**kwargs)
        self.max_skipped_updates_in_a_row = max_skipped_updates_in_a_row
        self.skipped_updates_in_a_row = 0
        self.skip_next_update = False

    def state_dict(self):
        return dict(skipped_updates_in_a_row=self.skipped_updates_in_a_row)

    def load_state_dict(self, state_dict):
        self.skipped_updates_in_a_row = state_dict["skipped_updates_in_a_row"]

    def after_every_backward(self, total_loss, **kwargs):
        if torch.isnan(total_loss):
            self.logger.info(f"encountered NaN loss -> skip update")
            self.skip_next_update = True

    def before_every_optim_step(self, model, **kwargs):
        if self.skip_next_update:
            if self.skipped_updates_in_a_row > self.max_skipped_updates_in_a_row:
                raise RuntimeError(f"skipped {self.max_skipped_updates_in_a_row} in a row due to NaN loss -> exit")
            self.skipped_updates_in_a_row += 1
        else:
            self.skipped_updates_in_a_row = 0

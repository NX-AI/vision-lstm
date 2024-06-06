from distributed.config import get_rank

from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class KillOnLossSpikeCallback(PeriodicCallback):
    def __init__(self, tolerance_factor=10, recovery_tolerance=5, **kwargs):
        super().__init__(**kwargs)
        self.tolerance_factor = tolerance_factor
        self.recovery_tolerance = recovery_tolerance
        self.best_loss = float("inf")
        self.recovery_tolerance_counter = 0

    def state_dict(self):
        return dict(
            best_loss=self.best_loss,
            recovery_tolerance_counter=self.recovery_tolerance_counter,
        )

    def load_state_dict(self, state_dict):
        if state_dict is None:
            self.logger.error(f"state_dict of KillOnLossSpikeCallback is None on rank {get_rank()} -> skip loading")
            return
        self.best_loss = state_dict["best_loss"]
        self.recovery_tolerance_counter = state_dict["recovery_tolerance_counter"]

    def _periodic_callback(self, model, **kwargs):
        # extract loss from log_cache (produced by OnlineLossCallback)
        loss = self.writer.log_cache[f"loss/online/total/{self.to_short_interval_string()}"]
        if self.recovery_tolerance_counter > 0:
            # loss has <recovery_tolerance> log intervals time to recover after a loss spike
            if loss < self.best_loss:
                # loss recovered
                self.recovery_tolerance_counter = 0
                self.best_loss = loss
                self.logger.info(f"loss recovered from spike")
            else:
                # loss didnt recover -> increase counter
                self.recovery_tolerance_counter += 1
                self.logger.info(f"loss hasnt recovered from spike")
                self.logger.info(f"tolerance: {self.recovery_tolerance_counter}/{self.recovery_tolerance}")
                if self.recovery_tolerance_counter >= self.recovery_tolerance:
                    # tolerance exceeded
                    raise RuntimeError(f"couldnt recover from loss spike within tolerance")
        elif loss > self.tolerance_factor * self.best_loss:
            # detect loss spikes
            if self.recovery_tolerance == 0:
                # no tolerance -> instantly kill
                raise RuntimeError(
                    f"loss is higher than {self.tolerance_factor} * best loss "
                    f"({loss:.6f} > {self.tolerance_factor * self.best_loss:.6f})"
                )
            # start tracking tolerance log intervals
            self.recovery_tolerance_counter += 1
            self.logger.warning(f"detected loss spike -> loss has {self.recovery_tolerance} intervals to recover")
        else:
            # update best_loss
            if loss < self.best_loss:
                self.best_loss = loss

from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class DebugMonitorCallback(PeriodicCallback):
    def __init__(self, verbose=False, **kwargs):
        super().__init__(**kwargs)
        self.verbose = verbose

    def before_every_update(self, model, **kwargs):
        for name, param in model.named_parameters():
            norm = param.norm().item()
            absmax = param.abs().max().item()
            mean = param.mean().item()
            self.writer.add_scalar(f"param/{name}/norm", norm, logger=self.logger if self.verbose else None)
            self.writer.add_scalar(f"param/{name}/absmax", absmax, logger=self.logger if self.verbose else None)
            self.writer.add_scalar(f"param/{name}/mean", mean, logger=self.logger if self.verbose else None)

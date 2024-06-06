from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class WeightMonitorCallback(PeriodicCallback):
    def _track_after_update_step(self, model, **kwargs):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            param_norm = param.norm()
            param_absmax = param.abs().max()
            param_mean = param.abs().mean()
            self.writer.add_scalar(f"weight/{name}/norm", param_norm)
            self.writer.add_scalar(f"weight/{name}/absmax", param_absmax)
            self.writer.add_scalar(f"weight/{name}/mean", param_mean)
            if param.numel() > 1:
                param_std = param.std()
                self.writer.add_scalar(f"weight/{name}/std", param_std)

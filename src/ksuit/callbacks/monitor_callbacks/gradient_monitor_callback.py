from ksuit.callbacks.base.periodic_callback import PeriodicCallback


class GradientMonitorCallback(PeriodicCallback):
    def before_every_optim_step(self, model, **kwargs):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            grad_norm = param.grad.norm()
            grad_absmax = param.grad.abs().max()
            grad_mean = param.grad.abs().mean()
            grad_std = param.grad.std()
            self.writer.add_scalar(f"gradient/{name}/norm", grad_norm)
            self.writer.add_scalar(f"gradient/{name}/absmax", grad_absmax)
            self.writer.add_scalar(f"gradient/{name}/mean", grad_mean)
            if grad_std.numel() > 1:
                self.writer.add_scalar(f"gradient/{name}/std", grad_std)

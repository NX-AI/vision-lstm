class LrScalerBase:
    def __str__(self):
        raise NotImplementedError

    def scale_lr(self, base_lr, lr_scale_factor):
        raise NotImplementedError

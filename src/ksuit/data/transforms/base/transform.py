class Transform:
    @property
    def is_deterministic(self):
        return True

    def set_rng(self, rng):
        return self

    def worker_init_fn(self, *args, **kwargs):
        pass

    def __call__(self, x, ctx=None):
        raise NotImplementedError

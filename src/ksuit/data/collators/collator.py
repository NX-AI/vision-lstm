from torch.utils.data import default_collate


class Collator:
    def worker_init_fn(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        return default_collate(batch)

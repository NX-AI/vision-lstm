from .collator import Collator


class ComposeCollator(Collator):
    def __init__(self, collators):
        self.collators = collators

    def __call__(self, batch):
        for collator in self.collators:
            batch = collator(batch)
        return batch

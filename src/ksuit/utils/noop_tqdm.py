class NoopTqdm:
    def __init__(self, iterable):
        self.iterable = iterable

    def __enter__(self):
        return self

    def __exit__(self, *_, **__):
        pass

    def noop(self, *_, **__):
        pass

    def __getattr__(self, item):
        return self.noop

    def __iter__(self):
        yield from self.iterable

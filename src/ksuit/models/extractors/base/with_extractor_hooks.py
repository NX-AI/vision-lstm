from contextlib import contextmanager

from .stop_forward_exception import StopForwardException


@contextmanager
def with_extractor_hooks(extractors):
    assert isinstance(extractors, (tuple, list))
    for pooling in extractors:
        pooling.enable_hooks()
    try:
        yield
    except StopForwardException:
        for pooling in extractors:
            pooling.disable_hooks()
        raise
    for pooling in extractors:
        pooling.disable_hooks()

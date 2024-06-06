import uuid

from .stop_forward_exception import StopForwardException


class ForwardHook:
    def __init__(self, outputs: dict, raise_exception: bool = False):
        self.key = uuid.uuid4()
        self.outputs = outputs
        self.raise_exception = raise_exception
        self.enabled = True

    def __call__(self, _, __, output):
        if not self.enabled:
            return
        self.outputs[self.key] = output
        if self.raise_exception:
            raise StopForwardException()

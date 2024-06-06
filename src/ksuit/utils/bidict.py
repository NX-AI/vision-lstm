class Bidict:
    def __init__(self, forward=None, backward=None):
        assert forward is None or backward is None
        self._forward = {}
        self._backward = {}
        if forward is not None:
            for key, value in forward.items():
                self.set_forward(key, value)
        if backward is not None:
            for key, value in backward.items():
                self.set_backward(key, value)

    def to_forward(self):
        return self._forward.copy()

    def to_backward(self):
        return self._backward.copy()

    def get_forward(self, key):
        return self._forward[key]

    def get_backward(self, key):
        return self._backward[key]

    def set_forward(self, key, value):
        self._forward[key] = value
        self._backward[value] = key

    def set_backward(self, key, value):
        self._backward[key] = value
        self._forward[value] = key

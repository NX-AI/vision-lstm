from ksuit.factory import MasterFactory
from ksuit.pattern_matchers import PatternMatcher, FnmatchPatternMatcher
from .base import FreezerBase


class FullFreezer(FreezerBase):
    def __init__(
            self,
            exclude_patterns: list[str] = None,
            pattern_matcher: PatternMatcher = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.exclude_patterns = exclude_patterns
        self.pattern_matcher = MasterFactory.get("pattern_matcher").create(pattern_matcher) or FnmatchPatternMatcher()

    def __str__(self):
        return type(self).__name__

    def _update_state(self, model, requires_grad):
        model.train(requires_grad)
        if self.exclude_patterns is None:
            # freeze everything
            for param in model.parameters():
                param.requires_grad = requires_grad
        else:
            # dont freeze parameters that are excluded
            for name, param in model.named_parameters():
                if self.pattern_matcher.match_patterns(string=name, patterns=self.exclude_patterns):
                    # param is excluded
                    pass
                else:
                    # param is included
                    param.requires_grad = requires_grad

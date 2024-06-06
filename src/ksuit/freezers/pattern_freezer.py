from ksuit.factory import MasterFactory
from ksuit.pattern_matchers import PatternMatcher, FnmatchPatternMatcher
from .base.freezer_base import FreezerBase


class PatternFreezer(FreezerBase):
    def __init__(
            self,
            patterns: list[str],
            pattern_matcher: PatternMatcher = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.patterns = patterns
        self.pattern_matcher = MasterFactory.get("pattern_matcher").create(pattern_matcher) or FnmatchPatternMatcher()

    def _update_state(self, model, requires_grad):
        modules_to_freeze = []
        for name, module in model.named_modules():
            if name == "":
                continue
            # freeze all modules that match
            if self.patterns is not None:
                if self.pattern_matcher.match_patterns(string=name, patterns=self.patterns):
                    modules_to_freeze.append(module)

        for module in modules_to_freeze:
            module.train(requires_grad)
            for p in module.parameters():
                p.requires_grad = requires_grad

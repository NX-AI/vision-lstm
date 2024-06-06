import logging

from ksuit.factory import MasterFactory
from ksuit.pattern_matchers import PatternMatcher, FnmatchPatternMatcher


class MetricPropertyProvider:
    def __init__(self, pattern_matcher: PatternMatcher = None):
        self.logger = logging.getLogger(type(self).__name__)
        self.pattern_matcher = MasterFactory.get("pattern_matcher").create(pattern_matcher) or FnmatchPatternMatcher()
        self.lower_is_better_patterns = [
            "loss/*",
        ]
        self.higher_is_better_patterns = [
            "accuracy*",
            "knn_accuracy/*",
            "auroc/*",
            "miou/*",
            "acc*",
            "iou/*",
            "dice/*",
            "all_acc*",
        ]
        self.neutral_patterns = [
            "profiler/*",
            "optim/*",
            "freezers/*",
            "profiling/*",
            "ctx/*",
            "loss_weight/*",
            "gradient/*",
            "detach/*",
        ]

    def is_neutral_key(self, key):
        return self.pattern_matcher.match_patterns(string=key.lower(), patterns=self.neutral_patterns)

    def higher_is_better(self, key):
        if self.pattern_matcher.match_patterns(string=key.lower(), patterns=self.higher_is_better_patterns):
            return True
        if self.pattern_matcher.match_patterns(string=key.lower(), patterns=self.lower_is_better_patterns):
            return False
        self.logger.warning(f"{key} has no defined behavior for higher_is_better -> using True")
        return True

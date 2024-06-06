import fnmatch

from .base import PatternMatcher


class FnmatchPatternMatcher(PatternMatcher):
    @staticmethod
    def match_pattern(string: str, pattern: str) -> bool:
        return fnmatch.fnmatch(name=string, pat=pattern)

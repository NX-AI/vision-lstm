import re

from .base import PatternMatcher


class RegexPatternMatcher(PatternMatcher):
    @staticmethod
    def match_pattern(string: str, pattern: str) -> bool:
        return re.search(string=string, pattern=pattern)

class PatternMatcher:
    def match_pattern(self, string: str, pattern: str) -> bool:
        raise NotImplementedError

    def match_patterns(self, string: str, patterns: list[str]) -> bool:
        assert isinstance(patterns, list)
        for pattern in patterns:
            if self.match_pattern(string=string, pattern=pattern):
                return True
        return False

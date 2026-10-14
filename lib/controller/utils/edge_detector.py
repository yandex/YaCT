class EdgeDetector:
    """Tracks multiple boolean signals, returns True only on rising edge."""

    def __init__(self):
        self._prev = {}

    def rising(self, name: str, value: bool) -> bool:
        prev = self._prev.get(name, False)
        self._prev[name] = value
        return value and not prev

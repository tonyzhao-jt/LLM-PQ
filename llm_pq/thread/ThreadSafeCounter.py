import threading
class ThreadSafeCounter:
    """Thread-safe counter."""

    def __init__(self, value: int=0):
        self._value = value
        self._cond = threading.Condition()

    @property
    def value(self) -> int:
        """Current counter value."""
        with self._cond:
            val = self._value
            self._cond.notify_all()
        return val

    def add(self, quantity: int=1) -> None:
        """Add to counter atomically."""
        with self._cond:
            self._value += quantity
            self._cond.notify_all()

    def set(self, value: int=0) -> None:
        """Set (or reset) counter value."""
        with self._cond:
            self._value = value
            self._cond.notify_all()

    def wait_gte(self, threshold: int) -> None:
        """Wait until counter >= threshold."""
        with self._cond:
            while self._value < threshold:
                self._cond.wait()

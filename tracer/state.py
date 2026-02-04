"""Global state management for Tracer."""

import threading
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from tracer.config import TracerConfig
    from tracer.core.tracer import TrainingTracer


class GlobalState:
    """
    Thread-safe global state singleton.

    Stores the current Tracer configuration and instance. Uses double-checked
    locking pattern for thread-safe singleton initialization.
    """

    _instance: Optional["GlobalState"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "GlobalState":
        """Create or return singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        """Initialize state (only once)."""
        if self._initialized:
            return

        self.config: Optional["TracerConfig"] = None
        self.tracer: Optional["TrainingTracer"] = None
        self.run_id: Optional[str] = None
        self.step_count: int = 0

        self._initialized = True

    def reset(self) -> None:
        """Reset global state."""
        self.config = None
        self.tracer = None
        self.run_id = None
        self.step_count = 0

    def is_initialized(self) -> bool:
        """Check if Tracer is currently initialized."""
        return self.tracer is not None


# Global singleton instance
_state = GlobalState()

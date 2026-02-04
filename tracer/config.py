"""Configuration management for Tracer."""

import datetime
import socket
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class TracerConfig:
    """
    Global configuration for Tracer.

    Attributes:
        project_name: Project identifier for grouping runs
        run_name: Unique run name (auto-generated if None)
        tags: List of tags for organizing runs
        notes: Optional notes about this run
        storage_path: Path to storage directory
        retention_days: Days to keep run data before cleanup
        checkpoint_interval: Steps between checkpoints
        capture_samples: Number of samples to capture per checkpoint
        track_gradients: Enable gradient statistics tracking
        track_system_metrics: Enable CPU/GPU/memory tracking
        gradient_layers: Specific layers to track (None = all)
        async_writes: Use async background writes
        write_buffer_size: Buffer size for async writes
        max_workers: Thread pool size for async operations
        custom_metrics: Additional user-defined configuration
    """

    # Project metadata
    project_name: str
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None

    # Storage configuration
    storage_path: str = "./.tracer"
    retention_days: int = 30

    # Capture settings
    checkpoint_interval: int = 1000
    capture_samples: int = 5
    track_gradients: bool = True
    track_system_metrics: bool = True

    # Gradient tracking
    gradient_layers: Optional[List[str]] = None  # None = all layers

    # Performance
    async_writes: bool = True
    write_buffer_size: int = 100
    max_workers: int = 4

    # Advanced
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = self._generate_run_name()

        # Ensure storage path is absolute
        self.storage_path = str(Path(self.storage_path).resolve())

        # Validate numeric values
        if self.checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be >= 1")
        if self.capture_samples < 0:
            raise ValueError("capture_samples must be >= 0")
        if self.retention_days < 0:
            raise ValueError("retention_days must be >= 0")
        if self.write_buffer_size < 1:
            raise ValueError("write_buffer_size must be >= 1")
        if self.max_workers < 1:
            raise ValueError("max_workers must be >= 1")

    @staticmethod
    def _generate_run_name() -> str:
        """
        Generate a unique run name.

        Returns:
            Generated run name in format: run_YYYYMMDD_HHMMSS_hostname
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname().split(".")[0]  # Short hostname
        return f"run_{timestamp}_{hostname}"

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Returns:
            Configuration as dict
        """
        from dataclasses import asdict

        return asdict(self)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TracerConfig(project={self.project_name}, "
            f"run={self.run_name}, "
            f"checkpoint_interval={self.checkpoint_interval})"
        )

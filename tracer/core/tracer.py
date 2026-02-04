"""Main Tracer orchestrator."""

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Any, Dict, Optional

import torch.nn as nn

from tracer.config import TracerConfig
from tracer.core.hooks import HookManager
from tracer.core.system_metrics import SystemMetrics
from tracer.storage.backend import StorageBackend


class TrainingTracer:
    """
    Main orchestrator for training observation.

    Coordinates hooks, metrics collection, and storage to provide comprehensive
    training observability.
    """

    def __init__(self, config: TracerConfig) -> None:
        """
        Initialize training tracer.

        Args:
            config: Tracer configuration
        """
        self.config = config
        self.run_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.step_count = 0
        self.checkpoint_count = 0

        # Components
        self.hook_manager = HookManager(config)
        self.system_metrics = SystemMetrics(config) if config.track_system_metrics else None
        self.storage = StorageBackend(config)

        # Metric buffer
        self.metric_buffer: Dict[str, Any] = {}
        self._buffer_lock = threading.Lock()

        # Async writes
        self.write_queue: Optional[Queue[Optional[Dict[str, Any]]]] = None
        self.executor: Optional[ThreadPoolExecutor] = None
        self.writer_thread: Optional[threading.Thread] = None

        if config.async_writes:
            self.write_queue = Queue(maxsize=config.write_buffer_size)
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            self._start_async_writer()

        self.model: Optional[nn.Module] = None

        # Initialize run in storage
        self._initialize_run()

    def _initialize_run(self) -> None:
        """Create run record in storage."""
        self.storage.create_run(
            run_id=self.run_id,
            project_name=self.config.project_name,
            run_name=self.config.run_name or "unknown",
            start_time=self.start_time,
            config=self.config.to_dict(),
        )

        print(f"🔍 Tracer initialized: {self.config.run_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Storage: {self.config.storage_path}")

    def attach_hooks(self, model: nn.Module) -> None:
        """
        Attach tracking hooks to model.

        Args:
            model: PyTorch model to track
        """
        self.model = model
        self.hook_manager.attach_all(model)

    def log(self, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for current step.

        Args:
            metrics: Metrics to log
        """
        with self._buffer_lock:
            self.metric_buffer.update(metrics)

    def step(self) -> None:
        """Increment step and checkpoint if needed."""
        self.step_count += 1

        if self.step_count % self.config.checkpoint_interval == 0:
            self._checkpoint()

    def _checkpoint(self) -> None:
        """Create a checkpoint."""
        checkpoint_data: Dict[str, Any] = {
            "run_id": self.run_id,
            "step": self.step_count,
            "timestamp": time.time(),
            "metrics": {},
            "gradient_stats": {},
            "system_metrics": {},
            "samples": None,
        }

        # Get metrics from buffer
        with self._buffer_lock:
            checkpoint_data["metrics"] = self.metric_buffer.copy()
            self.metric_buffer.clear()

        # Get gradients
        if self.config.track_gradients:
            hook_data = self.hook_manager.get_all_data()
            checkpoint_data["gradient_stats"] = hook_data.get("GradientHook", {})
            self.hook_manager.reset_all()

        # Get system metrics
        if self.config.track_system_metrics and self.system_metrics:
            checkpoint_data["system_metrics"] = self.system_metrics.capture()

        # Write (async or sync)
        if self.config.async_writes and self.write_queue is not None:
            self.write_queue.put(checkpoint_data)
        else:
            self._write_checkpoint(checkpoint_data)

        self.checkpoint_count += 1

    def _write_checkpoint(self, data: Dict[str, Any]) -> None:
        """
        Write checkpoint to storage.

        Args:
            data: Checkpoint data
        """
        self.storage.write_checkpoint(
            run_id=data["run_id"],
            step=data["step"],
            timestamp=data["timestamp"],
            metrics=data["metrics"],
            gradient_stats=data["gradient_stats"],
            system_metrics=data["system_metrics"],
            samples=data["samples"],
        )

    def _start_async_writer(self) -> None:
        """Start background writer thread."""

        def writer_loop() -> None:
            """Worker loop for async writes."""
            while True:
                data = self.write_queue.get() if self.write_queue else None  # type: ignore
                if data is None:
                    break
                self._write_checkpoint(data)

        self.writer_thread = threading.Thread(target=writer_loop, daemon=False)
        self.writer_thread.start()

    def finish(self) -> None:
        """Finalize run and cleanup."""
        # Final checkpoint if there are buffered metrics
        if self.metric_buffer:
            self._checkpoint()

        # Stop async writer
        if self.config.async_writes and self.write_queue is not None:
            self.write_queue.put(None)
            if self.writer_thread:
                self.writer_thread.join(timeout=30)
            if self.executor:
                self.executor.shutdown(wait=True)

        # Detach hooks
        self.hook_manager.detach_all()

        # Update run status
        self.storage.finalize_run(
            run_id=self.run_id,
            end_time=time.time(),
            status="completed",
            total_steps=self.step_count,
        )

        duration = time.time() - self.start_time
        print(f"\n✓ Run completed: {self.config.run_name}")
        print(f"  Steps: {self.step_count}")
        print(f"  Checkpoints: {self.checkpoint_count}")
        print(f"  Duration: {duration:.2f}s")

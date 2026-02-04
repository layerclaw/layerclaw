"""
Tracer - Deep Training Observability for PyTorch

A lightweight, powerful observability tool for tracking, debugging, and optimizing
PyTorch training runs with comprehensive gradient tracking, anomaly detection,
and beautiful CLI tools.

Example:
    >>> import tracer
    >>> tracer.init(project="my-project", checkpoint_interval=100)
    >>> # Your training loop
    >>> tracer.log({"loss": 0.5})
    >>> tracer.step()
    >>> tracer.finish()
"""

__version__ = "0.1.0"
__author__ = "Tracer Contributors"
__license__ = "MIT"

from typing import Any, Dict, Optional

from tracer.config import TracerConfig
from tracer.core.tracer import TrainingTracer
from tracer.state import _state

__all__ = [
    "__version__",
    "init",
    "log",
    "step",
    "finish",
    "get_run_id",
    "get_step",
    "watch",
    "TracerConfig",
    "TrainingTracer",
]


def init(
    project: str,
    run_name: Optional[str] = None,
    **kwargs: Any,
) -> TrainingTracer:
    """
    Initialize Tracer for a training run.

    Creates a new training run with the specified configuration and begins
    tracking metrics, gradients, and system resources.

    Args:
        project: Project name for grouping related runs
        run_name: Unique run identifier (auto-generated if None)
        **kwargs: Additional configuration options (see TracerConfig)

    Returns:
        TrainingTracer instance for this run

    Raises:
        RuntimeError: If Tracer is already initialized

    Example:
        >>> import tracer
        >>> tracer.init(
        ...     project="my-llm",
        ...     run_name="experiment-1",
        ...     checkpoint_interval=500,
        ...     capture_samples=10,
        ...     track_gradients=True
        ... )
    """
    if _state.tracer is not None:
        raise RuntimeError(
            "Tracer already initialized. Call tracer.finish() before starting a new run."
        )

    config_dict: Dict[str, Any] = {"project_name": project}

    if run_name:
        config_dict["run_name"] = run_name

    config_dict.update(kwargs)

    # Create config
    config = TracerConfig(**config_dict)

    # Create tracer
    tracer_instance = TrainingTracer(config)

    # Store in global state
    _state.config = config
    _state.tracer = tracer_instance
    _state.run_id = tracer_instance.run_id

    return tracer_instance


def log(metrics: Dict[str, Any]) -> None:
    """
    Log metrics for the current step.

    Metrics are buffered and written to storage at the next checkpoint.

    Args:
        metrics: Dictionary of metric_name -> value pairs

    Raises:
        RuntimeError: If Tracer not initialized

    Example:
        >>> tracer.log({"loss": 2.34, "accuracy": 0.89, "lr": 1e-4})
    """
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")

    _state.tracer.log(metrics)


def step() -> None:
    """
    Increment step counter and checkpoint if needed.

    Should be called once per training step. Automatically triggers
    checkpointing based on configured checkpoint_interval.

    Raises:
        RuntimeError: If Tracer not initialized

    Example:
        >>> for batch in dataloader:
        ...     loss = train_step(batch)
        ...     tracer.log({"loss": loss.item()})
        ...     tracer.step()  # Increment and maybe checkpoint
    """
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")

    _state.tracer.step()
    _state.step_count = _state.tracer.step_count


def finish() -> None:
    """
    Finalize current run and flush all data.

    Writes any remaining buffered data, detaches hooks, and closes
    storage connections. Always call this at the end of training.

    Raises:
        RuntimeError: If Tracer not initialized

    Example:
        >>> try:
        ...     # Training loop
        ...     for epoch in range(epochs):
        ...         train()
        ... finally:
        ...     tracer.finish()  # Always cleanup
    """
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")

    _state.tracer.finish()
    _state.reset()


def get_run_id() -> str:
    """
    Get the current run ID.

    Returns:
        Unique identifier for current run

    Raises:
        RuntimeError: If Tracer not initialized
    """
    if _state.run_id is None:
        raise RuntimeError("Tracer not initialized.")
    return _state.run_id


def get_step() -> int:
    """
    Get the current step count.

    Returns:
        Number of steps since run started
    """
    return _state.step_count


def watch(model: Any) -> None:
    """
    Attach gradient tracking hooks to a model.

    Convenience function to attach hooks after initialization.
    Alternatively, hooks can be attached via tracer._state.tracer.attach_hooks().

    Args:
        model: PyTorch model to watch

    Raises:
        RuntimeError: If Tracer not initialized

    Example:
        >>> model = MyModel()
        >>> tracer.init(project="test")
        >>> tracer.watch(model)  # Start tracking gradients
    """
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")

    _state.tracer.attach_hooks(model)

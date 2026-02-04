"""PyTorch hook implementations for gradient tracking."""

import threading
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn

from tracer.config import TracerConfig


class GradientHook:
    """
    Captures gradient statistics during backpropagation.

    Registers hooks on model parameters to track gradient norms, statistics,
    and anomalies (NaN/Inf) in real-time.
    """

    def __init__(self, config: TracerConfig) -> None:
        """
        Initialize gradient hook.

        Args:
            config: Tracer configuration
        """
        self.config = config
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.gradient_data: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()

    def attach(self, model: nn.Module) -> None:
        """
        Attach hooks to model parameters.

        Args:
            model: PyTorch model to attach hooks to
        """
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue

            # Filter layers if specified
            if self.config.gradient_layers is not None:
                if not any(layer in name for layer in self.config.gradient_layers):
                    continue

            # Register backward hook
            handle = param.register_hook(self._create_hook_fn(name))
            self.handles.append(handle)

    def _create_hook_fn(self, param_name: str) -> Callable[[torch.Tensor], None]:
        """
        Create hook function for parameter.

        Args:
            param_name: Name of the parameter

        Returns:
            Hook function
        """

        def hook_fn(grad: torch.Tensor) -> None:
            """Capture gradient statistics."""
            self._capture_gradient(param_name, grad)

        return hook_fn

    def _capture_gradient(self, param_name: str, grad: Optional[torch.Tensor]) -> None:
        """
        Capture gradient statistics.

        Args:
            param_name: Name of parameter
            grad: Gradient tensor
        """
        if grad is None:
            return

        with torch.no_grad():
            # Calculate statistics
            # Handle edge case where grad has only one element (std would fail)
            std_val = grad.std().item() if grad.numel() > 1 else 0.0
            
            stats: Dict[str, float] = {
                "norm": grad.norm().item(),
                "mean": grad.mean().item(),
                "std": std_val,
                "max": grad.max().item(),
                "min": grad.min().item(),
                "num_zeros": int((grad == 0).sum().item()),
                "num_nans": int(torch.isnan(grad).sum().item()),
                "num_infs": int(torch.isinf(grad).sum().item()),
            }

        with self._lock:
            self.gradient_data[param_name] = stats

    def detach(self) -> None:
        """Remove all hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def get_data(self) -> Dict[str, Dict[str, float]]:
        """
        Get captured gradient data.

        Returns:
            Dictionary mapping parameter names to gradient statistics
        """
        with self._lock:
            return self.gradient_data.copy()

    def reset(self) -> None:
        """Clear captured data."""
        with self._lock:
            self.gradient_data.clear()


class HookManager:
    """
    Manages all hooks for a training run.

    Coordinates multiple hook types (gradients, activations, etc.) and
    provides unified interface for attachment/detachment.
    """

    def __init__(self, config: TracerConfig) -> None:
        """
        Initialize hook manager.

        Args:
            config: Tracer configuration
        """
        self.config = config
        self.hooks: List[Any] = []

        # Register hooks based on config
        if config.track_gradients:
            self.hooks.append(GradientHook(config))

    def attach_all(self, model: nn.Module) -> None:
        """
        Attach all enabled hooks to model.

        Args:
            model: PyTorch model
        """
        for hook in self.hooks:
            hook.attach(model)

    def detach_all(self) -> None:
        """Detach all hooks."""
        for hook in self.hooks:
            hook.detach()

    def get_all_data(self) -> Dict[str, Any]:
        """
        Get data from all hooks.

        Returns:
            Dictionary mapping hook type to captured data
        """
        data: Dict[str, Any] = {}
        for hook in self.hooks:
            hook_type = type(hook).__name__
            data[hook_type] = hook.get_data()
        return data

    def reset_all(self) -> None:
        """Reset all hooks."""
        for hook in self.hooks:
            hook.reset()

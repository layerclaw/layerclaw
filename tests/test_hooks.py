"""Tests for hook implementations."""

import torch
import torch.nn as nn

from tracer.config import TracerConfig
from tracer.core.hooks import GradientHook, HookManager


class TestGradientHook:
    """Test GradientHook class."""

    def test_hook_attachment(self, tracer_config: TracerConfig, simple_model: nn.Module):
        """Test attaching hooks to model."""
        hook = GradientHook(tracer_config)
        hook.attach(simple_model)

        # Should have hooks for all trainable parameters
        assert len(hook.handles) > 0

    def test_gradient_capture(
        self, tracer_config: TracerConfig, simple_model: nn.Module, sample_data: tuple
    ):
        """Test gradient statistics capture."""
        X, y = sample_data

        hook = GradientHook(tracer_config)
        hook.attach(simple_model)

        # Forward and backward pass
        outputs = simple_model(X)
        loss = outputs.sum()
        loss.backward()

        # Check captured data
        data = hook.get_data()
        assert len(data) > 0

        # Check gradient statistics
        for param_name, stats in data.items():
            assert "norm" in stats
            assert "mean" in stats
            assert "std" in stats
            assert "max" in stats
            assert "min" in stats
            assert stats["norm"] > 0  # Should have gradients

    def test_hook_detachment(self, tracer_config: TracerConfig, simple_model: nn.Module):
        """Test hook removal."""
        hook = GradientHook(tracer_config)
        hook.attach(simple_model)

        assert len(hook.handles) > 0

        hook.detach()

        assert len(hook.handles) == 0

    def test_gradient_layer_filtering(self, temp_dir):
        """Test filtering specific layers."""
        config = TracerConfig(
            project_name="test",
            storage_path=str(temp_dir),
            gradient_layers=["linear1"],  # Only track linear1
        )

        model = nn.Sequential(nn.Linear(10, 20, bias=True), nn.Linear(20, 1, bias=True))

        hook = GradientHook(config)
        hook.attach(model)

        # Should only have hooks for linear1 (weight + bias = 2 parameters)
        # Since we're filtering by "linear1", no parameters will match in Sequential
        # Let's use a named model instead
        class NamedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 1)

            def forward(self, x):
                return self.linear2(self.linear1(x))

        named_model = NamedModel()
        hook2 = GradientHook(config)
        hook2.attach(named_model)

        # Should only track linear1 parameters
        assert len(hook2.handles) == 2  # weight and bias


class TestHookManager:
    """Test HookManager class."""

    def test_manager_initialization(self, tracer_config: TracerConfig):
        """Test hook manager initialization."""
        manager = HookManager(tracer_config)

        # Should have gradient hook if enabled
        assert len(manager.hooks) > 0

    def test_attach_all_hooks(
        self, tracer_config: TracerConfig, simple_model: nn.Module
    ):
        """Test attaching all hooks."""
        manager = HookManager(tracer_config)
        manager.attach_all(simple_model)

        # Hooks should be attached
        for hook in manager.hooks:
            if isinstance(hook, GradientHook):
                assert len(hook.handles) > 0

    def test_get_all_data(
        self,
        tracer_config: TracerConfig,
        simple_model: nn.Module,
        sample_data: tuple,
    ):
        """Test getting data from all hooks."""
        X, y = sample_data

        manager = HookManager(tracer_config)
        manager.attach_all(simple_model)

        # Forward and backward
        outputs = simple_model(X)
        loss = outputs.sum()
        loss.backward()

        # Get all data
        data = manager.get_all_data()

        assert "GradientHook" in data
        assert len(data["GradientHook"]) > 0

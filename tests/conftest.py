"""Pytest configuration and fixtures."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch
import torch.nn as nn

import tracer
from tracer.config import TracerConfig
from tracer.storage.backend import StorageBackend


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def tracer_config(temp_dir: Path) -> TracerConfig:
    """Create test configuration."""
    return TracerConfig(
        project_name="test_project",
        run_name="test_run",
        storage_path=str(temp_dir),
        checkpoint_interval=10,
        track_gradients=True,
        track_system_metrics=True,
        async_writes=False,  # Synchronous for easier testing
    )


@pytest.fixture
def storage_backend(tracer_config: TracerConfig) -> StorageBackend:
    """Create storage backend for tests."""
    return StorageBackend(tracer_config)


@pytest.fixture
def simple_model() -> nn.Module:
    """Create simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear1 = nn.Linear(10, 20)
            self.relu = nn.ReLU()
            self.linear2 = nn.Linear(20, 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.linear1(x)
            x = self.relu(x)
            return self.linear2(x)

    return SimpleModel()


@pytest.fixture
def sample_data() -> tuple:
    """Create sample input/target data."""
    X = torch.randn(32, 10)
    y = torch.randn(32, 1)
    return X, y


@pytest.fixture(autouse=True)
def reset_tracer_state() -> Generator[None, None, None]:
    """Reset Tracer global state before and after each test."""
    tracer._state.reset()
    yield
    tracer._state.reset()

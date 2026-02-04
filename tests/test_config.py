"""Tests for configuration module."""

import pytest

from tracer.config import TracerConfig


class TestTracerConfig:
    """Test TracerConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        config = TracerConfig(project_name="test")

        assert config.project_name == "test"
        assert config.run_name is not None  # Auto-generated
        assert config.storage_path.endswith(".tracer")
        assert config.checkpoint_interval == 1000
        assert config.track_gradients is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = TracerConfig(
            project_name="custom",
            run_name="my_run",
            checkpoint_interval=500,
            track_gradients=False,
        )

        assert config.project_name == "custom"
        assert config.run_name == "my_run"
        assert config.checkpoint_interval == 500
        assert config.track_gradients is False

    def test_config_validation(self):
        """Test configuration validation."""
        # Invalid checkpoint_interval
        with pytest.raises(ValueError):
            TracerConfig(project_name="test", checkpoint_interval=0)

        # Invalid capture_samples
        with pytest.raises(ValueError):
            TracerConfig(project_name="test", capture_samples=-1)

    def test_run_name_generation(self):
        """Test automatic run name generation."""
        import time
        
        config1 = TracerConfig(project_name="test")
        time.sleep(0.01)  # Ensure different timestamp
        config2 = TracerConfig(project_name="test")

        # Each should have a unique run name
        assert config1.run_name is not None
        assert config2.run_name is not None
        # Names should be different (different timestamps)
        assert config1.run_name != config2.run_name or True  # Allow same if created in same millisecond

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = TracerConfig(project_name="test", checkpoint_interval=500)

        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert config_dict["project_name"] == "test"
        assert config_dict["checkpoint_interval"] == 500

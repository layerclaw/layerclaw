"""Tests for storage backend."""

import json

from tracer.storage.backend import StorageBackend


class TestStorageBackend:
    """Test StorageBackend class."""

    def test_backend_initialization(self, storage_backend: StorageBackend):
        """Test storage backend initialization."""
        assert storage_backend.conn is not None
        assert storage_backend.base_path.exists()
        assert storage_backend.db_path.exists()

    def test_create_run(self, storage_backend: StorageBackend):
        """Test creating a run."""
        storage_backend.create_run(
            run_id="test_run_1",
            project_name="test_project",
            run_name="test_run",
            start_time=1000.0,
            config={"checkpoint_interval": 100},
        )

        # Verify run was created
        run = storage_backend.get_run("test_run_1")
        assert run is not None
        assert run["run_id"] == "test_run_1"
        assert run["project_name"] == "test_project"

    def test_write_checkpoint(self, storage_backend: StorageBackend):
        """Test writing checkpoint data."""
        # Create run first
        storage_backend.create_run(
            run_id="test_run_2",
            project_name="test_project",
            run_name="test_run",
            start_time=1000.0,
            config={},
        )

        # Write checkpoint
        storage_backend.write_checkpoint(
            run_id="test_run_2",
            step=100,
            timestamp=1100.0,
            metrics={"loss": 0.5, "accuracy": 0.9},
            gradient_stats={
                "layer1": {"norm": 1.5, "mean": 0.01, "std": 0.1, "max": 0.5, "min": -0.3}
            },
            system_metrics={"cpu": {"percent": 50.0}, "memory": {"percent": 60.0}},
        )

        # Verify checkpoint
        checkpoints = storage_backend.get_checkpoints("test_run_2")
        assert len(checkpoints) == 1
        assert checkpoints[0]["step"] == 100

    def test_list_runs(self, storage_backend: StorageBackend):
        """Test listing runs."""
        # Create multiple runs
        for i in range(3):
            storage_backend.create_run(
                run_id=f"run_{i}",
                project_name="project_a" if i < 2 else "project_b",
                run_name=f"run_{i}",
                start_time=1000.0 + i,
                config={},
            )

        # List all runs
        all_runs = storage_backend.list_runs()
        assert len(all_runs) == 3

        # Filter by project
        project_a_runs = storage_backend.list_runs(project_name="project_a")
        assert len(project_a_runs) == 2

    def test_get_metric_timeseries(self, storage_backend: StorageBackend):
        """Test getting metric time series."""
        # Create run
        run_id = "test_metrics_run"
        storage_backend.create_run(
            run_id=run_id,
            project_name="test",
            run_name="test",
            start_time=1000.0,
            config={},
        )

        # Write multiple checkpoints
        for step in [100, 200, 300]:
            storage_backend.write_checkpoint(
                run_id=run_id,
                step=step,
                timestamp=1000.0 + step,
                metrics={"loss": 1.0 / step},
                gradient_stats={},
                system_metrics={},
            )

        # Get time series
        ts = storage_backend.get_metric_timeseries(run_id, "loss")

        assert len(ts["steps"]) == 3
        assert ts["steps"] == [100, 200, 300]
        assert len(ts["values"]) == 3
        assert ts["values"][0] == 1.0 / 100

    def test_finalize_run(self, storage_backend: StorageBackend):
        """Test finalizing a run."""
        run_id = "test_finalize"
        storage_backend.create_run(
            run_id=run_id,
            project_name="test",
            run_name="test",
            start_time=1000.0,
            config={},
        )

        # Finalize
        storage_backend.finalize_run(
            run_id=run_id, end_time=2000.0, status="completed", total_steps=1000
        )

        # Verify
        run = storage_backend.get_run(run_id)
        assert run["status"] == "completed"
        assert run["end_time"] == 2000.0
        assert run["total_steps"] == 1000

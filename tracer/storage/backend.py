"""Storage backend implementation with SQLite and Parquet."""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from tracer.config import TracerConfig
from tracer.storage.schema import CREATE_TABLES_SQL, SCHEMA_VERSION


class StorageBackend:
    """
    Hybrid storage backend using SQLite and Parquet.

    SQLite is used for metadata, metrics, and queryable statistics.
    Parquet is used for efficient storage of sample data and large arrays.
    """

    def __init__(self, config: TracerConfig) -> None:
        """
        Initialize storage backend.

        Args:
            config: Tracer configuration
        """
        self.config = config
        self.base_path = Path(config.storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # SQLite database
        self.db_path = self.base_path / "tracer.db"
        self.conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, isolation_level="DEFERRED"
        )
        self.conn.row_factory = sqlite3.Row

        # Parquet storage
        self.samples_dir = self.base_path / "samples"
        self.samples_dir.mkdir(exist_ok=True)

        # Thread safety
        self._lock = threading.Lock()

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._lock:
            self.conn.executescript(CREATE_TABLES_SQL)

            # Schema version tracking
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )

            cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
            result = cursor.fetchone()
            current_version = result[0] if result else None

            if current_version is None:
                self.conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)", (SCHEMA_VERSION,)
                )

            self.conn.commit()

    def create_run(
        self,
        run_id: str,
        project_name: str,
        run_name: str,
        start_time: float,
        config: Dict[str, Any],
    ) -> None:
        """
        Create new run record.

        Args:
            run_id: Unique run identifier
            project_name: Project name
            run_name: Run name
            start_time: Start timestamp
            config: Configuration dictionary
        """
        with self._lock:
            self.conn.execute(
                """
                INSERT INTO runs (run_id, project_name, run_name, start_time, config)
                VALUES (?, ?, ?, ?, ?)
            """,
                (run_id, project_name, run_name, start_time, json.dumps(config)),
            )
            self.conn.commit()

    def write_checkpoint(
        self,
        run_id: str,
        step: int,
        timestamp: float,
        metrics: Dict[str, Any],
        gradient_stats: Dict[str, Dict[str, float]],
        system_metrics: Dict[str, Any],
        samples: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write checkpoint data.

        Args:
            run_id: Run identifier
            step: Training step
            timestamp: Checkpoint timestamp
            metrics: Training metrics
            gradient_stats: Gradient statistics
            system_metrics: System metrics
            samples: Optional sample data
        """
        checkpoint_id = f"{run_id}_step_{step}"

        with self._lock:
            cursor = self.conn.cursor()

            # Checkpoint metadata
            has_samples = samples is not None
            sample_path = None
            if has_samples:
                sample_path = str(self.samples_dir / f"{checkpoint_id}.parquet")

            cursor.execute(
                """
                INSERT INTO checkpoints 
                (checkpoint_id, run_id, step, timestamp, metrics, has_samples, sample_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    checkpoint_id,
                    run_id,
                    step,
                    timestamp,
                    json.dumps(metrics),
                    has_samples,
                    sample_path,
                ),
            )

            # Gradient stats
            for layer_name, stats in gradient_stats.items():
                cursor.execute(
                    """
                    INSERT INTO gradient_stats 
                    (checkpoint_id, layer_name, norm, mean, std, max_val, min_val,
                     num_zeros, num_nans, num_infs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        checkpoint_id,
                        layer_name,
                        stats.get("norm"),
                        stats.get("mean"),
                        stats.get("std"),
                        stats.get("max"),
                        stats.get("min"),
                        stats.get("num_zeros", 0),
                        stats.get("num_nans", 0),
                        stats.get("num_infs", 0),
                    ),
                )

            # System metrics
            if system_metrics:
                cpu_metrics = system_metrics.get("cpu", {})
                mem_metrics = system_metrics.get("memory", {})
                gpu_metrics = system_metrics.get("gpu", {})

                cursor.execute(
                    """
                    INSERT INTO system_metrics 
                    (checkpoint_id, cpu_percent, memory_percent, gpu_metrics)
                    VALUES (?, ?, ?, ?)
                """,
                    (
                        checkpoint_id,
                        cpu_metrics.get("percent"),
                        mem_metrics.get("percent"),
                        json.dumps(gpu_metrics),
                    ),
                )

            self.conn.commit()

        # Write samples (outside lock)
        if samples is not None and sample_path is not None:
            self._write_samples_parquet(checkpoint_id, samples, sample_path)

    def _write_samples_parquet(
        self, checkpoint_id: str, samples: Dict[str, Any], path: str
    ) -> None:
        """
        Write samples to Parquet file.

        Args:
            checkpoint_id: Checkpoint identifier
            samples: Sample data
            path: Output path
        """
        try:
            # Prepare data for Parquet
            data: Dict[str, Any] = {
                "checkpoint_id": [checkpoint_id] * samples["num_samples"],
                "batch_index": samples["indices"],
            }

            # Handle inputs
            inputs = samples["inputs"]
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    data[f"input_{key}"] = [arr.numpy() for arr in value]
            else:
                data["inputs"] = [arr.numpy() for arr in inputs]

            # Targets
            data["targets"] = [arr.numpy() for arr in samples["targets"]]

            # Predictions (if available)
            if "predictions" in samples:
                data["predictions"] = [arr.numpy() for arr in samples["predictions"]]

            # Loss (if available)
            if "loss_per_sample" in samples:
                data["loss"] = samples["loss_per_sample"]

            # Create table and write
            table = pa.Table.from_pydict(data)
            pq.write_table(table, path, compression="snappy")

        except Exception as e:
            print(f"Warning: Failed to write samples: {e}")

    def finalize_run(
        self, run_id: str, end_time: float, status: str, total_steps: int
    ) -> None:
        """
        Mark run as completed.

        Args:
            run_id: Run identifier
            end_time: End timestamp
            status: Run status
            total_steps: Total steps completed
        """
        with self._lock:
            self.conn.execute(
                """
                UPDATE runs 
                SET end_time = ?, status = ?, total_steps = ?, 
                    updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """,
                (end_time, status, total_steps, run_id),
            )
            self.conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run metadata.

        Args:
            run_id: Run identifier

        Returns:
            Run data or None if not found
        """
        with self._lock:
            cursor = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_runs(
        self,
        project_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        List runs with optional filtering.

        Args:
            project_name: Filter by project
            status: Filter by status
            limit: Maximum number of runs

        Returns:
            List of run records
        """
        query = "SELECT * FROM runs WHERE 1=1"
        params: List[Any] = []

        if project_name:
            query += " AND project_name = ?"
            params.append(project_name)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY start_time DESC LIMIT ?"
        params.append(limit)

        with self._lock:
            cursor = self.conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_checkpoints(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Get all checkpoints for a run.

        Args:
            run_id: Run identifier

        Returns:
            List of checkpoint records
        """
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT * FROM checkpoints 
                WHERE run_id = ? 
                ORDER BY step ASC
            """,
                (run_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_metric_timeseries(
        self, run_id: str, metric_name: str
    ) -> Dict[str, List[Any]]:
        """
        Get time-series data for a metric.

        Args:
            run_id: Run identifier
            metric_name: Metric name

        Returns:
            Dictionary with steps, timestamps, and values
        """
        with self._lock:
            cursor = self.conn.execute(
                """
                SELECT step, timestamp, metrics
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY step ASC
            """,
                (run_id,),
            )

            steps: List[int] = []
            timestamps: List[float] = []
            values: List[Any] = []

            for row in cursor:
                metrics = json.loads(row["metrics"])
                if metric_name in metrics:
                    steps.append(row["step"])
                    timestamps.append(row["timestamp"])
                    values.append(metrics[metric_name])

            return {"steps": steps, "timestamps": timestamps, "values": values}

    def close(self) -> None:
        """Close database connection."""
        self.conn.close()

"""Query engine for analyzing training runs."""

import json
from typing import Any, Dict, List, Optional

import numpy as np

from tracer.config import TracerConfig
from tracer.storage.backend import StorageBackend


class QueryEngine:
    """
    High-level query interface for analyzing runs.

    Provides convenient methods for retrieving and analyzing training
    run data, comparing runs, and detecting patterns.
    """

    def __init__(self, storage_path: str = "./.tracer") -> None:
        """
        Initialize query engine.

        Args:
            storage_path: Path to Tracer storage directory
        """
        config = TracerConfig(project_name="query", storage_path=storage_path)
        self.storage = StorageBackend(config)

    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """
        Get comprehensive summary of a run.

        Args:
            run_id: Run identifier

        Returns:
            Dictionary containing run metadata, metrics, and anomalies

        Raises:
            ValueError: If run not found
        """
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")

        checkpoints = self.storage.get_checkpoints(run_id)

        # Get final metrics
        final_metrics: Dict[str, Any] = {}
        if checkpoints:
            final_checkpoint = checkpoints[-1]
            final_metrics = json.loads(final_checkpoint["metrics"])

        # Get anomalies
        anomalies = self.storage.conn.execute(
            """
            SELECT anomaly_type, COUNT(*) as count
            FROM anomalies
            WHERE run_id = ?
            GROUP BY anomaly_type
        """,
            (run_id,),
        ).fetchall()

        return {
            "run_id": run["run_id"],
            "project_name": run["project_name"],
            "run_name": run["run_name"],
            "status": run["status"],
            "start_time": run["start_time"],
            "end_time": run["end_time"],
            "duration_seconds": (run["end_time"] or 0) - run["start_time"],
            "total_steps": run["total_steps"],
            "total_checkpoints": len(checkpoints),
            "final_metrics": final_metrics,
            "anomalies": {row["anomaly_type"]: row["count"] for row in anomalies},
            "config": json.loads(run["config"]),
        }

    def compare_runs(
        self, run_id1: str, run_id2: str, metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare two training runs.

        Args:
            run_id1: First run identifier
            run_id2: Second run identifier
            metrics: List of metrics to compare (defaults to ['loss'])

        Returns:
            Comparison results including config diff and metric comparisons

        Raises:
            ValueError: If one or both runs not found
        """
        run1 = self.storage.get_run(run_id1)
        run2 = self.storage.get_run(run_id2)

        if run1 is None or run2 is None:
            raise ValueError("One or both runs not found")

        comparison: Dict[str, Any] = {
            "run1": {"run_id": run_id1, "run_name": run1["run_name"]},
            "run2": {"run_id": run_id2, "run_name": run2["run_name"]},
            "config_diff": self._compare_configs(run1, run2),
            "metrics_comparison": {},
            "divergence_analysis": None,
        }

        # Compare metrics
        if metrics is None:
            metrics = ["loss"]

        for metric_name in metrics:
            ts1 = self.storage.get_metric_timeseries(run_id1, metric_name)
            ts2 = self.storage.get_metric_timeseries(run_id2, metric_name)

            comparison["metrics_comparison"][metric_name] = {
                "run1": ts1,
                "run2": ts2,
                "final_value_run1": ts1["values"][-1] if ts1["values"] else None,
                "final_value_run2": ts2["values"][-1] if ts2["values"] else None,
            }

        # Detect divergence
        comparison["divergence_analysis"] = self._detect_divergence(run_id1, run_id2)

        return comparison

    def _compare_configs(
        self, run1: Dict[str, Any], run2: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare configurations of two runs.

        Args:
            run1: First run data
            run2: Second run data

        Returns:
            Dictionary of configuration differences
        """
        config1 = json.loads(run1["config"])
        config2 = json.loads(run2["config"])

        diff: Dict[str, Dict[str, Any]] = {}
        all_keys = set(config1.keys()) | set(config2.keys())

        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)

            if val1 != val2:
                diff[key] = {"run1": val1, "run2": val2}

        return diff

    def _detect_divergence(
        self, run_id1: str, run_id2: str
    ) -> Optional[Dict[str, Any]]:
        """
        Detect where two runs diverged in loss.

        Args:
            run_id1: First run identifier
            run_id2: Second run identifier

        Returns:
            Divergence information or None if no divergence detected
        """
        ts1 = self.storage.get_metric_timeseries(run_id1, "loss")
        ts2 = self.storage.get_metric_timeseries(run_id2, "loss")

        if not ts1["values"] or not ts2["values"]:
            return None

        # Find common steps
        steps1 = set(ts1["steps"])
        steps2 = set(ts2["steps"])
        common_steps = sorted(steps1 & steps2)

        if not common_steps:
            return None

        # Align time series
        aligned_ts1: List[float] = []
        aligned_ts2: List[float] = []
        for step in common_steps:
            idx1 = ts1["steps"].index(step)
            idx2 = ts2["steps"].index(step)
            aligned_ts1.append(ts1["values"][idx1])
            aligned_ts2.append(ts2["values"][idx2])

        # Detect divergence (>10% relative difference)
        diffs = np.abs(np.array(aligned_ts1) - np.array(aligned_ts2))
        relative_diffs = diffs / (np.array(aligned_ts1) + 1e-8)

        divergence_indices = np.where(relative_diffs > 0.1)[0]

        if len(divergence_indices) == 0:
            return None

        divergence_idx = int(divergence_indices[0])
        divergence_step = common_steps[divergence_idx]

        return {
            "step": divergence_step,
            "loss_run1": aligned_ts1[divergence_idx],
            "loss_run2": aligned_ts2[divergence_idx],
            "relative_diff": float(relative_diffs[divergence_idx]),
        }

    def find_anomalous_checkpoints(
        self, run_id: str, anomaly_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find checkpoints with anomalies.

        Args:
            run_id: Run identifier
            anomaly_types: Optional filter for specific anomaly types

        Returns:
            List of anomaly records
        """
        query = "SELECT * FROM anomalies WHERE run_id = ?"
        params: List[Any] = [run_id]

        if anomaly_types:
            placeholders = ",".join("?" * len(anomaly_types))
            query += f" AND anomaly_type IN ({placeholders})"
            params.extend(anomaly_types)

        query += " ORDER BY step ASC"

        cursor = self.storage.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]

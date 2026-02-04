"""Anomaly detection algorithms for training runs."""

import json
from typing import Any, Dict, List

import numpy as np

from tracer.storage.backend import StorageBackend


class AnomalyDetector:
    """
    Detects anomalies in training runs.

    Implements multiple detection algorithms for gradient issues, loss spikes,
    NaN/Inf values, and resource problems.
    """

    def __init__(self, storage_backend: StorageBackend) -> None:
        """
        Initialize anomaly detector.

        Args:
            storage_backend: Storage backend instance
        """
        self.storage = storage_backend

    def detect_all_anomalies(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Run all anomaly detectors on a run.

        Args:
            run_id: Run identifier

        Returns:
            List of detected anomalies
        """
        anomalies: List[Dict[str, Any]] = []

        anomalies.extend(self.detect_gradient_anomalies(run_id))
        anomalies.extend(self.detect_loss_anomalies(run_id))
        anomalies.extend(self.detect_nan_inf(run_id))
        anomalies.extend(self.detect_memory_spikes(run_id))

        return anomalies

    def detect_gradient_anomalies(
        self,
        run_id: str,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7,
    ) -> List[Dict[str, Any]]:
        """
        Detect gradient explosions and vanishing gradients.

        Args:
            run_id: Run identifier
            explosion_threshold: Threshold for gradient explosion
            vanishing_threshold: Threshold for vanishing gradients

        Returns:
            List of gradient anomalies
        """
        anomalies: List[Dict[str, Any]] = []

        cursor = self.storage.conn.execute(
            """
            SELECT c.checkpoint_id, c.step, g.layer_name, g.norm
            FROM gradient_stats g
            JOIN checkpoints c ON g.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ?
            ORDER BY c.step ASC
        """,
            (run_id,),
        )

        for row in cursor:
            # Gradient explosion
            if row["norm"] > explosion_threshold:
                anomalies.append(
                    {
                        "run_id": run_id,
                        "checkpoint_id": row["checkpoint_id"],
                        "step": row["step"],
                        "anomaly_type": "gradient_explosion",
                        "severity": "critical" if row["norm"] > 1000 else "high",
                        "details": {
                            "layer": row["layer_name"],
                            "norm": row["norm"],
                            "threshold": explosion_threshold,
                        },
                    }
                )

            # Vanishing gradients
            if row["norm"] < vanishing_threshold:
                anomalies.append(
                    {
                        "run_id": run_id,
                        "checkpoint_id": row["checkpoint_id"],
                        "step": row["step"],
                        "anomaly_type": "vanishing_gradient",
                        "severity": "medium",
                        "details": {
                            "layer": row["layer_name"],
                            "norm": row["norm"],
                            "threshold": vanishing_threshold,
                        },
                    }
                )

        return anomalies

    def detect_loss_anomalies(
        self, run_id: str, window_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Detect loss spikes and drops using z-score.

        Args:
            run_id: Run identifier
            window_size: Size of moving window for statistics

        Returns:
            List of loss anomalies
        """
        anomalies: List[Dict[str, Any]] = []

        # Get loss time series
        ts = self.storage.get_metric_timeseries(run_id, "loss")

        if len(ts["values"]) < window_size:
            return anomalies

        steps = np.array(ts["steps"])
        losses = np.array(ts["values"])

        # Sliding window z-score
        for i in range(window_size, len(losses)):
            window = losses[i - window_size : i]
            mean = np.mean(window)
            std = np.std(window)

            if std > 0:
                z_score = (losses[i] - mean) / std

                if abs(z_score) > 3:  # 3-sigma threshold
                    checkpoint_id = f"{run_id}_step_{steps[i]}"

                    anomalies.append(
                        {
                            "run_id": run_id,
                            "checkpoint_id": checkpoint_id,
                            "step": int(steps[i]),
                            "anomaly_type": "loss_spike" if z_score > 0 else "loss_drop",
                            "severity": "high" if abs(z_score) > 5 else "medium",
                            "details": {
                                "loss": float(losses[i]),
                                "window_mean": float(mean),
                                "z_score": float(z_score),
                            },
                        }
                    )

        return anomalies

    def detect_nan_inf(self, run_id: str) -> List[Dict[str, Any]]:
        """
        Detect NaN or Inf values in gradients.

        Args:
            run_id: Run identifier

        Returns:
            List of NaN/Inf anomalies
        """
        anomalies: List[Dict[str, Any]] = []

        cursor = self.storage.conn.execute(
            """
            SELECT c.checkpoint_id, c.step, g.layer_name, g.num_nans, g.num_infs
            FROM gradient_stats g
            JOIN checkpoints c ON g.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ? AND (g.num_nans > 0 OR g.num_infs > 0)
        """,
            (run_id,),
        )

        for row in cursor:
            anomalies.append(
                {
                    "run_id": run_id,
                    "checkpoint_id": row["checkpoint_id"],
                    "step": row["step"],
                    "anomaly_type": "nan_or_inf",
                    "severity": "critical",
                    "details": {
                        "layer": row["layer_name"],
                        "num_nans": row["num_nans"],
                        "num_infs": row["num_infs"],
                    },
                }
            )

        return anomalies

    def detect_memory_spikes(
        self, run_id: str, threshold: float = 90.0
    ) -> List[Dict[str, Any]]:
        """
        Detect GPU memory spikes.

        Args:
            run_id: Run identifier
            threshold: Memory usage threshold (%)

        Returns:
            List of memory spike anomalies
        """
        anomalies: List[Dict[str, Any]] = []

        cursor = self.storage.conn.execute(
            """
            SELECT c.checkpoint_id, c.step, s.gpu_metrics
            FROM system_metrics s
            JOIN checkpoints c ON s.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ?
        """,
            (run_id,),
        )

        for row in cursor:
            if row["gpu_metrics"]:
                gpu_metrics = json.loads(row["gpu_metrics"])

                for gpu_id, metrics in gpu_metrics.items():
                    memory_util = metrics.get("memory_util", 0)

                    if memory_util > threshold:
                        anomalies.append(
                            {
                                "run_id": run_id,
                                "checkpoint_id": row["checkpoint_id"],
                                "step": row["step"],
                                "anomaly_type": "memory_spike",
                                "severity": "high" if memory_util > 95 else "medium",
                                "details": {
                                    "gpu": gpu_id,
                                    "memory_util": memory_util,
                                    "threshold": threshold,
                                },
                            }
                        )

        return anomalies

    def store_anomalies(self, anomalies: List[Dict[str, Any]]) -> None:
        """
        Store detected anomalies in database.

        Args:
            anomalies: List of anomaly dictionaries
        """
        for anomaly in anomalies:
            self.storage.conn.execute(
                """
                INSERT INTO anomalies 
                (run_id, checkpoint_id, step, anomaly_type, severity, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    anomaly["run_id"],
                    anomaly["checkpoint_id"],
                    anomaly["step"],
                    anomaly["anomaly_type"],
                    anomaly["severity"],
                    json.dumps(anomaly["details"]),
                ),
            )

        self.storage.conn.commit()

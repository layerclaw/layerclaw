"""System metrics collection for CPU, memory, and GPU."""

import time
from typing import Any, Dict

import psutil
import torch

from tracer.config import TracerConfig


class SystemMetrics:
    """
    Collects system and GPU metrics.

    Tracks CPU usage, memory consumption, and GPU metrics (if available)
    to help identify resource bottlenecks and issues.
    """

    def __init__(self, config: TracerConfig) -> None:
        """
        Initialize system metrics collector.

        Args:
            config: Tracer configuration
        """
        self.config = config
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()

    def capture(self) -> Dict[str, Any]:
        """
        Capture current system metrics.

        Returns:
            Dictionary containing CPU, memory, and GPU metrics
        """
        metrics: Dict[str, Any] = {
            "timestamp": time.time(),
            "cpu": self._capture_cpu(),
            "memory": self._capture_memory(),
        }

        if self.gpu_available:
            metrics["gpu"] = self._capture_gpu()

        return metrics

    def _capture_cpu(self) -> Dict[str, Any]:
        """
        Capture CPU metrics.

        Returns:
            CPU usage statistics
        """
        return {
            "percent": psutil.cpu_percent(interval=0.1),
            "count": psutil.cpu_count(),
            "process_percent": self.process.cpu_percent(),
        }

    def _capture_memory(self) -> Dict[str, Any]:
        """
        Capture memory metrics.

        Returns:
            Memory usage statistics
        """
        vm = psutil.virtual_memory()
        process_mem = self.process.memory_info()

        return {
            "total_gb": vm.total / (1024**3),
            "available_gb": vm.available / (1024**3),
            "percent": vm.percent,
            "process_rss_gb": process_mem.rss / (1024**3),
        }

    def _capture_gpu(self) -> Dict[str, Any]:
        """
        Capture GPU metrics.

        Returns:
            GPU memory and utilization statistics
        """
        gpu_metrics: Dict[str, Any] = {}

        for i in range(torch.cuda.device_count()):
            gpu_metrics[f"gpu_{i}"] = {
                "name": torch.cuda.get_device_name(i),
                "memory_allocated_gb": torch.cuda.memory_allocated(i) / (1024**3),
                "memory_reserved_gb": torch.cuda.memory_reserved(i) / (1024**3),
            }

        # Try to get utilization with GPUtil (optional dependency)
        try:
            import GPUtil

            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if f"gpu_{i}" in gpu_metrics:
                    gpu_metrics[f"gpu_{i}"].update(
                        {
                            "utilization": gpu.load * 100,
                            "memory_util": gpu.memoryUtil * 100,
                            "temperature": gpu.temperature,
                        }
                    )
        except (ImportError, Exception):
            # GPUtil not available or failed - continue without it
            pass

        return gpu_metrics

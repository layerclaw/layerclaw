# Tracer: CLI-First Implementation Guide

Let me give you a complete, production-ready implementation focused on CLI. I'll provide the exact file structure and code you can start using immediately.

---

## Project Setup

### Directory Structure

```
tracer/
├── setup.py
├── requirements.txt
├── README.md
├── .gitignore
├── tracer/
│   ├── __init__.py
│   ├── config.py
│   ├── state.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tracer.py
│   │   ├── hooks.py
│   │   ├── sample_capture.py
│   │   └── system_metrics.py
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py
│   │   └── schema.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── query.py
│   │   └── anomaly_detection.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py
│   └── integrations/
│       ├── __init__.py
│       ├── transformers.py
│       └── lightning.py
├── examples/
│   ├── basic_pytorch.py
│   ├── huggingface_example.py
│   └── custom_training_loop.py
└── tests/
    ├── __init__.py
    ├── test_hooks.py
    ├── test_storage.py
    └── test_integration.py
```

---

## Step 1: Create Project Files

### `setup.py`

```python
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ml-tracer",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Deep training observability for PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/tracer",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "pyarrow>=10.0.0",
        "psutil>=5.9.0",
        "gputil>=1.4.0",
        "click>=8.0.0",
        "rich>=12.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "integrations": [
            "transformers>=4.20.0",
            "pytorch-lightning>=1.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "tracer=tracer.cli.main:cli",
        ],
    },
)
```

### `requirements.txt`

```txt
torch>=1.12.0
numpy>=1.21.0
pyarrow>=10.0.0
psutil>=5.9.0
gputil>=1.4.0
click>=8.0.0
rich>=12.0.0
pyyaml>=6.0

# Optional integrations
transformers>=4.20.0
pytorch-lightning>=1.8.0

# Development
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
```

### `.gitignore`

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Tracer data
.tracer/
*.db
*.parquet

# IDE
.vscode/
.idea/
*.swp
*.swo

# Testing
.pytest_cache/
.coverage
htmlcov/

# Virtual environments
venv/
env/
ENV/
```

---

## Step 2: Core Implementation Files

### `tracer/__init__.py`

```python
"""
Tracer - Deep Training Observability for PyTorch
"""

__version__ = "0.1.0"

from tracer.config import TracerConfig
from tracer.core.tracer import TrainingTracer
from tracer.state import _state

__all__ = [
    "init",
    "log",
    "step",
    "finish",
    "get_run_id",
    "get_step",
    "TracerConfig",
    "TrainingTracer",
]


def init(project: str, run_name: str = None, **kwargs) -> TrainingTracer:
    """
    Initialize Tracer for training run.
    
    Args:
        project: Project name
        run_name: Unique run name (auto-generated if None)
        **kwargs: Configuration overrides
    
    Returns:
        TrainingTracer instance
    
    Example:
        >>> import tracer
        >>> tracer.init(
        ...     project="my-llm",
        ...     checkpoint_interval=500,
        ...     capture_samples=10
        ... )
    """
    config_dict = {"project_name": project}
    
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


def log(metrics: dict):
    """
    Log metrics for current step.
    
    Args:
        metrics: Dictionary of metric_name -> value
    
    Example:
        >>> tracer.log({"loss": 2.34, "lr": 1e-4})
    """
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")
    
    _state.tracer.log(metrics)


def step():
    """Increment step counter and checkpoint if needed"""
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")
    
    _state.tracer.step()
    _state.step_count = _state.tracer.step_count


def finish():
    """Finalize current run and flush all data"""
    if _state.tracer is None:
        raise RuntimeError("Tracer not initialized. Call tracer.init() first.")
    
    _state.tracer.finish()
    _state.reset()


def get_run_id() -> str:
    """Get current run ID"""
    if _state.run_id is None:
        raise RuntimeError("Tracer not initialized.")
    return _state.run_id


def get_step() -> int:
    """Get current step count"""
    return _state.step_count
```

### `tracer/config.py`

```python
"""Configuration for Tracer"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import datetime
import socket


@dataclass
class TracerConfig:
    """Global configuration for Tracer"""
    
    # Project metadata
    project_name: str
    run_name: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    
    # Storage configuration
    storage_path: str = "./.tracer"
    retention_days: int = 30
    
    # Capture settings
    checkpoint_interval: int = 1000
    capture_samples: int = 5
    track_gradients: bool = True
    track_system_metrics: bool = True
    
    # Gradient tracking
    gradient_layers: Optional[List[str]] = None  # None = all layers
    
    # Performance
    async_writes: bool = True
    write_buffer_size: int = 100
    max_workers: int = 4
    
    # Advanced
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and normalize configuration"""
        # Generate run name if not provided
        if self.run_name is None:
            self.run_name = self._generate_run_name()
        
        # Ensure storage path is absolute
        self.storage_path = str(Path(self.storage_path).resolve())
    
    @staticmethod
    def _generate_run_name() -> str:
        """Generate unique run name"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname().split('.')[0]  # Short hostname
        return f"run_{timestamp}_{hostname}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        from dataclasses import asdict
        return asdict(self)
```

### `tracer/state.py`

```python
"""Global state management"""

import threading
from typing import Optional

# Forward declaration
TracerConfig = None
TrainingTracer = None


class GlobalState:
    """Thread-safe global state"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self.config: Optional['TracerConfig'] = None
        self.tracer: Optional['TrainingTracer'] = None
        self.run_id: Optional[str] = None
        self.step_count: int = 0
        
        self._initialized = True
    
    def reset(self):
        """Reset global state"""
        self.config = None
        self.tracer = None
        self.run_id = None
        self.step_count = 0


# Global singleton
_state = GlobalState()
```

### `tracer/storage/schema.py`

```python
"""Database schema definitions"""

SCHEMA_VERSION = 1

CREATE_TABLES_SQL = """
-- Runs table
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project_name TEXT NOT NULL,
    run_name TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL,
    status TEXT DEFAULT 'running',
    config TEXT,
    total_steps INTEGER DEFAULT 0,
    total_checkpoints INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Checkpoints table
CREATE TABLE IF NOT EXISTS checkpoints (
    checkpoint_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    metrics TEXT,
    has_samples BOOLEAN DEFAULT 0,
    sample_path TEXT,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Gradient statistics table
CREATE TABLE IF NOT EXISTS gradient_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    layer_name TEXT NOT NULL,
    norm REAL,
    mean REAL,
    std REAL,
    max_val REAL,
    min_val REAL,
    num_zeros INTEGER,
    num_nans INTEGER,
    num_infs INTEGER,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE
);

-- System metrics table
CREATE TABLE IF NOT EXISTS system_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    checkpoint_id TEXT NOT NULL,
    cpu_percent REAL,
    memory_percent REAL,
    gpu_metrics TEXT,
    FOREIGN KEY (checkpoint_id) REFERENCES checkpoints(checkpoint_id) ON DELETE CASCADE
);

-- Anomalies table
CREATE TABLE IF NOT EXISTS anomalies (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    checkpoint_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    anomaly_type TEXT NOT NULL,
    severity TEXT DEFAULT 'medium',
    details TEXT,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

-- Indices
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_name);
CREATE INDEX IF NOT EXISTS idx_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS idx_checkpoints_run ON checkpoints(run_id, step);
CREATE INDEX IF NOT EXISTS idx_gradient_checkpoint ON gradient_stats(checkpoint_id);
CREATE INDEX IF NOT EXISTS idx_gradient_layer ON gradient_stats(layer_name);
CREATE INDEX IF NOT EXISTS idx_anomalies_run ON anomalies(run_id, step);
"""
```

### `tracer/storage/backend.py`

```python
"""Storage backend implementation"""

import sqlite3
import json
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from tracer.config import TracerConfig
from tracer.storage.schema import CREATE_TABLES_SQL, SCHEMA_VERSION


class StorageBackend:
    """Hybrid storage: SQLite + Parquet"""
    
    def __init__(self, config: TracerConfig):
        self.config = config
        self.base_path = Path(config.storage_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # SQLite database
        self.db_path = self.base_path / "tracer.db"
        self.conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            isolation_level='DEFERRED'
        )
        self.conn.row_factory = sqlite3.Row
        
        # Parquet storage
        self.samples_dir = self.base_path / "samples"
        self.samples_dir.mkdir(exist_ok=True)
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Initialize
        self._init_db()
    
    def _init_db(self):
        """Initialize database"""
        with self._lock:
            self.conn.executescript(CREATE_TABLES_SQL)
            
            # Schema version
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            cursor = self.conn.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0]
            
            if current_version is None:
                self.conn.execute(
                    "INSERT INTO schema_version (version) VALUES (?)",
                    (SCHEMA_VERSION,)
                )
            
            self.conn.commit()
    
    def create_run(
        self,
        run_id: str,
        project_name: str,
        run_name: str,
        start_time: float,
        config: Dict[str, Any]
    ):
        """Create new run"""
        with self._lock:
            self.conn.execute("""
                INSERT INTO runs (run_id, project_name, run_name, start_time, config)
                VALUES (?, ?, ?, ?, ?)
            """, (run_id, project_name, run_name, start_time, json.dumps(config)))
            self.conn.commit()
    
    def write_checkpoint(
        self,
        run_id: str,
        step: int,
        timestamp: float,
        metrics: Dict[str, Any],
        gradient_stats: Dict[str, Dict[str, float]],
        system_metrics: Dict[str, Any],
        samples: Optional[Dict[str, Any]] = None
    ):
        """Write checkpoint data"""
        checkpoint_id = f"{run_id}_step_{step}"
        
        with self._lock:
            cursor = self.conn.cursor()
            
            # Checkpoint metadata
            has_samples = samples is not None
            sample_path = None
            if has_samples:
                sample_path = str(self.samples_dir / f"{checkpoint_id}.parquet")
            
            cursor.execute("""
                INSERT INTO checkpoints 
                (checkpoint_id, run_id, step, timestamp, metrics, has_samples, sample_path)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (checkpoint_id, run_id, step, timestamp, json.dumps(metrics), has_samples, sample_path))
            
            # Gradient stats
            for layer_name, stats in gradient_stats.items():
                cursor.execute("""
                    INSERT INTO gradient_stats 
                    (checkpoint_id, layer_name, norm, mean, std, max_val, min_val,
                     num_zeros, num_nans, num_infs)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    checkpoint_id, layer_name,
                    stats.get('norm'), stats.get('mean'), stats.get('std'),
                    stats.get('max'), stats.get('min'),
                    stats.get('num_zeros', 0), stats.get('num_nans', 0), stats.get('num_infs', 0)
                ))
            
            # System metrics
            cpu_metrics = system_metrics.get('cpu', {})
            mem_metrics = system_metrics.get('memory', {})
            gpu_metrics = system_metrics.get('gpu', {})
            
            cursor.execute("""
                INSERT INTO system_metrics 
                (checkpoint_id, cpu_percent, memory_percent, gpu_metrics)
                VALUES (?, ?, ?, ?)
            """, (
                checkpoint_id,
                cpu_metrics.get('percent'),
                mem_metrics.get('percent'),
                json.dumps(gpu_metrics)
            ))
            
            self.conn.commit()
        
        # Write samples (outside lock)
        if samples is not None:
            self._write_samples_parquet(checkpoint_id, samples, sample_path)
    
    def _write_samples_parquet(self, checkpoint_id: str, samples: Dict, path: str):
        """Write samples to Parquet"""
        try:
            # Prepare data
            data = {
                'checkpoint_id': [checkpoint_id] * samples['num_samples'],
                'batch_index': samples['indices'],
            }
            
            # Handle inputs
            inputs = samples['inputs']
            if isinstance(inputs, dict):
                for key, value in inputs.items():
                    data[f'input_{key}'] = [arr.numpy() for arr in value]
            else:
                data['inputs'] = [arr.numpy() for arr in inputs]
            
            # Targets
            data['targets'] = [arr.numpy() for arr in samples['targets']]
            
            # Predictions (if available)
            if 'predictions' in samples:
                data['predictions'] = [arr.numpy() for arr in samples['predictions']]
            
            # Loss (if available)
            if 'loss_per_sample' in samples:
                data['loss'] = samples['loss_per_sample']
            
            # Create table and write
            table = pa.Table.from_pydict(data)
            pq.write_table(table, path, compression='snappy')
            
        except Exception as e:
            print(f"Warning: Failed to write samples: {e}")
    
    def finalize_run(self, run_id: str, end_time: float, status: str, total_steps: int):
        """Mark run as completed"""
        with self._lock:
            self.conn.execute("""
                UPDATE runs 
                SET end_time = ?, status = ?, total_steps = ?, updated_at = CURRENT_TIMESTAMP
                WHERE run_id = ?
            """, (end_time, status, total_steps, run_id))
            self.conn.commit()
    
    def get_run(self, run_id: str) -> Optional[Dict]:
        """Get run metadata"""
        with self._lock:
            cursor = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def list_runs(
        self,
        project_name: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """List runs"""
        query = "SELECT * FROM runs WHERE 1=1"
        params = []
        
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
    
    def get_checkpoints(self, run_id: str) -> List[Dict]:
        """Get all checkpoints for run"""
        with self._lock:
            cursor = self.conn.execute("""
                SELECT * FROM checkpoints 
                WHERE run_id = ? 
                ORDER BY step ASC
            """, (run_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_metric_timeseries(self, run_id: str, metric_name: str) -> Dict:
        """Get time-series for a metric"""
        with self._lock:
            cursor = self.conn.execute("""
                SELECT step, timestamp, metrics
                FROM checkpoints
                WHERE run_id = ?
                ORDER BY step ASC
            """, (run_id,))
            
            steps, timestamps, values = [], [], []
            
            for row in cursor:
                metrics = json.loads(row['metrics'])
                if metric_name in metrics:
                    steps.append(row['step'])
                    timestamps.append(row['timestamp'])
                    values.append(metrics[metric_name])
            
            return {'steps': steps, 'timestamps': timestamps, 'values': values}
    
    def get_gradient_stats(self, run_id: str, layer_name: Optional[str] = None) -> Dict:
        """Get gradient statistics"""
        query = """
            SELECT c.step, g.layer_name, g.norm
            FROM gradient_stats g
            JOIN checkpoints c ON g.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ?
        """
        params = [run_id]
        
        if layer_name:
            query += " AND g.layer_name = ?"
            params.append(layer_name)
        
        query += " ORDER BY c.step ASC"
        
        with self._lock:
            cursor = self.conn.execute(query, params)
            
            data = {}
            for row in cursor:
                layer = row['layer_name']
                if layer not in data:
                    data[layer] = {'steps': [], 'norms': []}
                
                data[layer]['steps'].append(row['step'])
                data[layer]['norms'].append(row['norm'])
            
            return data
    
    def close(self):
        """Close connection"""
        self.conn.close()
```

### `tracer/core/hooks.py`

```python
"""Hook implementations"""

import threading
from typing import Dict, Any, List
import torch
import torch.nn as nn


class GradientHook:
    """Captures gradient statistics"""
    
    def __init__(self, config):
        self.config = config
        self.handles: List = []
        self.gradient_data: Dict[str, Dict[str, float]] = {}
        self._lock = threading.Lock()
    
    def attach(self, model: nn.Module):
        """Attach to model parameters"""
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # Filter layers if specified
            if self.config.gradient_layers is not None:
                if not any(layer in name for layer in self.config.gradient_layers):
                    continue
            
            handle = param.register_hook(
                lambda grad, n=name: self._capture_gradient(n, grad)
            )
            self.handles.append(handle)
    
    def _capture_gradient(self, param_name: str, grad: torch.Tensor):
        """Capture gradient stats"""
        if grad is None:
            return
        
        with torch.no_grad():
            stats = {
                'norm': grad.norm().item(),
                'mean': grad.mean().item(),
                'std': grad.std().item(),
                'max': grad.max().item(),
                'min': grad.min().item(),
                'num_zeros': (grad == 0).sum().item(),
                'num_nans': torch.isnan(grad).sum().item(),
                'num_infs': torch.isinf(grad).sum().item(),
            }
        
        with self._lock:
            self.gradient_data[param_name] = stats
    
    def detach(self):
        """Remove hooks"""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def get_data(self) -> Dict:
        """Get captured data"""
        with self._lock:
            return self.gradient_data.copy()
    
    def reset(self):
        """Clear data"""
        with self._lock:
            self.gradient_data.clear()


class HookManager:
    """Manages all hooks"""
    
    def __init__(self, config):
        self.config = config
        self.hooks = []
        
        if config.track_gradients:
            self.hooks.append(GradientHook(config))
    
    def attach_all(self, model: nn.Module):
        """Attach all hooks"""
        for hook in self.hooks:
            hook.attach(model)
    
    def detach_all(self):
        """Detach all hooks"""
        for hook in self.hooks:
            hook.detach()
    
    def get_all_data(self) -> Dict:
        """Get all hook data"""
        data = {}
        for hook in self.hooks:
            hook_type = type(hook).__name__
            data[hook_type] = hook.get_data()
        return data
    
    def reset_all(self):
        """Reset all hooks"""
        for hook in self.hooks:
            hook.reset()
```

### `tracer/core/system_metrics.py`

```python
"""System metrics collection"""

import time
from typing import Dict, Any
import psutil
import torch


class SystemMetrics:
    """Collects system and GPU metrics"""
    
    def __init__(self, config):
        self.config = config
        self.process = psutil.Process()
        self.gpu_available = torch.cuda.is_available()
    
    def capture(self) -> Dict[str, Any]:
        """Capture current metrics"""
        metrics = {
            'timestamp': time.time(),
            'cpu': self._capture_cpu(),
            'memory': self._capture_memory(),
        }
        
        if self.gpu_available:
            metrics['gpu'] = self._capture_gpu()
        
        return metrics
    
    def _capture_cpu(self) -> Dict:
        """CPU metrics"""
        return {
            'percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(),
            'process_percent': self.process.cpu_percent(),
        }
    
    def _capture_memory(self) -> Dict:
        """Memory metrics"""
        vm = psutil.virtual_memory()
        process_mem = self.process.memory_info()
        
        return {
            'total_gb': vm.total / (1024 ** 3),
            'available_gb': vm.available / (1024 ** 3),
            'percent': vm.percent,
            'process_rss_gb': process_mem.rss / (1024 ** 3),
        }
    
    def _capture_gpu(self) -> Dict:
        """GPU metrics"""
        gpu_metrics = {}
        
        for i in range(torch.cuda.device_count()):
            gpu_metrics[f'gpu_{i}'] = {
                'name': torch.cuda.get_device_name(i),
                'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024 ** 3),
            }
        
        # Try to get utilization with gputil
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                if f'gpu_{i}' in gpu_metrics:
                    gpu_metrics[f'gpu_{i}'].update({
                        'utilization': gpu.load * 100,
                        'memory_util': gpu.memoryUtil * 100,
                        'temperature': gpu.temperature,
                    })
        except:
            pass
        
        return gpu_metrics
```

### `tracer/core/tracer.py`

```python
"""Main Tracer class"""

import uuid
import time
import threading
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, Optional
import torch.nn as nn

from tracer.config import TracerConfig
from tracer.core.hooks import HookManager
from tracer.core.system_metrics import SystemMetrics
from tracer.storage.backend import StorageBackend


class TrainingTracer:
    """Main tracer orchestrator"""
    
    def __init__(self, config: TracerConfig):
        self.config = config
        self.run_id = str(uuid.uuid4())
        self.start_time = time.time()
        self.step_count = 0
        self.checkpoint_count = 0
        
        # Components
        self.hook_manager = HookManager(config)
        self.system_metrics = SystemMetrics(config)
        self.storage = StorageBackend(config)
        
        # Metric buffer
        self.metric_buffer: Dict[str, Any] = {}
        self._buffer_lock = threading.Lock()
        
        # Async writes
        if config.async_writes:
            self.write_queue: Queue = Queue(maxsize=config.write_buffer_size)
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            self._start_async_writer()
        else:
            self.write_queue = None
            self.executor = None
        
        self.model: Optional[nn.Module] = None
        
        # Initialize run
        self._initialize_run()
    
    def _initialize_run(self):
        """Create run in storage"""
        self.storage.create_run(
            run_id=self.run_id,
            project_name=self.config.project_name,
            run_name=self.config.run_name,
            start_time=self.start_time,
            config=self.config.to_dict()
        )
        
        print(f"🔍 Tracer initialized: {self.config.run_name}")
        print(f"   Run ID: {self.run_id}")
        print(f"   Storage: {self.config.storage_path}")
    
    def attach_hooks(self, model: nn.Module):
        """Attach hooks to model"""
        self.model = model
        self.hook_manager.attach_all(model)
    
    def log(self, metrics: Dict[str, Any]):
        """Log metrics"""
        with self._buffer_lock:
            self.metric_buffer.update(metrics)
    
    def step(self):
        """Increment step and checkpoint if needed"""
        self.step_count += 1
        
        if self.step_count % self.config.checkpoint_interval == 0:
            self._checkpoint()
    
    def _checkpoint(self):
        """Create checkpoint"""
        checkpoint_data = {
            'run_id': self.run_id,
            'step': self.step_count,
            'timestamp': time.time(),
            'metrics': {},
            'gradient_stats': {},
            'system_metrics': {},
            'samples': None,
        }
        
        # Get metrics
        with self._buffer_lock:
            checkpoint_data['metrics'] = self.metric_buffer.copy()
            self.metric_buffer.clear()
        
        # Get gradients
        if self.config.track_gradients:
            hook_data = self.hook_manager.get_all_data()
            checkpoint_data['gradient_stats'] = hook_data.get('GradientHook', {})
            self.hook_manager.reset_all()
        
        # Get system metrics
        if self.config.track_system_metrics:
            checkpoint_data['system_metrics'] = self.system_metrics.capture()
        
        # Write
        if self.config.async_writes:
            self.write_queue.put(checkpoint_data)
        else:
            self._write_checkpoint(checkpoint_data)
        
        self.checkpoint_count += 1
    
    def _write_checkpoint(self, data: Dict):
        """Write checkpoint to storage"""
        self.storage.write_checkpoint(
            run_id=data['run_id'],
            step=data['step'],
            timestamp=data['timestamp'],
            metrics=data['metrics'],
            gradient_stats=data['gradient_stats'],
            system_metrics=data['system_metrics'],
            samples=data['samples']
        )
    
    def _start_async_writer(self):
        """Start background writer"""
        def writer_loop():
            while True:
                data = self.write_queue.get()
                if data is None:
                    break
                self._write_checkpoint(data)
        
        self.writer_thread = threading.Thread(target=writer_loop, daemon=False)
        self.writer_thread.start()
    
    def finish(self):
        """Finalize run"""
        # Final checkpoint
        if self.metric_buffer:
            self._checkpoint()
        
        # Stop async writer
        if self.config.async_writes:
            self.write_queue.put(None)
            self.writer_thread.join(timeout=30)
            self.executor.shutdown(wait=True)
        
        # Detach hooks
        self.hook_manager.detach_all()
        
        # Update run
        self.storage.finalize_run(
            run_id=self.run_id,
            end_time=time.time(),
            status='completed',
            total_steps=self.step_count
        )
        
        duration = time.time() - self.start_time
        print(f"\n✓ Run completed: {self.config.run_name}")
        print(f"  Steps: {self.step_count}")
        print(f"  Checkpoints: {self.checkpoint_count}")
        print(f"  Duration: {duration:.2f}s")
```

# Tracer Implementation (Continued)

## CLI Implementation

### `tracer/cli/main.py`

```python
"""CLI tool for Tracer"""

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
from pathlib import Path
import json
import datetime

from tracer.analysis.query import QueryEngine
from tracer.analysis.anomaly_detection import AnomalyDetector

console = Console()


@click.group()
@click.option('--storage', default='./.tracer', help='Storage path')
@click.pass_context
def cli(ctx, storage):
    """
    Tracer CLI - Deep Training Observability Tool
    
    Use 'tracer COMMAND --help' for more information on a command.
    """
    ctx.ensure_object(dict)
    ctx.obj['storage'] = storage
    ctx.obj['query'] = QueryEngine(storage)


@cli.command()
@click.option('--project', help='Filter by project name')
@click.option('--status', help='Filter by status')
@click.option('--limit', default=20, help='Max runs to show')
@click.pass_context
def list(ctx, project, status, limit):
    """List all training runs"""
    query = ctx.obj['query']
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading runs...", total=None)
        runs = query.storage.list_runs(
            project_name=project,
            status=status,
            limit=limit
        )
        progress.update(task, completed=True)
    
    if not runs:
        console.print("[yellow]No runs found[/yellow]")
        return
    
    # Create table
    table = Table(
        title=f"Training Runs ({len(runs)} found)",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta"
    )
    
    table.add_column("Run Name", style="cyan", no_wrap=True)
    table.add_column("Status", justify="center")
    table.add_column("Steps", justify="right", style="green")
    table.add_column("Duration", justify="right")
    table.add_column("Started", style="dim")
    
    for run in runs:
        # Status with emoji
        status_map = {
            'completed': ('✓', 'green'),
            'running': ('▶', 'yellow'),
            'failed': ('✗', 'red'),
            'crashed': ('💥', 'red bold')
        }
        status_emoji, status_style = status_map.get(run['status'], ('?', 'white'))
        status_str = f"[{status_style}]{status_emoji} {run['status']}[/{status_style}]"
        
        # Duration
        if run['end_time']:
            duration = run['end_time'] - run['start_time']
            if duration < 60:
                duration_str = f"{duration:.0f}s"
            elif duration < 3600:
                duration_str = f"{duration/60:.1f}m"
            else:
                duration_str = f"{duration/3600:.1f}h"
        else:
            duration_str = "[dim]running[/dim]"
        
        # Start time
        start_dt = datetime.datetime.fromtimestamp(run['start_time'])
        start_str = start_dt.strftime("%Y-%m-%d %H:%M")
        
        table.add_row(
            run['run_name'],
            status_str,
            str(run['total_steps']),
            duration_str,
            start_str
        )
    
    console.print(table)


@cli.command()
@click.argument('run_name')
@click.option('--metrics', '-m', multiple=True, help='Show specific metrics')
@click.pass_context
def show(ctx, run_name, metrics):
    """Show detailed information about a run"""
    query = ctx.obj['query']
    
    # Find run
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading run...", total=None)
        runs = query.storage.list_runs()
        run = next((r for r in runs if r['run_name'] == run_name), None)
        progress.update(task, completed=True)
    
    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        console.print("\n[dim]Use 'tracer list' to see available runs[/dim]")
        return
    
    summary = query.get_run_summary(run['run_id'])
    
    # Header
    console.print(Panel(
        f"[bold cyan]{summary['run_name']}[/bold cyan]\n"
        f"[dim]{summary['run_id']}[/dim]",
        title="Run Details",
        border_style="cyan"
    ))
    
    # Status info
    status_map = {
        'completed': '✓ Completed',
        'running': '▶ Running',
        'failed': '✗ Failed',
        'crashed': '💥 Crashed'
    }
    status_str = status_map.get(summary['status'], summary['status'])
    
    console.print(f"\n[bold]Status:[/bold] {status_str}")
    console.print(f"[bold]Project:[/bold] {summary['project_name']}")
    console.print(f"[bold]Steps:[/bold] {summary['total_steps']:,}")
    console.print(f"[bold]Checkpoints:[/bold] {summary['total_checkpoints']}")
    console.print(f"[bold]Duration:[/bold] {summary['duration_seconds']:.2f}s")
    
    # Start/End time
    start_dt = datetime.datetime.fromtimestamp(summary['start_time'])
    console.print(f"[bold]Started:[/bold] {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    if summary['end_time']:
        end_dt = datetime.datetime.fromtimestamp(summary['end_time'])
        console.print(f"[bold]Ended:[/bold] {end_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Configuration (collapsed by default)
    if summary['config']:
        console.print("\n[bold]Configuration:[/bold]")
        config_table = Table(show_header=False, box=None, padding=(0, 2))
        config_table.add_column(style="cyan")
        config_table.add_column()
        
        important_keys = ['checkpoint_interval', 'track_gradients', 'track_system_metrics']
        for key in important_keys:
            if key in summary['config']:
                config_table.add_row(key, str(summary['config'][key]))
        
        console.print(config_table)
    
    # Final metrics
    if summary['final_metrics']:
        console.print("\n[bold]Final Metrics:[/bold]")
        metrics_table = Table(show_header=False, box=None, padding=(0, 2))
        metrics_table.add_column(style="green")
        metrics_table.add_column(justify="right")
        
        for key, value in summary['final_metrics'].items():
            if isinstance(value, float):
                value_str = f"{value:.6f}"
            else:
                value_str = str(value)
            metrics_table.add_row(key, value_str)
        
        console.print(metrics_table)
    
    # Anomalies
    if summary['anomalies']:
        console.print("\n[bold yellow]⚠ Anomalies Detected:[/bold yellow]")
        anomaly_table = Table(show_header=False, box=None, padding=(0, 2))
        anomaly_table.add_column(style="red")
        anomaly_table.add_column(justify="right", style="yellow")
        
        for anomaly_type, count in summary['anomalies'].items():
            anomaly_table.add_row(anomaly_type.replace('_', ' ').title(), str(count))
        
        console.print(anomaly_table)
    
    # Show specific metrics if requested
    if metrics:
        console.print("\n[bold]Metric Time Series:[/bold]")
        for metric_name in metrics:
            ts = query.storage.get_metric_timeseries(run['run_id'], metric_name)
            if ts['values']:
                console.print(f"\n[cyan]{metric_name}:[/cyan]")
                console.print(f"  Initial: {ts['values'][0]:.6f}")
                console.print(f"  Final: {ts['values'][-1]:.6f}")
                console.print(f"  Min: {min(ts['values']):.6f}")
                console.print(f"  Max: {max(ts['values']):.6f}")
            else:
                console.print(f"\n[yellow]No data for metric '{metric_name}'[/yellow]")


@cli.command()
@click.argument('run1')
@click.argument('run2')
@click.option('--metric', '-m', default='loss', help='Metric to compare')
@click.pass_context
def compare(ctx, run1, run2, metric):
    """Compare two training runs"""
    query = ctx.obj['query']
    
    # Find runs
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Loading runs...", total=None)
        runs = query.storage.list_runs()
        run1_obj = next((r for r in runs if r['run_name'] == run1), None)
        run2_obj = next((r for r in runs if r['run_name'] == run2), None)
        progress.update(task, completed=True)
    
    if run1_obj is None or run2_obj is None:
        console.print("[red]✗ One or both runs not found[/red]")
        return
    
    # Compare
    console.print(Panel(
        f"[cyan]{run1}[/cyan] vs [cyan]{run2}[/cyan]",
        title="Run Comparison",
        border_style="cyan"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Comparing runs...", total=None)
        comparison = query.compare_runs(run1_obj['run_id'], run2_obj['run_id'], [metric])
        progress.update(task, completed=True)
    
    # Configuration differences
    if comparison['config_diff']:
        console.print("\n[bold]⚙ Configuration Differences:[/bold]")
        
        diff_table = Table(box=box.SIMPLE)
        diff_table.add_column("Parameter", style="cyan")
        diff_table.add_column(run1, style="green")
        diff_table.add_column(run2, style="yellow")
        
        for key, diff in comparison['config_diff'].items():
            diff_table.add_row(
                key,
                str(diff['run1']),
                str(diff['run2'])
            )
        
        console.print(diff_table)
    else:
        console.print("\n[green]✓ Identical configurations[/green]")
    
    # Metric comparison
    if metric in comparison['metrics_comparison']:
        metric_data = comparison['metrics_comparison'][metric]
        
        console.print(f"\n[bold]📊 Metric: {metric}[/bold]")
        
        # Final values
        final1 = metric_data['final_value_run1']
        final2 = metric_data['final_value_run2']
        
        if final1 is not None and final2 is not None:
            console.print(f"  {run1}: {final1:.6f}")
            console.print(f"  {run2}: {final2:.6f}")
            
            diff = abs(final1 - final2)
            pct_diff = (diff / final1) * 100 if final1 != 0 else 0
            
            if final1 < final2:
                console.print(f"  [green]✓ {run1} is better by {pct_diff:.2f}%[/green]")
            else:
                console.print(f"  [green]✓ {run2} is better by {pct_diff:.2f}%[/green]")
    
    # Divergence analysis
    if comparison['divergence_analysis']:
        div = comparison['divergence_analysis']
        
        console.print(f"\n[bold red]⚠ Divergence Detected[/bold red]")
        console.print(f"  Step: {div['step']}")
        console.print(f"  {run1} loss: {div['loss_run1']:.6f}")
        console.print(f"  {run2} loss: {div['loss_run2']:.6f}")
        console.print(f"  Relative diff: {div['relative_diff']:.2%}")
    else:
        console.print("\n[green]✓ No significant divergence detected[/green]")


@cli.command()
@click.argument('run_name')
@click.option('--auto', is_flag=True, help='Auto-detect anomalies')
@click.option('--type', '-t', multiple=True, help='Filter by anomaly type')
@click.pass_context
def anomalies(ctx, run_name, auto, type):
    """Show or detect anomalies in a run"""
    query = ctx.obj['query']
    
    # Find run
    runs = query.storage.list_runs()
    run = next((r for r in runs if r['run_name'] == run_name), None)
    
    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        return
    
    # Auto-detect if requested
    if auto:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Detecting anomalies...", total=None)
            detector = AnomalyDetector(query.storage)
            detected = detector.detect_all_anomalies(run['run_id'])
            detector.store_anomalies(detected)
            progress.update(task, completed=True)
        
        console.print(f"[green]✓ Detected {len(detected)} anomalies[/green]\n")
    
    # Get anomalies
    anomaly_types = list(type) if type else None
    anomalies = query.find_anomalous_checkpoints(run['run_id'], anomaly_types)
    
    if not anomalies:
        console.print("[green]✓ No anomalies detected[/green]")
        console.print("[dim]Use --auto to run anomaly detection[/dim]")
        return
    
    # Display anomalies
    console.print(Panel(
        f"[yellow]Found {len(anomalies)} anomalies[/yellow]",
        title=f"Anomalies: {run_name}",
        border_style="yellow"
    ))
    
    table = Table(box=box.ROUNDED)
    table.add_column("Step", justify="right", style="cyan")
    table.add_column("Type", style="yellow")
    table.add_column("Severity", justify="center")
    table.add_column("Details", style="dim")
    
    severity_colors = {
        'low': 'green',
        'medium': 'yellow',
        'high': 'red',
        'critical': 'red bold'
    }
    
    for anomaly in anomalies:
        details = json.loads(anomaly['details'])
        details_str = ", ".join(f"{k}={v}" for k, v in list(details.items())[:3])
        
        severity_color = severity_colors.get(anomaly['severity'], 'white')
        severity_str = f"[{severity_color}]{anomaly['severity'].upper()}[/{severity_color}]"
        
        table.add_row(
            str(anomaly['step']),
            anomaly['anomaly_type'].replace('_', ' ').title(),
            severity_str,
            details_str
        )
    
    console.print(table)


@cli.command()
@click.argument('run_name')
@click.argument('metric')
@click.option('--export', '-e', help='Export to file (csv/json)')
@click.pass_context
def plot(ctx, run_name, metric, export):
    """Show metric plot (requires matplotlib)"""
    query = ctx.obj['query']
    
    # Find run
    runs = query.storage.list_runs()
    run = next((r for r in runs if r['run_name'] == run_name), None)
    
    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        return
    
    # Get time series
    ts = query.storage.get_metric_timeseries(run['run_id'], metric)
    
    if not ts['values']:
        console.print(f"[yellow]No data for metric '{metric}'[/yellow]")
        return
    
    # Export if requested
    if export:
        import csv
        
        if export.endswith('.csv'):
            with open(export, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['step', 'timestamp', 'value'])
                for i in range(len(ts['steps'])):
                    writer.writerow([ts['steps'][i], ts['timestamps'][i], ts['values'][i]])
            console.print(f"[green]✓ Exported to {export}[/green]")
        
        elif export.endswith('.json'):
            with open(export, 'w') as f:
                json.dump(ts, f, indent=2)
            console.print(f"[green]✓ Exported to {export}[/green]")
        
        return
    
    # Try to plot with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(ts['steps'], ts['values'], linewidth=2)
        plt.xlabel('Step')
        plt.ylabel(metric)
        plt.title(f"{metric} - {run_name}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        console.print("[yellow]matplotlib not installed[/yellow]")
        console.print("[dim]Install with: pip install matplotlib[/dim]")
        console.print("\nMetric values:")
        for i in range(min(10, len(ts['values']))):
            console.print(f"  Step {ts['steps'][i]}: {ts['values'][i]:.6f}")
        if len(ts['values']) > 10:
            console.print(f"  ... and {len(ts['values']) - 10} more")


@cli.command()
@click.argument('run_name')
@click.pass_context
def delete(ctx, run_name):
    """Delete a training run"""
    query = ctx.obj['query']
    
    # Find run
    runs = query.storage.list_runs()
    run = next((r for r in runs if r['run_name'] == run_name), None)
    
    if run is None:
        console.print(f"[red]✗ Run '{run_name}' not found[/red]")
        return
    
    # Confirm
    if not click.confirm(f"Delete run '{run_name}'?"):
        console.print("[yellow]Cancelled[/yellow]")
        return
    
    # Delete from database
    query.storage.conn.execute("DELETE FROM runs WHERE run_id = ?", (run['run_id'],))
    query.storage.conn.commit()
    
    console.print(f"[green]✓ Deleted run '{run_name}'[/green]")


@cli.command()
@click.pass_context
def info(ctx):
    """Show Tracer installation info"""
    from tracer import __version__
    
    console.print(Panel(
        f"[bold cyan]Tracer v{__version__}[/bold cyan]\n"
        f"Deep Training Observability for PyTorch",
        title="Tracer Info",
        border_style="cyan"
    ))
    
    # Storage info
    storage_path = Path(ctx.obj['storage'])
    console.print(f"\n[bold]Storage Path:[/bold] {storage_path}")
    
    if storage_path.exists():
        db_path = storage_path / "tracer.db"
        if db_path.exists():
            db_size = db_path.stat().st_size / (1024 ** 2)
            console.print(f"[bold]Database Size:[/bold] {db_size:.2f} MB")
        
        samples_dir = storage_path / "samples"
        if samples_dir.exists():
            sample_files = list(samples_dir.glob("*.parquet"))
            console.print(f"[bold]Sample Files:[/bold] {len(sample_files)}")
    else:
        console.print("[yellow]Storage directory does not exist[/yellow]")
    
    # Check dependencies
    console.print("\n[bold]Dependencies:[/bold]")
    
    deps = {
        'torch': 'PyTorch',
        'pyarrow': 'Arrow/Parquet',
        'psutil': 'System Metrics',
        'GPUtil': 'GPU Metrics (optional)',
        'matplotlib': 'Plotting (optional)',
    }
    
    for module, name in deps.items():
        try:
            __import__(module)
            console.print(f"  [green]✓[/green] {name}")
        except ImportError:
            console.print(f"  [yellow]✗[/yellow] {name}")


if __name__ == '__main__':
    cli()
```

---

## Analysis Components

### `tracer/analysis/query.py`

```python
"""Query engine for analysis"""

from typing import Dict, List, Any, Optional
import json
from tracer.config import TracerConfig
from tracer.storage.backend import StorageBackend


class QueryEngine:
    """High-level query interface"""
    
    def __init__(self, storage_path: str = "./.tracer"):
        config = TracerConfig(project_name="query", storage_path=storage_path)
        self.storage = StorageBackend(config)
    
    def get_run_summary(self, run_id: str) -> Dict[str, Any]:
        """Get comprehensive run summary"""
        run = self.storage.get_run(run_id)
        if run is None:
            raise ValueError(f"Run {run_id} not found")
        
        checkpoints = self.storage.get_checkpoints(run_id)
        
        # Get final metrics
        final_metrics = {}
        if checkpoints:
            final_checkpoint = checkpoints[-1]
            final_metrics = json.loads(final_checkpoint['metrics'])
        
        # Get anomalies
        anomalies = self.storage.conn.execute("""
            SELECT anomaly_type, COUNT(*) as count
            FROM anomalies
            WHERE run_id = ?
            GROUP BY anomaly_type
        """, (run_id,)).fetchall()
        
        return {
            'run_id': run['run_id'],
            'project_name': run['project_name'],
            'run_name': run['run_name'],
            'status': run['status'],
            'start_time': run['start_time'],
            'end_time': run['end_time'],
            'duration_seconds': (run['end_time'] or 0) - run['start_time'],
            'total_steps': run['total_steps'],
            'total_checkpoints': len(checkpoints),
            'final_metrics': final_metrics,
            'anomalies': {row['anomaly_type']: row['count'] for row in anomalies},
            'config': json.loads(run['config'])
        }
    
    def compare_runs(
        self,
        run_id1: str,
        run_id2: str,
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare two runs"""
        run1 = self.storage.get_run(run_id1)
        run2 = self.storage.get_run(run_id2)
        
        if run1 is None or run2 is None:
            raise ValueError("One or both runs not found")
        
        comparison = {
            'run1': {'run_id': run_id1, 'run_name': run1['run_name']},
            'run2': {'run_id': run_id2, 'run_name': run2['run_name']},
            'config_diff': self._compare_configs(run1, run2),
            'metrics_comparison': {},
            'divergence_analysis': None
        }
        
        # Compare metrics
        if metrics is None:
            metrics = ['loss']
        
        for metric_name in metrics:
            ts1 = self.storage.get_metric_timeseries(run_id1, metric_name)
            ts2 = self.storage.get_metric_timeseries(run_id2, metric_name)
            
            comparison['metrics_comparison'][metric_name] = {
                'run1': ts1,
                'run2': ts2,
                'final_value_run1': ts1['values'][-1] if ts1['values'] else None,
                'final_value_run2': ts2['values'][-1] if ts2['values'] else None,
            }
        
        # Detect divergence
        comparison['divergence_analysis'] = self._detect_divergence(run_id1, run_id2)
        
        return comparison
    
    def _compare_configs(self, run1: Dict, run2: Dict) -> Dict:
        """Compare configurations"""
        config1 = json.loads(run1['config'])
        config2 = json.loads(run2['config'])
        
        diff = {}
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                diff[key] = {'run1': val1, 'run2': val2}
        
        return diff
    
    def _detect_divergence(self, run_id1: str, run_id2: str) -> Optional[Dict]:
        """Detect where runs diverged"""
        import numpy as np
        
        ts1 = self.storage.get_metric_timeseries(run_id1, 'loss')
        ts2 = self.storage.get_metric_timeseries(run_id2, 'loss')
        
        if not ts1['values'] or not ts2['values']:
            return None
        
        # Find common steps
        steps1 = set(ts1['steps'])
        steps2 = set(ts2['steps'])
        common_steps = sorted(steps1 & steps2)
        
        if not common_steps:
            return None
        
        # Align
        aligned_ts1 = []
        aligned_ts2 = []
        for step in common_steps:
            idx1 = ts1['steps'].index(step)
            idx2 = ts2['steps'].index(step)
            aligned_ts1.append(ts1['values'][idx1])
            aligned_ts2.append(ts2['values'][idx2])
        
        # Detect divergence
        diffs = np.abs(np.array(aligned_ts1) - np.array(aligned_ts2))
        relative_diffs = diffs / (np.array(aligned_ts1) + 1e-8)
        
        divergence_indices = np.where(relative_diffs > 0.1)[0]
        
        if len(divergence_indices) == 0:
            return None
        
        divergence_idx = divergence_indices[0]
        divergence_step = common_steps[divergence_idx]
        
        return {
            'step': divergence_step,
            'loss_run1': aligned_ts1[divergence_idx],
            'loss_run2': aligned_ts2[divergence_idx],
            'relative_diff': float(relative_diffs[divergence_idx])
        }
    
    def find_anomalous_checkpoints(
        self,
        run_id: str,
        anomaly_types: Optional[List[str]] = None
    ) -> List[Dict]:
        """Find checkpoints with anomalies"""
        query = "SELECT * FROM anomalies WHERE run_id = ?"
        params = [run_id]
        
        if anomaly_types:
            placeholders = ','.join('?' * len(anomaly_types))
            query += f" AND anomaly_type IN ({placeholders})"
            params.extend(anomaly_types)
        
        query += " ORDER BY step ASC"
        
        cursor = self.storage.conn.execute(query, params)
        return [dict(row) for row in cursor.fetchall()]
```

### `tracer/analysis/anomaly_detection.py`

```python
"""Anomaly detection algorithms"""

from typing import Dict, List, Any
import json
import numpy as np


class AnomalyDetector:
    """Detect training anomalies"""
    
    def __init__(self, storage_backend):
        self.storage = storage_backend
    
    def detect_all_anomalies(self, run_id: str) -> List[Dict[str, Any]]:
        """Run all detectors"""
        anomalies = []
        
        anomalies.extend(self.detect_gradient_anomalies(run_id))
        anomalies.extend(self.detect_loss_anomalies(run_id))
        anomalies.extend(self.detect_nan_inf(run_id))
        anomalies.extend(self.detect_memory_spikes(run_id))
        
        return anomalies
    
    def detect_gradient_anomalies(
        self,
        run_id: str,
        explosion_threshold: float = 100.0,
        vanishing_threshold: float = 1e-7
    ) -> List[Dict]:
        """Detect gradient explosions/vanishing"""
        anomalies = []
        
        cursor = self.storage.conn.execute("""
            SELECT c.checkpoint_id, c.step, g.layer_name, g.norm
            FROM gradient_stats g
            JOIN checkpoints c ON g.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ?
            ORDER BY c.step ASC
        """, (run_id,))
        
        for row in cursor:
            # Explosion
            if row['norm'] > explosion_threshold:
                anomalies.append({
                    'run_id': run_id,
                    'checkpoint_id': row['checkpoint_id'],
                    'step': row['step'],
                    'anomaly_type': 'gradient_explosion',
                    'severity': 'critical' if row['norm'] > 1000 else 'high',
                    'details': {
                        'layer': row['layer_name'],
                        'norm': row['norm'],
                        'threshold': explosion_threshold
                    }
                })
            
            # Vanishing
            if row['norm'] < vanishing_threshold:
                anomalies.append({
                    'run_id': run_id,
                    'checkpoint_id': row['checkpoint_id'],
                    'step': row['step'],
                    'anomaly_type': 'vanishing_gradient',
                    'severity': 'medium',
                    'details': {
                        'layer': row['layer_name'],
                        'norm': row['norm'],
                        'threshold': vanishing_threshold
                    }
                })
        
        return anomalies
    
    def detect_loss_anomalies(self, run_id: str, window_size: int = 10) -> List[Dict]:
        """Detect loss spikes/drops"""
        anomalies = []
        
        from tracer.analysis.query import QueryEngine
        query = QueryEngine(str(self.storage.base_path))
        ts = query.storage.get_metric_timeseries(run_id, 'loss')
        
        if len(ts['values']) < window_size:
            return anomalies
        
        steps = np.array(ts['steps'])
        losses = np.array(ts['values'])
        
        for i in range(window_size, len(losses)):
            window = losses[i-window_size:i]
            mean = np.mean(window)
            std = np.std(window)
            
            if std > 0:
                z_score = (losses[i] - mean) / std
                
                if abs(z_score) > 3:
                    checkpoint_id = f"{run_id}_step_{steps[i]}"
                    
                    anomalies.append({
                        'run_id': run_id,
                        'checkpoint_id': checkpoint_id,
                        'step': int(steps[i]),
                        'anomaly_type': 'loss_spike' if z_score > 0 else 'loss_drop',
                        'severity': 'high' if abs(z_score) > 5 else 'medium',
                        'details': {
                            'loss': float(losses[i]),
                            'window_mean': float(mean),
                            'z_score': float(z_score)
                        }
                    })
        
        return anomalies
    
    def detect_nan_inf(self, run_id: str) -> List[Dict]:
        """Detect NaN/Inf in gradients"""
        anomalies = []
        
        cursor = self.storage.conn.execute("""
            SELECT c.checkpoint_id, c.step, g.layer_name, g.num_nans, g.num_infs
            FROM gradient_stats g
            JOIN checkpoints c ON g.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ? AND (g.num_nans > 0 OR g.num_infs > 0)
        """, (run_id,))
        
        for row in cursor:
            anomalies.append({
                'run_id': run_id,
                'checkpoint_id': row['checkpoint_id'],
                'step': row['step'],
                'anomaly_type': 'nan_or_inf',
                'severity': 'critical',
                'details': {
                    'layer': row['layer_name'],
                    'num_nans': row['num_nans'],
                    'num_infs': row['num_infs']
                }
            })
        
        return anomalies
    
    def detect_memory_spikes(self, run_id: str, threshold: float = 90.0) -> List[Dict]:
        """Detect GPU memory spikes"""
        anomalies = []
        
        cursor = self.storage.conn.execute("""
            SELECT c.checkpoint_id, c.step, s.gpu_metrics
            FROM system_metrics s
            JOIN checkpoints c ON s.checkpoint_id = c.checkpoint_id
            WHERE c.run_id = ?
        """, (run_id,))
        
        for row in cursor:
            gpu_metrics = json.loads(row['gpu_metrics'])
            
            for gpu_id, metrics in gpu_metrics.items():
                memory_util = metrics.get('memory_util', 0)
                
                if memory_util > threshold:
                    anomalies.append({
                        'run_id': run_id,
                        'checkpoint_id': row['checkpoint_id'],
                        'step': row['step'],
                        'anomaly_type': 'memory_spike',
                        'severity': 'high' if memory_util > 95 else 'medium',
                        'details': {
                            'gpu': gpu_id,
                            'memory_util': memory_util,
                            'threshold': threshold
                        }
                    })
        
        return anomalies
    
    def store_anomalies(self, anomalies: List[Dict]):
        """Store detected anomalies"""
        for anomaly in anomalies:
            self.storage.conn.execute("""
                INSERT INTO anomalies 
                (run_id, checkpoint_id, step, anomaly_type, severity, details)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                anomaly['run_id'],
                anomaly['checkpoint_id'],
                anomaly['step'],
                anomaly['anomaly_type'],
                anomaly['severity'],
                json.dumps(anomaly['details'])
            ))
        
        self.storage.conn.commit()
```

---

## Example Usage Files

### `examples/basic_pytorch.py`

```python
"""Basic PyTorch training example with Tracer"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import tracer


# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.layers(x)


def main():
    # Initialize Tracer
    tracer.init(
        project="basic-example",
        run_name="simple-training",
        checkpoint_interval=100,  # Checkpoint every 100 steps
        track_gradients=True,
        track_system_metrics=True
    )
    
    # Create model and optimizer
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Attach Tracer hooks to model (for gradient tracking)
    tracer._state.tracer.attach_hooks(model)
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Training loop
    num_epochs = 10
    
    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Log metrics
            tracer.log({
                "loss": loss.item(),
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]['lr']
            })
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Step tracer (checkpoints automatically)
            tracer.step()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Finish tracer
    tracer.finish()
    
    print("\nTraining complete!")
    print("View results with: tracer show simple-training")


if __name__ == "__main__":
    main()
```

### `examples/huggingface_example.py`

```python
"""HuggingFace Transformers example with Tracer"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
from datasets import load_dataset

import tracer


class TracerCallback(TrainerCallback):
    """Custom callback for HuggingFace Trainer"""
    
    def __init__(self, project_name: str, run_name: str = None):
        self.tracer_instance = None
        self.project_name = project_name
        self.run_name = run_name
    
    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Initialize Tracer"""
        self.tracer_instance = tracer.init(
            project=self.project_name,
            run_name=self.run_name,
            checkpoint_interval=100,
            track_gradients=True
        )
        
        if model is not None:
            self.tracer_instance.attach_hooks(model)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics"""
        if logs and self.tracer_instance:
            tracer.log(logs)
            tracer.step()
    
    def on_train_end(self, args, state, control, **kwargs):
        """Finish Tracer"""
        if self.tracer_instance:
            tracer.finish()


def main():
    # Load dataset
    dataset = load_dataset("imdb", split="train[:1000]")  # Small subset
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Load model and tokenizer
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )
    
    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        logging_steps=10,
        save_steps=1000,
        evaluation_strategy="steps",
        eval_steps=100,
    )
    
    # Create trainer with Tracer callback
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=[TracerCallback(project_name="huggingface-example")]
    )
    
    # Train
    trainer.train()
    
    print("\nTraining complete!")
    print("View results with: tracer list")


if __name__ == "__main__":
    main()
```

---

## Installation and Usage

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/tracer.git
cd tracer

# Install in development mode
pip install -e .

# Or install from PyPI (when published)
pip install ml-tracer
```

### Quick Start

```python
import tracer
import torch
import torch.nn as nn

# Initialize
tracer.init(project="my-project")

# Your model
model = nn.Linear(10, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Attach hooks
tracer._state.tracer.attach_hooks(model)

# Training loop
for epoch in range(10):
    for batch in dataloader:
        loss = train_step(model, batch)
        
        tracer.log({"loss": loss.item()})
        tracer.step()

# Finish
tracer.finish()
```

### CLI Usage

```bash
# List all runs
tracer list

# Show run details
tracer show my-run-name

# Compare two runs
tracer compare run1 run2

# Detect anomalies
tracer anomalies my-run --auto

# Plot metrics
tracer plot my-run loss

# Delete a run
tracer delete my-run

# Show info
tracer info
```

---



1. Copy all files into your project structure
2. Install dependencies: `pip install -e .`
3. Run the examples: `python examples/basic_pytorch.py`
4. Use the CLI: `tracer list`


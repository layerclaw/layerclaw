# API Reference

Complete API documentation for Tracer.

## Core API

### `tracer.init()`

Initialize Tracer for a training run.

```python
tracer.init(
    project: str,
    run_name: Optional[str] = None,
    **kwargs
) -> TrainingTracer
```

**Parameters:**
- `project` (str): Project name for grouping runs
- `run_name` (str, optional): Unique run identifier (auto-generated if None)
- `**kwargs`: Additional configuration (see Configuration)

**Returns:**
- `TrainingTracer`: Tracer instance

**Example:**
```python
tracer.init(
    project="gpt-finetuning",
    run_name="experiment-001",
    checkpoint_interval=500,
    track_gradients=True
)
```

---

### `tracer.log()`

Log metrics for the current step.

```python
tracer.log(metrics: Dict[str, Any]) -> None
```

**Parameters:**
- `metrics` (dict): Dictionary of metric_name -> value

**Example:**
```python
tracer.log({
    "loss": 2.34,
    "accuracy": 0.89,
    "learning_rate": 1e-4
})
```

---

### `tracer.step()`

Increment step counter and checkpoint if needed.

```python
tracer.step() -> None
```

**Example:**
```python
for batch in dataloader:
    loss = train_step(batch)
    tracer.log({"loss": loss.item()})
    tracer.step()  # Auto-checkpoints based on interval
```

---

### `tracer.watch()`

Attach gradient tracking to a model.

```python
tracer.watch(model: nn.Module) -> None
```

**Parameters:**
- `model` (nn.Module): PyTorch model to track

**Example:**
```python
model = MyModel()
tracer.watch(model)
```

---

### `tracer.finish()`

Finalize run and flush all data.

```python
tracer.finish() -> None
```

**Example:**
```python
try:
    # Training loop
    train()
finally:
    tracer.finish()  # Always cleanup
```

---

## Configuration

### `TracerConfig`

```python
from tracer import TracerConfig

config = TracerConfig(
    # Project metadata
    project_name: str,
    run_name: Optional[str] = None,
    tags: List[str] = [],
    notes: Optional[str] = None,
    
    # Storage
    storage_path: str = "./.tracer",
    retention_days: int = 30,
    
    # Capture settings
    checkpoint_interval: int = 1000,
    capture_samples: int = 5,
    track_gradients: bool = True,
    track_system_metrics: bool = True,
    
    # Gradient tracking
    gradient_layers: Optional[List[str]] = None,
    
    # Performance
    async_writes: bool = True,
    write_buffer_size: int = 100,
    max_workers: int = 4,
)
```

---

## Analysis API

### `QueryEngine`

```python
from tracer.analysis import QueryEngine

query = QueryEngine(storage_path="./.tracer")
```

#### Methods

**`get_run_summary(run_id: str) -> Dict`**

Get comprehensive run summary.

```python
summary = query.get_run_summary("abc-123")
print(summary["total_steps"])
print(summary["final_metrics"])
```

**`compare_runs(run_id1: str, run_id2: str, metrics: List[str]) -> Dict`**

Compare two runs.

```python
comparison = query.compare_runs(
    "run1", "run2", 
    metrics=["loss", "accuracy"]
)
```

**`find_anomalous_checkpoints(run_id: str, anomaly_types: List[str]) -> List`**

Find checkpoints with anomalies.

```python
anomalies = query.find_anomalous_checkpoints(
    "run1",
    anomaly_types=["gradient_explosion", "nan_or_inf"]
)
```

---

### `AnomalyDetector`

```python
from tracer.analysis import AnomalyDetector

detector = AnomalyDetector(storage_backend)
```

#### Methods

**`detect_all_anomalies(run_id: str) -> List[Dict]`**

Run all anomaly detectors.

```python
anomalies = detector.detect_all_anomalies("run1")
detector.store_anomalies(anomalies)
```

**`detect_gradient_anomalies(run_id: str) -> List[Dict]`**

Detect gradient explosions/vanishing.

**`detect_loss_anomalies(run_id: str) -> List[Dict]`**

Detect loss spikes/drops.

**`detect_nan_inf(run_id: str) -> List[Dict]`**

Detect NaN/Inf in gradients.

**`detect_memory_spikes(run_id: str) -> List[Dict]`**

Detect GPU memory issues.

---

## Framework Integrations

### HuggingFace Transformers

```python
from tracer.integrations import TransformersCallback
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[
        TransformersCallback(
            project="my-project",
            checkpoint_interval=100
        )
    ]
)
trainer.train()
```

### PyTorch Lightning

```python
from tracer.integrations import LightningCallback
import pytorch_lightning as pl

trainer = pl.Trainer(
    callbacks=[
        LightningCallback(
            project="my-project",
            checkpoint_interval=100
        )
    ]
)
trainer.fit(model)
```

---

## Storage Backend

### `StorageBackend`

Low-level storage interface (advanced usage).

```python
from tracer.storage import StorageBackend
from tracer.config import TracerConfig

config = TracerConfig(project_name="test")
storage = StorageBackend(config)

# Create run
storage.create_run(
    run_id="123",
    project_name="test",
    run_name="run1",
    start_time=1000.0,
    config={}
)

# Write checkpoint
storage.write_checkpoint(
    run_id="123",
    step=100,
    timestamp=1100.0,
    metrics={"loss": 0.5},
    gradient_stats={},
    system_metrics={}
)

# Query
runs = storage.list_runs(project_name="test")
checkpoints = storage.get_checkpoints("123")
```

---

## Exceptions

### `RuntimeError`

Raised when Tracer operations are called without initialization.

```python
try:
    tracer.log({"loss": 1.0})
except RuntimeError as e:
    print("Tracer not initialized!")
```

---

## Type Hints

Tracer is fully typed. Import types for type checking:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tracer.config import TracerConfig
    from tracer.core.tracer import TrainingTracer
```

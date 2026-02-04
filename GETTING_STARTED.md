# Getting Started with LayerClaw

Welcome to LayerClaw! This guide will help you set up and start using the library.

## 🚀 Installation

### From Source (Development)

```bash
# Clone the repository
git clone https://github.com/layerclaw/layerclaw.git
cd layerclaw

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[all]"

# Install pre-commit hooks
pre-commit install
```

### From PyPI (When Published)

```bash
# Basic installation
pip install layerclaw

# With GPU support
pip install layerclaw[gpu]

# With framework integrations
pip install layerclaw[integrations]

# Everything
pip install layerclaw[all]
```

## 📚 Your First Training Run

### Step 1: Basic Example

Create a file `my_training.py`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tracer

# 1. Initialize Tracer
tracer.init(
    project="my-first-project",
    checkpoint_interval=50,
    track_gradients=True
)

# 2. Create your model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# 3. Attach Tracer to track gradients
tracer.watch(model)

# 4. Setup training
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Create dummy data
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataloader = DataLoader(TensorDataset(X, y), batch_size=32)

# 5. Training loop
print("Training started...")
for epoch in range(5):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Log metrics
        tracer.log({
            "loss": loss.item(),
            "epoch": epoch,
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Step Tracer
        tracer.step()
    
    print(f"Epoch {epoch + 1} complete")

# 6. Finish
tracer.finish()
print("\nTraining complete!")
print("Run 'tracer list' to see your runs")
```

### Step 2: Run the Script

```bash
python my_training.py
```

You should see output like:
```
🔍 Tracer initialized: run_20260204_143022_hostname
   Run ID: abc-123-def
   Storage: /path/to/.tracer
Training started...
...
✓ Run completed: run_20260204_143022_hostname
  Steps: 156
  Checkpoints: 3
  Duration: 12.45s
```

### Step 3: Explore Your Run

```bash
# List all runs
tracer list

# Show details about your run
tracer show run_20260204_143022_hostname

# Detect anomalies
tracer anomalies run_20260204_143022_hostname --auto

# Export metrics
tracer plot run_20260204_143022_hostname loss --export metrics.csv
```

## 🎯 Next Steps

### 1. Try Different Features

**Custom Metrics:**
```python
tracer.log({
    "train/loss": loss.item(),
    "train/accuracy": accuracy,
    "lr": lr,
    "gradient_norm": grad_norm
})
```

**Specific Layer Tracking:**
```python
tracer.init(
    project="selective-tracking",
    gradient_layers=["encoder", "decoder"]  # Only track these layers
)
```

**Tags and Notes:**
```python
tracer.init(
    project="experiments",
    tags=["baseline", "v1"],
    notes="Testing new architecture with dropout"
)
```

### 2. Try Framework Integrations

**HuggingFace Transformers:**
```python
from tracer.integrations import TransformersCallback
from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    callbacks=[TransformersCallback(project="my-nlp-project")]
)
trainer.train()
```

**PyTorch Lightning:**
```python
from tracer.integrations import LightningCallback
import pytorch_lightning as pl

trainer = pl.Trainer(
    callbacks=[LightningCallback(project="my-lightning-project")]
)
trainer.fit(model)
```

### 3. Run the Examples

```bash
# Basic PyTorch example
python examples/basic_pytorch.py

# Advanced training loop
python examples/custom_training_loop.py
```

### 4. Read the Documentation

- [Quick Start Guide](docs/quickstart.md)
- [API Reference](docs/api.md)
- [Contributing Guide](CONTRIBUTING.md)

## 🔧 Development Workflow

If you're contributing to Tracer:

```bash
# Setup
make install-dev

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code
make format

# Run linters
make lint

# Run all checks
make dev-test

# Clean build artifacts
make clean
```

## 📊 Understanding the Output

### Storage Structure

Tracer creates a `.tracer/` directory:

```
.tracer/
├── tracer.db          # SQLite database (metadata, metrics)
└── samples/           # Parquet files (sample data)
    └── run_id_step_1000.parquet
```

### Database Schema

- **runs**: Training run metadata
- **checkpoints**: Checkpoint records
- **gradient_stats**: Per-layer gradient statistics
- **system_metrics**: CPU/GPU/memory metrics
- **anomalies**: Detected anomalies

## 🐛 Troubleshooting

### Issue: "Tracer not initialized"

**Solution:** Make sure to call `tracer.init()` before any other Tracer operations.

### Issue: No gradients captured

**Solution:** Call `tracer.watch(model)` after creating your model and after `tracer.init()`.

### Issue: GPU metrics not showing

**Solution:** Install GPUtil:
```bash
pip install gputil
```

### Issue: Storage path permission denied

**Solution:** Specify a custom storage path:
```python
tracer.init(project="test", storage_path="./my_data")
```

### Issue: Pre-commit hooks failing

**Solution:** Format your code:
```bash
make format
```

## 💡 Tips and Best Practices

1. **Always use `tracer.finish()`**: Wrap your training in try/finally to ensure cleanup:
   ```python
   try:
       # Training code
       train()
   finally:
       tracer.finish()
   ```

2. **Checkpoint interval**: Balance between overhead and granularity:
   - Fine-grained: 10-100 steps
   - Standard: 100-500 steps
   - Coarse: 1000+ steps

3. **Gradient tracking**: Only track specific layers for large models:
   ```python
   tracer.init(
       project="large-model",
       gradient_layers=["attention", "ffn"]
   )
   ```

4. **Async writes**: Keep enabled (default) for minimal overhead:
   ```python
   tracer.init(project="fast-training", async_writes=True)
   ```

## 🎓 Learning Resources

- **Examples**: Check `examples/` directory
- **Tests**: Look at `tests/` for usage patterns
- **API Docs**: See `docs/api.md`
- **Source Code**: Explore `tracer/` for implementation details

## 🤝 Getting Help

- **Issues**: [GitHub Issues](https://github.com/yourusername/tracer/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/tracer/discussions)
- **Documentation**: [Read the Docs](https://tracer.readthedocs.io)

## 🎉 You're Ready!

You now have everything you need to start using Tracer. Happy training!

For more advanced usage, check out:
- [Configuration Guide](docs/quickstart.md)
- [CLI Reference](docs/api.md)
- [Contributing](CONTRIBUTING.md)

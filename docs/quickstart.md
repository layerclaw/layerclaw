# Quick Start Guide

Get started with Tracer in 5 minutes.

## Installation

```bash
pip install layerclaw
```

For GPU support:
```bash
pip install layerclaw[gpu]
```

For all features:
```bash
pip install layerclaw[all]
```

## Basic Usage

### 1. Initialize Tracer

```python
import tracer

tracer.init(
    project="my-project",
    checkpoint_interval=100,  # Checkpoint every 100 steps
    track_gradients=True,
)
```

### 2. Attach to Your Model

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 1)
)

# Watch the model for gradient tracking
tracer.watch(model)
```

### 3. Training Loop

```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # Your training code
        loss = train_step(model, batch)
        
        # Log metrics
        tracer.log({"loss": loss.item()})
        
        # Increment step
        tracer.step()
```

### 4. Finish Training

```python
tracer.finish()
```

## Complete Example

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tracer

# Initialize Tracer
tracer.init(project="demo", checkpoint_interval=50)

# Create model
model = nn.Linear(10, 1)
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

# Attach Tracer
tracer.watch(model)

# Create dataset
X = torch.randn(1000, 10)
y = torch.randn(1000, 1)
dataloader = DataLoader(TensorDataset(X, y), batch_size=32)

# Training loop
for epoch in range(5):
    for inputs, targets in dataloader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        tracer.log({"loss": loss.item(), "epoch": epoch})
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tracer.step()

tracer.finish()
```

## CLI Commands

After training, explore your runs:

```bash
# List all runs
tracer list

# Show run details
tracer show demo

# Detect anomalies
tracer anomalies demo --auto

# Compare runs
tracer compare run1 run2
```

## Next Steps

- [Configuration Guide](configuration.md) - Learn about all config options
- [CLI Reference](cli.md) - Complete CLI documentation
- [Integrations](integrations.md) - Use with HuggingFace, Lightning
- [Examples](../examples/) - More complex examples

## Common Issues

### GPU Metrics Not Showing

Install GPUtil:
```bash
pip install gputil
```

### Permission Errors

Check storage path permissions:
```python
tracer.init(project="test", storage_path="./my_data")
```

### Hook Not Capturing Gradients

Make sure to call `tracer.watch(model)` after model creation and before training.

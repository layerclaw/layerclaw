# Tracer Examples

This directory contains example scripts demonstrating various use cases of Tracer.

## Examples

### 1. `basic_pytorch.py`
**Simple PyTorch training with Tracer**

The most basic example showing how to integrate Tracer into a vanilla PyTorch training loop.

```bash
python examples/basic_pytorch.py
```

Features demonstrated:
- Basic initialization
- Gradient tracking
- System metrics
- Automatic checkpointing

### 2. `custom_training_loop.py`
**Advanced training loop**

Shows more advanced features like gradient clipping, learning rate scheduling, and custom metrics.

```bash
python examples/custom_training_loop.py
```

Features demonstrated:
- Gradient clipping tracking
- Learning rate scheduling
- Custom tags and notes
- Advanced metrics logging

### 3. `huggingface_example.py` (TODO)
**HuggingFace Transformers integration**

Example using Tracer with HuggingFace's Trainer API.

```bash
python examples/huggingface_example.py
```

Features demonstrated:
- TracerCallback for Transformers
- Automatic integration
- Dataset tracking

## After Running Examples

View your training runs:
```bash
# List all runs
tracer list

# Show specific run details
tracer show <run-name>

# Detect anomalies
tracer anomalies <run-name> --auto

# Compare two runs
tracer compare run1 run2
```

## Creating Your Own Example

```python
import tracer
import torch.nn as nn

# 1. Initialize
tracer.init(project="my-project")

# 2. Attach to model
tracer.watch(model)

# 3. Training loop
for batch in dataloader:
    loss = train_step(batch)
    tracer.log({"loss": loss.item()})
    tracer.step()

# 4. Finish
tracer.finish()
```

## Directory Structure

```
examples/
├── README.md                  # This file
├── basic_pytorch.py          # Basic example
├── custom_training_loop.py   # Advanced example
├── huggingface_example.py    # HuggingFace integration
└── requirements.txt          # Additional dependencies
```

## Additional Dependencies

Some examples may require additional packages:
```bash
pip install transformers datasets
```

Or install all example dependencies:
```bash
pip install -e ".[integrations]"
```

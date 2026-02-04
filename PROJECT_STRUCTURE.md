# LayerClaw Project Structure

Complete overview of the LayerClaw library structure.

## 📁 Directory Layout

```
tracer/
├── .github/
│   └── workflows/
│       ├── ci.yml                    # CI/CD pipeline
│       └── publish.yml               # PyPI publishing workflow
│
├── docs/
│   ├── quickstart.md                # Quick start guide
│   └── api.md                       # API reference
│
├── examples/
│   ├── README.md                    # Examples documentation
│   ├── basic_pytorch.py             # Basic usage example
│   └── custom_training_loop.py      # Advanced usage example
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py                  # Pytest fixtures
│   ├── test_config.py               # Configuration tests
│   ├── test_hooks.py                # Hook tests
│   ├── test_storage.py              # Storage tests
│   └── test_integration.py          # Integration tests
│
├── tracer/
│   ├── __init__.py                  # Main API exports
│   ├── config.py                    # Configuration management
│   ├── state.py                     # Global state management
│   ├── py.typed                     # Type hints marker
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── tracer.py                # Main Tracer orchestrator
│   │   ├── hooks.py                 # PyTorch hooks
│   │   └── system_metrics.py        # System metrics collection
│   │
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── backend.py               # SQLite + Parquet storage
│   │   └── schema.py                # Database schema
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── query.py                 # Query engine
│   │   └── anomaly_detection.py    # Anomaly detection
│   │
│   ├── cli/
│   │   ├── __init__.py
│   │   └── main.py                  # CLI commands
│   │
│   └── integrations/
│       ├── __init__.py
│       ├── transformers.py          # HuggingFace integration
│       └── lightning.py             # PyTorch Lightning integration
│
├── .gitignore                       # Git ignore rules
├── .pre-commit-config.yaml          # Pre-commit hooks
├── CHANGELOG.md                     # Version history
├── CONTRIBUTING.md                  # Contribution guidelines
├── GETTING_STARTED.md               # Getting started guide
├── LICENSE                          # MIT license
├── MANIFEST.in                      # Package manifest
├── Makefile                         # Development commands
├── PROJECT_STRUCTURE.md             # This file
├── README.md                        # Main documentation
├── pyproject.toml                   # Modern Python packaging
├── setup.py                         # Setup file
└── tracer.md                        # Original design document
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **Main API (`tracer/__init__.py`)**
- `init()` - Initialize Tracer
- `log()` - Log metrics
- `step()` - Increment step counter
- `watch()` - Attach to model
- `finish()` - Cleanup

#### 2. **Configuration (`tracer/config.py`)**
- `TracerConfig` - Configuration dataclass
- Validation and defaults
- Run name generation

#### 3. **State Management (`tracer/state.py`)**
- `GlobalState` - Thread-safe singleton
- Stores current tracer instance
- Manages run lifecycle

#### 4. **Core Module (`tracer/core/`)**

**`tracer.py` - Main Orchestrator**
- `TrainingTracer` class
- Coordinates all components
- Manages lifecycle

**`hooks.py` - PyTorch Hooks**
- `GradientHook` - Gradient statistics
- `HookManager` - Hook coordination
- Thread-safe data capture

**`system_metrics.py` - System Monitoring**
- `SystemMetrics` class
- CPU, memory, GPU tracking
- Optional GPUtil integration

#### 5. **Storage Module (`tracer/storage/`)**

**`backend.py` - Storage Implementation**
- `StorageBackend` class
- SQLite for metadata
- Parquet for samples
- Thread-safe operations

**`schema.py` - Database Schema**
- Table definitions
- Indices for performance
- Schema versioning

#### 6. **Analysis Module (`tracer/analysis/`)**

**`query.py` - Query Engine**
- `QueryEngine` class
- Run summarization
- Run comparison
- Divergence detection

**`anomaly_detection.py` - Anomaly Detection**
- `AnomalyDetector` class
- Gradient anomalies
- Loss spikes/drops
- NaN/Inf detection
- Memory spikes

#### 7. **CLI Module (`tracer/cli/`)**

**`main.py` - CLI Commands**
- `list` - List runs
- `show` - Show run details
- `compare` - Compare runs
- `anomalies` - Detect anomalies
- `delete` - Delete runs
- `info` - Show info
- Rich terminal UI

#### 8. **Integrations Module (`tracer/integrations/`)**

**`transformers.py` - HuggingFace**
- `TracerCallback` for Trainer
- Automatic integration

**`lightning.py` - PyTorch Lightning**
- `TracerCallback` for Lightning
- Automatic integration

## 🔄 Data Flow

```
User Code
    ↓
tracer.init()
    ↓
TrainingTracer
    ├→ HookManager → GradientHook
    ├→ SystemMetrics
    └→ StorageBackend
           ├→ SQLite (metadata)
           └→ Parquet (samples)
    ↓
tracer.log() / tracer.step()
    ↓
Checkpointing (async)
    ↓
tracer.finish()
    ↓
CLI / Analysis
```

## 📊 Database Schema

### Tables

1. **runs**
   - run_id (PK)
   - project_name
   - run_name
   - start_time, end_time
   - status
   - config (JSON)
   - total_steps

2. **checkpoints**
   - checkpoint_id (PK)
   - run_id (FK)
   - step
   - timestamp
   - metrics (JSON)
   - sample_path

3. **gradient_stats**
   - id (PK)
   - checkpoint_id (FK)
   - layer_name
   - norm, mean, std, max, min
   - num_zeros, num_nans, num_infs

4. **system_metrics**
   - id (PK)
   - checkpoint_id (FK)
   - cpu_percent
   - memory_percent
   - gpu_metrics (JSON)

5. **anomalies**
   - id (PK)
   - run_id (FK)
   - checkpoint_id (FK)
   - step
   - anomaly_type
   - severity
   - details (JSON)

## 🧪 Testing Structure

```
tests/
├── conftest.py          # Shared fixtures
├── test_config.py       # Config validation
├── test_hooks.py        # Hook functionality
├── test_storage.py      # Storage operations
└── test_integration.py  # End-to-end tests
```

### Test Coverage Goals
- Core: >95%
- Storage: >90%
- CLI: >80%
- Integrations: >85%

## 🛠️ Development Tools

### Pre-commit Hooks
- trailing-whitespace
- end-of-file-fixer
- black (formatting)
- ruff (linting)
- mypy (type checking)

### CI/CD Pipeline
1. **Lint Job**
   - Black formatting check
   - Ruff linting
   - MyPy type checking

2. **Test Job**
   - Matrix: Python 3.8-3.12
   - Matrix: Ubuntu, macOS, Windows
   - Coverage reporting

3. **Publish Job**
   - Triggered on release
   - Build and publish to PyPI

## 📦 Package Distribution

### Build Artifacts
```
dist/
├── ml_tracer-0.1.0-py3-none-any.whl
└── ml_tracer-0.1.0.tar.gz
```

### Installation Extras
- `[dev]` - Development dependencies
- `[integrations]` - Framework integrations
- `[gpu]` - GPU monitoring (GPUtil)
- `[viz]` - Visualization tools
- `[all]` - Everything

## 🔐 Type Safety

- Full type hints throughout
- `py.typed` marker file
- MyPy strict mode
- Compatible with Pyright/Pylance

## 📝 Documentation Files

1. **README.md** - Main documentation
2. **CONTRIBUTING.md** - How to contribute
3. **CHANGELOG.md** - Version history
4. **GETTING_STARTED.md** - Setup guide
5. **docs/quickstart.md** - Quick start
6. **docs/api.md** - API reference

## 🎯 Design Principles

1. **Minimal Overhead**: Async writes, smart sampling
2. **Easy Integration**: One-line initialization
3. **Framework Agnostic**: Works with any PyTorch code
4. **Type Safe**: Full type hints
5. **Well Tested**: Comprehensive test suite
6. **Extensible**: Plugin architecture for integrations
7. **CLI First**: Powerful command-line tools
8. **Production Ready**: Battle-tested components

## 🚀 Release Process

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Run tests: `make test`
4. Build: `make build`
5. Create GitHub release
6. Publish: `make publish`

## 📈 Future Enhancements

See [CHANGELOG.md](CHANGELOG.md) for planned features.

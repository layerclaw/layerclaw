# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned Features
- Web UI for run visualization
- Distributed training support
- Remote storage backends (S3, GCS, Azure)
- Real-time monitoring dashboard
- Jupyter notebook integration
- TensorBoard export
- Model checkpointing integration

## [0.1.0] - 2026-02-04

### Added
- **Core Features**
  - Deep gradient tracking with statistics (norm, mean, std, min, max)
  - Comprehensive metrics logging with automatic checkpointing
  - System metrics collection (CPU, memory, GPU)
  - Hybrid SQLite + Parquet storage for efficiency
  - Async background writes for minimal training overhead
  - Thread-safe global state management

- **Anomaly Detection**
  - Gradient explosion detection
  - Vanishing gradient detection
  - Loss spike/drop detection using z-score
  - NaN/Inf detection in gradients
  - GPU memory spike detection

- **CLI Tools**
  - `tracer list` - List all training runs
  - `tracer show` - Show detailed run information
  - `tracer compare` - Compare two runs
  - `tracer anomalies` - Detect and display anomalies
  - `tracer delete` - Delete runs
  - `tracer info` - Show installation info
  - Beautiful Rich-based terminal UI

- **Framework Integrations**
  - HuggingFace Transformers callback
  - PyTorch Lightning callback
  - Vanilla PyTorch hook system

- **Analysis Tools**
  - Query engine for analyzing runs
  - Run comparison with divergence detection
  - Time-series metric extraction
  - Configuration diff analysis

- **Developer Experience**
  - Type hints throughout codebase
  - Comprehensive test suite
  - Pre-commit hooks
  - Example scripts
  - Detailed documentation

### Technical Details
- Python 3.8+ support
- PyTorch 1.12+ compatibility
- SQLite for metadata and queryable stats
- Parquet for efficient sample storage
- Async I/O for minimal training impact
- Thread-safe operations

[Unreleased]: https://github.com/yourusername/layerclaw/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yourusername/layerclaw/releases/tag/v0.1.0

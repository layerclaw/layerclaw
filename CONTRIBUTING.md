# Contributing to LayerClaw

Thank you for your interest in contributing to LayerClaw! This document provides guidelines and instructions for contributing.

## 🌟 Ways to Contribute

- **Report bugs** and suggest features via [Issues](https://github.com/yourusername/tracer/issues)
- **Submit pull requests** with bug fixes or new features
- **Improve documentation** - typo fixes, clarifications, examples
- **Write tutorials** and blog posts
- **Help others** in discussions and issues

## 🚀 Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- (Optional) PyTorch with CUDA for GPU testing

### Setup Steps

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/layerclaw/layerclaw.git
   cd layerclaw
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   # Install in editable mode with all dev dependencies
   pip install -e ".[all]"
   ```

4. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

5. **Verify setup**
   ```bash
   pytest tests/
   ```

## 📝 Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/bug-description
```

Branch naming conventions:
- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation changes
- `refactor/` - Code refactoring
- `test/` - Test additions/changes

### 2. Make Changes

- Write clear, concise code
- Follow existing code style (enforced by Black and Ruff)
- Add type hints to all functions
- Write docstrings for public APIs

### 3. Write Tests

- Add tests for new functionality
- Ensure existing tests pass
- Aim for >90% code coverage

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=tracer tests/

# Run specific test file
pytest tests/test_hooks.py

# Run with markers
pytest -m "not slow"  # Skip slow tests
```

### 4. Format and Lint

```bash
# Format code with Black
black tracer/ tests/

# Lint with Ruff
ruff tracer/ tests/

# Type check with mypy
mypy tracer/
```

Or let pre-commit do it all:
```bash
pre-commit run --all-files
```

### 5. Commit Changes

Write clear, descriptive commit messages:

```bash
git commit -m "feat: add gradient clipping detection"
git commit -m "fix: resolve memory leak in async writer"
git commit -m "docs: update quick start example"
```

Commit message prefixes:
- `feat:` - New feature
- `fix:` - Bug fix
- `docs:` - Documentation
- `test:` - Test changes
- `refactor:` - Code refactoring
- `perf:` - Performance improvement
- `chore:` - Maintenance tasks

### 6. Push and Create PR

```bash
git push origin feature/your-feature-name
```

Then create a Pull Request on GitHub with:
- Clear title and description
- Reference related issues
- Screenshots/examples if applicable

## 🧪 Testing Guidelines

### Test Structure

```python
"""Test module for feature X"""
import pytest
from tracer import init, log, step

class TestFeatureX:
    """Tests for feature X"""
    
    def test_basic_functionality(self):
        """Test basic use case"""
        # Arrange
        ...
        # Act
        ...
        # Assert
        ...
    
    def test_edge_case(self):
        """Test edge case handling"""
        ...
    
    @pytest.mark.slow
    def test_performance(self):
        """Test performance characteristics"""
        ...
```

### Test Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.gpu` - Tests requiring GPU

### Fixtures

Use fixtures for common setup:

```python
@pytest.fixture
def temp_tracer():
    """Create temporary tracer instance"""
    tracer = init(project="test", storage_path="/tmp/test_tracer")
    yield tracer
    tracer.finish()
```

## 📚 Documentation

### Docstring Format

Use Google-style docstrings:

```python
def example_function(param1: str, param2: int) -> bool:
    """
    Brief description of function.
    
    Longer description if needed, explaining the purpose,
    behavior, and any important details.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param2 is negative
        RuntimeError: If operation fails
    
    Example:
        >>> result = example_function("test", 42)
        >>> print(result)
        True
    """
    ...
```

### README Updates

If your change affects user-facing functionality, update:
- README.md - Usage examples
- docs/ - Detailed documentation
- CHANGELOG.md - Note the change

## 🎯 Code Style Guidelines

### General Principles

1. **Readability**: Code is read more than written
2. **Simplicity**: Prefer simple solutions over clever ones
3. **Consistency**: Follow existing patterns in the codebase
4. **Documentation**: Document why, not what

### Python Style

- Follow PEP 8
- Use type hints everywhere
- Maximum line length: 100 characters
- Use f-strings for formatting
- Prefer comprehensions over map/filter

### Good Examples

```python
# Good: Clear, typed, documented
def calculate_gradient_norm(
    gradients: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """Calculate L2 norm for each gradient tensor."""
    return {
        name: grad.norm().item()
        for name, grad in gradients.items()
    }

# Bad: No types, unclear
def calc_norm(g):
    return {n: g[n].norm().item() for n in g}
```

## 🐛 Bug Reports

### Before Submitting

1. Check if the bug is already reported
2. Try to reproduce with the latest version
3. Gather relevant information

### Bug Report Template

```markdown
## Bug Description
Clear description of the issue

## To Reproduce
1. Step 1
2. Step 2
3. See error

## Expected Behavior
What should happen

## Environment
- OS: [e.g. Ubuntu 22.04]
- Python version: [e.g. 3.10.5]
- PyTorch version: [e.g. 2.0.0]
- Tracer version: [e.g. 0.1.0]

## Additional Context
Logs, screenshots, etc.
```

## ✨ Feature Requests

### Before Submitting

1. Check if the feature is already requested
2. Consider if it fits Tracer's scope
3. Think about implementation approach

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature

## Use Case
Why is this feature needed? What problem does it solve?

## Proposed Solution
How might this be implemented?

## Alternatives
Other approaches considered

## Additional Context
Mockups, examples from other tools, etc.
```

## 🔒 Security

If you discover a security vulnerability:
1. **DO NOT** open a public issue
2. Email: security@tracer-project.org (or your email)
3. Include detailed information
4. Allow time for patching before disclosure

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for everyone.

### Our Standards

**Positive behavior:**
- Being respectful and inclusive
- Accepting constructive criticism
- Focusing on what's best for the community

**Unacceptable behavior:**
- Harassment, discrimination, or trolling
- Publishing others' private information
- Other unprofessional conduct

### Enforcement

Violations can be reported to: conduct@tracer-project.org

## 📖 Additional Resources

- [GitHub Flow Guide](https://guides.github.com/introduction/flow/)
- [How to Write Good Commit Messages](https://chris.beams.io/posts/git-commit/)
- [Python Type Hints](https://docs.python.org/3/library/typing.html)

## ❓ Questions?

Feel free to ask in:
- [GitHub Discussions](https://github.com/yourusername/tracer/discussions)
- [Issue Tracker](https://github.com/yourusername/tracer/issues)

---

Thank you for contributing to Tracer! 🎉

# Makefile for Tracer development

.PHONY: help install install-dev test lint format clean build publish docs

help:
	@echo "Tracer Development Commands"
	@echo "==========================="
	@echo "install        - Install package"
	@echo "install-dev    - Install package with dev dependencies"
	@echo "test           - Run tests"
	@echo "test-cov       - Run tests with coverage"
	@echo "lint           - Run linters"
	@echo "format         - Format code"
	@echo "clean          - Clean build artifacts"
	@echo "build          - Build package"
	@echo "publish        - Publish to PyPI"
	@echo "docs           - Build documentation"

install:
	pip install -e .

install-dev:
	pip install -e ".[all]"
	pre-commit install

test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=tracer --cov-report=html --cov-report=term-missing

test-fast:
	pytest tests/ -v -m "not slow"

lint:
	ruff tracer/ tests/ examples/
	mypy tracer/ --ignore-missing-imports

format:
	black tracer/ tests/ examples/
	ruff tracer/ tests/ examples/ --fix

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .tracer/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

build: clean
	python -m build

publish: build
	twine upload dist/*

publish-test: build
	twine upload --repository testpypi dist/*

check:
	pre-commit run --all-files

example-basic:
	python examples/basic_pytorch.py

example-custom:
	python examples/custom_training_loop.py

cli-list:
	tracer list

cli-info:
	tracer info

# Development shortcuts
dev-setup: install-dev
	@echo "Development environment ready!"

dev-test: format lint test
	@echo "All checks passed!"

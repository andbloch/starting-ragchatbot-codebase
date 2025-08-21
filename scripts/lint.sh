#!/bin/bash

# Code linting script
# This script runs all linting and type checking tools

set -e

echo "🔍 Running code quality checks..."

echo "  → Running flake8..."
uv run flake8 backend/ main.py

echo "  → Running mypy type checking..."
uv run mypy backend/ main.py

echo "  → Checking import sorting..."
uv run isort --check-only --diff backend/ main.py

echo "  → Checking code formatting..."
uv run black --check --diff backend/ main.py

echo "✅ All quality checks passed!"
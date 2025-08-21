#!/bin/bash

# Code formatting script
# This script formats Python code using Black and sorts imports with isort

set -e

echo "🎨 Formatting Python code..."

echo "  → Running Black..."
uv run black backend/ main.py

echo "  → Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"
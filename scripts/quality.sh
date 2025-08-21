#!/bin/bash

# Complete quality check and formatting script
# This script formats code first, then runs all quality checks

set -e

echo "🚀 Running complete code quality workflow..."

# Format code first
./scripts/format.sh

echo ""

# Run quality checks
./scripts/lint.sh

echo ""
echo "🎉 Quality workflow complete!"
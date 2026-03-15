#!/bin/bash
# Publish DeepGym to PyPI
set -e

echo "Building wheel..."
python -m build --wheel

echo "Publishing to PyPI..."
python -m twine upload dist/*

echo "Done! pip install deepgym should now work."

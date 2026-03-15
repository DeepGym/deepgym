# Contributing to DeepGym

## Development Setup

```bash
git clone https://github.com/abhishekgahlot2/deepgym.git
cd deepgym
pip install -e ".[dev]"
pytest
ruff check src/
```

## Running Tests

```bash
pytest tests/ -v
```

## Code Style

- Python 3.10+
- Ruff for linting and formatting
- Single quotes for strings
- Type annotations on all public functions
- Docstrings on all public classes and methods

## Submitting Changes

1. Fork the repo
2. Create a feature branch
3. Make your changes with tests
4. Run pytest and ruff
5. Submit a PR

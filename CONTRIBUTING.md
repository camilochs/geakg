# Contributing to NS-SE

Thank you for your interest in contributing to NS-SE!

## Development Setup

1. Fork and clone the repository:

```bash
git clone https://github.com/camilochs/geakg.git
cd geakg
```

2. Install dependencies with dev extras:

```bash
uv sync --all-extras
```

3. Set up pre-commit hooks:

```bash
uv run pre-commit install
```

## Code Style

We use automated tools to maintain code quality:

- **Ruff**: Linting and formatting (replaces black, isort, flake8)
- **Mypy**: Static type checking
- **Pydantic**: Runtime type validation

### Running Checks

```bash
# Run all pre-commit checks
uv run pre-commit run --all-files

# Run specific tools
uv run ruff check src/
uv run ruff format src/
uv run mypy src/
```

## Type Hints

All code must include type hints. Use Pydantic models for data structures:

```python
from pydantic import BaseModel, Field

class OperatorNode(BaseModel):
    id: str
    name: str
    parameters: dict[str, Any] = Field(default_factory=dict)
```

## Testing

Write tests for all new functionality:

```bash
# Run tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/unit/test_akg.py
```

### Test Structure

```
tests/
├── unit/           # Unit tests for individual components
├── integration/    # Integration tests for component interactions
└── conftest.py     # Shared fixtures
```

## Documentation

### Docstrings

Use Google-style docstrings:

```python
def get_valid_transitions(self, current_node: str) -> list[str]:
    """Get valid next operations from current node.

    Args:
        current_node: ID of the current operator node.

    Returns:
        List of valid operator IDs that can follow current_node.

    Raises:
        NodeNotFoundError: If current_node doesn't exist in graph.
    """
```

### Documentation Files

Update relevant documentation when making changes:

- `docs/architecture/` - System design docs
- `docs/experiments/` - Experiment documentation
- `README.md` - Main project readme

## Pull Request Process

1. Create a feature branch:

```bash
git checkout -b feature/your-feature-name
```

2. Make your changes with clear commits:

```bash
git commit -m "Add: description of feature"
git commit -m "Fix: description of fix"
```

3. Ensure all checks pass:

```bash
uv run pre-commit run --all-files
uv run pytest
```

4. Push and create PR:

```bash
git push origin feature/your-feature-name
```

5. Fill out the PR template with:
   - Description of changes
   - Related issues
   - Testing performed

## Commit Message Convention

Use prefixes for clarity:

- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Enhancement to existing feature
- `Refactor:` Code restructuring
- `Docs:` Documentation changes
- `Test:` Test additions/modifications
- `Chore:` Maintenance tasks

## Project Structure

When adding new components:

```
src/
├── akg/           # Knowledge graph components
├── retrieval/     # GraphRAG retrieval
├── constraints/   # Symbolic validation
├── llm/           # LLM integration
├── evolution/     # Evolution engine
├── domains/       # Problem domains
├── baselines/     # Baseline implementations
└── utils/         # Shared utilities
```

## Questions?

Open an issue for:
- Bug reports
- Feature requests
- Questions about the codebase

We appreciate your contributions!

#!/usr/bin/env python3
"""Verify NS-SE setup is complete."""

import sys
from pathlib import Path


def check_imports() -> bool:
    """Check if core dependencies can be imported."""
    print("Checking imports...")

    required = [
        ("networkx", "NetworkX"),
        ("numpy", "NumPy"),
        ("scipy", "SciPy"),
        ("sklearn", "scikit-learn"),
        ("pydantic", "Pydantic"),
        ("loguru", "Loguru"),
        ("tqdm", "tqdm"),
        ("yaml", "PyYAML"),
    ]

    all_ok = True
    for module, name in required:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except ImportError:
            print(f"  [FAIL] {name} - not installed")
            all_ok = False

    # Optional: check Ollama
    try:
        import ollama

        print(f"  [OK] Ollama")
    except ImportError:
        print(f"  [WARN] Ollama - not installed (optional for setup)")

    return all_ok


def check_structure() -> bool:
    """Check if project structure is complete."""
    print("\nChecking project structure...")

    root = Path(__file__).parent.parent
    required_dirs = [
        "src/akg",
        "src/retrieval",
        "src/constraints",
        "src/llm",
        "src/evolution",
        "src/domains",
        "src/baselines",
        "src/utils",
        "experiments/configs",
        "experiments/runs",
        "experiments/analysis",
        "data/instances/tsp",
        "data/instances/jssp",
        "data/instances/vrp",
        "tests/unit",
        "tests/integration",
        "docker",
        "docs",
    ]

    all_ok = True
    for dir_path in required_dirs:
        full_path = root / dir_path
        if full_path.exists():
            print(f"  [OK] {dir_path}/")
        else:
            print(f"  [FAIL] {dir_path}/ - missing")
            all_ok = False

    return all_ok


def check_files() -> bool:
    """Check if required files exist."""
    print("\nChecking required files...")

    root = Path(__file__).parent.parent
    required_files = [
        "pyproject.toml",
        "README.md",
        "INSTALLATION.md",
        "CONTRIBUTING.md",
        "LICENSE",
        ".gitignore",
        ".env.example",
        ".pre-commit-config.yaml",
        "docker/Dockerfile",
        "docker/docker-compose.yml",
        "src/__init__.py",
        "src/utils/logging.py",
    ]

    all_ok = True
    for file_path in required_files:
        full_path = root / file_path
        if full_path.exists():
            print(f"  [OK] {file_path}")
        else:
            print(f"  [FAIL] {file_path} - missing")
            all_ok = False

    return all_ok


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible."""
    print("\nChecking Ollama connection...")

    try:
        import ollama

        client = ollama.Client()
        models = client.list()
        print(f"  [OK] Ollama is running")

        # Check for recommended models
        model_names = [m.get("name", "") for m in models.get("models", [])]
        recommended = ["qwen2.5:7b", "llama3.1:8b", "gemma2:9b"]

        for model in recommended:
            if any(model in name for name in model_names):
                print(f"  [OK] Model {model} available")
            else:
                print(f"  [WARN] Model {model} not found (run: ollama pull {model})")

        return True
    except Exception as e:
        print(f"  [WARN] Ollama not accessible: {e}")
        print("         Run 'ollama serve' to start Ollama")
        return False


def main():
    """Run all checks."""
    print("=" * 60)
    print("NS-SE Setup Verification")
    print("=" * 60)

    results = []
    results.append(("Imports", check_imports()))
    results.append(("Structure", check_structure()))
    results.append(("Files", check_files()))
    results.append(("Ollama", check_ollama_connection()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed and name != "Ollama":  # Ollama is optional
            all_passed = False

    print()
    if all_passed:
        print("Setup verification PASSED!")
        print("You can now proceed with FASE 1: AKG + TSP")
        return 0
    else:
        print("Setup verification FAILED!")
        print("Please fix the issues above before continuing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

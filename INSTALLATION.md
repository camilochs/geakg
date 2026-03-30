# Installation Guide

## Prerequisites

### 1. Python 3.11+

Ensure you have Python 3.11 or higher installed:

```bash
python --version  # Should be 3.11+
```

### 2. uv Package Manager

Install uv (fast Python package manager):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with pip
pip install uv
```

### 3. LLM Backend (Choose One)

#### Option A: OpenAI API (Recommended)

```bash
# Set your API key
export OPENAI_API_KEY="your-api-key-here"

# Or add to .env file
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

#### Option B: Ollama (Local LLM)

Install Ollama for local LLM inference:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

# Start Ollama service
ollama serve
```

Pull required models:

```bash
# Principal model
ollama pull qwen2.5:7b

# Alternative models
ollama pull llama3.1:8b
ollama pull gemma2:9b
```

## Installation

```bash
# Clone repository
git clone https://github.com/camilochs/geakg.git
cd geakg

# Install dependencies
uv sync

# Install with dev dependencies (for testing/linting)
uv sync --all-extras

# Verify installation
uv run python -c "from src.geakg.meta_graph import MetaGraph; print('OK')"
```

## Configuration

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```bash
# OpenAI configuration (recommended)
OPENAI_API_KEY=your-api-key-here

# OR Ollama configuration (local)
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OLLAMA_TEMPERATURE=0.7

# General settings
LOG_LEVEL=INFO
CACHE_DIR=.cache/
RANDOM_SEED=42
```

## Quick Start

### Run TSP Experiment

```bash
# With OpenAI (recommended)
uv run python scripts/exp_pure_mode_tsp.py \
    --timeout 300 \
    --async \
    --openai \
    --model gpt-4o-mini

# With Ollama (local)
uv run python scripts/exp_pure_mode_tsp.py \
    --timeout 300
```

### Run Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src
```

## NAS Benchmarks (Optional)

### NAS-Bench-201 (Vision)

For real NAS-Bench-201 data (CIFAR-10/100, ImageNet-16-120):

```bash
# Download NATS-tss-v1_0-3ffb9-simple.tar from
# https://drive.google.com/drive/folders/1zjB6wMANiKwB2A1yil2hQ8H_qyeSe2yt
# and extract to data/

pip install nats_bench
export NASBENCH_PATH=/path/to/NATS-tss-v1_0-3ffb9-simple
```

### NAS-Bench-Graph (GNN)

For NAS-Bench-Graph data (GNN architecture search on 9 graph datasets):

```bash
# Install the nas_bench_graph package
pip install nas_bench_graph
# or with uv:
uv add nas_bench_graph
```

Supported datasets: `cora`, `citeseer`, `pubmed`, `cs`, `physics`, `photo`, `computers`, `arxiv`, `proteins`.

Both benchmarks can run in **proxy mode** (no data needed) for development and smoke tests:

```bash
# NAS-Bench-201 proxy
uv run python scripts/run_nas_benchmark.py --use-proxy --quick

# NAS-Bench-Graph proxy
uv run python scripts/run_nas_graph_benchmark.py --use-proxy --quick
```

## Troubleshooting

### OpenAI API Issues

If you see authentication errors:

```bash
# Verify API key is set
echo $OPENAI_API_KEY

# Test connection
uv run python -c "from openai import OpenAI; c = OpenAI(); print(c.models.list())"
```

### Ollama Connection Issues

If you see connection errors:

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
ollama serve
```

### Memory Issues

For large models (qwen2.5:14b), ensure sufficient RAM:

- Minimum: 16GB RAM
- Recommended: 32GB RAM

### uv Lock File Issues

If dependency resolution fails:

```bash
# Remove lock file and reinstall
rm uv.lock
uv sync
```

## Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8GB | 16GB+ |
| CPU | 4 cores | 8+ cores |
| Disk | 10GB | 20GB |
| GPU | Not required | Optional (speeds up Ollama) |

## Project Structure

```
geakg/
├── src/
│   ├── akg/              # Core framework
│   │   ├── meta_graph.py # MetaGraph definition
│   │   ├── aco.py        # ACO-based traversal
│   │   ├── roles.py      # Abstract Roles
│   │   └── agents/       # Multi-agent system
│   ├── domains/          # Domain implementations
│   └── llm/              # LLM clients (OpenAI, Ollama)
├── scripts/              # Experiment scripts
├── tests/                # Test suite
└── data/                 # TSP instances
```

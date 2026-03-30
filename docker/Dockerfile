FROM python:3.11-slim

WORKDIR /app

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml ./
COPY uv.lock* ./

# Install dependencies
RUN uv sync --frozen --no-dev || uv sync --no-dev

# Copy source code
COPY src/ ./src/
COPY experiments/ ./experiments/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["uv", "run", "python", "-c", "print('NS-SE ready. Run experiments with: uv run python experiments/run_experiment.py')"]

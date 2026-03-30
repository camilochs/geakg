# GEAKG: Generative Executable Algorithm Knowledge Graphs

Knowledge graphs where nodes store executable operators, edges encode learned composition patterns, and traversal generates solutions.

A GEAKG is:
- **Generative** — topology and operators are synthesized by an LLM (offline)
- **Executable** — every node is runnable code, not a static entity
- **Transferable** — learned patterns generalize zero-shot across domains

```
OFFLINE (LLM)                          ONLINE (no LLM)
┌──────────────────────────────┐      ┌─────────────────────┐
│ L0: MetaGraph topology       │      │ Symbolic Executor   │
│ L1: Operator code generation │ ───> │ (pure ACO traversal,│
│ L2: ACO pheromone learning   │      │  zero tokens)       │
└──────────────────────────────┘      └─────────────────────┘
```

## Paper

**GEAKG: Generative Executable Algorithm Knowledge Graphs**
Camilo Chacón Sartori, José H. García, Andrei Voicu Tomut, Christian Blum.
ICN2 (CSIC) & IIIA-CSIC. *(preprint coming soon)*

Two case studies — sharing no domain-specific framework code:
1. **Neural Architecture Search**: 70 cross-dataset transfer pairs on NAS-Bench-201 and NAS-Bench-Graph
2. **Combinatorial Optimization**: TSP → JSSP, QAP (zero-shot, zero tokens)

## Setup

```bash
git clone https://github.com/camilochs/geakg.git
cd geakg
uv sync
cp .env.example .env  # add your OPENAI_API_KEY
```

Python 3.11+. See [INSTALLATION.md](INSTALLATION.md) for NAS benchmarks and Ollama setup.

**Note**: NAS benchmark data (NATS-tss, ~2.3GB) is not included due to size. See INSTALLATION.md for download instructions.

## Usage

Train on TSP (offline — uses LLM):
```bash
uv run python scripts/run_iterative_refinement.py \
    --instances-dir data/instances/tsp_diverse \
    --model gpt-4o-mini
```

Transfer to other domains (online — no LLM, zero tokens):
```bash
uv run python scripts/run_jssp_transfer.py --instance data/instances/jssp/ft06.txt
uv run python scripts/run_qap_transfer.py --instance data/instances/qap/nug12.txt
```

NAS benchmarks:
```bash
uv run python scripts/run_nas_benchmark.py
uv run python scripts/run_nas_graph_benchmark.py
```

## Architecture

```
src/geakg/
├── layers/
│   ├── l0/         # MetaGraph topology (roles, transitions, conditions)
│   ├── l1/         # Operator generation (executable code per role)
│   └── l2/         # Learned knowledge (pheromones, symbolic rules)
├── core/
│   ├── role_schema.py   # RoleSchema ontology (11 abstract roles)
│   └── schemas/         # Domain-specific schemas
├── contexts/       # Problem domains (TSP, JSSP, QAP, NAS)
├── offline/        # Training pipeline (iterative refinement)
├── online/         # Symbolic executor (no LLM at runtime)
├── transfer/       # Cross-domain transfer via GEAKG snapshots
├── aco.py          # Ant Colony Optimization engine
└── execution.py    # Symbolic execution runtime
```

The 11 abstract roles (RoleSchema) define the ontological primitives:
`initializer`, `constructor`, `local_search`, `perturbation`, `crossover`, `mutation`, `selection`, `evaluation`, `repair`, `decoder`, `acceptance_criterion`.

## Tests

```bash
uv run pytest tests/ -v
```

## License

MIT


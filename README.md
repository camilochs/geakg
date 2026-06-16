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

## Reproducibility

The headline results run **offline from a frozen GEAKG snapshot with zero LLM
tokens**. The canonical snapshots are committed under [`artifacts/`](artifacts/),
so the commands below reproduce paper numbers from a fresh clone — no API key, no
GPU, no benchmark download.

**Combinatorial optimization — TSP Symbolic Executor (zero tokens):**
```bash
uv run python scripts/run_symbolic_tsp.py data/instances/tsp/berlin52.tsp \
    --snapshot artifacts/tsp/akg_snapshot.json \
    --pool artifacts/tsp/refined_pool.json \
    --optimal 7542 -t 20
# best gap ~0.03–0.1% on berlin52 (stochastic multistart; the paper reports the
# multi-run figure for the hybrid-50k TSP result).
```

**Knowledge-graph analysis — from the canonical NAS run (`artifacts/nas/`):**
```bash
uv run python scripts/canonical_cora_stats.py   # pheromone convergence: 42 edges, tau in [0.04,1.0]
uv run python scripts/rule_quality_metrics.py   # 42 candidate -> 18 non-redundant rules; confidence/support
uv run python scripts/spectral_analysis.py      # spectral gap 0.187, attractor act_mixed (pi=0.14), entropy 0.71
```

> The full multi-domain transfer tables (TSP→JSSP/QAP) and the 64-pair NAS
> aggregate require the complete trained-result bundle, which is not yet included
> in the repository (data deposit pending). The offline training pipeline
> (`run_iterative_refinement.py`) regenerates snapshots from scratch and **does**
> use LLM tokens.

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

The 11 abstract roles (RoleSchema, `src/geakg/layers/l0/roles.py`) define the
ontological primitives, grouped into 3 categories:
- **Construction**: `const_greedy`, `const_insertion`, `const_savings`, `const_random`
- **Local search**: `ls_intensify_small`, `ls_intensify_medium`, `ls_intensify_large`, `ls_chain`
- **Perturbation**: `pert_escape_small`, `pert_escape_large`, `pert_adaptive`

## Tests

```bash
uv run pytest tests/ -v
```

## License

MIT


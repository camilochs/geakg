# GEAKG — Reproducibility Artifact

This archive reproduces every headline number in the paper *"Transferable knowledge
graphs with executable learned operators for algorithm design"* (GEAKG). All
optimization experiments are **zero-token** (no language-model calls): they run a
frozen, pre-learned snapshot. Nothing here needs a GPU or an API key.

## 1. What is in this archive

The archive **mirrors the repository layout**, so every command below runs with its
default paths — no flags needed.

```
geakg-artifact/
├── README.md                      ← this file
├── pyproject.toml, uv.lock        ← pinned environment
├── scripts/                       ← run_*.py that regenerate each result file
├── src/                           ← the GEAKG codebase the scripts import
├── data/instances/
│   ├── tsp/        (TSPLIB)       └── jssp/parsed/  (Fisher–Thompson, Lawrence, ABZ)
├── experiments/
│   ├── iterative/20260125_121313_iterative/      ← CANONICAL learned snapshot (base)
│   │   ├── akg_snapshot.json      ←   L0 topology + L2 pheromones + symbolic rules (17 KB)
│   │   └── refined_pool.json      ←   L1 operator pool (98 KB)
│   ├── nsse-llamea/knowledge/nsse_50k_gpt5_2_llamea/   ← HYBRID snapshot (base + 1 LLaMEA op, for E3)
│   └── nsse-transfer/instances/   ← qap/  (QAPLIB)  ·  lop/  (reduced-size LOLIB variants)
└── results/
    ├── case_study_2_optimization/ ← E1–E6 (the paper's optimization numbers)
    ├── ablation/                  ← L0/L1/L2 layer-ablation JSON
    └── nas_bench/                 ← NAS-Bench-201 result files
```

This archive is **self-contained**: code (`src/`, `scripts/`), data (`experiments/`,
`data/`, `results/`), and the pinned environment (`pyproject.toml`, `uv.lock`).
Download, `uv sync`, and run any command in Section 3 from the archive root.

## 2. Setup (one command)

```bash
pip install uv          # if not present
uv sync                 # installs the pinned environment from env/uv.lock
```

## 3. Claim → evidence → command (the crystal-clear map)

Each row is one claim in the paper, the result file that backs it, and the exact
command that regenerates that file. Run from the archive root.

| Paper claim | Result file (`results/optimization/`) | Command to regenerate |
|---|---|---|
| **L2 weights do not transfer cross-domain; JSSP transfer is size-dependent** | `E1_jssp_ablation_FIXED.json` | `uv run python scripts/run_jssp_ablation_multiseed.py --instances ft06 ft10 ft20 la01 la06 la16 la21 la26 la31 la36 la40 abz5 abz7 abz9 --seeds 8 --time 30` |
| **QAP: learned ≈ uniform (L2 inert); ILS stronger** (small) | `E2_qap_ablation_FIXED.json` | `uv run python scripts/run_qap_ablation_multiseed.py --instances nug12 nug20 nug30 chr15a chr20a chr25a tai20a tai50a --seeds 5 --time 20` |
| **QAP large: no crossover for pure transfer** | `E2_qap_LARGE.json` | `uv run python scripts/run_qap_ablation_multiseed.py --instances tai80a tai100a sko100a wil100 tai150b tai256c --seeds 3 --time 60 --out results/optimization/E2_qap_LARGE.json` |
| **Hybrid (absorbed LLaMEA op) ≈ pure at matched budget** | `E3_jssp_hybrid_{small,large}.json`, `E3_qap_hybrid_LARGE.json` | same `run_*_ablation_multiseed.py` commands **with** `--snapshot experiments/nsse-llamea/knowledge/nsse_50k_gpt5_2_llamea/akg_snapshot.json` |
| **Win is not a wall-clock artifact: GEAKG beats ILS even at 5× ILS budget** | `E4_jssp_evalparity.json` | `uv run python scripts/run_jssp_evalparity.py --instances abz7 la31 abz9 --seeds 3 --base 30 --mults 1 5` |
| **LOP is a second boundary: ILS higher on every instance** | `E5_lop_ablation.json` | `uv run python scripts/run_lop_ablation_multiseed.py --instances be75np be75oi be75tot stabu70 stabu74 t59b11xx t65b11xx --seeds 3 --time 10` |
| **Boundary is budget-invariant: ILS at 1/5 budget still beats GEAKG (QAP, LOP)** | `E6_qaplop_budget.json` | `uv run python scripts/run_qaplop_budget.py --seeds 3 --time 30` |
| **L0 ontology constraint lowers the search gap** | `ablation/L0_dynamic_*.json` | `uv run python scripts/run_l0_ontology_ablation.py` |
| **L1 operators are load-bearing; robust to noisy operators** | `ablation/l1_robustness.json` | `uv run python scripts/run_l1_robustness.py --instances berlin52 kroA100 ch150 pr226 --seeds 8 --time 10` |
| **No catastrophic forgetting under sequential transfer** | (printed) | `uv run python scripts/check_no_forgetting.py` |
| **Operators in the same role are behaviorally consistent** | (printed) | `uv run python scripts/analyze_role_consistency.py` |

> All commands read `experiments/iterative/20260125_121313_iterative/akg_snapshot.json`
> by default. Run them from the archive root; if you move files, pass `--snapshot <path>`.

## 4. Instance provenance (honest)

- **TSP**: TSPLIB (public).
- **JSSP**: Fisher–Thompson (ft), Lawrence (la), Adams–Balas–Zawack (abz) — standard, public.
- **QAP**: QAPLIB (public).
- **LOP**: the linear-ordering instances are **reduced-size variants (n = 44–60) derived
  from LOLIB matrices**, *not* the full standard LOLIB instances; they are provided in
  `instances/lop/` so the LOP numbers are exactly reproducible from this archive.

## 5. Notes

- Solution values vary slightly run-to-run because the executor is **wall-clock-bounded**
  (not iteration-bounded); the reported numbers are means/medians over the stated seeds.
- The learned snapshot's one-time generation cost is recorded in
  `experiments/iterative/20260125_121313_iterative/akg_snapshot.json` under `token_usage`
  (~50k tokens, output ~24k).
- NAS results are provided for completeness; the NAS-Bench-Graph benchmark data is loaded
  via the `nas_bench_graph` package (a dependency in `env/`).

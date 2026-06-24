#!/usr/bin/env bash
# Assemble the self-contained GEAKG reproducibility archive for Zenodo.
# The archive MIRRORS the repository layout, so every command in README.md runs
# with its default paths (no --snapshot needed). Produces ./zenodo_deposit/geakg-artifact/
# and, with --zip, geakg-artifact.zip ready to upload.
#
#   bash scripts/make_zenodo_deposit.sh           # build the folder
#   bash scripts/make_zenodo_deposit.sh --zip     # build + zip
#   INCLUDE_NAS_GRAPH=1 bash scripts/make_zenodo_deposit.sh   # also include the 132 MB graph data
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

OUT="zenodo_deposit/geakg-artifact"
BASE="experiments/iterative/20260125_121313_iterative"
HYB="experiments/nsse-llamea/knowledge/nsse_50k_gpt5_2_llamea"

echo ">> assembling $OUT (mirrors repo layout so commands run with defaults)"
rm -rf zenodo_deposit
mkdir -p "$OUT"

copy() { mkdir -p "$OUT/$(dirname "$1")"; cp -R "$1" "$OUT/$1"; }   # copy preserving repo path

# 1. README + environment
cp ZENODO_README.md "$OUT/README.md"
copy pyproject.toml; copy uv.lock

# 2. Code: scripts + the src/ they import
copy scripts; copy src

# 3. Learned snapshots (base = canonical TSP-learned; hybrid = base + 1 LLaMEA operator)
copy "$BASE/akg_snapshot.json"; copy "$BASE/refined_pool.json"
copy "$HYB/akg_snapshot.json";  copy "$HYB/refined_pool.json"

# 4. Instances (at their real paths)
copy data/instances/tsp
copy data/instances/jssp/parsed
copy experiments/nsse-transfer/instances/qap
copy experiments/nsse-transfer/instances/lop

# 5. Results — the paper's numbers
mkdir -p "$OUT/results/case_study_2_optimization" "$OUT/results/ablation"
cp results/case_study_2_optimization/E[1-6]*.json "$OUT/results/case_study_2_optimization/"
cp results/ablation/L0_dynamic_*.json results/ablation/l1_robustness.json \
   results/ablation/l2_table*.json results/ablation/bifurcation.json \
   results/ablation/pheromone_ablation_confirm.json "$OUT/results/ablation/" 2>/dev/null || true
copy results/nas_bench
[ "${INCLUDE_NAS_GRAPH:-0}" = "1" ] && copy results/nas_bench_graph

# strip caches
find "$OUT" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$OUT" -name '*.pyc' -delete 2>/dev/null || true

# 6. Summary
echo ">> done."
echo ">> optimization result files: $(ls "$OUT"/results/case_study_2_optimization/*.json | wc -l | tr -d ' ')  (E1-E6)"
echo ">> instances tsp/jssp/qap/lop: $(ls "$OUT"/data/instances/tsp | wc -l|tr -d ' ')/$(ls "$OUT"/data/instances/jssp/parsed|wc -l|tr -d ' ')/$(ls "$OUT"/experiments/nsse-transfer/instances/qap|wc -l|tr -d ' ')/$(ls "$OUT"/experiments/nsse-transfer/instances/lop|wc -l|tr -d ' ')"
echo ">> base snapshot present: $([ -f "$OUT/$BASE/akg_snapshot.json" ] && echo yes || echo NO)"
echo ">> total size: $(du -sh "$OUT" | cut -f1)"

if [ "${1:-}" = "--zip" ]; then
  ( cd zenodo_deposit && zip -qr ../geakg-artifact.zip geakg-artifact )
  echo ">> wrote geakg-artifact.zip ($(du -h geakg-artifact.zip | cut -f1)) — upload this to Zenodo"
fi

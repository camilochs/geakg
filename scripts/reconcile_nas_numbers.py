#!/usr/bin/env python3
"""Recompute the canonical NAS-Bench-Graph aggregate numbers from the result data,
to reconcile the discrepancy between the paper text and REPORT_NAS_BENCH_GRAPH.md.

Canonical set = latest complete batch (timestamps 20260208_2040xx), 10 runs / 200
evals / 8 targets, with reg_evolution present, for the 8 source datasets EXCLUDING
proteins-as-source (paper convention) -> 8 x 8 = 64 (source,target) configurations.

For each pair: win = symbolic mean > baseline mean; significance = one-sided
Wilcoxon signed-rank on the 10 per-run best accuracies (paper uses Wilcoxon).
"""

from __future__ import annotations

import glob
import json
import statistics
from pathlib import Path

from scipy.stats import wilcoxon

SOURCES = ["arxiv", "citeseer", "computers", "cora", "cs", "photo", "physics", "pubmed"]
# proteins is a TARGET but excluded as SOURCE (paper convention)


def latest_file(source):
    fs = sorted(glob.glob(f"results/nas_bench_graph/nas_symbolic_{source}_20260208_2040*.json"))
    return fs[-1] if fs else None


def runs_acc(method_block):
    """Per-run best accuracies; fall back to [mean] if runs absent."""
    runs = method_block.get("runs")
    if runs:
        return [r.get("best_accuracy") for r in runs if r.get("best_accuracy") is not None]
    return [method_block.get("mean_accuracy")]


def sig_greater(a, b):
    """One-sided Wilcoxon signed-rank: is a > b (paired)? Returns p, else nan."""
    try:
        if len(a) != len(b) or len(a) < 3:
            return float("nan")
        diff = [x - y for x, y in zip(a, b)]
        if all(d == 0 for d in diff):
            return 1.0
        return wilcoxon(a, b, alternative="greater").pvalue
    except Exception:
        return float("nan")


def main():
    pairs = []
    for src in SOURCES:
        f = latest_file(src)
        if not f:
            print(f"!! no canonical file for source {src}")
            continue
        d = json.load(open(f))
        for tgt, block in d.get("targets", {}).items():
            m = block.get("methods", {})
            sym = m.get("symbolic_executor"); rnd = m.get("random_search"); re_ = m.get("reg_evolution")
            if not (sym and rnd):
                continue
            sa = runs_acc(sym); ra = runs_acc(rnd)
            row = {"src": src, "tgt": tgt,
                   "sym": statistics.mean(sa), "rnd": statistics.mean(ra),
                   "win_rnd": statistics.mean(sa) > statistics.mean(ra),
                   "sig_rnd": sig_greater(sa, ra)}
            if re_:
                ea = runs_acc(re_)
                row["reg"] = statistics.mean(ea)
                row["win_reg"] = statistics.mean(sa) > statistics.mean(ea)
                row["sig_reg"] = sig_greater(sa, ea)
            pairs.append(row)

    n = len(pairs)
    win_rnd = sum(p["win_rnd"] for p in pairs)
    sig_rnd = sum(1 for p in pairs if p.get("sig_rnd", 1) < 0.05)
    delta_rnd = statistics.mean(p["sym"] - p["rnd"] for p in pairs)
    have_reg = [p for p in pairs if "reg" in p]
    win_reg = sum(p["win_reg"] for p in have_reg)
    sig_reg = sum(1 for p in have_reg if p.get("sig_reg", 1) < 0.05)
    delta_reg = statistics.mean(p["sym"] - p["reg"] for p in have_reg) if have_reg else float("nan")

    print(f"canonical set: {len(SOURCES)} sources x 8 targets = {n} configs "
          f"(reg_evolution present in {len(have_reg)})\n")
    print("=== CANONICAL (recomputed from data) ===")
    print(f"vs Random:  wins {win_rnd}/{n} ({100*win_rnd/n:.0f}%)   "
          f"significant {sig_rnd}/{n} ({100*sig_rnd/n:.0f}%)   mean Δ {delta_rnd:+.2f} pp")
    if have_reg:
        m = len(have_reg)
        print(f"vs RegEvo:  wins {win_reg}/{m} ({100*win_reg/m:.0f}%)   "
              f"significant {sig_reg}/{m} ({100*sig_reg/m:.0f}%)   mean Δ {delta_reg:+.2f} pp")
    print("\n=== vs PAPER / REPORT (for reconciliation) ===")
    print("            paper-text     report.md      RECOMPUTED")
    print(f"vs Rnd win  64/64          64/64          {win_rnd}/{n}")
    print(f"vs Rnd sig  57/64 (89%)    59/64 (92%)    {sig_rnd}/{n} ({100*sig_rnd/n:.0f}%)")
    print(f"vs RE  win  39/64 (61%)    36/64 (56%)    {win_reg}/{len(have_reg)} ({100*win_reg/max(1,len(have_reg)):.0f}%)")
    print(f"vs RE  sig  10/64 (16%)    15/64 (23%)    {sig_reg}/{len(have_reg)} ({100*sig_reg/max(1,len(have_reg)):.0f}%)")

    out = Path("results/nas_bench_graph/RECONCILED_canonical.json")
    with open(out, "w") as f:
        json.dump({"n": n, "vs_random": {"wins": win_rnd, "sig": sig_rnd, "mean_delta": delta_rnd},
                   "vs_regevo": {"wins": win_reg, "n": len(have_reg), "sig": sig_reg, "mean_delta": delta_reg},
                   "pairs": pairs}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()

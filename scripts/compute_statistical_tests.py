#!/usr/bin/env python3
"""Compute Mann-Whitney U tests for optimization experiment results (W6).

Generates p-values for:
  - JSSP: GEAKG (nsse-llamea) vs ILS per instance
  - QAP: GEAKG (nsse) vs ILS and vs Gilmore-Lawler per instance
  - TSP: GEAKG (nsse) vs LLaMEA per instance (from nsse-llamea CSVs)

Note: SPT/LPT baselines are deterministic (single value per instance),
so statistical tests are not applicable to them.

Usage:
    uv run python scripts/compute_statistical_tests.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu


# =============================================================================
# Data loaders
# =============================================================================


def load_transfer_csvs(pattern: str, base_dir: Path) -> list[dict]:
    """Load all semicolon-delimited transfer CSVs matching a glob pattern."""
    rows: list[dict] = []
    for csv_path in sorted(base_dir.glob(pattern)):
        with open(csv_path) as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                if row.get("status") == "ok":
                    try:
                        row["gap"] = float(row["gap"])
                    except (ValueError, KeyError):
                        continue
                    rows.append(row)
    return rows


def load_tsp_csvs(base_dir: Path) -> list[dict]:
    """Load TSP experiment CSVs (headerless: method;model;budget;instance;gap)."""
    rows: list[dict] = []
    for csv_path in sorted(base_dir.glob("*.csv")):
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) >= 5:
                    method, model, budget, instance, gap_str = parts[:5]
                    try:
                        gap = float(gap_str)
                    except ValueError:
                        continue
                    rows.append({
                        "method": method,
                        "model": model,
                        "budget": budget,
                        "instance": instance,
                        "gap": gap,
                    })
    return rows


def load_tsp_jsons(base_dir: Path) -> list[dict]:
    """Load TSP experiment JSONs (array of result dicts)."""
    rows: list[dict] = []
    for json_path in sorted(base_dir.glob("*.json")):
        try:
            with open(json_path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for entry in data:
                    if entry.get("status") == "ok" and "gap" in entry:
                        rows.append(entry)
        except (json.JSONDecodeError, KeyError):
            continue
    return rows


# =============================================================================
# Statistical tests
# =============================================================================


def mann_whitney_test(
    group_a: list[float],
    group_b: list[float],
) -> dict:
    """Perform two-sided Mann-Whitney U test between two groups."""
    if len(group_a) < 2 or len(group_b) < 2:
        return {"U": None, "p": None, "note": "insufficient_samples"}

    # Check if both groups are constant (identical values)
    if len(set(group_a)) == 1 and len(set(group_b)) == 1:
        if group_a[0] == group_b[0]:
            return {"U": None, "p": 1.0, "note": "both_constant_equal"}
        else:
            return {"U": None, "p": None, "note": "both_constant_different"}

    try:
        stat, p_value = mannwhitneyu(group_a, group_b, alternative="two-sided")
        return {
            "U": float(stat),
            "p": float(p_value),
            "significant": p_value < 0.05,
            "n_a": len(group_a),
            "n_b": len(group_b),
            "mean_a": float(np.mean(group_a)),
            "mean_b": float(np.mean(group_b)),
        }
    except ValueError as e:
        return {"U": None, "p": None, "note": str(e)}


def group_by_instance_method(rows: list[dict]) -> dict[str, dict[str, list[float]]]:
    """Group gap values by (instance, method)."""
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        instance = row.get("instance", "unknown")
        method = row.get("method", "unknown")
        grouped[instance][method].append(row["gap"])
    return grouped


# =============================================================================
# Domain-specific analysis
# =============================================================================


def analyze_jssp(base_dir: Path) -> list[dict]:
    """Analyze JSSP: GEAKG (nsse-llamea) vs ILS per instance."""
    rows = load_transfer_csvs("transfer_jssp_*.csv", base_dir)
    grouped = group_by_instance_method(rows)

    results = []
    for instance in sorted(grouped.keys()):
        methods = grouped[instance]
        nsse_gaps = methods.get("nsse-llamea", [])
        ils_gaps = methods.get("ils", [])

        if nsse_gaps and ils_gaps:
            test = mann_whitney_test(nsse_gaps, ils_gaps)
            results.append({
                "domain": "JSSP",
                "instance": instance,
                "comparison": "GEAKG vs ILS",
                **test,
            })

        # Note: SPT/LPT are deterministic
        for det_method in ["spt", "lpt"]:
            det_gaps = methods.get(det_method, [])
            if det_gaps and len(set(det_gaps)) == 1:
                results.append({
                    "domain": "JSSP",
                    "instance": instance,
                    "comparison": f"GEAKG vs {det_method.upper()}",
                    "U": None,
                    "p": None,
                    "note": "deterministic_baseline",
                    "mean_a": float(np.mean(nsse_gaps)) if nsse_gaps else None,
                    "mean_b": det_gaps[0] if det_gaps else None,
                })

    return results


def analyze_qap(base_dir: Path) -> list[dict]:
    """Analyze QAP: GEAKG (nsse) vs ILS and vs Gilmore-Lawler per instance."""
    rows = load_transfer_csvs("transfer_qap_*.csv", base_dir)
    grouped = group_by_instance_method(rows)

    results = []
    for instance in sorted(grouped.keys()):
        methods = grouped[instance]

        # Identify GEAKG method name (could be 'nsse' or 'nsse-llamea')
        nsse_gaps = methods.get("nsse", []) or methods.get("nsse-llamea", [])

        # vs ILS
        ils_gaps = methods.get("ils_basic", []) or methods.get("ils", [])
        if nsse_gaps and ils_gaps:
            test = mann_whitney_test(nsse_gaps, ils_gaps)
            results.append({
                "domain": "QAP",
                "instance": instance,
                "comparison": "GEAKG vs ILS",
                **test,
            })

        # vs Gilmore-Lawler (deterministic)
        gl_gaps = methods.get("gilmore_lawler", [])
        if nsse_gaps and gl_gaps:
            if len(set(gl_gaps)) == 1:
                results.append({
                    "domain": "QAP",
                    "instance": instance,
                    "comparison": "GEAKG vs Gilmore-Lawler",
                    "U": None,
                    "p": None,
                    "note": "deterministic_baseline",
                    "mean_a": float(np.mean(nsse_gaps)),
                    "mean_b": gl_gaps[0],
                })
            else:
                test = mann_whitney_test(nsse_gaps, gl_gaps)
                results.append({
                    "domain": "QAP",
                    "instance": instance,
                    "comparison": "GEAKG vs Gilmore-Lawler",
                    **test,
                })

    return results


def analyze_tsp(base_dir: Path) -> list[dict]:
    """Analyze TSP: GEAKG (nsse) vs LLaMEA per instance from JSON files."""
    rows = load_tsp_jsons(base_dir)
    grouped = group_by_instance_method(rows)

    results = []
    for instance in sorted(grouped.keys()):
        methods = grouped[instance]
        nsse_gaps = methods.get("nsse", [])
        llamea_gaps = methods.get("llamea", [])

        # Also try nsse-llamea key
        if not nsse_gaps:
            nsse_gaps = methods.get("nsse-llamea", [])

        if nsse_gaps and llamea_gaps:
            test = mann_whitney_test(nsse_gaps, llamea_gaps)
            results.append({
                "domain": "TSP",
                "instance": instance,
                "comparison": "GEAKG vs LLaMEA",
                **test,
            })

    # If JSON gave no results, try CSV format
    if not results:
        csv_rows = load_tsp_csvs(base_dir)
        grouped = group_by_instance_method(csv_rows)
        for instance in sorted(grouped.keys()):
            methods = grouped[instance]
            nsse_gaps = methods.get("nsse", []) or methods.get("nsse-llamea", [])
            llamea_gaps = methods.get("llamea", [])

            if nsse_gaps and llamea_gaps:
                test = mann_whitney_test(nsse_gaps, llamea_gaps)
                results.append({
                    "domain": "TSP",
                    "instance": instance,
                    "comparison": "GEAKG vs LLaMEA",
                    **test,
                })

    return results


# =============================================================================
# Output formatting
# =============================================================================


def print_results_table(results: list[dict], domain: str) -> None:
    """Print a formatted table of statistical test results."""
    domain_results = [r for r in results if r["domain"] == domain]
    if not domain_results:
        print(f"\n  No results for {domain}")
        return

    print(f"\n{'='*80}")
    print(f"  {domain} — Mann-Whitney U Tests (two-sided)")
    print(f"{'='*80}")
    print(
        f"  {'Instance':<15} {'Comparison':<25} {'Mean GEAKG':>10} "
        f"{'Mean Base':>10} {'U':>8} {'p-value':>10} {'Sig.':>5}"
    )
    print(f"  {'-'*75}")

    for r in domain_results:
        instance = r.get("instance", "?")
        comparison = r.get("comparison", "?")
        mean_a = r.get("mean_a", None)
        mean_b = r.get("mean_b", None)
        u_stat = r.get("U", None)
        p_val = r.get("p", None)
        note = r.get("note", "")
        sig = r.get("significant", None)

        mean_a_str = f"{mean_a:10.4f}" if mean_a is not None else f"{'N/A':>10}"
        mean_b_str = f"{mean_b:10.4f}" if mean_b is not None else f"{'N/A':>10}"
        u_str = f"{u_stat:8.1f}" if u_stat is not None else f"{'N/A':>8}"
        p_str = f"{p_val:10.4f}" if p_val is not None else f"{'N/A':>10}"

        if note:
            sig_str = note[:5]
        elif sig is True:
            sig_str = "  *"
        elif sig is False:
            sig_str = "   "
        else:
            sig_str = " N/A"

        print(
            f"  {instance:<15} {comparison:<25} {mean_a_str} "
            f"{mean_b_str} {u_str} {p_str} {sig_str}"
        )


def generate_latex_pvalues(
    results: list[dict], domain: str, comparison_filter: str = "ILS"
) -> dict[str, str]:
    """Generate LaTeX-ready p-value strings for each instance.

    Only includes comparisons matching comparison_filter (e.g. 'ILS' or 'LLaMEA').
    """
    latex_pvals: dict[str, str] = {}
    domain_results = [
        r for r in results
        if r["domain"] == domain and comparison_filter in r.get("comparison", "")
    ]

    for r in domain_results:
        instance = r.get("instance", "?")
        p_val = r.get("p", None)
        note = r.get("note", "")

        if note == "deterministic_baseline":
            latex_pvals[instance] = "N/A$^\\dagger$"
        elif note in ("both_constant_equal",):
            latex_pvals[instance] = "$=$"
        elif p_val is not None:
            if p_val < 0.001:
                latex_pvals[instance] = f"$<$0.001"
            elif p_val < 0.01:
                latex_pvals[instance] = f"{p_val:.3f}"
            else:
                latex_pvals[instance] = f"{p_val:.2f}"
        else:
            latex_pvals[instance] = "---"

    return latex_pvals


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    transfer_dir = base / "experiments" / "nsse-transfer" / "results"
    tsp_dir = base / "experiments" / "nsse-llamea" / "results"

    print("=" * 80)
    print("  Statistical Tests for GEAKG Paper (W6)")
    print("  Mann-Whitney U test, two-sided, alpha=0.05")
    print("=" * 80)

    all_results: list[dict] = []

    # JSSP
    print("\n[1/3] Analyzing JSSP transfer results...")
    jssp_results = analyze_jssp(transfer_dir)
    all_results.extend(jssp_results)
    print_results_table(all_results, "JSSP")

    # QAP
    print("\n[2/3] Analyzing QAP transfer results...")
    qap_results = analyze_qap(transfer_dir)
    all_results.extend(qap_results)
    print_results_table(all_results, "QAP")

    # TSP
    print("\n[3/3] Analyzing TSP results...")
    tsp_results = analyze_tsp(tsp_dir)
    all_results.extend(tsp_results)
    print_results_table(all_results, "TSP")

    # Summary
    stochastic = [r for r in all_results if r.get("p") is not None]
    significant = [r for r in stochastic if r.get("significant")]
    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  Total comparisons: {len(all_results)}")
    print(f"  Stochastic (testable): {len(stochastic)}")
    print(f"  Significant (p < 0.05): {len(significant)}/{len(stochastic)}")

    # LaTeX p-value strings for paper tables
    print(f"\n{'='*80}")
    print(f"  LaTeX p-value strings (for paper tables)")
    print(f"{'='*80}")
    for domain, comp_filter in [("JSSP", "ILS"), ("QAP", "ILS"), ("TSP", "LLaMEA")]:
        latex = generate_latex_pvalues(all_results, domain, comp_filter)
        if latex:
            print(f"\n  {domain} (GEAKG vs {comp_filter}):")
            for inst, pstr in sorted(latex.items()):
                print(f"    {inst}: {pstr}")

    # Save JSON output
    output_path = base / "results" / "statistical_tests.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved to: {output_path}")


if __name__ == "__main__":
    main()

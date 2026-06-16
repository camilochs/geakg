#!/usr/bin/env python3
"""Spectral analysis of the learned pheromone-transition matrix (R5: stability
boundaries, attractor formation, traversal entropy).

Builds the row-stochastic role-transition matrix T from the canonical NAS Cora
run, where T[i,j] = P(r_j | r_i) = tau_ij / sum_k tau_ik, and reports:
  - eigenvalue spectrum and the spectral gap 1 - |lambda_2| (SLEM): mixing rate /
    stability margin of the learned traversal dynamics;
  - the stationary distribution pi (dominant left eigenvector): the attractor the
    traversal concentrates on;
  - mean per-row transition entropy (normalized): exploration vs exploitation.

A small uniform teleport (epsilon) makes the chain irreducible so pi is unique.
"""

from __future__ import annotations

import json
from collections import defaultdict

import numpy as np

SRC = "artifacts/nas/nasbench_graph_cora_canonical.json"
EPS = 0.01  # teleport for irreducibility


def main():
    pher = json.load(open(SRC))["runs"][0]["pheromones_display"]
    roles = sorted({r for e in pher for r in e.split("->")})
    idx = {r: i for i, r in enumerate(roles)}
    n = len(roles)

    # weighted adjacency -> row-stochastic transition matrix
    W = np.zeros((n, n))
    for e, tau in pher.items():
        a, b = e.split("->")
        W[idx[a], idx[b]] = tau
    T = np.zeros((n, n))
    for i in range(n):
        s = W[i].sum()
        T[i] = W[i] / s if s > 0 else np.ones(n) / n  # terminal row -> uniform
    # teleport for irreducibility
    Tt = (1 - EPS) * T + EPS * np.ones((n, n)) / n

    # eigenvalues (sorted by modulus, descending)
    eig = np.linalg.eigvals(Tt)
    mod = np.sort(np.abs(eig))[::-1]
    lam2 = mod[1]
    spectral_gap = 1.0 - lam2

    # stationary distribution: left eigenvector for lambda=1 (power iteration)
    pi = np.ones(n) / n
    for _ in range(2000):
        pi = pi @ Tt
        pi /= pi.sum()
    order = np.argsort(pi)[::-1]

    # mean per-row transition entropy (normalized by log of out-degree)
    ents = []
    for i in range(n):
        p = T[i][T[i] > 1e-12]
        if len(p) > 1:
            h = -(p * np.log(p)).sum() / np.log(len(p))
            ents.append(h)
    mean_entropy = float(np.mean(ents))
    pi_entropy = float(-(pi * np.log(pi)).sum() / np.log(n))

    print(f"roles (states): {n}   edges: {len(pher)}")
    print(f"eigenvalue moduli (top 5): {np.round(mod[:5], 3)}")
    print(f"spectral gap (1 - |lambda_2|): {spectral_gap:.3f}  (|lambda_2|={lam2:.3f})")
    print(f"mean per-row transition entropy (norm.): {mean_entropy:.3f}")
    print(f"stationary-distribution entropy (norm.): {pi_entropy:.3f}")
    print(f"\nstationary distribution (attractor) top-6 roles:")
    for j in order[:6]:
        print(f"  {roles[j]:20} pi={pi[j]:.3f}")

    json.dump({"n_states": n, "n_edges": len(pher),
               "eig_moduli_top5": [float(x) for x in mod[:5]],
               "spectral_gap": float(spectral_gap), "lambda2": float(lam2),
               "mean_transition_entropy": mean_entropy, "stationary_entropy": pi_entropy,
               "stationary_top": [(roles[j], float(pi[j])) for j in order[:6]]},
              open("results/nas_bench_graph/spectral_cora.json", "w"), indent=2)


if __name__ == "__main__":
    main()

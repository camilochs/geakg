#!/usr/bin/env python3
"""Regenerate the RoleSchema MetaGraph figures (fig_metagraph_nas / fig_metagraph_opt)
from the real snapshots, in the pink paper palette. Replaces the old static green
PDFs that had no regeneratable source.

Nodes are roles colored by category (pink family); edges are learned transitions,
width/opacity scaled by pheromone weight; a left-to-right multipartite layout
exposes the category pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from paper_palette import PALETTE, apply_paper_style

CAT_COLORS = {
    "topology": "#D6336C", "activation": "#8093F1", "training": "#B388EB",
    "regularization": "#8E1F4F", "evaluation": "#FDC5F5",
    "construction": "#D6336C", "local_search": "#8093F1", "perturbation": "#B388EB",
}
PREFIX_CAT = {"topo": "topology", "act": "activation", "train": "training",
              "reg": "regularization", "eval": "evaluation",
              "const": "construction", "ls": "local_search", "pert": "perturbation"}


def category(role):
    p = role.split("_")[0]
    if p == "ls":
        return "local_search"
    return PREFIX_CAT.get(p, "other")


def short(role):
    s = role.split("_", 1)[1] if "_" in role else role
    return s.replace("intensify_", "int. ").replace("escape_", "esc. ")


def build(edges, cat_order, title, out, figsize):
    apply_paper_style()
    G = nx.DiGraph()
    for (a, b, w) in edges:
        G.add_edge(a, b, weight=w)
    for n in G.nodes():
        G.nodes[n]["layer"] = cat_order.index(category(n))
    pos = nx.multipartite_layout(G, subset_key="layer", align="vertical")

    fig, ax = plt.subplots(figsize=figsize)
    wmax = max((d["weight"] for *_, d in G.edges(data=True)), default=1.0)
    for a, b, d in G.edges(data=True):
        w = d["weight"] / wmax
        rad = 0.18 if category(a) == category(b) else 0.06
        ax.add_patch(FancyArrowPatch(
            pos[a], pos[b], arrowstyle="-|>", mutation_scale=11,
            connectionstyle=f"arc3,rad={rad}", lw=0.6 + 2.4 * w,
            color=PALETTE["geakg"] if w > 0.66 else PALETTE["ils"],
            alpha=0.25 + 0.6 * w, zorder=1, shrinkA=11, shrinkB=11))
    for n in G.nodes():
        ax.scatter(*pos[n], s=430, color=CAT_COLORS.get(category(n), PALETTE["classic"]),
                   edgecolor=PALETTE["ink"], lw=0.8, zorder=3)
        ax.annotate(short(n), pos[n], xytext=(0, -12), textcoords="offset points",
                    fontsize=6.6, ha="center", va="top", color=PALETTE["ink"], zorder=4)
    # category legend
    handles = [plt.Line2D([], [], ls="", marker="o", mfc=CAT_COLORS[c], mec=PALETTE["ink"],
                          ms=10, label=c.replace("_", " ").title()) for c in cat_order]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.08),
              ncol=len(cat_order), frameon=False, fontsize=8, handletextpad=0.2,
              columnspacing=1.0)
    ax.set_axis_off()
    ax.margins(0.12)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"saved {out}  ({G.number_of_nodes()} roles, {G.number_of_edges()} edges)")


def main():
    # Optimization metagraph (TSP snapshot, 11 roles / 3 categories)
    opt = json.load(open("artifacts/tsp/akg_snapshot.json"))
    oe = [(e["source"], e["target"], e.get("weight", 1.0))
          for e in opt["metagraph"]["edges"]]
    build(oe, ["construction", "local_search", "perturbation"],
          "Optimization RoleSchema MetaGraph",
          "paper_els_cas/fig_metagraph_opt.pdf", (5.2, 4.2))

    # NAS metagraph (canonical Cora run, 18 roles / 5 categories)
    nas = json.load(open("artifacts/nas/nasbench_graph_cora_canonical.json"))
    run = nas["runs"][0] if "runs" in nas else nas
    ne = [(k.split("->")[0], k.split("->")[1], v)
          for k, v in run["pheromones_display"].items()]
    build(ne, ["topology", "activation", "training", "regularization", "evaluation"],
          "NAS RoleSchema MetaGraph",
          "paper_els_cas/fig_metagraph_nas.pdf", (6.6, 4.6))


if __name__ == "__main__":
    main()

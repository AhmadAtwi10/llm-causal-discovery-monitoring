"""Compare causal graphs saved by run.py (no ground truth needed).

For every pair of graphs it prints the number of edges, the overlap and the three metrics of the paper:
    Jaccard  = |A ∩ B| / |A ∪ B|
    EDR      = |A △ B| / |A ∪ B|     (edge difference ratio)
    SHD      = |A △ B|               (edges to add or remove to go from one graph to the other)

Usage:
    python scripts/compare_graphs.py logs/cloud_edge_kwollect_10/llm_batch_*_all_bs4.npy
    python scripts/compare_graphs.py a.npy b.npy --heatmap a_vs_b.png
"""
import argparse
import itertools
import json
import os

import numpy as np


def edge_set(adj):
    return {tuple(e) for e in np.argwhere(np.asarray(adj) != 0)}


def compare(adj_a, adj_b):
    a, b = edge_set(adj_a), edge_set(adj_b)
    overlap = len(a & b)
    union = len(a | b)
    diff = len(a ^ b)
    return {
        "edges_a": len(a),
        "edges_b": len(b),
        "overlap": overlap,
        "jaccard": overlap / union if union else 1.0,
        "edr": diff / union if union else 0.0,
        "shd": diff,
    }


def label(path):
    return os.path.splitext(os.path.basename(path))[0]


def variable_names(path, n):
    meta = os.path.splitext(path)[0] + ".json"
    if os.path.exists(meta):
        with open(meta) as f:
            return json.load(f)["variables"]
    return [str(i) for i in range(n)]


def heatmap(path_a, path_b, out):
    import matplotlib.pyplot as plt
    import seaborn as sns

    a, b = np.load(path_a), np.load(path_b)
    names = variable_names(path_a, a.shape[0])
    fig, axs = plt.subplots(1, 3, figsize=(24, 8))
    for ax, mat, title, cmap in [(axs[0], a, label(path_a), "Blues"),
                                 (axs[1], b, label(path_b), "Blues"),
                                 (axs[2], a - b, "difference (A - B)", "bwr")]:
        sns.heatmap(mat, cbar=False, cmap=cmap, center=0 if cmap == "bwr" else None, ax=ax,
                    xticklabels=names, yticklabels=names, square=True)
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"heatmap saved to {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graphs", nargs="+", help="two or more .npy adjacency matrices")
    p.add_argument("--heatmap", help="save a side-by-side heatmap of the first two graphs")
    p.add_argument("--csv", help="also write the table to this CSV file")
    args = p.parse_args()

    if len(args.graphs) < 2:
        p.error("give at least two graphs")

    rows = []
    for path_a, path_b in itertools.combinations(args.graphs, 2):
        r = compare(np.load(path_a), np.load(path_b))
        rows.append({"graph_a": label(path_a), "graph_b": label(path_b), **r})

    w = max(len(r["graph_a"]) for r in rows), max(len(r["graph_b"]) for r in rows)
    print(f"{'graph A':<{w[0]}}  {'graph B':<{w[1]}}  edges A  edges B  overlap  jaccard    EDR  SHD")
    for r in rows:
        print(f"{r['graph_a']:<{w[0]}}  {r['graph_b']:<{w[1]}}  {r['edges_a']:7d}  {r['edges_b']:7d}  "
              f"{r['overlap']:7d}  {r['jaccard']:7.3f}  {r['edr']:5.3f}  {r['shd']:3d}")

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    if args.heatmap:
        heatmap(args.graphs[0], args.graphs[1], args.heatmap)


if __name__ == "__main__":
    main()

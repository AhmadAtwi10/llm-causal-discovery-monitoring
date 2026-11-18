"""Interactive view of a graph saved by run.py: click on an edge to highlight it.

Usage:
    python scripts/plot_graph.py logs/cloud_edge_kwollect_10/llm_batch_deepseek-r1-70b_all_bs7.npy
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def plot_interactive(adj_mat, var_names=None, save_path=None):
    """
    Plots an interactive DAG where clicking on an edge highlights it.

    Parameters:
    - adj_mat (numpy.ndarray): Adjacency matrix of the DAG.
    - var_names (list, optional): List of variable names for labeling nodes. If None, indices are used.
    - save_path (str, optional): Path to save the last viewed state of the DAG.
    """
    G = nx.DiGraph()
    num_nodes = adj_mat.shape[0]

    # Use variable names if provided, otherwise use indices
    if var_names is None:
        var_names = [str(i) for i in range(num_nodes)]

    # Add edges from adjacency matrix
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if adj_mat[i, j] == 1:
                G.add_edge(var_names[i], var_names[j])
                edges.append((var_names[i], var_names[j]))

    # Increase figure size
    fig, ax = plt.subplots(figsize=(20, 12))

    # Improve node positioning
    pos = nx.spring_layout(G, k=2, seed=42)

    # Draw the initial graph
    nodes = nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000, ax=ax)
    labels = nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", 
                                     bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"), ax=ax)
    edges_drawn = nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.8, arrows=True, ax=ax)

    # Function to highlight clicked edge
    def on_click(event):
        if event.inaxes != ax:  # Ignore clicks outside the graph
            return
        
        # Get the click position
        x, y = event.xdata, event.ydata

        # Find the closest edge
        min_dist = float("inf")
        selected_edge = None
        for (u, v), line in zip(edges, edges_drawn):
            x1, y1 = pos[u]
            x2, y2 = pos[v]
            dist = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
            if dist < min_dist and dist < 0.1:  # Adjust 0.1 if necessary
                min_dist = dist
                selected_edge = (u, v)

        if selected_edge:
            # Clear previous edges
            ax.clear()
            
            # Redraw all elements
            nx.draw_networkx_nodes(G, pos, node_color="lightblue", node_size=3000, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold", 
                                    bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"), ax=ax)
            
            # Draw all edges in gray
            nx.draw_networkx_edges(G, pos, edge_color="gray", width=1.5, alpha=0.5, arrows=True, ax=ax)
            
            # Highlight the selected edge in red
            nx.draw_networkx_edges(G, pos, edgelist=[selected_edge], edge_color="red", width=3, alpha=1, arrows=True, ax=ax)
            
            # Display selected edge information
            ax.set_title(f"Selected Edge: {selected_edge[0]} → {selected_edge[1]}", fontsize=16, fontweight="bold")
            
            plt.draw()  # Update the plot dynamically

            # Save the highlighted graph if save_path is provided
            if save_path:
                plt.savefig(save_path, format='png', dpi=300, bbox_inches="tight")
                print(f"Saved highlighted DAG to {save_path}")

    fig.canvas.mpl_connect("button_press_event", on_click)

    plt.title("Interactive Causal DAG - Click to Highlight an Edge", fontsize=16, fontweight="bold")
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("graph", help=".npy adjacency matrix saved by run.py")
    p.add_argument("--save", help="save the highlighted view to this file")
    args = p.parse_args()

    adj = np.load(args.graph)
    meta = os.path.splitext(args.graph)[0] + ".json"
    var_names = json.load(open(meta))["variables"] if os.path.exists(meta) else None
    plot_interactive(adj, var_names, save_path=args.save)


if __name__ == "__main__":
    main()

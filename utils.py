import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def is_dag(W: np.ndarray) -> bool:
    """Return True if the adjacency matrix W is a DAG."""
    G = ig.Graph.Weighted_Adjacency(np.asarray(W).tolist())
    return G.is_dag()


def adj_mat_to_edge_list(adj_mat):
    """List of (i, j) pairs for which adj_mat[i, j] == 1."""
    return [tuple(e) for e in np.argwhere(np.asarray(adj_mat) == 1)]


def plot_dag(adj_mat, var_names=None, save_path=None):
    """Draw the graph of an adjacency matrix ([i, j] = 1 means i -> j); save it if save_path is given."""
    G = nx.DiGraph()
    n = adj_mat.shape[0]
    var_names = list(var_names) if var_names is not None else [str(i) for i in range(n)]
    G.add_nodes_from(var_names)
    for i, j in adj_mat_to_edge_list(adj_mat):
        G.add_edge(var_names[i], var_names[j])

    plt.figure(figsize=(20, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray",
            node_size=2000, font_size=10, font_weight="bold", arrowsize=15)
    plt.title("Discovered causal graph")
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

from causallearn.search.ConstraintBased.FCI import fci as causallearn_fci
import numpy as np


def dag_to_adj_mat(dag):
    # Transform causal learn format into bn format
    n = len(dag.graph)
    adj_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if dag.graph[i][j] == -1:
                adj_mat[i][j] = 1
    return adj_mat


def fci(data):
    """
    Wrapper for fci algorithm from causallearn package. Takes in data and returns adjacency matrix of a DAG
    """
    predicted_dag, _ = causallearn_fci(data)
    predicted_adj_mat = dag_to_adj_mat(predicted_dag)
    return predicted_adj_mat    

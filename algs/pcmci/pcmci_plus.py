import numpy as np
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.gpdc import GPDC
from sklearn.gaussian_process.kernels import RBF

def normalize_data(data, eps=1e-8):
    """
    Z-score normalization: (x - mean) / std
    """
    return (data - data.mean(axis=0)) / (data.std(axis=0) + eps)

def pcmci_plus_wrapper(data, var_names, tau_max=3, pc_alpha=0.05, p_threshold=0.05, normalize=True):
    """
    Wrapper for the PCMCI+ algorithm using GPDC independence test.

    Args:
        data (np.ndarray): Time-series data (T, D)
        var_names (list): List of variable names in the correct order
        tau_max (int): Maximum time lag to consider
        pc_alpha (float): Significance level for PC-stable step
        threshold (float): P-value threshold to binarize causal edges
        normalize (bool): Whether to normalize the data

    Returns:
        np.ndarray: Binary adjacency matrix (D, D), where [i, j] = 1 means i -> j (at some lag)
    """
    if normalize:
        data = normalize_data(data)

    dataframe = pp.DataFrame(data, var_names=var_names)

    kernel = 1.0 * RBF(length_scale_bounds=(1e-7, 1e5))
    gpdc = GPDC(significance='analytic', gp_params={'kernel': kernel, 'alpha': 1e-3})

    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=gpdc)
    results = pcmci.run_pcmciplus(tau_max=tau_max, pc_alpha=pc_alpha)

    p_matrix = results['p_matrix']  # p_matrix[i, j, tau]: link i (at t - tau) -> j (at t)

    # Edge i -> j if the link is significant at any lag 1..tau_max; self-loops (autocorrelation) are dropped
    predicted_adj_mat = (p_matrix[:, :, 1:tau_max + 1] < p_threshold).any(axis=2).astype(int)
    np.fill_diagonal(predicted_adj_mat, 0)
    return predicted_adj_mat

import numpy as np
from lingam import VARLiNGAM


def normalize_data(data, eps=1e-8):
    """Z-score normalization: (x - mean) / std."""
    return (data - data.mean(axis=0)) / (data.std(axis=0) + eps)


def varlingam_wrapper(data, lag=1, threshold=0.05, normalize=True):
    """
    Run VARLiNGAM and return a binary adjacency matrix.

    The absolute coefficients of the first `lag` matrices returned by VARLiNGAM (index 0 is the
    instantaneous matrix B0, index k the lag-k matrix) are summed, and an edge is kept when the total
    is >= `threshold`.

    Args:
        data (np.ndarray): time series, shape (T, D)
        lag (int): number of lags of the VAR model
        threshold (float): minimum aggregated coefficient for an edge
        normalize (bool): z-score the data before fitting

    Returns:
        np.ndarray: binary matrix (D, D) where [i, j] = 1 means i -> j (self-loops removed)
    """
    if normalize:
        data = normalize_data(data)
    model = VARLiNGAM(lags=lag)
    model.fit(data)
    # lingam convention: B[i, j] != 0 means j -> i, so transpose to get [cause, effect]
    total = sum(np.abs(B) for B in model.adjacency_matrices_[:lag]).T
    adj = (total >= threshold).astype(int)
    np.fill_diagonal(adj, 0)
    return adj

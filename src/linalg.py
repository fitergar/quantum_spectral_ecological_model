# src/linalg.py
from __future__ import annotations

import numpy as np
import torch
from numba import njit


def laplacian_4nbrs(df, step: int = 10, xcol: str = "X", ycol: str = "Y"):
    """
    Dense 4-neighbor combinatorial Laplacian on a rectilinear grid.

    Neighbor criterion: |dx| + |dy| == step
    L = D - A (symmetric)
    """
    X = df[xcol].to_numpy()
    Y = df[ycol].to_numpy()

    dL1 = np.abs(X[:, None] - X[None, :]) + np.abs(Y[:, None] - Y[None, :])
    A = (dL1 == step).astype(np.float64)
    deg = A.sum(axis=1)

    L = -A
    np.fill_diagonal(L, deg)
    return torch.from_numpy(L)


@njit
def minevec_iter(H: np.ndarray, x: np.ndarray, iters: int = 100_000) -> np.ndarray:
    """
    Power iteration on H (largest |eig|). Use on H^{-1} or (H^{-1})^k
    to approximate smallest eigenvector of original H.
    """
    for _ in range(iters):
        x = H @ x
        x = x / np.linalg.norm(x)
    return x


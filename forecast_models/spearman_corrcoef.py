import numpy as np
from scipy.stats import spearmanr

def spearman_corrcoef(x, y=None):
    """
    Calculate the Spearman correlation coefficient matrix.

    Parameters:
    - x: array_like, shape (n, m)
        Array where each row is a dataset and each column represents a variable.
    - y: array_like, shape (n, m), optional
        An additional array representing another dataset. If provided, the correlation coefficient
        between x and y will be calculated. Otherwise, the correlation coefficient matrix for x
        will be calculated.

    Returns:
    - corr: ndarray, shape (m, m) or shape (m,)
        The Spearman correlation coefficient matrix or correlation coefficient
        vector if only one input array is provided.
    """
    if y is None:
        x = np.asarray(x)
        if x.ndim < 2:
            raise ValueError("x must have at least 2 dimensions")
        return spearmanr(x, axis=1)[0]
    else:
        x = np.asarray(x)
        y = np.asarray(y)
        if x.shape != y.shape:
            raise ValueError("x and y must have the same shape")
        return spearmanr(x, y, axis=1)[0]
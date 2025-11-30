"""
Utility functions shared across analysis modules
"""

import numpy as np
from scipy.optimize import fsolve


def solveNu(X, Y, id_US):
    """
    Solve for nu parameters using aggregate data
    
    Parameters:
    -----------
    X : array-like
        Trade flow matrix (N x N)
    Y : array-like
        GDP vector (N,)
    id_US : int
        US country ID (1-indexed as in MATLAB)
    
    Returns:
    --------
    nu : array
        nu parameters [nu_non_US, nu_US]
    """
    N = X.shape[0]
    
    # Create aggregation matrix: US vs non-US
    AggI = np.zeros((2, N))
    AggI[0, :] = 1.0
    AggI[0, id_US - 1] = 0  # MATLAB is 1-indexed, Python is 0-indexed
    AggI[1, id_US - 1] = 1
    
    # Aggregate trade flows and GDP
    X_agg = AggI @ X @ AggI.T
    Y_agg = AggI @ Y
    
    # Initial guess
    nu0 = np.array([0.1, 0.24])
    
    # Solve system
    def eqFun(nu):
        E_i = Y_agg + (1 - nu) * (X_agg.sum(axis=1) - (np.outer((1 - nu), np.ones(2)) * X_agg).sum(axis=1))
        
        r_11 = (E_i[0] - X_agg[1, 0]) / (E_i[0] - X_agg[1, 0] + X_agg[0, 1])
        r_22 = (E_i[1] - X_agg[0, 1]) / (E_i[1] - X_agg[0, 1] + X_agg[1, 0])
        
        F = np.zeros(2)
        F[0] = (1 - r_11) * nu[1] + r_11 * nu[0] - 0.12
        F[1] = r_22 * nu[1] + (1 - r_22) * nu[0] - 0.26
        return F
    
    nu = fsolve(eqFun, nu0, xtol=1e-10, maxfev=100000)
    nu = np.maximum(nu, 0)  # nu(nu<0) = 0
    return nu


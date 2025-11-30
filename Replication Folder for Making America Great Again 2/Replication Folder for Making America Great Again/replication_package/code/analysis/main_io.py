"""
Main IO analysis - Python conversion of main_io.m
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve, minimize, least_squares
import sys
import pickle
import time

from utils import solveNu


def Balanced_Trade_IO(x, data, param, debug=False):
    """
    Balanced trade equilibrium function with IO structure
    
    Args:
        x: Solution vector [w_i_h, E_i_h, L_i_h, P_i_h]
        data: [N, E_i, Y_i, lambda_ji, t_ji, nu, T_i]
        param: [eps, kappa, psi, phi, beta]
        debug: If True, print debug information
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data
    eps, kappa, psi, phi, beta = param
    
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    P_i_h = np.abs(x[N+N+N:N+N+N+N])
    
    # Clip to bounds to prevent extreme values (bounds are [0.1, 5.0])
    # This ensures the solver stays in reasonable regions
    w_i_h = np.clip(w_i_h, 0.1, 5.0)
    E_i_h = np.clip(E_i_h, 0.1, 5.0)
    L_i_h = np.clip(L_i_h, 0.1, 5.0)
    P_i_h = np.clip(P_i_h, 0.1, 5.0)
    
    if debug:
        print(f"      Balanced_Trade_IO inputs: w_i_h=[{w_i_h.min():.4f}, {w_i_h.max():.4f}], "
              f"P_i_h=[{P_i_h.min():.4f}, {P_i_h.max():.4f}]")
    
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    
    # Construct new trade values
    # Add safeguards only where needed to prevent numerical issues
    # Ensure P_i_h is positive to avoid division by zero
    P_i_h_safe = np.maximum(P_i_h, 1e-10)
    w_i_h_safe = np.maximum(w_i_h, 1e-10)
    L_i_h_safe = np.maximum(L_i_h, 1e-10)
    
    c_i_h = np.tile((w_i_h_safe ** beta) * (P_i_h_safe ** (1 - beta)), (N, 1))
    entry = np.tile((w_i_h_safe / P_i_h_safe) ** (1 - beta), (N, 1))
    # Prevent division by zero
    entry_safe = np.maximum(entry, 1e-10)
    L_entry = np.maximum(entry_safe * L_i_h_safe.reshape(-1, 1), 1e-10)
    p_ij_h = ((c_i_h / (L_entry ** psi)) ** -eps) * ((1 + t_ji) ** (-eps * phi_2D))
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(AUX0.sum(axis=0).reshape(1, -1), (N, 1))
    lambda_ji_new = AUX0 / AUX1
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = (lambda_ji_new * (t_ji / (1 + t_ji)) * np.tile(E_i_new.reshape(1, -1), (N, 1))).sum(axis=0)
    
    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    
    # Wage Income = Total Sales net of Taxes
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = (beta * (1 - nu_2D) * X_ji_new).sum(axis=1) + (nu_2D * X_ji_new).sum(axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = ((w_i_h - 1) * Y_i).mean()  # Replace one excess equation
    
    # Total Income = Total Sales
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + ((1 - beta) * (1 - nu_2D) * X_ji_new).sum(axis=1) + T_i * (X_global_new / X_global) - E_i_new
    
    # Labor supply equation
    # Avoid negative values in sqrt by ensuring the expression is positive
    labor_supply_expr = tau_i_h * w_i_h / P_i_h
    # Clip to positive values to avoid sqrt warnings
    labor_supply_expr = np.maximum(labor_supply_expr, 1e-10)
    ERR3 = L_i_h - labor_supply_expr ** kappa
    
    # Price equation
    # MATLAB: ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1 - phi)) .* ( sum(AUX0,1).^(-1./eps)');
    # sum(AUX0,1) in MATLAB sums along dimension 1 (columns), giving (1 x N), then ' transposes to (N x 1)
    # In Python, sum(axis=0) sums along rows (axis 0), giving (N,)
    AUX0_sum = AUX0.sum(axis=0)  # (N,) - sum over rows (imports for each country)
    # Ensure it's positive to avoid numerical issues
    AUX0_sum = np.maximum(AUX0_sum, 1e-10)
    price_index_term = AUX0_sum ** (-1.0 / eps)  # (N,)
    
    # Safeguard the (E_i_h / w_i_h) term to prevent extreme values
    E_w_ratio = E_i_h / np.maximum(w_i_h, 1e-10)
    E_w_ratio = np.clip(E_w_ratio, 1e-6, 1e6)  # Prevent extreme ratios
    
    # Safeguard (1 - phi) - if phi is very large, this could be negative or very large
    exp_term = 1 - phi
    exp_term = np.clip(exp_term, -5, 5)  # Limit exponent to prevent extreme values
    
    # Calculate price term with safeguards
    price_term = (E_w_ratio ** exp_term) * price_index_term
    price_term = np.clip(price_term, 1e-10, 1e10)
    
    ERR4 = P_i_h - price_term
    
    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])
    
    # Calculate results
    Ec_i = Y_i + T_i
    # Avoid division by zero in delta_i calculation
    delta_denom = Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa)
    delta_denom = np.maximum(delta_denom, 1e-6)  # Prevent division by zero
    delta_i = Ec_i / delta_denom
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
    # Avoid division by zero in welfare calculation
    P_i_h_welfare = np.maximum(P_i_h, 1e-10)  # Prevent division by zero
    W_i_h = delta_i * (Ec_i_h / P_i_h_welfare) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h_welfare)
    # Clip welfare to prevent negative or extreme values
    W_i_h = np.maximum(W_i_h, 0.01)  # Ensure minimum of 0.01 (1% of baseline)
    W_i_h = np.clip(W_i_h, 0.01, 100.0)  # Clip to reasonable range
    
    # Debug logging: Check for extreme values (only warn if truly extreme to reduce noise)
    # Only print warnings for very extreme values to avoid flooding output during optimization
    if np.any(np.isnan(W_i_h)) or np.any(np.isinf(W_i_h)):
        # Only print once per call using a simple check
        if not hasattr(Balanced_Trade_IO, '_warned_nan'):
            print(f"    ⚠️  WARNING: NaN or Inf in W_i_h!")
            Balanced_Trade_IO._warned_nan = True
    if np.any(W_i_h < 0.001) or np.any(W_i_h > 1000):
        if not hasattr(Balanced_Trade_IO, '_warned_w_extreme'):
            print(f"    ⚠️  WARNING: Extreme W_i_h values: min={W_i_h.min():.4f}, max={W_i_h.max():.4f}")
            Balanced_Trade_IO._warned_w_extreme = True
    if np.any(P_i_h < 0.001) or np.any(P_i_h > 1000):
        if not hasattr(Balanced_Trade_IO, '_warned_p_extreme'):
            print(f"    ⚠️  WARNING: Extreme P_i_h values: min={P_i_h.min():.4f}, max={P_i_h.max():.4f}")
            Balanced_Trade_IO._warned_p_extreme = True
    
    # Factual trade flows
    X_ji = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    eye_N = np.eye(N)
    # MATLAB uses opposite sign convention (surplus = positive, deficit = negative)
    D_i = X_ji.sum(axis=1) - X_ji.sum(axis=0)  # Flipped: exports - imports
    D_i_new = X_ji_new.sum(axis=1) - X_ji_new.sum(axis=0)  # Flipped: exports - imports
    
    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * (((X_ji_new * (1 - eye_N)).sum(axis=1) / Y_i_new) / 
                      ((X_ji * (1 - eye_N)).sum(axis=1) / Y_i) - 1)
    d_import = 100 * (((X_ji_new * (1 - eye_N)).sum(axis=0) / Y_i_new) / 
                      ((X_ji * (1 - eye_N)).sum(axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))
    
    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, 
                               d_employment, d_CPI, tariff_rev / E_i])
    
    trade = X_ji * (1 - eye_N)
    trade_new = X_ji_new * (1 + t_ji) * (1 - eye_N)
    d_trade = 100 * ((trade_new.sum() / trade.sum()) / 
                     (Y_i_new.sum() / Y_i.sum()) - 1)
    
    return ceq, results, d_trade


def const_mpec(x, data, param, id, tariff_case):
    """
    Constraint function for MPEC optimization
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data
    eps, kappa, psi, phi, beta = param
    
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    P_i_h = np.abs(x[N+N+N:N+N+N+N])
    t = np.abs(x[N+N+N+N:])
    
    t_ji_copy = t_ji.copy()
    eye_N = np.eye(N)
    
    if tariff_case == 1:
        non_id = np.setdiff1d(np.arange(N), [id - 1])  # Convert to 0-indexed
        t_ji_copy[non_id, id - 1] = t
    elif tariff_case == 2:
        non_id = np.setdiff1d(np.arange(N), [id - 1])
        t_ji_copy[id - 1, non_id] = t
    
    t_ji_copy[eye_N == 1] = 0
    
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    
    # Construct new trade values
    c_i_h = np.tile((w_i_h ** beta) * (P_i_h ** (1 - beta)), (N, 1))
    entry = np.tile((w_i_h / P_i_h) ** (1 - beta), (N, 1))
    p_ij_h = ((c_i_h / ((entry * L_i_h.reshape(-1, 1)) ** psi)) ** -eps) * ((1 + t_ji_copy) ** (-eps * phi_2D))
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(AUX0.sum(axis=0).reshape(1, -1), (N, 1))
    lambda_ji_new = AUX0 / AUX1
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji_copy)
    tariff_rev = (lambda_ji_new * (t_ji_copy / (1 + t_ji_copy)) * np.tile(E_i_new.reshape(1, -1), (N, 1))).sum(axis=0)
    
    tau_i = tariff_rev / Y_i_new
    tau_i_new = 0
    tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    
    # Wage Income = Total Sales net of Taxes
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = (beta * (1 - nu_2D) * X_ji_new).sum(axis=1) + (nu_2D * X_ji_new).sum(axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = ((w_i_h - 1) * Y_i).mean()  # Replace one excess equation
    
    # Total Income = Total Sales
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + ((1 - beta) * (1 - nu_2D) * X_ji_new).sum(axis=1) + T_i * (X_global_new / X_global) - E_i_new
    
    # Labor supply equation
    # Avoid negative values in sqrt by ensuring the expression is positive
    labor_supply_expr = tau_i_h * w_i_h / P_i_h
    # Clip to positive values to avoid sqrt warnings
    labor_supply_expr = np.maximum(labor_supply_expr, 1e-10)
    ERR3 = L_i_h - labor_supply_expr ** kappa
    
    # Price equation
    # MATLAB: ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1 - phi)) .* ( sum(AUX0,1).^(-1./eps)');
    # sum(AUX0,1) in MATLAB sums along dimension 1 (columns), giving (1 x N), then ' transposes to (N x 1)
    # In Python, sum(axis=0) sums along rows (axis 0), giving (N,)
    AUX0_sum = AUX0.sum(axis=0)  # (N,) - sum over rows (imports for each country)
    # Ensure it's positive to avoid numerical issues
    AUX0_sum = np.maximum(AUX0_sum, 1e-10)
    price_index_term = AUX0_sum ** (-1.0 / eps)  # (N,)
    
    # Safeguard the (E_i_h / w_i_h) term to prevent extreme values
    E_w_ratio = E_i_h / np.maximum(w_i_h, 1e-10)
    E_w_ratio = np.clip(E_w_ratio, 1e-6, 1e6)  # Prevent extreme ratios
    
    # Safeguard (1 - phi) - if phi is very large, this could be negative or very large
    exp_term = 1 - phi
    exp_term = np.clip(exp_term, -5, 5)  # Limit exponent to prevent extreme values
    
    # Calculate price term with safeguards
    price_term = (E_w_ratio ** exp_term) * price_index_term
    price_term = np.clip(price_term, 1e-10, 1e10)
    
    ERR4 = P_i_h - price_term
    
    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])
    c = np.array([])
    
    return c, ceq


def obj_mpec(x, data, param, id, tariff_case):
    """
    Objective function for MPEC optimization
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data
    eps, kappa, psi, phi, beta = param
    
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    P_i_h = np.abs(x[N+N+N:N+N+N+N])
    t = np.abs(x[N+N+N+N:])
    
    t_ji_copy = t_ji.copy()
    eye_N = np.eye(N)
    
    if tariff_case == 1:
        non_id = np.setdiff1d(np.arange(N), [id - 1])
        t_ji_copy[non_id, id - 1] = t
    elif tariff_case == 2:
        non_id = np.setdiff1d(np.arange(N), [id - 1])
        t_ji_copy[id - 1, non_id] = t
    
    t_ji_copy[eye_N == 1] = 0
    
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    
    # Construct new trade values
    c_i_h = np.tile((w_i_h ** beta) * (P_i_h ** (1 - beta)), (N, 1))
    entry = np.tile((w_i_h / P_i_h) ** (1 - beta), (N, 1))
    p_ij_h = ((c_i_h / ((entry * L_i_h.reshape(-1, 1)) ** psi)) ** -eps) * ((1 + t_ji_copy) ** (-eps * phi_2D))
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(AUX0.sum(axis=0).reshape(1, -1), (N, 1))
    lambda_ji_new = AUX0 / AUX1
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    tariff_rev = (lambda_ji_new * (t_ji_copy / (1 + t_ji_copy)) * np.tile(E_i_new.reshape(1, -1), (N, 1))).sum(axis=0)
    
    tau_i = tariff_rev / Y_i_new
    
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    Ec_i = Y_i + T_i
    delta_i = Ec_i / (Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i
    P_i_h_safe = np.maximum(P_i_h, 1e-10)
    W_i_h = delta_i * (Ec_i_h / P_i_h_safe) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h_safe)
    # Clip welfare to prevent negative or extreme values
    W_i_h = np.maximum(W_i_h, 0.01)
    W_i_h = np.clip(W_i_h, 0.01, 100.0)
    
    if tariff_case == 1:
        gains = -100 * (W_i_h[id - 1] - 1)
    elif tariff_case == 2:
        non_id_indices = np.concatenate([np.arange(id - 1), np.arange(id, N)])
        gains = -100 * (Y_i[non_id_indices] * (W_i_h[non_id_indices] - 1)).sum() / Y_i[non_id_indices].sum()
    
    return gains


def main(quick_test=False):
    """
    Main IO analysis
    
    Args:
        quick_test: If True, use reduced iterations for faster testing (500 instead of 5000)
    """
    # Read trade and GDP data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
    
    data = pd.read_csv(os.path.join(data_dir, 'trade_cepii.csv'))
    X_ji = data.values
    X_ji = np.nan_to_num(X_ji, nan=0.0)
    N = X_ji.shape[0]
    id_US = 185
    
    # GDP data
    gdp = pd.read_csv(os.path.join(data_dir, 'gdp.csv'))
    Y_i = gdp.values.flatten()
    Y_i = Y_i / 1000  # Trade flows are in 1000 of USD
    
    tot_exports = X_ji.sum(axis=1)
    tot_imports = X_ji.sum(axis=0)
    
    nu_eq = solveNu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US - 1] = nu_eq[1]
    
    T = (1 - nu) * (X_ji.sum(axis=0) - (np.outer((1 - nu), np.ones(N)) * X_ji).sum(axis=1))
    E_i = Y_i + T
    X_ii = E_i - tot_imports
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)
    
    beta = 0.49
    nu_IO = nu.copy()
    X_ji_IO = X_ji.copy()
    np.fill_diagonal(X_ji_IO, np.diag(X_ji_IO) / beta)
    E_i_IO = X_ji_IO.sum(axis=0)
    Y_i_IO = beta * (np.outer((1 - nu_IO), np.ones(N)) * X_ji_IO).sum(axis=1) + nu_IO * X_ji_IO.sum(axis=0)
    lambda_ji_IO = X_ji_IO / np.tile(E_i_IO.reshape(1, -1), (N, 1))
    T_IO = E_i_IO - (Y_i_IO + (1 - beta) * (np.outer((1 - nu_IO), np.ones(N)) * X_ji_IO).sum(axis=1))
    
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))
    
    # Read USTR tariffs
    reuters = pd.read_csv(os.path.join(data_dir, 'tariffs.csv'))
    new_ustariff = reuters.values.flatten()
    t_ji = np.zeros((N, N))
    t_ji[:, id_US - 1] = new_ustariff
    
    t_ji[:, id_US - 1] = np.maximum(0.1, t_ji[:, id_US - 1])
    t_ji[id_US - 1, id_US - 1] = 0
    tariff = [t_ji]
    
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    Phi = [1 + phi_tilde, 0.5 + phi_tilde]
    
    results = np.zeros((N, 7, 4))
    d_trade_IO = np.zeros(2)
    d_employment_IO = np.zeros(2)
    
    # Roundabout Production
    t_ji_new = tariff[0]
    phi_IO = Phi[0]
    
    # Use initial guess with bounds to prevent extreme values
    # Set reasonable bounds: [0.1, 5.0] for all variables (relative changes from baseline)
    print("  Using bounded solver to prevent extreme values...")
    x0 = np.ones(4 * N)
    bounds = ([0.1] * (4 * N), [5.0] * (4 * N))  # Reasonable bounds for w, E, L, P
    
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    syst = lambda x: Balanced_Trade_IO(x, data, param)[0]
    
    # For quick test, use faster unbounded solver; for full run, use bounded solver
    from scipy.optimize import least_squares
    
    if quick_test:
        print("  ⚡ QUICK TEST: Using bounded solver with reduced iterations...")
        print("  (Bounds prevent extreme values - using 1000 iterations for speed)")
        max_iter = 1000
        try:
            # Use bounded TRF method even for quick test (prevents extreme values)
            # But with fewer iterations and relaxed tolerance for speed
            res_ls = least_squares(syst, x0, method='trf', bounds=bounds, 
                                   xtol=1e-5, max_nfev=max_iter, verbose=1)
            if res_ls.success:
                x_fsolve_1 = res_ls.x
                ier = 1
                msg = "least_squares (bounded, quick test) converged"
                print(f"  ✓ least_squares (bounded) converged (cost: {res_ls.cost:.2e})")
            else:
                print(f"  ⚠ Bounded solver did not fully converge (cost: {res_ls.cost:.2e}), using result anyway...")
                x_fsolve_1 = res_ls.x
                ier = 4
                msg = f"least_squares (bounded) did not fully converge"
        except Exception as e:
            print(f"  ⚠ Bounded solver failed: {e}, trying unbounded LM...")
            # Fallback to unbounded if bounded fails
            res_ls = least_squares(syst, x0, method='lm', xtol=1e-5, max_nfev=1000, verbose=1)
            x_fsolve_1 = res_ls.x if res_ls.success else x0
            ier = 1 if res_ls.success else 4
            msg = "least_squares (LM fallback)" if res_ls.success else "solver failed"
    else:
        print("  Solving IO model equilibrium with bounds...")
        print("  (This may take several minutes - bounded solver is more robust)")
        max_iter = 50000
        try:
            # Use bounded TRF method for full run (more robust)
            res_ls = least_squares(syst, x0, method='trf', bounds=bounds, 
                                   xtol=1e-8, max_nfev=max_iter, verbose=1)
            if res_ls.success:
                x_fsolve_1 = res_ls.x
                ier = 1
                msg = "least_squares (bounded) converged"
                print(f"  ✓ least_squares (bounded) converged (cost: {res_ls.cost:.2e})")
            else:
                print(f"  least_squares (bounded) did not converge (cost: {res_ls.cost:.2e}), trying without bounds...")
                # Fallback to unbounded least_squares
                res_ls_unbounded = least_squares(syst, x0, method='lm', xtol=1e-8, max_nfev=20000, verbose=0)
                if res_ls_unbounded.success:
                    x_fsolve_1 = res_ls_unbounded.x
                    ier = 1
                    msg = "least_squares (unbounded) converged"
                    print(f"  ✓ least_squares (unbounded) converged (cost: {res_ls_unbounded.cost:.2e})")
                else:
                    x_fsolve_1 = res_ls.x  # Use bounded result even if not fully converged
                    ier = 4
                    msg = f"least_squares did not fully converge (cost: {res_ls_unbounded.cost:.2e})"
        except Exception as e:
            print(f"  least_squares failed: {e}, trying fsolve as fallback...")
            # Final fallback to fsolve
            x_fsolve_1, info, ier, msg = fsolve(syst, x0, xtol=1e-8, maxfev=50000, full_output=True)
    
    # Check if solver converged
    if ier != 1:
        print(f"  WARNING: Solver did not converge properly. ier={ier}, msg={msg}")
        print(f"  This may cause incorrect results. Continuing anyway...")
    
    # Validate solution and log intermediate values
    ceq_check, results_temp, _ = Balanced_Trade_IO(x_fsolve_1, data, param)
    max_error = np.abs(ceq_check).max()
    if max_error > 1e-6:
        print(f"  WARNING: Solution may be inaccurate. Max error: {max_error:.2e}")
    
    # Log intermediate values for debugging
    w_i_h = np.abs(x_fsolve_1[:N])
    E_i_h = np.abs(x_fsolve_1[N:N+N])
    L_i_h = np.abs(x_fsolve_1[N+N:N+N+N])
    P_i_h = np.abs(x_fsolve_1[N+N+N:N+N+N+N])
    
    print(f"  Initial solution validation:")
    print(f"    w_i_h range: [{w_i_h.min():.4f}, {w_i_h.max():.4f}]")
    print(f"    E_i_h range: [{E_i_h.min():.4f}, {E_i_h.max():.4f}]")
    print(f"    L_i_h range: [{L_i_h.min():.4f}, {L_i_h.max():.4f}]")
    print(f"    P_i_h range: [{P_i_h.min():.4f}, {P_i_h.max():.4f}]")
    print(f"    Max equilibrium error: {max_error:.2e}")
    
    # Check for extreme values
    if P_i_h.max() > 1000 or P_i_h.min() < 0.001:
        print(f"  ⚠️  WARNING: Extreme P_i_h values detected! This may cause incorrect results.")
    if w_i_h.max() > 1000 or w_i_h.min() < 0.001:
        print(f"  ⚠️  WARNING: Extreme w_i_h values detected!")
    
    _, results[:, :, 0], d_trade_IO[0] = Balanced_Trade_IO(x_fsolve_1, data, param)
    d_employment_IO[0] = (results[:, 4, 0] * Y_i_IO).sum() / Y_i_IO.sum()
    
    # Optimal tariff + IO
    # Note: fmincon optimization is complex - using scipy.optimize.minimize instead
    from scipy.optimize import minimize
    
    tariff_case = 1
    t_ji_new = np.zeros((N, N))
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    LB = np.concatenate([0.75 * x_fsolve_1, np.zeros(N - 1)])
    UB = np.concatenate([1.5 * x_fsolve_1, 0.25 * np.ones(N - 1)])
    x0_opt = np.concatenate([x_fsolve_1, 0.15 * np.ones(N - 1)])
    
    # Ensure LB < UB for all elements
    LB, UB = np.minimum(LB, UB), np.maximum(LB, UB)
    
    def constraint_fun(x):
        _, ceq = const_mpec(x, data, param, id_US, tariff_case)
        return ceq
    
    def objective_fun(x):
        return obj_mpec(x, data, param, id_US, tariff_case)
    
    # Progress callback for optimization with time estimation
    iteration_count = [0]
    start_time = [time.time()]
    
    def callback(xk):
        iteration_count[0] += 1
        if iteration_count[0] % 50 == 0:
            elapsed = time.time() - start_time[0]
            if iteration_count[0] > 50:
                # Estimate time remaining based on average time per iteration
                avg_time_per_iter = elapsed / iteration_count[0]
                remaining_iters = 5000 - iteration_count[0]
                est_remaining = avg_time_per_iter * remaining_iters
                
                # Format time
                if est_remaining < 60:
                    time_str = f"{est_remaining:.0f} seconds"
                elif est_remaining < 3600:
                    time_str = f"{est_remaining/60:.1f} minutes"
                else:
                    hours = int(est_remaining // 3600)
                    mins = int((est_remaining % 3600) // 60)
                    time_str = f"{hours}h {mins}m"
                
                progress_pct = (iteration_count[0] / 5000) * 100
                print(f"  Optimization progress: {iteration_count[0]}/5000 ({progress_pct:.1f}%) - Est. remaining: {time_str}")
            else:
                progress_pct = (iteration_count[0] / 5000) * 100
                print(f"  Optimization progress: {iteration_count[0]}/5000 ({progress_pct:.1f}%)...")
    
    constraints = {'type': 'eq', 'fun': constraint_fun}
    bounds = [(LB[i], UB[i]) for i in range(len(LB))]
    
    max_iter_opt1 = 100 if quick_test else 5000  # Reduced from 500 to 100 for faster quick test
    print(f"  Starting optimal tariff optimization (max {max_iter_opt1} iterations)...")
    if quick_test:
        print("  ⚡ QUICK TEST MODE: Using reduced iterations for faster testing")
    print("  Progress will be shown every 50 iterations with time estimates.")
    start_time[0] = time.time()
    res = minimize(objective_fun, x0_opt, method='SLSQP', bounds=bounds, 
                   constraints=constraints, options={'maxiter': max_iter_opt1, 'ftol': 1e-8},
                   callback=callback)
    total_time = time.time() - start_time[0]
    if total_time < 60:
        time_str = f"{total_time:.0f} seconds"
    elif total_time < 3600:
        time_str = f"{total_time/60:.1f} minutes"
    else:
        hours = int(total_time // 3600)
        mins = int((total_time % 3600) // 60)
        time_str = f"{hours}h {mins}m"
    print(f"  Optimization completed after {iteration_count[0]} iterations (Total time: {time_str})")
    # MATLAB implementation uses a single optimal tariff value.
    # Take the last element to mirror x_fmincon(end).
    t_optimal = res.x[-1]
    
    t_ji_new = np.zeros((N, N))
    t_ji_new[:, id_US - 1] = t_optimal
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    # Use previous solution as initial guess (should be close)
    x0 = np.concatenate([x_fsolve_1[:3*N], np.ones(N)])  # Use w, E, L from previous, P = 1.0
    syst = lambda x: Balanced_Trade_IO(x, data, param)[0]
    # Use least_squares for better convergence
    try:
        res_ls = least_squares(syst, x0, method='lm', xtol=1e-10, max_nfev=100000, verbose=0)
        x_fsolve = res_ls.x if res_ls.success else fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    except:
        x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 1], _ = Balanced_Trade_IO(x_fsolve, data, param)
    
    # Liberation Tariffs with optimal retaliation + IO
    tariff_case = 2
    t_ji_new = tariff[0]
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    non_us_idx = np.setdiff1d(np.arange(1, N+1), [id_US])
    LB = np.concatenate([0.75 * x_fsolve_1, np.zeros(len(non_us_idx))])
    UB = np.concatenate([1.5 * x_fsolve_1, 0.5 * np.ones(len(non_us_idx))])
    x0_opt = np.concatenate([x_fsolve_1, t_ji_new[non_us_idx - 1, id_US - 1]])
    
    # Ensure LB < UB for all elements
    LB, UB = np.minimum(LB, UB), np.maximum(LB, UB)
    
    def constraint_fun2(x):
        _, ceq = const_mpec(x, data, param, id_US, tariff_case)
        return ceq
    
    def objective_fun2(x):
        return obj_mpec(x, data, param, id_US, tariff_case)
    
    # Progress callback for second optimization with time estimation
    iteration_count2 = [0]
    start_time2 = [time.time()]
    
    def callback2(xk):
        iteration_count2[0] += 1
        if iteration_count2[0] % 25 == 0:
            elapsed = time.time() - start_time2[0]
            if iteration_count2[0] > 25:
                # Estimate time remaining
                avg_time_per_iter = elapsed / iteration_count2[0]
                remaining_iters = 200 - iteration_count2[0]
                est_remaining = avg_time_per_iter * remaining_iters
                
                # Format time
                if est_remaining < 60:
                    time_str = f"{est_remaining:.0f} seconds"
                elif est_remaining < 3600:
                    time_str = f"{est_remaining/60:.1f} minutes"
                else:
                    hours = int(est_remaining // 3600)
                    mins = int((est_remaining % 3600) // 60)
                    time_str = f"{hours}h {mins}m"
                
                progress_pct = (iteration_count2[0] / 200) * 100
                print(f"  Second optimization progress: {iteration_count2[0]}/200 ({progress_pct:.1f}%) - Est. remaining: {time_str}")
            else:
                progress_pct = (iteration_count2[0] / 200) * 100
                print(f"  Second optimization progress: {iteration_count2[0]}/200 ({progress_pct:.1f}%)...")
    
    constraints2 = {'type': 'eq', 'fun': constraint_fun2}
    bounds2 = [(LB[i], UB[i]) for i in range(len(LB))]
    
    max_iter_opt2 = 50 if quick_test else 200
    print(f"  Starting second optimal tariff optimization (max {max_iter_opt2} iterations)...")
    if quick_test:
        print("  ⚡ QUICK TEST MODE: Using reduced iterations for faster testing")
    print("  Progress will be shown every 25 iterations with time estimates.")
    start_time2[0] = time.time()
    res2 = minimize(objective_fun2, x0_opt, method='SLSQP', bounds=bounds2, 
                    constraints=constraints2, options={'maxiter': max_iter_opt2, 'ftol': 1e-6},
                    callback=callback2)
    total_time2 = time.time() - start_time2[0]
    if total_time2 < 60:
        time_str2 = f"{total_time2:.0f} seconds"
    elif total_time2 < 3600:
        time_str2 = f"{total_time2/60:.1f} minutes"
    else:
        hours = int(total_time2 // 3600)
        mins = int((total_time2 % 3600) // 60)
        time_str2 = f"{hours}h {mins}m"
    print(f"  Second optimization completed after {iteration_count2[0]} iterations (Total time: {time_str2})")
    t_optimal = res2.x[4*N:]
    
    t_ji_new = tariff[0].copy()
    t_ji_new[id_US - 1, non_us_idx - 1] = t_optimal
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    # Use first solution as initial guess (should be close)
    x0 = np.concatenate([x_fsolve_1[:3*N], np.ones(N)])  # Use w, E, L from first solution, P = 1.0
    syst = lambda x: Balanced_Trade_IO(x, data, param)[0]
    # Use least_squares for better convergence
    try:
        res_ls = least_squares(syst, x0, method='lm', xtol=1e-10, max_nfev=100000, verbose=0)
        x_fsolve = res_ls.x if res_ls.success else fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    except:
        x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 3], _ = Balanced_Trade_IO(x_fsolve, data, param)
    
    # Liberation Tariffs with reciprocal retaliation + IO
    t_ji_new = tariff[0].copy()
    t_ji_new[id_US - 1, :] = t_ji_new[:, id_US - 1].T
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i_IO, Y_i_IO, lambda_ji_IO, t_ji_new, nu_IO, T_IO]
    param = [eps, kappa, psi, phi_IO, beta]
    
    # Use first solution as initial guess (should be close)
    x0 = np.concatenate([x_fsolve_1[:3*N], np.ones(N)])  # Use w, E, L from first solution, P = 1.0
    syst = lambda x: Balanced_Trade_IO(x, data, param)[0]
    # Use least_squares for better convergence
    try:
        res_ls = least_squares(syst, x0, method='lm', xtol=1e-10, max_nfev=100000, verbose=0)
        x_fsolve = res_ls.x if res_ls.success else fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    except:
        x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 2], d_trade_IO[1] = Balanced_Trade_IO(x_fsolve, data, param)
    d_employment_IO[1] = (results[:, 4, 2] * Y_i_IO).sum() / Y_i_IO.sum()
    
    # Multi-Sector IO Model
    import sub_multisector_io
    N_multi = N
    N = 194
    
    # For quick test, skip multi-sector or use minimal iterations
    if quick_test:
        print("  ⚡ QUICK TEST: Skipping multi-sector IO (too slow for quick test)")
        print("  (Multi-sector requires full run for accurate results)")
        # Create dummy results to prevent errors
        results_multi_IO = np.zeros((N, 7, 2))
        id_US_new_IO = id_US
        E_i_multi_IO = E_i_IO
        Y_i_multi_IO = Y_i_IO
        d_trade_IO_multi = np.zeros(2)
        d_employment_IO_multi = np.zeros(2)
    else:
        # Run multi-sector IO analysis (full run only)
        results_multi_IO, id_US_new_IO, E_i_multi_IO, Y_i_multi_IO, d_trade_IO_multi, d_employment_IO_multi = \
            sub_multisector_io.main(results, d_trade_IO, d_employment_IO, Y_i_IO, E_i_IO,
                                    Phi, nu_IO, psi, kappa, id_US, quick_test=False)
    
    # Load Table_11 data (from baseline analysis)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    try:
        with open(os.path.join(output_dir, 'Table_11.pkl'), 'rb') as f:
            table_11_data = pickle.load(f)
            d_trade = table_11_data['d_trade']
            d_employment = table_11_data['d_employment']
    except:
        print("Warning: Table_11.pkl not found. Using default values.")
        d_trade = np.zeros(9)
        d_employment = np.zeros(9)
    
    # Import and run print_tables_io
    import print_tables_io
    
    # Pass multi-sector results if available
    print_tables_io.main(results, d_trade_IO, d_employment_IO, Y_i_IO, E_i_IO, id_US, d_trade, d_employment,
                        results_multi_IO=results_multi_IO, id_US_new=id_US_new_IO, E_i_multi=E_i_multi_IO,
                        d_trade_IO_multi=d_trade_IO_multi, d_employment_IO_multi=d_employment_IO_multi)
    
    return results, d_trade_IO, d_employment_IO


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run IO model analysis')
    parser.add_argument('--quick-test', action='store_true', 
                       help='Use reduced iterations for faster testing (500 instead of 5000)')
    args = parser.parse_args()
    main(quick_test=args.quick_test)


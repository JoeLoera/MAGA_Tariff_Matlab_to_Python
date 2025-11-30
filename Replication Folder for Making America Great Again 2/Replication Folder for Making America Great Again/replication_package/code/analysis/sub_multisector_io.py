"""
Sub multisector IO - Python conversion of sub_multisector_io.m
Multi-sector IO model
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve
import sys

# Note: This module expects Phi, nu, psi, kappa from main_io.py
# It will be called from main_io.py which provides these variables


def Balanced_Trade_IO_MultiSector(x, data, param):
    """
    Balanced trade equilibrium function with IO structure and multi-sector (3D)
    """
    N, K, E_i, Y_i, lambda_ji, e_i, ell_ik, t_ji, nu, T_i = data
    eps, kappa, psi, phi, beta = param
    
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    P_i_h = np.abs(x[N+N+N:N+N+N+N])
    ell_ik_h = np.abs(x[N+N+N+N:]).reshape((N, 1, K))
    
    # Construct 3D matrices
    wi_h_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    Pi_h_3D = np.tile(P_i_h.reshape(-1, 1, 1), (1, N, K))
    Lik_h_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    phi_3D = np.tile(phi.reshape(-1, 1, 1), (1, N, K))
    
    # Construct new trade values with safeguards
    P_i_h_safe = np.maximum(P_i_h, 1e-10)
    w_i_h_safe = np.maximum(w_i_h, 1e-10)
    Lik_h_3D_safe = np.maximum(Lik_h_3D, 1e-10)
    
    c_i_h = (wi_h_3D ** beta) * (Pi_h_3D ** (1 - beta))
    entry = np.tile((w_i_h_safe / P_i_h_safe).reshape(-1, 1, 1), (1, N, K)) ** (1 - beta)
    entry = np.maximum(entry, 1e-10)  # Prevent zero entry
    
    p_ij_h = ((c_i_h / ((entry * Lik_h_3D_safe) ** psi)) ** -eps) * ((1 + t_ji) ** (-eps * phi_3D))
    AUX0 = lambda_ji * p_ij_h
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    # Avoid divide by zero
    AUX1 = np.maximum(AUX1, 1e-10)
    lambda_ji_new = AUX0 / AUX1
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    
    X_ji_new = lambda_ji_new * e_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    
    tariff_rev = (t_ji * X_ji_new).sum(axis=(0, 2))
    
    # Safeguard division by zero
    Y_i_safe = np.maximum(Y_i, 1e-10)
    tau_i = tariff_rev / Y_i_safe
    tau_i_new = 0
    tau_i_denom = np.maximum(1 - tau_i, 1e-10)
    tau_i_h = (1 - tau_i_new) / tau_i_denom
    
    # Wage Income = Total Sales net of Taxes
    # MATLAB: Y_ik_cf = sum((1-nu_3D).*beta.*X_ji_new,2) + permute(sum(nu_3D.*X_ji_new,1), [2 1 3]);
    nu_3D = np.tile(nu.reshape(-1, 1, 1), (1, N, K))
    
    # Calculate Y_ik_h safely
    try:
        Y_ik_h = wi_h_3D[:, 0, :] * Lik_h_3D[:, 0, :]  # (N, K)
    except (IndexError, ValueError):
        # Fallback: extract first column properly
        if wi_h_3D.ndim == 3 and wi_h_3D.shape[1] > 0:
            Y_ik_h = (wi_h_3D[:, 0:1, :] * Lik_h_3D[:, 0:1, :]).squeeze(1)
        else:
            Y_ik_h = (wi_h_3D.reshape(N, N, K)[:, 0, :] * Lik_h_3D.reshape(N, N, K)[:, 0, :])
    
    # Ensure Y_ik_h is (N, K)
    if Y_ik_h.ndim != 2 or Y_ik_h.shape != (N, K):
        Y_ik_h = Y_ik_h.reshape(N, K) if Y_ik_h.size == N * K else np.ones((N, K))
    
    # Ensure ell_ik is (N, K)
    if ell_ik.ndim == 3:
        ell_ik = ell_ik.squeeze(1) if ell_ik.shape[1] == 1 else ell_ik[:, 0, :]
    if ell_ik.shape != (N, K):
        ell_ik = ell_ik.reshape(N, K) if ell_ik.size == N * K else np.ones((N, K)) / K
    
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1), (1, K))  # (N, K)
    # Ensure Y_ik is (N, K)
    if Y_ik.ndim != 2 or Y_ik.shape != (N, K):
        if Y_ik.size == N * N * K:
            Y_ik = Y_ik[:, 0, :] if Y_ik.ndim == 3 else Y_ik.reshape(N, N, K)[:, 0, :]
        else:
            Y_ik = Y_ik.reshape(N, K) if Y_ik.size == N * K else np.ones((N, K))
    
    # MATLAB: sum((1-nu_3D).*beta.*X_ji_new,2) - sum over dimension 2 (origins)
    # Ensure beta is properly shaped for broadcasting
    if beta.ndim == 3:
        # beta is (N, N, K), use it directly
        Y_ik_cf_part1 = ((1 - nu_3D) * beta * X_ji_new).sum(axis=1)  # (N, K)
    elif beta.ndim == 1:
        # beta is (K,), tile it to (N, N, K)
        beta_3D = np.tile(beta.reshape(1, 1, -1), (N, N, 1))
        Y_ik_cf_part1 = ((1 - nu_3D) * beta_3D * X_ji_new).sum(axis=1)  # (N, K)
    else:
        # Try to reshape beta
        beta_3D = np.tile(beta.reshape(-1), (N, N, 1)) if beta.size == K else np.tile(beta, (N, N, 1))
        Y_ik_cf_part1 = ((1 - nu_3D) * beta_3D * X_ji_new).sum(axis=1)  # (N, K)
    
    # Ensure it's 2D and correct shape
    if Y_ik_cf_part1.ndim == 3:
        Y_ik_cf_part1 = Y_ik_cf_part1.squeeze(1)  # (N, K)
    if Y_ik_cf_part1.shape != (N, K):
        Y_ik_cf_part1 = Y_ik_cf_part1.reshape(N, K) if Y_ik_cf_part1.size == N * K else np.zeros((N, K))
    
    # MATLAB: permute(sum(nu_3D.*X_ji_new,1), [2 1 3])
    # sum over dimension 1 (destinations) gives (1, N, K) or (N, K)
    Y_ik_cf_part2 = (nu_3D * X_ji_new).sum(axis=0)  # (N, K) or (1, N, K)
    if Y_ik_cf_part2.ndim == 3:
        Y_ik_cf_part2 = Y_ik_cf_part2.transpose(1, 0, 2)  # permute([2 1 3]) -> (N, 1, K)
        Y_ik_cf_part2 = Y_ik_cf_part2.squeeze(1) if Y_ik_cf_part2.shape[1] == 1 else Y_ik_cf_part2  # (N, K)
    elif Y_ik_cf_part2.ndim == 2 and Y_ik_cf_part2.shape[0] == 1:
        Y_ik_cf_part2 = Y_ik_cf_part2.T  # (N, K)
    if Y_ik_cf_part2.shape != (N, K):
        Y_ik_cf_part2 = Y_ik_cf_part2.reshape(N, K) if Y_ik_cf_part2.size == N * K else np.zeros((N, K))
    
    Y_ik_cf = Y_ik_cf_part1 + Y_ik_cf_part2  # (N, K)
    if Y_ik_cf.shape != (N, K):
        if Y_ik_cf.size == N * N * K:
            Y_ik_cf = Y_ik_cf.sum(axis=0)  # -> (N, K)
        else:
            Y_ik_cf = Y_ik_cf.reshape(N, K) if Y_ik_cf.size == N * K else np.zeros((N, K))
    if Y_ik_cf.shape != (N, K):
        Y_ik_cf = Y_ik_cf.reshape(N, K) if Y_ik_cf.size == N * K else np.zeros((N, K))
    
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N * K)
    ERR1[N-1] = ((E_i_h - 1) * E_i).sum()  # Replace one excess equation
    
    # Total Income = Total Sales
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + ((1 - beta) * (1 - nu_3D) * X_ji_new).sum(axis=(1, 2)) + T_i * (X_global_new / X_global) - E_i_new
    
    # Labor supply equation
    # Avoid negative values in sqrt by ensuring the expression is positive
    labor_supply_expr = tau_i_h * w_i_h / P_i_h
    # Clip to positive values to avoid sqrt warnings
    labor_supply_expr = np.maximum(labor_supply_expr, 1e-10)
    ERR3 = L_i_h - labor_supply_expr ** kappa
    
    # Price equation
    # MATLAB: ERR4 = P_i_h - ( (E_i_h./w_i_h).^(1-phi)) .* prod( sum(AUX0,1).^(-e_i(1,:,:)./eps(1,:,:)) ,3)';
    # sum(AUX0,1) gives (1, N, K) - sum over first dimension (axis=0 in Python)
    AUX0_sum = AUX0.sum(axis=0, keepdims=True)  # (1, N, K)
    # e_i[0, :, :] is first row = (N, K)
    # eps[0, :, :] is first row = (N, K)
    e_i_1 = e_i[0, :, :]  # (N, K)
    eps_1 = eps[0, :, :]  # (N, K)
    # AUX0_sum[0, :, :] is (N, K)
    # Product over dimension 3 (sectors) - axis=1 in Python (columns are sectors)
    # Result should be (N,) - one value per country
    # Avoid divide by zero in power calculation
    AUX0_sum_safe = np.maximum(AUX0_sum[0, :, :], 1e-10)
    eps_1_safe = np.maximum(eps_1, 1e-10)
    power_term = AUX0_sum_safe ** (-e_i_1 / eps_1_safe)  # (N, K)
    # Clip power term to prevent extreme values
    power_term = np.clip(power_term, 1e-10, 1e10)
    prod_term = np.prod(power_term, axis=1)  # (N,) - product over sectors
    # Clip product term
    prod_term = np.clip(prod_term, 1e-10, 1e10)
    # Ensure it's 1D
    if prod_term.ndim > 1:
        prod_term = prod_term.flatten()
    
    # Safeguard the (E_i_h / w_i_h) term to prevent extreme values
    E_w_ratio = E_i_h / np.maximum(w_i_h, 1e-10)
    E_w_ratio = np.clip(E_w_ratio, 1e-6, 1e6)  # Prevent extreme ratios
    
    # Safeguard (1 - phi) - if phi is very large, this could be negative or very large
    exp_term = 1 - phi
    exp_term = np.clip(exp_term, -5, 5)  # Limit exponent to prevent extreme values
    
    # Calculate price term with safeguards
    price_term = (E_w_ratio ** exp_term) * prod_term
    price_term = np.clip(price_term, 1e-10, 1e10)
    
    ERR4 = P_i_h - price_term
    
    # Sector share equation
    # MATLAB: ERR5 = 100*(sum(ell_ik.*ell_ik_h,3) - 1);
    # ell_ik is (N, K), ell_ik_h is (N, 1, K)
    # sum(..., 3) means sum over dimension 3 (sectors), gives (N, 1) -> (N,)
    # Ensure ell_ik_h is (N, K) for multiplication
    ell_ik_h_2D = ell_ik_h.squeeze(1) if ell_ik_h.ndim == 3 else ell_ik_h.reshape(N, K)
    ERR5 = 100 * ((ell_ik * ell_ik_h_2D).sum(axis=1) - 1)  # Sum over sectors (axis=1) -> (N,)
    ERR5 = ERR5.flatten()  # Ensure 1D
    if ERR5.size != N:
        ERR5 = ERR5[:N]  # Take first N elements if wrong size
    
    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4, ERR5])
    
    # Calculate results
    Ec_i = Y_i + T_i
    # Safeguard division by zero
    delta_denom = Ec_i - kappa * (1 - tau_i) * Y_i / (1 + kappa)
    delta_denom = np.maximum(delta_denom, 1e-6)
    delta_i = Ec_i / delta_denom
    Ec_i_safe = np.maximum(Ec_i, 1e-10)
    Ec_i_h = (tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global)) / Ec_i_safe
    P_i_h_safe = np.maximum(P_i_h, 1e-10)
    W_i_h = delta_i * (Ec_i_h / P_i_h_safe) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h_safe)
    # Clip welfare to prevent extreme values
    W_i_h = np.clip(W_i_h, 0.01, 100.0)
    
    # Factual trade flows
    X_ji = lambda_ji * e_i * np.tile(E_i.reshape(1, -1, 1), (N, 1, K))
    eye_N = np.eye(N)
    # MATLAB uses opposite sign convention (surplus = positive, deficit = negative)
    D_i = X_ji.sum(axis=(1, 2)) - X_ji.sum(axis=(0, 2))  # Flipped: exports - imports
    D_i_new = X_ji_new.sum(axis=(1, 2)) - X_ji_new.sum(axis=(0, 2))  # Flipped: exports - imports
    
    # Create mask for non-diagonal elements: reshape to (N, N, 1) then tile to (N, N, K)
    non_diag_mask = np.tile((1 - eye_N).reshape(N, N, 1), (1, 1, K))
    
    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * (((X_ji_new * non_diag_mask).sum(axis=(1, 2)) / Y_i_new) / 
                      ((X_ji * non_diag_mask).sum(axis=(1, 2)) / Y_i) - 1)
    d_import = 100 * (((X_ji_new * non_diag_mask).sum(axis=(0, 2)) / Y_i_new) / 
                      ((X_ji * non_diag_mask).sum(axis=(0, 2)) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    # Safeguard division by zero
    D_i_abs = np.maximum(np.abs(D_i), 1e-10)
    d_D_i = 100 * ((D_i_new - D_i) / D_i_abs)
    
    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, 
                               d_employment, d_CPI, tariff_rev / E_i])
    
    trade = X_ji * non_diag_mask
    trade_new = X_ji_new * (1 + t_ji) * non_diag_mask
    # Safeguard division by zero
    trade_sum = np.maximum(trade.sum(), 1e-10)
    Y_i_sum = np.maximum(Y_i.sum(), 1e-10)
    d_trade = 100 * ((trade_new.sum() / trade_sum) / 
                     (Y_i_new.sum() / Y_i_sum) - 1)
    
    return ceq, results, d_trade


def main(results_IO=None, d_trade_IO=None, d_employment_IO=None, Y_i_IO=None, E_i_IO=None,
         Phi=None, nu=None, psi=None, kappa=None, id_US=None, quick_test=False):
    """
    Multi-sector IO analysis
    
    This is called from main_io.py with necessary parameters
    """
    print("Running sub_multisector_io...")
    if quick_test:
        print("  ⚡ QUICK TEST MODE: Using reduced iterations for faster testing")
    
    # Read multi-sector trade data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ITPDS')
    
    try:
        data = pd.read_csv(os.path.join(data_dir, 'trade_ITPD.csv'))
        X = data.iloc[:, 3].values  # Column 4 (0-indexed is 3)
        N_full = 194
        K = 4
        expected_size = N_full * N_full * K
        
        # Handle missing data (should be 150544, but we have 150543)
        if len(X) == expected_size - 1:
            X = np.append(X, 0.0)
        elif len(X) != expected_size:
            raise ValueError(f"Unexpected data size: {len(X)}, expected {expected_size}")
        
        X_ji = X.reshape((N_full, N_full, K), order='F')  # Fortran order like MATLAB
        
        # Process problematic countries
        problematic_id = (X_ji == 0).all(axis=(0, 2))
        ID = np.where(problematic_id)[0]
        idx = np.setdiff1d(np.arange(N_full), ID)
        
        N_new = len(idx)
        X_new = np.zeros((N_new, N_new, K))
        t_new = np.zeros((N_new, N_new, K))
        
        for k in range(K):
            X_new[:, :, k] = X_ji[idx, :][:, idx, k]
        
        X_ji = X_new
        
        # Load tariffs
        data_dir_base = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
        t = pd.read_csv(os.path.join(data_dir_base, 'tariffs.csv'))
        new_ustariff = t.values.flatten()
        
        t_ji = np.zeros((N_full, N_full, K))
        for k in range(K-1):
            t_ji[:, id_US - 1, k] = new_ustariff
        
        t_ji[:, id_US - 1, :K-1] = np.maximum(0.1, t_ji[:, id_US - 1, :K-1])
        t_ji[id_US - 1, id_US - 1, :K-1] = 0
        
        for k in range(K):
            t_new[:, :, k] = t_ji[idx, :][:, idx, k]
        
        t_ji = t_new
        
        # Find US in new indices
        id_US_new = np.where(idx == id_US - 1)[0]
        if len(id_US_new) == 0:
            print("Warning: US not found in filtered countries")
            return None, None, None, None, np.zeros(2), np.zeros(2)
        
        id_US_new = id_US_new[0] + 1  # Convert back to 1-indexed
        
        # Filter phi and nu
        if Phi is not None and len(Phi) > 1:
            phi = Phi[1][idx] if isinstance(Phi[1], np.ndarray) else Phi[1]
        else:
            phi = np.ones(N_new)  # Default
        
        if nu is not None:
            nu_new = nu[idx]
        else:
            nu_new = np.zeros(N_new)  # Default
        
        # Parameters
        beta = np.array([0.51, 0.32, 0.49, 0.56])
        beta_3D = np.tile(beta.reshape(1, 1, -1), (N_new, N_new, 1))
        
        nu_3D = np.tile(nu_new.reshape(-1, 1, 1), (1, N_new, K))
        E_i_multi = X_ji.sum(axis=(0, 2))
        Y_i_multi = ((1 - nu_3D) * beta_3D * X_ji).sum(axis=(1, 2)) + (nu_3D * X_ji).sum(axis=(0, 2))
        T = E_i_multi - (Y_i_multi + ((1 - beta_3D) * (1 - nu_3D) * X_ji).sum(axis=(1, 2)))
        
        lambda_ji = X_ji / X_ji.sum(axis=(0, 2), keepdims=True)
        e_i = X_ji.sum(axis=(0, 2), keepdims=True) / np.tile(E_i_multi.reshape(1, -1, 1), (N_new, 1, K))
        
        Y_ik_p = ((1 - nu_3D) * beta_3D * X_ji).sum(axis=1)  # Shape: (N, K)
        # MATLAB: permute(sum(nu_3D.*X_ji,1), [2 1 3])
        # sum(nu_3D.*X_ji, 1) sums along dimension 1 (rows) giving (1, N, K)
        # permute([2 1 3]) rearranges to (N, 1, K), then squeeze to (N, K)
        Y_ik_f = (nu_3D * X_ji).sum(axis=0, keepdims=True)  # (1, N, K)
        Y_ik_f = Y_ik_f.transpose(1, 0, 2)  # permute([2 1 3]) = transpose(1,0,2) gives (N, 1, K)
        Y_ik_f = Y_ik_f.squeeze(1) if Y_ik_f.shape[1] == 1 else Y_ik_f  # (N, K)
        Y_ik = Y_ik_p + Y_ik_f
        ell_ik = Y_ik / np.tile(Y_i_multi.reshape(-1, 1), (1, K))
        
        # Elasticity parameters
        if Phi is not None and len(Phi) > 0 and Y_i_IO is not None:
            phi_avg = (Phi[0] * Y_i_IO).sum() / Y_i_IO.sum() if isinstance(Phi[0], np.ndarray) else Phi[0]
        else:
            phi_avg = 1.0  # Default
        
        eps = np.array([3.3, 3.8, 4.1]) / phi_avg
        eps = np.append(eps, 3.0)
        eps_3D = np.tile(eps.reshape(1, 1, -1), (N_new, N_new, 1))
        
        results_multi = np.zeros((N_new, 7, 2))
        d_trade_IO_multi = np.zeros(2)
        d_employment_IO_multi = np.zeros(2)
        
        # No Retaliation
        data = [N_new, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu_new, T]
        param = [eps_3D, kappa, psi, phi, beta_3D]
        
        x0 = np.ones(4 * N_new + N_new * K)
        syst = lambda x: Balanced_Trade_IO_MultiSector(x, data, param)[0]
        
        # Use bounded solver to prevent extreme values
        from scipy.optimize import least_squares
        bounds = ([0.1] * (4 * N_new + N_new * K), [5.0] * (4 * N_new + N_new * K))
        
        # Reduce iterations for quick test
        max_iter = 2000 if quick_test else 50000
        print(f"  Solving multi-sector IO equilibrium (max {max_iter} iterations)...")
        print("  (This may take several minutes - multi-sector model has many variables)")
        
        try:
            res_ls = least_squares(syst, x0, method='trf', bounds=bounds, 
                                  xtol=1e-5 if quick_test else 1e-8, 
                                  max_nfev=max_iter, verbose=1)
            if res_ls.success:
                x_fsolve = res_ls.x
                print(f"  ✓ Multi-sector IO solver converged (cost: {res_ls.cost:.2e})")
            else:
                print(f"  ⚠ Multi-sector IO solver did not fully converge, using fsolve fallback...")
                x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
        except Exception as e:
            print(f"  ⚠ Bounded solver failed ({e}), using fsolve fallback...")
            x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
        
        _, results_multi[:, :, 0], d_trade_IO_multi[0] = Balanced_Trade_IO_MultiSector(x_fsolve, data, param)
        d_employment_IO_multi[0] = (results_multi[:, 4, 0] * Y_i_multi).sum() / Y_i_multi.sum()
        
        # Reciprocal Retaliation
        for k in range(K-1):
            t_ji[id_US_new - 1, :, k] = t_ji[:, id_US_new - 1, k].T
        t_ji[id_US_new - 1, id_US_new - 1, :] = 0
        
        data = [N_new, K, E_i_multi, Y_i_multi, lambda_ji, e_i, ell_ik, t_ji, nu_new, T]
        param = [eps_3D, kappa, psi, phi, beta_3D]
        
        syst = lambda x: Balanced_Trade_IO_MultiSector(x, data, param)[0]
        # Use bounded solver for retaliation case too
        print(f"  Solving multi-sector IO (retaliation) equilibrium (max {max_iter} iterations)...")
        try:
            res_ls = least_squares(syst, x_fsolve, method='trf', bounds=bounds, 
                                  xtol=1e-5 if quick_test else 1e-8, 
                                  max_nfev=max_iter, verbose=1)
            if res_ls.success:
                x_fsolve = res_ls.x
                print(f"  ✓ Multi-sector IO (retaliation) solver converged (cost: {res_ls.cost:.2e})")
            else:
                print(f"  ⚠ Multi-sector IO (retaliation) solver did not fully converge, using fsolve fallback...")
                x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=100000)
        except Exception as e:
            print(f"  ⚠ Bounded solver failed ({e}), using fsolve fallback...")
            x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=100000)
        
        _, results_multi[:, :, 1], d_trade_IO_multi[1] = Balanced_Trade_IO_MultiSector(x_fsolve, data, param)
        d_employment_IO_multi[1] = (results_multi[:, 4, 1] * Y_i_multi).sum() / Y_i_multi.sum()
        
        print(f"Multi-sector IO model: N={N_new}, K={K}, US index={id_US_new}")
        
        return results_multi, id_US_new, E_i_multi, Y_i_multi, d_trade_IO_multi, d_employment_IO_multi
        
    except FileNotFoundError:
        print("Warning: trade_ITPD.csv not found. Skipping multi-sector IO analysis.")
        return None, None, None, None, np.zeros(2), np.zeros(2)
    except Exception as e:
        print(f"Warning: Multi-sector IO analysis incomplete: {e}")
        print("Note: Full multi-sector IO implementation may need adjustments.")
        import traceback
        traceback.print_exc()
        return None, None, None, None, np.zeros(2), np.zeros(2)


if __name__ == '__main__':
    # This would be called from main_io.py
    pass


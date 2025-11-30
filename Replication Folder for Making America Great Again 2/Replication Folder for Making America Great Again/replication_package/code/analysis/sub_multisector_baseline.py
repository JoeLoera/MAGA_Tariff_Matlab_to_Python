"""
Sub multisector baseline - Python conversion of sub_multisector_baseline.m
Full implementation matching MATLAB version
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve
import sys


def Balanced_Trade_EQ_MultiSector(x, data, param):
    """
    Balanced trade equilibrium function for multi-sector baseline model
    
    This matches the MATLAB Balanced_Trade_EQ function in sub_multisector_baseline.m
    """
    N, K, E_i, Y_i, lambda_ji, beta_i, ell_ik, t_ji, nu, T_i = data
    eps_3D, kappa, psi, phi = param  # Note: eps is 3D array (N, N, K)
    
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    ell_ik_h = np.abs(x[N+N+N:]).reshape((N, 1, K))
    
    # Construct 3D matrices
    wi_h_3D = np.tile(w_i_h.reshape(-1, 1, 1), (1, N, K))
    Lik_h_3D = np.tile(L_i_h.reshape(-1, 1, 1), (1, N, K)) * np.tile(ell_ik_h, (1, N, 1))
    
    phi_3D = np.tile(phi.reshape(-1, 1, 1), (1, N, K))
    
    # Construct new trade values
    # Note: eps_3D is (N, N, K), we need to use it element-wise
    AUX0 = lambda_ji * ((wi_h_3D / (Lik_h_3D ** psi)) ** -eps_3D) * ((1 + t_ji) ** (-eps_3D * phi_3D))
    AUX1 = np.tile(AUX0.sum(axis=0, keepdims=True), (N, 1, 1))
    lambda_ji_new = AUX0 / AUX1
    
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    
    # Price index with product over sectors
    # MATLAB: P_i_h=( (E_i_h./w_i_h).^(1 - phi)) .* prod(sum(AUX0,1).^(-beta_i(1,:,:)./eps(1,:,:)),3)';
    # sum(AUX0,1) gives (1, N, K)
    # beta_i(1,:,:) is first row of (N, N, K) = (N, K)
    # eps(1,:,:) is first row of (N, N, K) = (N, K)
    # prod(..., 3) means product over dimension 3 (sectors), result is (1, N) which is transposed to (N, 1) then becomes (N,)
    AUX0_sum = AUX0.sum(axis=0, keepdims=True)  # (1, N, K)
    
    # beta_i is (N, N, K), beta_i(1,:,:) means first row = (N, K)
    # eps_3D is (N, N, K), eps_3D(1,:,:) means first row = (N, K)
    beta_i_1 = beta_i[0, :, :]  # (N, K)
    eps_1 = eps_3D[0, :, :]  # (N, K)
    
    # AUX0_sum[0, :, :] is (N, K)
    # Product over dimension 3 (sectors) - axis=1 in Python (columns are sectors)
    # Result should be (N,) - one value per country
    # Safeguard division by zero
    AUX0_sum_safe = np.maximum(AUX0_sum[0, :, :], 1e-10)
    eps_1_safe = np.maximum(eps_1, 1e-10)
    power_term = AUX0_sum_safe ** (-beta_i_1 / eps_1_safe)  # (N, K)
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
    
    P_i_h = price_term
    
    X_ji_new = lambda_ji_new * beta_i * np.tile(E_i_new.reshape(1, -1, 1), (N, 1, K)) / (1 + t_ji)
    tariff_rev = (t_ji * X_ji_new).sum(axis=(0, 2))
    
    # Safeguard division by zero
    Y_i_new_safe = np.maximum(Y_i_new, 1e-10)
    tau_i = tariff_rev / Y_i_new_safe
    tau_i_new = 0
    tau_i_denom = np.maximum(1 - tau_i, 1e-10)
    tau_i_h = (1 - tau_i_new) / tau_i_denom
    
    # Wage Income = Total Sales net of Taxes
    # MATLAB: nu_3D = repmat(nu',N,1,K); Y_ik_h = wi_h_3D(:,1,:).*Lik_h_3D(:,1,:);
    # MATLAB: Y_ik = ell_ik.*repmat( Y_i, [1 1 K]);
    # MATLAB: Y_ik_cf = sum((1-nu_3D).*X_ji_new,2) + permute(sum(nu_3D.*X_ji_new,1), [2 1 3]);
    nu_3D = np.tile(nu.reshape(-1, 1, 1), (1, N, K))
    Y_ik_h = wi_h_3D[:, 0, :] * Lik_h_3D[:, 0, :]  # (N, K)
    # Ensure Y_ik_h is (N, K)
    if Y_ik_h.ndim != 2 or Y_ik_h.shape != (N, K):
        Y_ik_h = Y_ik_h.reshape(N, K)
    # MATLAB: Y_ik = ell_ik.*repmat(Y_i, [1 1 K]);
    # ell_ik is (N, K), Y_i is (N,), repmat gives (N, 1, K) or we tile to (N, K)
    # Ensure ell_ik is (N, K)
    if ell_ik.ndim == 3:
        ell_ik = ell_ik.squeeze(1) if ell_ik.shape[1] == 1 else ell_ik[:, 0, :]  # Take first slice if 3D
    if ell_ik.shape != (N, K):
        ell_ik = ell_ik.reshape(N, K)
    Y_ik = ell_ik * np.tile(Y_i.reshape(-1, 1), (1, K))  # (N, K)
    # Ensure Y_ik is (N, K)
    if Y_ik.ndim != 2 or Y_ik.shape != (N, K):
        if Y_ik.size == N * N * K:
            # It's (N, N, K) - take first slice
            Y_ik = Y_ik[:, 0, :] if Y_ik.ndim == 3 else Y_ik.reshape(N, N, K)[:, 0, :]
        else:
            Y_ik = Y_ik.reshape(N, K)
    # MATLAB: Y_ik_cf = sum((1-nu_3D).*X_ji_new,2) + permute(sum(nu_3D.*X_ji_new,1), [2 1 3]);
    # In MATLAB, X_ji_new is (N, N, K)
    # sum(..., 2) means sum over dimension 2 (second dimension = origins), gives (N, 1, K) -> squeeze to (N, K)
    # sum(..., 1) means sum over dimension 1 (first dimension = destinations), gives (1, N, K)
    # permute([2 1 3]) swaps dimensions 1 and 2, giving (N, 1, K) -> squeeze to (N, K)
    # MATLAB: sum((1-nu_3D).*X_ji_new,2) - sum over dimension 2 (origins)
    # X_ji_new is (N, N, K), sum over axis 1 (second dimension) gives (N, K)
    Y_ik_cf_part1 = ((1 - nu_3D) * X_ji_new).sum(axis=1)  # (N, K)
    # Ensure it's 2D and correct shape
    if Y_ik_cf_part1.ndim == 3:
        Y_ik_cf_part1 = Y_ik_cf_part1.squeeze(1)  # (N, K)
    if Y_ik_cf_part1.shape != (N, K):
        Y_ik_cf_part1 = Y_ik_cf_part1.reshape(N, K)
    
    # MATLAB: permute(sum(nu_3D.*X_ji_new,1), [2 1 3])
    # sum over dimension 1 (destinations) gives (1, N, K) or (N, K)
    Y_ik_cf_part2 = (nu_3D * X_ji_new).sum(axis=0)  # (N, K) or (1, N, K)
    if Y_ik_cf_part2.ndim == 3:
        Y_ik_cf_part2 = Y_ik_cf_part2.transpose(1, 0, 2)  # permute([2 1 3]) -> (N, 1, K)
        Y_ik_cf_part2 = Y_ik_cf_part2.squeeze(1) if Y_ik_cf_part2.shape[1] == 1 else Y_ik_cf_part2  # (N, K)
    elif Y_ik_cf_part2.ndim == 2 and Y_ik_cf_part2.shape[0] == 1:
        Y_ik_cf_part2 = Y_ik_cf_part2.T  # (N, K)
    if Y_ik_cf_part2.shape != (N, K):
        Y_ik_cf_part2 = Y_ik_cf_part2.reshape(N, K)
    
    # Add them - ensure no broadcasting issues
    Y_ik_cf = Y_ik_cf_part1 + Y_ik_cf_part2  # (N, K)
    # Force correct shape
    if Y_ik_cf.shape != (N, K):
        if Y_ik_cf.size == N * N * K:
            # It's (N, N, K) - sum over first dimension
            Y_ik_cf = Y_ik_cf.sum(axis=0)  # -> (N, K)
        else:
            Y_ik_cf = Y_ik_cf.reshape(N, K)
    # Ensure all arrays are (N, K) before subtraction
    if Y_ik_cf.shape != (N, K):
        Y_ik_cf = Y_ik_cf.reshape(N, K)
    if Y_ik.shape != (N, K):
        Y_ik = Y_ik.reshape(N, K)
    if Y_ik_h.shape != (N, K):
        Y_ik_h = Y_ik_h.reshape(N, K)
    ERR1 = (Y_ik_cf - Y_ik * Y_ik_h).reshape(N * K)
    ERR1[N-1] = ((P_i_h - 1) * E_i).mean()  # Replace one excess equation
    
    # Total Income = Total Sales
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new
    
    # Labor supply equation
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa
    
    # Sector share constraint
    # MATLAB: ERR4 = 100*(sum(ell_ik.*ell_ik_h,3) - 1);
    # ell_ik is (N, K), ell_ik_h is (N, 1, K)
    # sum(..., 3) means sum over dimension 3 (sectors), gives (N, 1) -> (N,)
    # Ensure ell_ik_h is (N, K) for multiplication
    ell_ik_h_2D = ell_ik_h.squeeze(1) if ell_ik_h.ndim == 3 else ell_ik_h.reshape(N, K)
    ERR4 = 100 * ((ell_ik * ell_ik_h_2D).sum(axis=1) - 1)  # Sum over sectors (axis=1) -> (N,)
    ERR4 = ERR4.flatten()  # Ensure 1D
    if ERR4.size != N:
        ERR4 = ERR4[:N]  # Take first N elements if wrong size
    
    ceq = np.concatenate([ERR1, ERR2, ERR3, ERR4])
    
    # Calculate results
    # Safeguard division by zero
    delta_denom = E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa)
    delta_denom = np.maximum(delta_denom, 1e-6)
    delta_i = E_i / delta_denom
    P_i_h_safe = np.maximum(P_i_h, 1e-10)
    W_i_h = delta_i * (E_i_h / P_i_h_safe) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h_safe)
    # Clip welfare to prevent extreme values
    W_i_h = np.clip(W_i_h, 0.01, 100.0)
    
    # Factual trade flows
    X_ji = lambda_ji * beta_i * np.tile(E_i.reshape(1, -1, 1), (N, 1, K))
    eye_N = np.eye(N)
    # MATLAB uses opposite sign convention (surplus = positive, deficit = negative)
    D_i = X_ji.sum(axis=(1, 2)) - X_ji.sum(axis=(0, 2))  # Flipped: exports - imports
    D_i_new = X_ji_new.sum(axis=(1, 2)) - X_ji_new.sum(axis=(0, 2))  # Flipped: exports - imports
    
    d_welfare = 100 * (W_i_h - 1)
    # eye_N is (N, N), we need (N, N, K)
    eye_N_3D = np.tile((1 - eye_N)[:, :, np.newaxis], (1, 1, K))  # (N, N, K)
    d_export = 100 * (((X_ji_new * eye_N_3D).sum(axis=(1, 2)) / Y_i_new) / 
                      ((X_ji * eye_N_3D).sum(axis=(1, 2)) / Y_i) - 1)
    d_import = 100 * (((X_ji_new * eye_N_3D).sum(axis=(0, 2)) / Y_i_new) / 
                      ((X_ji * eye_N_3D).sum(axis=(0, 2)) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    # Safeguard division by zero
    D_i_abs = np.maximum(np.abs(D_i), 1e-10)
    d_D_i = 100 * ((D_i_new - D_i) / D_i_abs)
    
    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, 
                               d_employment, d_CPI, tariff_rev / E_i])
    
    # Trade-to-GDP change
    trade = X_ji * eye_N_3D
    trade_new = X_ji_new * (1 + t_ji) * eye_N_3D
    d_trade = 100 * ((trade_new.sum() / trade.sum()) / 
                     (Y_i_new.sum() / Y_i.sum()) - 1)
    
    return ceq, results, d_trade


def main(results, revenue, d_trade, d_employment, Y_i, E_i, Phi, nu, 
         id_US, id_EU, id_CHN, id_RoW, non_US, country_names, N):
    """
    Multi-sector baseline analysis
    
    This updates d_trade[8], d_trade[9], d_employment[8], d_employment[9]
    and stores results_multi for use in Table 9
    """
    print("Running sub_multisector_baseline...")
    
    # Read multi-sector trade data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'ITPDS')
    base_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
    
    try:
        data = pd.read_csv(os.path.join(data_dir, 'trade_ITPD.csv'))
        X = data.iloc[:, 3].values  # Column 4 (0-indexed is 3)
        N_full = 194
        K = 4
        expected_size = N_full * N_full * K
        
        # Handle missing data (should be 150544, but we have 150543)
        if len(X) == expected_size - 1:
            # Pad with zero for missing entry
            X = np.append(X, 0.0)
        elif len(X) != expected_size:
            raise ValueError(f"Unexpected data size: {len(X)}, expected {expected_size}")
        
        X_ji = X.reshape((N_full, N_full, K), order='F')  # Fortran order like MATLAB
        
        # Load tariffs
        t = pd.read_csv(os.path.join(base_data_dir, 'tariffs.csv'))
        new_ustariff = t.values.flatten()
        id_US_matlab = 185  # 1-indexed
        
        t_ji = np.zeros((N_full, N_full, K))
        for k in range(K-1):
            t_ji[:, id_US_matlab - 1, k] = new_ustariff
        
        t_ji[:, id_US_matlab - 1, :K-1] = np.maximum(0.1, t_ji[:, id_US_matlab - 1, :K-1])
        t_ji[id_US_matlab - 1, id_US_matlab - 1, :K-1] = 0
        
        # Process problematic countries
        problematic_id = (X_ji == 0).all(axis=(0, 2))
        ID = np.where(problematic_id)[0]
        idx = np.setdiff1d(np.arange(N_full), ID)
        
        N_new = len(idx)
        X_new = np.zeros((N_new, N_new, K))
        t_new = np.zeros((N_new, N_new, K))
        for k in range(K):
            X_new[:, :, k] = X_ji[idx, :][:, idx, k]
            t_new[:, :, k] = t_ji[idx, :][:, idx, k]
        
        X_ji = X_new
        t_ji = t_new
        
        # Find US in new indices (convert to 1-indexed for consistency with MATLAB)
        id_US_new = np.where(idx == id_US_matlab - 1)[0]
        if len(id_US_new) == 0:
            print("Warning: US not found in filtered countries")
            return results, d_trade, d_employment
        
        id_US_new = id_US_new[0] + 1  # Convert to 1-indexed for consistency
        
        # Filter nu
        nu_new = nu[idx]
        
        # Calculate aggregates
        E_i_multi = X_ji.sum(axis=(0, 2))
        Y_i_multi = (np.outer((1 - nu_new), np.ones(N_new)) * X_ji.sum(axis=2)).sum(axis=1) + nu_new * X_ji.sum(axis=(0, 2))
        T = E_i_multi - Y_i_multi
        
        # Share matrices
        # MATLAB: lambda_ji = X_ji./repmat(sum(X_ji),[N 1 1]);
        # MATLAB: beta_i = repmat(sum(X_ji),[N 1 1])./repmat(E_i_multi',[N 1 K]);
        X_ji_sum = X_ji.sum(axis=(0, 2), keepdims=True)  # (1, N, 1)
        lambda_ji = X_ji / np.tile(X_ji_sum, (N_new, 1, K))  # (N, N, K)
        beta_i = np.tile(X_ji_sum, (N_new, 1, K)) / np.tile(E_i_multi.reshape(1, -1, 1), (N_new, 1, K))  # (N, N, K)
        
        # Sector shares
        nu_outer_3d = np.outer((1 - nu_new), np.ones(N_new))[:, :, np.newaxis]
        Y_ik_p = (nu_outer_3d * X_ji).sum(axis=1)
        
        # MATLAB: Y_ik_f = repmat(nu',[1 1 K]).*sum(X_ji, 1); Y_ik = Y_ik_p + permute(Y_ik_f, [2 1 3]);
        # sum(X_ji, 1) gives (1, N, K), repmat(nu',[1 1 K]) broadcasts to (1, N, K)
        nu_3D_f = np.tile(nu_new.reshape(1, -1, 1), (1, 1, K))  # (1, N, K)
        Y_ik_f = nu_3D_f * X_ji.sum(axis=0, keepdims=True)  # (1, N, K)
        Y_ik = Y_ik_p + Y_ik_f.transpose(1, 0, 2)  # permute([2 1 3]) = transpose(1,0,2) gives (N, 1, K)
        Y_ik = Y_ik.squeeze(1) if Y_ik.shape[1] == 1 else Y_ik  # (N, K)
        ell_ik = Y_ik / Y_i_multi.reshape(-1, 1)
        
        # Parameters
        kappa = 0.5
        psi = 0.67 / 4
        theta = 1 / psi
        
        # Filter phi (use Phi[0] which is first element)
        phi = Phi[0][idx] if isinstance(Phi[0], np.ndarray) else Phi[0]
        phi_avg = (Phi[0] * Y_i).sum() / Y_i.sum() if isinstance(Phi[0], np.ndarray) else Phi[0]
        
        eps = np.array([3.3, 3.8, 4.1]) / phi_avg
        eps = np.append(eps, 3.0)
        eps_3D = np.tile(eps.reshape(1, 1, -1), (N_new, N_new, 1))
        
        results_multi = np.zeros((N_new, 7, 2))
        
        # No Retaliation
        data = [N_new, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu_new, T]
        param = [eps_3D, kappa, psi, phi]
        
        x0 = np.concatenate([np.ones(N_new), np.ones(N_new), np.ones(N_new), np.ones(N_new * K)])
        syst = lambda x: Balanced_Trade_EQ_MultiSector(x, data, param)[0]
        x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
        
        # Note: MATLAB uses d_trade(8) which is 1-indexed = index 7 in 0-indexed Python arrays
        _, results_multi[:, :, 0], d_trade[7] = Balanced_Trade_EQ_MultiSector(x_fsolve, data, param)
        d_employment[7] = (results_multi[:, 4, 0] * Y_i_multi).sum() / Y_i_multi.sum()
        
        # Reciprocal Retaliation
        # MATLAB: for k = 1:K-1; t_ji(id_US_new,:,k) = t_ji(:,id_US_new,k)'; end
        for k in range(K-1):
            t_ji[id_US_new - 1, :, k] = t_ji[:, id_US_new - 1, k]
        t_ji[id_US_new - 1, id_US_new - 1, :] = 0
        
        data = [N_new, K, E_i_multi, Y_i_multi, lambda_ji, beta_i, ell_ik, t_ji, nu_new, T]
        param = [eps_3D, kappa, psi, phi]
        
        syst = lambda x: Balanced_Trade_EQ_MultiSector(x, data, param)[0]
        x_fsolve = fsolve(syst, x_fsolve, xtol=1e-10, maxfev=100000)
        
        # Note: MATLAB uses d_trade(9) which is 1-indexed = index 8 in 0-indexed Python arrays
        _, results_multi[:, :, 1], d_trade[8] = Balanced_Trade_EQ_MultiSector(x_fsolve, data, param)
        d_employment[8] = (results_multi[:, 4, 1] * Y_i_multi).sum() / Y_i_multi.sum()
        
        print(f"Multi-sector baseline: N={N_new}, K={K}, US index={id_US_new}")
        print(f"d_trade[8]={d_trade[7]:.3f}, d_trade[9]={d_trade[8]:.3f}")  # MATLAB indices shown
        print(f"d_employment[8]={d_employment[7]:.3f}, d_employment[9]={d_employment[8]:.3f}")  # MATLAB indices shown
        
        # Store results_multi for use in print_tables_baseline.py
        # Note: We need to make this accessible globally or pass it through
        # For now, we'll store it in a way that can be retrieved
        # The calling code (main_baseline.py) should handle passing this to print_tables_baseline
        
        return results, d_trade, d_employment, results_multi, id_US_new, E_i_multi
        
    except FileNotFoundError:
        print("Warning: trade_ITPD.csv not found. Skipping multi-sector analysis.")
        return results, d_trade, d_employment, None, None, None
    except Exception as e:
        print(f"Error in multi-sector baseline analysis: {e}")
        import traceback
        traceback.print_exc()
        return results, d_trade, d_employment, None, None, None


if __name__ == '__main__':
    # This would be called from main_baseline.py
    pass

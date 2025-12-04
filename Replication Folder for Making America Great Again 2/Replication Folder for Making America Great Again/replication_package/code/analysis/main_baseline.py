"""
Main baseline analysis - Python conversion of main_baseline.m
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve, least_squares
import sys

# Import shared utilities
from utils import solveNu


def Balanced_Trade_EQ(x, data, param, lump_sum=0):
    """
    Balanced trade equilibrium function
    """
    N, E_i, Y_i, lambda_ji, t_ji, nu, T_i = data
    eps, kappa, psi, phi = param
    
    # Extract variables
    w_i_h = np.abs(x[:N])
    E_i_h = np.abs(x[N:N+N])
    L_i_h = np.abs(x[N+N:N+N+N])
    
    # Construct 2D matrices
    wi_h_2D = np.tile(w_i_h.reshape(-1, 1), (1, N))
    phi_2D = np.tile(phi.reshape(1, -1), (N, 1))
    
    # Construct new trade values
    AUX0 = lambda_ji * ((wi_h_2D / (L_i_h.reshape(-1, 1) ** psi)) ** -eps) * ((1 + t_ji) ** (-eps * phi_2D))
    AUX1 = np.tile(AUX0.sum(axis=0).reshape(1, -1), (N, 1))
    lambda_ji_new = AUX0 / AUX1
    Y_i_h = w_i_h * L_i_h
    Y_i_new = Y_i_h * Y_i
    E_i_new = E_i * E_i_h
    
    P_i_h = ((E_i_h / w_i_h) ** (1 - phi)) * (AUX0.sum(axis=0) ** (-1.0 / eps))
    
    X_ji_new = lambda_ji_new * np.tile(E_i_new.reshape(1, -1), (N, 1)) / (1 + t_ji)
    tariff_rev = (lambda_ji_new * (t_ji / (1 + t_ji)) * np.tile(E_i_new.reshape(1, -1), (N, 1))).sum(axis=0)
    
    if lump_sum == 0:
        tau_i = tariff_rev / Y_i_new
        tau_i_new = 0
        tau_i_h = (1 - tau_i_new) / (1 - tau_i)
    elif lump_sum == 1:
        tau_i = np.zeros(N)
        tau_i_h = np.ones(N)
    
    # Wage Income = Total Sales net of Taxes
    nu_2D = np.tile(nu.reshape(1, -1), (N, 1))
    ERR1 = ((1 - nu_2D) * X_ji_new).sum(axis=1) + (nu_2D * X_ji_new).sum(axis=0) - w_i_h * L_i_h * Y_i
    ERR1[N-1] = ((P_i_h - 1) * E_i).mean()  # Replace one excess equation
    
    # Total Income = Total Sales
    X_global = Y_i.sum()
    X_global_new = Y_i_new.sum()
    
    ERR2 = tariff_rev + (w_i_h * L_i_h * Y_i) + T_i * (X_global_new / X_global) - E_i_new
    
    # Labor supply equation
    ERR3 = L_i_h - (tau_i_h * w_i_h / P_i_h) ** kappa
    
    ceq = np.concatenate([ERR1, ERR2, ERR3])
    
    # Calculate results
    delta_i = E_i / (E_i - kappa * (1 - tau_i) * Y_i / (1 + kappa))
    W_i_h = delta_i * (E_i_h / P_i_h) + (1 - delta_i) * (w_i_h * L_i_h / P_i_h)
    
    # Factual trade flows
    X_ji = lambda_ji * np.tile(E_i.reshape(1, -1), (N, 1))
    eye_N = np.eye(N)
    # MATLAB: D_i = sum(X_ji,1)' - sum(X_ji,2)
    # In MATLAB: sum(X_ji,1) sums along dimension 1 (columns) = imports per destination
    #            sum(X_ji,2) sums along dimension 2 (rows) = exports per origin
    # In Python: sum(axis=0) sums along rows = imports per destination
    #           sum(axis=1) sums along columns = exports per origin
    # So: D_i = imports - exports (deficit is positive when imports > exports)
    D_i = X_ji.sum(axis=0) - X_ji.sum(axis=1)  # imports - exports = deficit
    D_i_new = X_ji_new.sum(axis=0) - X_ji_new.sum(axis=1)  # imports - exports = deficit
    
    d_welfare = 100 * (W_i_h - 1)
    d_export = 100 * (((X_ji_new * (1 - eye_N)).sum(axis=1) / Y_i_new) / 
                      ((X_ji * (1 - eye_N)).sum(axis=1) / Y_i) - 1)
    d_import = 100 * (((X_ji_new * (1 - eye_N)).sum(axis=0) / Y_i_new) / 
                      ((X_ji * (1 - eye_N)).sum(axis=0) / Y_i) - 1)
    d_employment = 100 * (L_i_h - 1)
    d_CPI = 100 * (P_i_h - 1)
    d_D_i = 100 * ((D_i_new - D_i) / np.abs(D_i))
    
    trade = X_ji * (1 - eye_N)
    trade_new = X_ji_new * (1 + t_ji) * (1 - eye_N)
    d_trade = 100 * ((trade_new.sum() / trade.sum()) / 
                     (Y_i_new.sum() / Y_i.sum()) - 1)
    
    results = np.column_stack([d_welfare, d_D_i, d_export, d_import, 
                               d_employment, d_CPI, tariff_rev / E_i])
    
    return ceq, results, d_trade


def main():
    """Main baseline analysis"""
    print('pwd =', os.getcwd())
    
    # Read trade and GDP data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
    
    data = pd.read_csv(os.path.join(data_dir, 'trade_cepii.csv'))
    X_ji = data.values
    X_ji = np.nan_to_num(X_ji, nan=0.0)
    N = X_ji.shape[0]
    
    id_US = 185
    id_CAN = 31
    id_MEX = 115
    id_CHN = 34
    id_EU = np.array([10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 88, 107, 108, 109, 119, 133, 144, 145, 149, 164, 165])
    id_RoW = np.setdiff1d(np.arange(1, N+1), np.concatenate([[id_US], [id_CHN], id_EU]))
    non_US = np.setdiff1d(np.arange(1, N+1), [id_US])
    
    # GDP data
    gdp = pd.read_csv(os.path.join(data_dir, 'gdp.csv'))
    Y_i = gdp.values.flatten()
    Y_i = Y_i / 1000  # Trade flows are in 1000 of USD
    
    tot_exports = X_ji.sum(axis=1)
    tot_imports = X_ji.sum(axis=0)
    
    nu_eq = solveNu(X_ji, Y_i, id_US)
    nu = nu_eq[0] * np.ones(N)
    nu[id_US - 1] = nu_eq[1]  # Convert to 0-indexed
    
    T = (1 - nu) * (X_ji.sum(axis=0) - (np.outer((1 - nu), np.ones(N)) * X_ji).sum(axis=1))
    E_i = Y_i + T
    X_ii = E_i - tot_imports
    X_ii[X_ii < 0] = 0
    np.fill_diagonal(X_ji, X_ii)
    
    E_i = X_ji.sum(axis=0)
    Y_i = (np.outer((1 - nu), np.ones(N)) * X_ji).sum(axis=1) + nu * X_ji.sum(axis=0)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))
    
    # Read US tariffs
    reuters = pd.read_csv(os.path.join(data_dir, 'tariffs.csv'))
    new_ustariff = reuters.values.flatten()
    t_ji = np.zeros((N, N))
    t_ji[:, id_US - 1] = new_ustariff  # Convert to 0-indexed
    
    t_ji[:, id_US - 1] = np.maximum(0.1, t_ji[:, id_US - 1])
    t_ji[id_US - 1, id_US - 1] = 0
    tariff = [t_ji]
    
    # Trade elasticity
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    
    Phi = [1 + phi_tilde, 0.5 + phi_tilde, 0.25 + phi_tilde]
    
    # Create array to save results
    results = np.zeros((N, 7, 9))
    revenue = np.zeros(9)
    d_trade = np.zeros(9)
    d_employment = np.zeros(9)
    
    # Baseline Analysis
    for i in range(2):
        t_ji_new = tariff[0]  # Use Reuters

        if i == 0:
            phi = Phi[0]
        elif i == 1:
            phi = Phi[2]  # MATLAB uses Phi{3} when i==2
        
        data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
        param = [eps, kappa, psi, phi]
        lump_sum = 0
        
        x0 = np.ones(3 * N)
        syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
        x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
        
        _, results[:, :, i], d_trade[i] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
        
        revenue[i] = results[id_US - 1, 6, i]  # Convert to 0-indexed
        d_employment[i] = (results[:, 4, i] * Y_i).sum() / Y_i.sum()
    
    # Eaton-Kortum Specification
    Y_i_EK = X_ji.sum(axis=1)
    T_EK = E_i - Y_i_EK
    
    t_ji_new = tariff[0]  # Use Reuters
    phi = np.ones(N)
    nu_EK = np.zeros(N)
    
    data = [N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 2], d_trade[2] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    revenue[2] = results[id_US - 1, 6, 2]
    d_employment[2] = (results[:, 4, 2] * Y_i).sum() / Y_i.sum()
    
    # Lump-sum rebate of tariff revenue
    t_ji_new = tariff[0]  # Use USTR tariffs
    phi = Phi[0]
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 1
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 7], _ = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    # Higher trade elasticity
    t_ji_new = tariff[0]  # Use USTR tariffs
    phi = Phi[0]
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [2 * eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 8], _ = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    # Optimal Tariff w/o retaliation
    t_ji_new = np.zeros((N, N))
    phi = Phi[0]
    
    eye_N = np.eye(N)
    delta = ((X_ji * np.tile((1 - nu).reshape(-1, 1), (1, N)) * 
              (1 - eye_N) * (1 - lambda_ji)).sum(axis=1)) / \
            ((1 - nu) * (E_i - np.diag(X_ji)))
    t_ji_new[:, id_US - 1] = 1 / ((1 + delta[id_US - 1] * eps) * phi[id_US - 1] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 3], d_trade[3] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    revenue[3] = results[id_US - 1, 6, 3]
    d_employment[3] = (results[:, 4, 3] * Y_i).sum() / Y_i.sum()
    
    # Liberation Tariffs with optimal retaliation
    t_ji_new = tariff[0]
    phi = Phi[0]
    
    AggI = np.zeros((2, N))
    AggI[0, :] = 1.0
    AggI[0, id_US - 1] = 0
    AggI[1, id_US - 1] = 1
    X = AggI @ X_ji @ AggI.T
    Y = AggI @ Y_i
    lambda_agg = X / np.tile(X.sum(axis=0).reshape(1, -1), (2, 1))
    
    eye_2 = np.eye(2)
    delta = ((X * np.tile((1 - nu_eq).reshape(-1, 1), (1, 2)) * 
              (1 - eye_2) * (1 - lambda_agg)).sum(axis=1)) / \
            ((1 - nu_eq) * (Y - np.diag(X)))
    
    t_ji_new[id_US - 1, :] = 1 / ((1 + delta[0] * eps) * phi[0] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 4], d_trade[4] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    revenue[4] = results[id_US - 1, 6, 4]
    d_employment[4] = (results[:, 4, 4] * Y_i).sum() / Y_i.sum()
    
    # Liberation Tariffs with reciprocal retaliation
    t_ji_new = tariff[0]
    phi = Phi[0]
    
    t_ji_new[id_US - 1, :] = t_ji_new[:, id_US - 1].T
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 5], d_trade[5] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    revenue[5] = results[id_US - 1, 6, 5]
    d_employment[5] = (results[:, 4, 5] * Y_i).sum() / Y_i.sum()
    
    # Optimal Tariff w/ optimal retaliation
    t_ji_new = np.zeros((N, N))
    phi = Phi[0]
    
    delta = ((X_ji * np.tile((1 - nu).reshape(-1, 1), (1, N)) * 
              (1 - eye_N) * (1 - lambda_ji)).sum(axis=1)) / \
            ((1 - nu) * (Y_i - np.diag(X_ji)))
    t_ji_new[:, id_US - 1] = 1 / ((1 + delta[id_US - 1] * eps) * phi[id_US - 1] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    AggI = np.zeros((2, N))
    AggI[0, :] = 1.0
    AggI[0, id_US - 1] = 0
    AggI[1, id_US - 1] = 1
    X = AggI @ X_ji @ AggI.T
    Y = AggI @ Y_i
    lambda_agg = X / np.tile(X.sum(axis=0).reshape(1, -1), (2, 1))
    
    eye_2 = np.eye(2)
    delta = ((X * np.tile((1 - nu_eq).reshape(-1, 1), (1, 2)) * 
              (1 - eye_2) * (1 - lambda_agg)).sum(axis=1)) / \
            ((1 - nu_eq) * (Y - np.diag(X)))
    
    t_ji_new[id_US - 1, :] = 1 / ((1 + delta[0] * eps) * phi[0] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 6], d_trade[6] = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    d_employment[6] = (results[:, 4, 6] * Y_i).sum() / Y_i.sum()
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    countries = pd.read_csv(os.path.join(data_dir, 'country_labels.csv'))
    country_names = countries['iso3'].values
    
    Data_base = results[:, :, 0]
    Tab = pd.DataFrame({'Country': country_names, 'Value': [Data_base[i, 0] for i in range(N)]})
    Tab.to_csv(os.path.join(output_dir, 'output_map.csv'), index=False)
    
    Data_retal = results[:, :, 4]
    Tab = pd.DataFrame({'Country': country_names, 'Value': [Data_retal[i, 0] for i in range(N)]})
    Tab.to_csv(os.path.join(output_dir, 'output_map_retal.csv'), index=False)
    
    # Store variables for other scripts
    globals_dict = globals()
    globals_dict.update({
        'results': results,
        'revenue': revenue,
        'd_trade': d_trade,
        'd_employment': d_employment,
        'Y_i': Y_i,
        'E_i': E_i,
        'Phi': Phi,
        'nu': nu,
        'id_US': id_US,
        'id_EU': id_EU,
        'id_CHN': id_CHN,
        'id_RoW': id_RoW,
        'non_US': non_US,
        'country_names': country_names,
        'N': N
    })
    
    # Import and run sub_multisector_baseline
    import sub_multisector_baseline
    results_multi = None
    id_US_new = None
    E_i_multi = None
    try:
        results, d_trade, d_employment, results_multi, id_US_new, E_i_multi = \
            sub_multisector_baseline.main(results, revenue, d_trade, d_employment, Y_i, E_i, Phi, nu, 
                                          id_US, id_EU, id_CHN, id_RoW, non_US, country_names, N)
    except Exception as e:
        print(f"Warning: Multi-sector baseline analysis incomplete: {e}")
        import traceback
        traceback.print_exc()
    
    # Save Table_11.mat equivalent (using pickle or npz)
    import pickle
    with open(os.path.join(output_dir, 'Table_11.pkl'), 'wb') as f:
        pickle.dump({'d_trade': d_trade, 'd_employment': d_employment}, f)
    
    # Import and run print_tables_baseline
    import print_tables_baseline
    print_tables_baseline.main(results, revenue, d_trade, d_employment, Y_i, E_i, id_US, id_EU, 
                              id_CHN, id_RoW, non_US, country_names, N,
                              results_multi=results_multi, id_US_new=id_US_new, E_i_multi=E_i_multi)
    
    return results, revenue, d_trade, d_employment


if __name__ == '__main__':
    main()


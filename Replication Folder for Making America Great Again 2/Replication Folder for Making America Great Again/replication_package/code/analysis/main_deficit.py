"""
Main deficit analysis - Python conversion of main_deficit.m
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve
import sys

from utils import solveNu
from main_baseline import Balanced_Trade_EQ


def Balanced_Trade_EQ_NoLumpSum(x, data, param):
    """
    Balanced trade equilibrium function without lump_sum parameter
    Same as Balanced_Trade_EQ but with lump_sum=0 hardcoded
    """
    result = Balanced_Trade_EQ(x, data, param, lump_sum=0)
    if len(result) == 3:
        return result[0], result[1]  # Return ceq and results only
    else:
        return result[0], result[1]  # Handle both cases


def main():
    """Main deficit framework analysis"""
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
    
    E_i = X_ji.sum(axis=0)
    Y_i = (np.outer((1 - nu), np.ones(N)) * X_ji).sum(axis=1) + nu * X_ji.sum(axis=0)
    T = E_i - Y_i
    lambda_ji = X_ji / np.tile(E_i.reshape(1, -1), (N, 1))
    
    # Read USTR tariffs
    reuters = pd.read_csv(os.path.join(data_dir, 'tariffs.csv'))
    new_ustariff = reuters.values.flatten()
    t_ji = np.zeros((N, N))
    t_ji[:, id_US - 1] = new_ustariff
    
    t_ji[:, id_US - 1] = np.maximum(0.1, t_ji[:, id_US - 1])
    t_ji[id_US - 1, id_US - 1] = 0
    tariff = [t_ji]
    
    # Trade elasticity
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    
    theta = eps / 0.67
    phi = (1 + theta) / ((1 - nu) * theta) - (1 / theta)
    
    # Create array to save results
    results = np.zeros((N, 7, 4))
    
    # Fixed Deficit
    Y_i_EK = X_ji.sum(axis=1)
    T_EK = E_i - Y_i_EK
    nu_EK = np.zeros(N)
    
    t_ji_new = tariff[0]  # Use Reuters
    phi_EK = np.ones(N)
    
    data = [N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK]
    param = [eps, kappa, psi, phi_EK]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 0] = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    # Zero Deficit
    # Balance trade with the US
    T_i_new = T - (X_ji[id_US - 1, :] - X_ji[:, id_US - 1])  # Subtract imbalance w/ US from deficit
    T_i_new[id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, np.zeros((N, N)), nu, T_i_new]
    param = [eps, kappa, psi, phi]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, temp_a = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    t_ji_new = tariff[0]  # Use Reuters
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T_i_new]
    param = [eps, kappa, psi, phi]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, temp_b = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    results[:, :, 1] = temp_b - temp_a
    
    # Fixed Deficit + Retaliation
    Y_i_EK = X_ji.sum(axis=1)
    T_EK = E_i - Y_i_EK
    
    t_ji_new = tariff[0]
    t_ji_new[id_US - 1, :] = 1 / ((1 + eps) * phi_EK[0] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i_EK, lambda_ji, t_ji_new, nu_EK, T_EK]
    param = [eps, kappa, psi, phi_EK]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 2] = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    # Zero Deficit + Retaliation
    T_i_new = T - (X_ji[id_US - 1, :] - X_ji[:, id_US - 1])  # Subtract imbalance w/ US from deficit
    T_i_new[id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, np.zeros((N, N)), nu, T_i_new]
    param = [eps, kappa, psi, phi]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, temp_a = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    t_ji_new = tariff[0]
    t_ji_new[id_US - 1, :] = 1 / ((1 + eps) * phi[id_US - 1] - 1)
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T_i_new]
    param = [eps, kappa, psi, phi]
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ_NoLumpSum(x, data, param)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, temp_b = Balanced_Trade_EQ_NoLumpSum(x_fsolve, data, param)
    
    results[:, :, 3] = temp_b - temp_a
    
    # Print Output (Table 10)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    countries = pd.read_csv(os.path.join(data_dir, 'country_labels.csv'))
    country_names = countries['iso3'].values
    
    print_table_10(results, E_i, id_US, country_names, N, output_dir)
    
    return results


def print_table_10(results, E_i, id_US, country_names, N, output_dir):
    """Print Table 10 to LaTeX file"""
    file_path = os.path.join(output_dir, 'Table_10.tex')
    us_idx = id_US - 1
    non_us_indices = np.concatenate([np.arange(us_idx), np.arange(us_idx + 1, N)])
    
    with open(file_path, 'w') as f:
        # Table preamble - Case 1
        f.write('\\begin{tabular}{lccccccc}\n')
        f.write('        \\toprule\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{(1) Pre-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }}  \\\\\n')
        f.write('        \\midrule\n')
        f.write('        Country &\n')
        f.write('        \\specialcell{$\\Delta$ welfare} &\n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('        \\specialcell{$\\Delta$ employment} &\n')
        f.write('        \\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 1
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 2, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 0]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 1
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_indices, np.newaxis] * results[non_us_indices, :, 0]).sum(axis=0) / E_i[non_us_indices].sum()
        avg_non_US[0] = results[non_us_indices, 0, 0].mean()  # Average welfare (unweighted)
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 2
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{(2) Pre-retaliation: balanced trade (Ossa, 2014) }} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 2
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 2, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 1]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 2
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_indices, np.newaxis] * results[non_us_indices, :, 1]).sum(axis=0) / E_i[non_us_indices].sum()
        avg_non_US[0] = results[non_us_indices, 0, 1].mean()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 3
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[25pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{(3) Post-retaliation: fixed transfers to global GDP (Dekle et al., 2008) }} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 3
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 2, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 2]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 3
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_indices, np.newaxis] * results[non_us_indices, :, 2]).sum(axis=0) / E_i[non_us_indices].sum()
        avg_non_US[0] = results[non_us_indices, 0, 2].mean()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 4
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{(4) Post-retaliation: balanced trade (Ossa, 2014) }} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 4
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 2, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 3]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 4
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_indices, np.newaxis] * results[non_us_indices, :, 3]).sum(axis=0) / E_i[non_us_indices].sum()
        avg_non_US[0] = results[non_us_indices, 0, 3].mean()  # Note: using case 3 index as in MATLAB
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.2f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 10 generated: {file_path}")


if __name__ == '__main__':
    main()


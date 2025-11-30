"""
Main regional analysis - Python conversion of main_regional.m
"""

import numpy as np
import pandas as pd
import os
from scipy.optimize import fsolve
import sys

from utils import solveNu
from main_baseline import Balanced_Trade_EQ


def main():
    """Main regional trade war analysis"""
    # Read trade and GDP data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
    
    data = pd.read_csv(os.path.join(data_dir, 'trade_cepii.csv'))
    X_ji = data.values
    X_ji = np.nan_to_num(X_ji, nan=0.0)
    N = X_ji.shape[0]
    
    id_US = 185
    id_CHN = 34
    id_CAN = 31
    id_MEX = 115
    id_EU = np.array([10, 13, 17, 45, 47, 50, 56, 57, 59, 61, 71, 78, 80, 83, 
                      88, 107, 108, 109, 119, 133, 144, 145, 149, 164, 165])
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
    tariff_USTR = t_ji
    
    # Trade elasticity
    eps = 4
    kappa = 0.5
    psi = 0.67 / eps
    
    theta = eps / 0.67
    phi_tilde = (1 + theta) / ((1 - nu) * theta) - (1 / theta) - 1
    
    Phi = [1 + phi_tilde, 0.5 + phi_tilde]
    
    # Create array to save results
    results = np.zeros((N, 7, 3))
    
    # USTR tariff on China/EU + 10 percent tariff on others
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US - 1, id_US - 1] = 0.1
    t_ji_new[id_US - 1, non_US - 1] = 0.1
    
    t_ji_new[id_CHN - 1, id_US - 1] = tariff_USTR[id_CHN - 1, id_US - 1]
    t_ji_new[id_EU - 1, id_US - 1] = tariff_USTR[id_EU - 1, id_US - 1]
    
    t_ji_new[id_US - 1, id_CHN - 1] = tariff_USTR[id_CHN - 1, id_US - 1]
    t_ji_new[id_US - 1, id_EU - 1] = tariff_USTR[id_EU - 1, id_US - 1]
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    phi = Phi[0]
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 0], _ = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    # USTR tariff on China + 10 percent tariff on others
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US - 1, id_US - 1] = 0.1
    t_ji_new[id_US - 1, non_US - 1] = 0.1
    
    t_ji_new[id_CHN - 1, id_US - 1] = tariff_USTR[id_CHN - 1, id_US - 1]
    t_ji_new[id_US - 1, id_CHN - 1] = tariff_USTR[id_CHN - 1, id_US - 1]
    t_ji_new[id_US - 1, id_US - 1] = 0
    
    phi = Phi[0]
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 1], _ = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    # 108 percent tariff on China + 10 percent tariff on others
    t_ji_new = np.zeros((N, N))
    t_ji_new[non_US - 1, id_US - 1] = 0.1
    t_ji_new[id_US - 1, non_US - 1] = 0.1
    
    t_ji_new[id_CHN - 1, id_US - 1] = 1.08
    t_ji_new[id_US - 1, id_CHN - 1] = 1.08
    t_ji_new[id_US - 1, id_US - 1] = 0
    phi = Phi[0]
    
    data = [N, E_i, Y_i, lambda_ji, t_ji_new, nu, T]
    param = [eps, kappa, psi, phi]
    lump_sum = 0
    
    x0 = np.ones(3 * N)
    syst = lambda x: Balanced_Trade_EQ(x, data, param, lump_sum)[0]
    x_fsolve = fsolve(syst, x0, xtol=1e-10, maxfev=100000)
    
    _, results[:, :, 2], _ = Balanced_Trade_EQ(x_fsolve, data, param, lump_sum)
    
    # Save results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    countries = pd.read_csv(os.path.join(data_dir, 'country_labels.csv'))
    country_names = countries['iso3'].values
    
    # Print Table 8
    print_table_8(results, country_names, E_i, id_US, id_CHN, id_EU, id_RoW, output_dir)
    
    return results


def print_table_8(results, country_names, E_i, id_US, id_CHN, id_EU, id_RoW, output_dir):
    """Print Table 8 to LaTeX file"""
    file_path = os.path.join(output_dir, 'Table_8.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccccc}\n')
        f.write('        \\toprule\n')
        f.write('        \\multicolumn{4}{l}{\\textbf{Case 1: US trade war with EU \\& China}}  \\\\\n')
        f.write('        \\midrule\n')
        f.write('        Country &\n')
        f.write('        \\specialcell{$\\Delta$ welfare} &\n')
        f.write('        \\specialcell{$\\Delta$ deficit} &\n')
        f.write('        \\specialcell{$\\Delta$ employment} &\n')
        f.write('        \\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('        \\midrule\n')
        
        # Results for US and China
        for i in [id_US, id_CHN]:
            idx = i - 1  # Convert to 0-indexed
            f.write(f'        {country_names[idx]} & ')
            f.write(f'{results[idx, 0, 0]:.2f}\\% & ')
            f.write(f'{results[idx, 1, 0]:.1f}\\% & ')
            f.write(f'{results[idx, 4, 0]:.2f}\\% & ')
            f.write(f'{results[idx, 5, 0]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        eu_idx = id_EU - 1
        avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 0]).sum(axis=0) / E_i[eu_idx].sum()
        
        f.write('        EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
        f.write('         \\addlinespace[3pt]\n')
        
        # RoW average
        row_idx = id_RoW - 1
        avg_RoW = (E_i[row_idx, np.newaxis] * results[row_idx, :, 0]).sum(axis=0) / E_i[row_idx].sum()
        
        f.write('        RoW & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Case 2
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{4}{l}{\\textbf{Case 2: US trade war with China}}  \\\\\n')
        f.write('        \\midrule\n')
        
        for i in [id_US, id_CHN]:
            idx = i - 1
            f.write(f'        {country_names[idx]} & ')
            f.write(f'{results[idx, 0, 1]:.2f}\\% & ')
            f.write(f'{results[idx, 1, 1]:.1f}\\% & ')
            f.write(f'{results[idx, 4, 1]:.2f}\\% & ')
            f.write(f'{results[idx, 5, 1]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 1]).sum(axis=0) / E_i[eu_idx].sum()
        
        f.write('        EU  & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
        f.write('         \\addlinespace[3pt]\n')
        
        # RoW average
        avg_RoW = (E_i[row_idx, np.newaxis] * results[row_idx, :, 1]).sum(axis=0) / E_i[row_idx].sum()
        
        f.write('        RoW & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Case 3
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{4}{l}{\\textbf{Case 3: US trade war with China (108\\% tariff)}} \\\\\n')
        f.write('        \\midrule\n')
        
        for i in [id_US, id_CHN]:
            idx = i - 1
            f.write(f'        {country_names[idx]} & ')
            f.write(f'{results[idx, 0, 2]:.2f}\\% & ')
            f.write(f'{results[idx, 1, 2]:.1f}\\% & ')
            f.write(f'{results[idx, 4, 2]:.2f}\\% & ')
            f.write(f'{results[idx, 5, 2]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 2]).sum(axis=0) / E_i[eu_idx].sum()
        
        f.write('        EU & ')
        f.write(f'{avg_EU[0]:.2f}\\%  & ')
        f.write(f'{avg_EU[1]:.1f}\\% & ')
        f.write(f'{avg_EU[4]:.2f}\\% & ')
        f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
        f.write('         \\addlinespace[3pt]\n')
        
        # RoW average
        avg_RoW = (E_i[row_idx, np.newaxis] * results[row_idx, :, 2]).sum(axis=0) / E_i[row_idx].sum()
        
        f.write('        RoW& ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')


if __name__ == '__main__':
    main()


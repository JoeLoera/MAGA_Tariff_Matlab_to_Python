"""
Print tables IO - Python conversion of print_tables_io.m
Generates LaTeX tables: Tables 4, 11
"""

import numpy as np
import os


def main(results, d_trade_IO, d_employment_IO, Y_i_IO, E_i_IO, id_US, d_trade, d_employment,
         results_multi_IO=None, id_US_new=None, E_i_multi=None, d_trade_IO_multi=None, d_employment_IO_multi=None):
    """
    Generate LaTeX tables for IO analysis
    
    Tables: 4, 11
    
    Parameters:
    -----------
    results : array (N, 7, 4)
        IO results array (4 scenarios)
    d_trade_IO : array (2,)
        Trade changes for IO model (before/after retaliation)
    d_employment_IO : array (2,)
        Employment changes for IO model (before/after retaliation)
    Y_i_IO : array (N,)
        GDP for IO model
    E_i_IO : array (N,)
        Expenditure for IO model
    id_US : int
        US country ID (1-indexed)
    d_trade : array (9,)
        Trade changes from baseline (includes multi-sector)
    d_employment : array (9,)
        Employment changes from baseline (includes multi-sector)
    """
    print("Generating IO tables...")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Read country names (needed for Table 4)
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'base_data')
    import pandas as pd
    countries = pd.read_csv(os.path.join(data_dir, 'country_labels.csv'))
    country_names = countries['iso3'].values
    N = len(country_names)
    
    # Use multi-sector results if provided (default to None/zeros if not available)
    if d_trade_IO_multi is None:
        d_trade_IO_multi = np.zeros(2)
    if d_employment_IO_multi is None:
        d_employment_IO_multi = np.zeros(2)
    
    # Table 4
    try:
        generate_table_4(results, Y_i_IO, id_US, country_names, N, 
                        results_multi_IO, id_US_new, E_i_multi, output_dir)
    except Exception as e:
        print(f"Error generating Table 4: {e}")
        import traceback
        traceback.print_exc()
    
    # Table 11
    try:
        generate_table_11(d_trade, d_employment, d_trade_IO, d_employment_IO,
                         d_trade_IO_multi, d_employment_IO_multi, output_dir)
    except Exception as e:
        print(f"Error generating Table 11: {e}")
        import traceback
        traceback.print_exc()


def generate_table_4(results, Y_i_IO, id_US, country_names, N, 
                     results_multi_IO=None, id_US_new=None, E_i_multi=None, output_dir=None):
    """Generate Table 4"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    
    file_path = os.path.join(output_dir, 'Table_4.tex')
    us_idx = id_US - 1
    non_us_indices = np.concatenate([np.arange(us_idx), np.arange(us_idx + 1, N)])
    
    with open(file_path, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lrrrrrr}\n')
        f.write('        \\toprule\n')
        f.write('        &\n')
        f.write('        \\specialcell{$\\Delta$ welfare} &\n')
        f.write('        \\specialcell{$\\Delta$ deficit} &\n')
        f.write('        \\specialcell{$\\Delta$$\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('        \\specialcell{$\\Delta$$\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('        \\specialcell{$\\Delta$ emp} &\n')
        f.write('        \\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('        \\addlinespace[-8pt]\n')
        f.write('        \\multicolumn{7}{l}{\\textbf{Pre-Retaliation Scenarios}} \\\\\n')
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[5pt]\n')
        f.write('        \\textbf{(1)} \\textit{USTR tariffs + one sector} \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        f.write('        \\addlinespace[3pt]\n')
        
        # Case 1: USTR tariffs + one sector
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 0]:.1f}\\% \\\\\n')
        
        # Non-US average
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (Y_i_IO[non_us_indices, np.newaxis] * results[non_us_indices, :, 0]).sum(axis=0) / Y_i_IO[non_us_indices].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 2: Optimal tariff + one sector
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\textbf{(2)}  \\textit{Optimal tariff + one sector} \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        f.write('        \\addlinespace[3pt]\n')
        
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 1]:.1f}\\% \\\\\n')
        
        # Non-US average
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (Y_i_IO[non_us_indices, np.newaxis] * results[non_us_indices, :, 1]).sum(axis=0) / Y_i_IO[non_us_indices].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 3: USTR tariffs + multiple sectors
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\textbf{(3)}  \\textit{USTR tariffs + multiple sectors} \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        f.write('        \\addlinespace[3pt]\n')
        
        if results_multi_IO is not None and id_US_new is not None and E_i_multi is not None:
            us_new_idx = id_US_new - 1 if id_US_new > 0 else 0
            non_us_multi_idx = np.setdiff1d(np.arange(results_multi_IO.shape[0]), [us_new_idx])
            
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results_multi_IO[us_new_idx, 0, 0]:.2f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 1, 0]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 2, 0]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 3, 0]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 4, 0]:.2f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 5, 0]:.1f}\\% \\\\\n')
            
            # Non-US average
            f.write('         \\addlinespace[3pt]\n')
            avg_non_US = (E_i_multi[non_us_multi_idx, np.newaxis] * results_multi_IO[non_us_multi_idx, :, 0]).sum(axis=0) / E_i_multi[non_us_multi_idx].sum()
            
            f.write('        non-US (average) & ')
            f.write(f'{avg_non_US[0]:.2f}\\%  & ')
            f.write(f'{avg_non_US[1]:.1f}\\% & ')
            f.write(f'{avg_non_US[2]:.1f}\\% & ')
            f.write(f'{avg_non_US[3]:.1f}\\% & ')
            f.write(f'{avg_non_US[4]:.2f}\\% & ')
            f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        else:
            # Placeholder if multi-sector not available
            f.write(f'        {country_names[us_idx]} & ')
            f.write('N/A & N/A & N/A & N/A & N/A & N/A \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
            f.write('        non-US (average) & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n')
        
        # Post-Retaliation Scenarios
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{7}{l}{\\textbf{Post-Retaliation Scenarios}} \\\\\n')
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[5pt]\n')
        f.write('        \\textbf{(1)}  \\textit{reciprocal retaliation + one sector} \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        f.write('        \\addlinespace[3pt]\n')
        
        # Case 1 Post-retaliation: reciprocal retaliation + one sector
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 2]:.1f}\\% \\\\\n')
        
        # Non-US average
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (Y_i_IO[non_us_indices, np.newaxis] * results[non_us_indices, :, 2]).sum(axis=0) / Y_i_IO[non_us_indices].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 2 Post-retaliation: optimal retaliation + one sector
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[5pt]\n')
        f.write('        \\textbf{(2)}  \\textit{optimal retaliation + one sector} \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        f.write('        \\addlinespace[3pt]\n')
        
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 3]:.1f}\\% \\\\\n')
        
        # Non-US average
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (Y_i_IO[non_us_indices, np.newaxis] * results[non_us_indices, :, 3]).sum(axis=0) / Y_i_IO[non_us_indices].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 3 Post-retaliation: reciprocal retaliation + multiple sectors
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[5pt]\n')
        f.write('        \\textbf{(3)}  \\textit{reciprocal retaliation + multiple sectors}  \\\\\n')
        f.write('        \\cmidrule(lr){1-1}\n')
        
        if results_multi_IO is not None and id_US_new is not None and E_i_multi is not None:
            us_new_idx = id_US_new - 1 if id_US_new > 0 else 0
            non_us_multi_idx = np.setdiff1d(np.arange(results_multi_IO.shape[0]), [us_new_idx])
            
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results_multi_IO[us_new_idx, 0, 1]:.2f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 1, 1]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 2, 1]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 3, 1]:.1f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 4, 1]:.2f}\\% & ')
            f.write(f'{results_multi_IO[us_new_idx, 5, 1]:.1f}\\% \\\\\n')
            
            # Non-US average
            f.write('         \\addlinespace[3pt]\n')
            avg_non_US = (E_i_multi[non_us_multi_idx, np.newaxis] * results_multi_IO[non_us_multi_idx, :, 1]).sum(axis=0) / E_i_multi[non_us_multi_idx].sum()
            
            f.write('        non-US (average) & ')
            f.write(f'{avg_non_US[0]:.2f}\\%  & ')
            f.write(f'{avg_non_US[1]:.1f}\\% & ')
            f.write(f'{avg_non_US[2]:.1f}\\% & ')
            f.write(f'{avg_non_US[3]:.1f}\\% & ')
            f.write(f'{avg_non_US[4]:.2f}\\% & ')
            f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        else:
            # Placeholder if multi-sector not available
            f.write(f'        {country_names[us_idx]} & ')
            f.write('N/A & N/A & N/A & N/A & N/A & N/A \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
            f.write('        non-US (average) & N/A & N/A & N/A & N/A & N/A & N/A \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 4 generated: {file_path}")


def generate_table_11(d_trade, d_employment, d_trade_IO, d_employment_IO,
                      d_trade_IO_multi, d_employment_IO_multi, output_dir=None):
    """Generate Table 11"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    
    file_path = os.path.join(output_dir, 'Table_11.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lcccccccccc}\n')
        f.write('        \\toprule\n')
        f.write('        & \\multicolumn{4}{c}{before retaliation} && \\multicolumn{4}{c}{after retaliation} \\\\\n')
        f.write('        \\cmidrule(lr){2-5} \\cmidrule(lr){7-10}\n')
        f.write('        main & IO & multi & multi + IO  && main & IO & multi & multi + IO \\\\\n')
        f.write('        \\midrule\n')
        
        # Delta global trade-to-GDP row
        f.write('        $\\Delta$ global trade-to-GDP &')
        # Before retaliation: main (d_trade[0]), IO (d_trade_IO[0]), multi (d_trade[7]), multi+IO (d_trade_IO_multi[0])
        f.write(f'{d_trade[0]:.1f}\\% & ')
        f.write(f'{d_trade_IO[0]:.1f}\\% & ')
        f.write(f'{d_trade[7]:.1f}\\% & ')
        f.write(f'{d_trade_IO_multi[0]:.1f}\\% && ')
        # After retaliation: main (d_trade[5]), IO (d_trade_IO[1]), multi (d_trade[8]), multi+IO (d_trade_IO_multi[1])
        f.write(f'{d_trade[5]:.1f}\\% & ')
        f.write(f'{d_trade_IO[1]:.1f}\\% & ')
        f.write(f'{d_trade[8]:.1f}\\% & ')
        f.write(f'{d_trade_IO_multi[1]:.1f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        
        # Delta global employment row
        f.write('        $\\Delta$ global employment &')
        # Before retaliation
        f.write(f'{d_employment[0]:.2f}\\% & ')
        f.write(f'{d_employment_IO[0]:.2f}\\% & ')
        f.write(f'{d_employment[7]:.2f}\\% & ')
        f.write(f'{d_employment_IO_multi[0]:.2f}\\% && ')
        # After retaliation
        f.write(f'{d_employment[5]:.2f}\\% & ')
        f.write(f'{d_employment_IO[1]:.2f}\\% & ')
        f.write(f'{d_employment[8]:.2f}\\% & ')
        f.write(f'{d_employment_IO_multi[1]:.2f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 11 generated: {file_path}")
    
    # Delete Table_11.pkl if it exists (MATLAB version deletes .mat file)
    pkl_path = os.path.join(output_dir, 'Table_11.pkl')
    if os.path.exists(pkl_path):
        try:
            os.remove(pkl_path)
        except:
            pass  # Don't fail if can't delete


if __name__ == '__main__':
    # This would be called from main_io.py
    pass


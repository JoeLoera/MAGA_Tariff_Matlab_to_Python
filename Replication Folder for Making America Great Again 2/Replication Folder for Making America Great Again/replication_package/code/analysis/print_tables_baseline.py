"""
Print tables baseline - Python conversion of print_tables_baseline.m
Generates LaTeX tables: Tables 1, 2, 3, 9
"""

import numpy as np
import os


def main(results, revenue, d_trade, d_employment, Y_i, E_i, id_US, id_EU, 
         id_CHN, id_RoW, non_US, country_names, N,
         results_multi=None, id_US_new=None, E_i_multi=None):
    """
    Generate LaTeX tables for baseline analysis
    
    Tables: 1, 2, 3, 9
    """
    print("Generating baseline tables...")
    
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to 0-indexed for array access
    us_idx = id_US - 1
    chn_idx = id_CHN - 1 if id_CHN else None
    eu_idx = id_EU - 1 if len(id_EU) > 0 else None
    non_us_idx = non_US - 1
    
    # Table 1
    try:
        generate_table_1(results, revenue, E_i, us_idx, non_us_idx, country_names, N, output_dir)
    except Exception as e:
        print(f"Error generating Table 1: {e}")
        import traceback
        traceback.print_exc()
    
    # Table 2
    try:
        generate_table_2(results, E_i, us_idx, chn_idx, eu_idx, non_us_idx, country_names, N, output_dir)
    except Exception as e:
        print(f"Error generating Table 2: {e}")
        import traceback
        traceback.print_exc()
    
    # Table 3
    try:
        generate_table_3(revenue, output_dir)
    except Exception as e:
        print(f"Error generating Table 3: {e}")
        import traceback
        traceback.print_exc()
    
    # Table 9
    try:
        # Try to get results_multi if available (from sub_multisector_baseline)
        results_multi = None
        id_US_new = None
        E_i_multi = None
        try:
            import sub_multisector_baseline
            # This will fail if multi-sector didn't run, but that's OK
        except:
            pass
        
        generate_table_9(results, E_i, us_idx, non_us_idx, country_names, N, 
                        results_multi, id_US_new, E_i_multi, output_dir)
    except Exception as e:
        print(f"Error generating Table 9: {e}")
        import traceback
        traceback.print_exc()


def generate_table_1(results, revenue, E_i, us_idx, non_us_idx, country_names, N, output_dir):
    """Generate Table 1"""
    file_path = os.path.join(output_dir, 'Table_1.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble - Case 1
        f.write('\\begin{tabular}{lcccccccc}\n')
        f.write('        \\toprule\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Case 1: USTR tariffs + income tax relief + no retaliation}}  \\\\\n')
        f.write('        \\midrule\n')
        f.write('        Country &\n')
        f.write('        \\specialcell{$\\Delta$ welfare} &\n')
        f.write('        \\specialcell{$\\Delta$ deficit} &\n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('        \\specialcell{$\\Delta$ employment} &\n')
        f.write('        \\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 1
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 0]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 1
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 0]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 2
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Case 2: USTR tariffs + lump-sum rebate + no retaliation}} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 2 (results[:,:,8])
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 7]:.2f}\\% & ')  # Note: results[:,:,8] is index 7 in 0-indexed
        f.write(f'{results[us_idx, 1, 7]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 7]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 7]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 7]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 7]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 2
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 7]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Case 3
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Case 3: optimal US tariffs + income tax relief + no retaliation}} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Case 3 (results[:,:,4])
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 3]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 3]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 3]:.1f}\\% \\\\\n')
        
        # Non-US average - Case 3
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 3]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 1 generated: {file_path}")


def generate_table_2(results, E_i, us_idx, chn_idx, eu_idx, non_us_idx, country_names, N, output_dir):
    """Generate Table 2"""
    file_path = os.path.join(output_dir, 'Table_2.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lccccc}\n')
        f.write('        \\toprule\n')
        f.write('        \\multicolumn{5}{l}{\\textbf{(1) USTR tariff + reciprocal retaliation}} \\\\\n')
        f.write('        \\midrule\n')
        f.write('        Country &\n')
        f.write('        $\\Delta$ welfare &\n')
        f.write('        $\\Delta$ deficit &\n')
        f.write('        $\\Delta$ employment &\n')
        f.write('        $\\Delta$ real prices \\\\\n')
        f.write('        \\midrule\n')
        
        # Results for US and China - Case 1 (results[:,:,6])
        if chn_idx is not None:
            for i_idx in [us_idx, chn_idx]:
                f.write(f'        {country_names[i_idx]} & ')
                f.write(f'{results[i_idx, 0, 5]:.2f}\\% & ')
                f.write(f'{results[i_idx, 1, 5]:.1f}\\% & ')
                f.write(f'{results[i_idx, 4, 5]:.2f}\\% & ')
                f.write(f'{results[i_idx, 5, 5]:.1f}\\% \\\\\n')
                f.write('         \\addlinespace[3pt]\n')
        else:
            # Just US
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results[us_idx, 0, 5]:.2f}\\% & ')
            f.write(f'{results[us_idx, 1, 5]:.1f}\\% & ')
            f.write(f'{results[us_idx, 4, 5]:.2f}\\% & ')
            f.write(f'{results[us_idx, 5, 5]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        if eu_idx is not None and len(eu_idx) > 0:
            avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 5]).sum(axis=0) / E_i[eu_idx].sum()
            
            f.write('        EU & ')
            f.write(f'{avg_EU[0]:.2f}\\%  & ')
            f.write(f'{avg_EU[1]:.1f}\\% & ')
            f.write(f'{avg_EU[4]:.2f}\\% & ')
            f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # Non-US average
        avg_RoW = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 5]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Case 3: USTR tariff + optimal retaliation
        f.write('        \\bottomrule\n')
        f.write('        \\addlinespace[15pt]\n')
        f.write('        \\multicolumn{5}{l}{\\textbf{(3) USTR tariff + optimal retaliation}}  \\\\\n')
        f.write('        \\midrule\n')
        
        # Results for US and China - Case 3 (results[:,:,5])
        if chn_idx is not None:
            for i_idx in [us_idx, chn_idx]:
                f.write(f'        {country_names[i_idx]} & ')
                f.write(f'{results[i_idx, 0, 4]:.2f}\\% & ')
                f.write(f'{results[i_idx, 1, 4]:.1f}\\% & ')
                f.write(f'{results[i_idx, 4, 4]:.2f}\\% & ')
                f.write(f'{results[i_idx, 5, 4]:.1f}\\% \\\\\n')
                f.write('         \\addlinespace[3pt]\n')
        else:
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results[us_idx, 0, 4]:.2f}\\% & ')
            f.write(f'{results[us_idx, 1, 4]:.1f}\\% & ')
            f.write(f'{results[us_idx, 4, 4]:.2f}\\% & ')
            f.write(f'{results[us_idx, 5, 4]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        if eu_idx is not None and len(eu_idx) > 0:
            avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 4]).sum(axis=0) / E_i[eu_idx].sum()
            
            f.write('        EU & ')
            f.write(f'{avg_EU[0]:.2f}\\%  & ')
            f.write(f'{avg_EU[1]:.1f}\\% & ')
            f.write(f'{avg_EU[4]:.2f}\\% & ')
            f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # Non-US average
        avg_RoW = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 4]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Case 4: optimal tariff + optimal retaliation
        f.write('        \\bottomrule\n')
        f.write('        \\addlinespace[15pt]\n')
        f.write('        \\multicolumn{5}{l}{\\textbf{(4) optimal tariff + optimal retaliation}}  \\\\\n')
        f.write('        \\midrule\n')
        
        # Results for US and China - Case 4 (results[:,:,7])
        if chn_idx is not None:
            for i_idx in [us_idx, chn_idx]:
                f.write(f'        {country_names[i_idx]} & ')
                f.write(f'{results[i_idx, 0, 6]:.2f}\\% & ')
                f.write(f'{results[i_idx, 1, 6]:.1f}\\% & ')
                f.write(f'{results[i_idx, 4, 6]:.2f}\\% & ')
                f.write(f'{results[i_idx, 5, 6]:.1f}\\% \\\\\n')
                f.write('         \\addlinespace[3pt]\n')
        else:
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results[us_idx, 0, 6]:.2f}\\% & ')
            f.write(f'{results[us_idx, 1, 6]:.1f}\\% & ')
            f.write(f'{results[us_idx, 4, 6]:.2f}\\% & ')
            f.write(f'{results[us_idx, 5, 6]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # EU average
        if eu_idx is not None and len(eu_idx) > 0:
            avg_EU = (E_i[eu_idx, np.newaxis] * results[eu_idx, :, 6]).sum(axis=0) / E_i[eu_idx].sum()
            
            f.write('        EU & ')
            f.write(f'{avg_EU[0]:.2f}\\%  & ')
            f.write(f'{avg_EU[1]:.1f}\\% & ')
            f.write(f'{avg_EU[4]:.2f}\\% & ')
            f.write(f'{avg_EU[5]:.1f}\\% \\\\\n')
            f.write('         \\addlinespace[3pt]\n')
        
        # Non-US average
        avg_RoW = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 6]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_RoW[0]:.2f}\\%  & ')
        f.write(f'{avg_RoW[1]:.1f}\\% & ')
        f.write(f'{avg_RoW[4]:.2f}\\% & ')
        f.write(f'{avg_RoW[5]:.1f}\\% \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 2 generated: {file_path}")


def generate_table_3(revenue, output_dir):
    """Generate Table 3"""
    file_path = os.path.join(output_dir, 'Table_3.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble
        f.write('\\begin{tabular}{lccccc}\n')
        f.write('        \\toprule\n')
        f.write('        & & & \\multicolumn{2}{c}{retaliation} \\\\\n')
        f.write('        \\cmidrule(lr){4-5}\n')
        f.write('        & USTR tariff$ & optimal tariff &  optimal & reciprocal \\\'\n')
        f.write('        \\midrule\n')
        
        # % of GDP row
        f.write('        \\% of GDP &')
        f.write(f'{100*revenue[0]:.2f}\\% & ')
        f.write(f'{100*revenue[3]:.2f}\\% & ')
        f.write(f'{100*revenue[4]:.2f}\\% & ')
        f.write(f'{100*revenue[5]:.2f}\\% \\\\\n')
        
        # % of Federal Budget row
        f.write('        \\% of Federal Budget &')
        f.write(f'{100*revenue[0]/0.23:.2f}\\% & ')
        f.write(f'{100*revenue[3]/0.23:.2f}\\% & ')
        f.write(f'{100*revenue[4]/0.23:.2f}\\% & ')
        f.write(f'{100*revenue[5]/0.23:.2f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 3 generated: {file_path}")


def generate_table_9(results, E_i, us_idx, non_us_idx, country_names, N, 
                     results_multi=None, id_US_new=None, E_i_multi=None, output_dir=None):
    """Generate Table 9"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output')
    
    file_path = os.path.join(output_dir, 'Table_9.tex')
    
    with open(file_path, 'w') as f:
        # Table preamble - Baseline
        f.write('\\begin{tabular}{lcccccccc}\n')
        f.write('        \\toprule\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Baseline model:($\\tilde{\\varphi}=1$, $\\varphi>1$, $\\varepsilon=4$)}}  \\\\\n')
        f.write('        \\midrule\n')
        f.write('        Country &\n')
        f.write('        \\specialcell{$\\Delta$ welfare} &\n')
        f.write('        \\specialcell{$\\Delta$ deficit} &\n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{exports}}{\\textrm{GDP}}$} & \n')
        f.write('        \\specialcell{$\\Delta$ $\\frac{\\textrm{imports}}{\\textrm{GDP}}$} &\n')
        f.write('        \\specialcell{$\\Delta$ employment} &\n')
        f.write('        \\specialcell{$\\Delta$ prices} \\\\\n')
        f.write('        \\midrule\n')
        
        # US results - Baseline
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 0]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 0]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 0]:.1f}\\% \\\\\n')
        
        # Non-US average - Baseline
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 0]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Alternative 1: Multiple sectors (if available)
        if results_multi is not None and id_US_new is not None and E_i_multi is not None:
            f.write('        \\midrule\n')
            f.write('        \\addlinespace[10pt]\n')
            f.write('        \\multicolumn{6}{l}{\\textbf{Alternative 1: multiple sectors}}  \\\\\n')
            f.write('        \\midrule\n')
            
            us_new_idx = id_US_new - 1 if id_US_new > 0 else 0
            non_us_multi_idx = np.setdiff1d(np.arange(results_multi.shape[0]), [us_new_idx])
            
            f.write(f'        {country_names[us_idx]} & ')
            f.write(f'{results_multi[us_new_idx, 0]:.2f}\\% & ')
            f.write(f'{results_multi[us_new_idx, 1]:.1f}\\% & ')
            f.write(f'{results_multi[us_new_idx, 2]:.1f}\\% & ')
            f.write(f'{results_multi[us_new_idx, 3]:.1f}\\% & ')
            f.write(f'{results_multi[us_new_idx, 4]:.2f}\\% & ')
            f.write(f'{results_multi[us_new_idx, 5]:.1f}\\% \\\\\n')
            
            f.write('         \\addlinespace[3pt]\n')
            avg_non_US = (E_i_multi[non_us_multi_idx, np.newaxis] * results_multi[non_us_multi_idx, :]).sum(axis=0) / E_i_multi[non_us_multi_idx].sum()
            
            f.write('        non-US (average) & ')
            f.write(f'{avg_non_US[0]:.2f}\\%  & ')
            f.write(f'{avg_non_US[1]:.1f}\\% & ')
            f.write(f'{avg_non_US[2]:.1f}\\% & ')
            f.write(f'{avg_non_US[3]:.1f}\\% & ')
            f.write(f'{avg_non_US[4]:.2f}\\% & ')
            f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Alternative 2: Incomplete passthrough
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Alternative 2: incomplete passthrough to firm-level prices ($\\tilde{\\varphi}=0.25$)}}  \\\\\n')
        f.write('        \\midrule\n')
        
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 1]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 1]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 1]:.1f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 1]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Alternative 3: Higher trade elasticity
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Alternative 3: higher trade elasticity ($\\varepsilon=8$)}}  \\\\\n')
        f.write('        \\midrule\n')
        
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 8]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 8]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 8]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 8]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 8]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 8]:.1f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 8]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Alternative 4: Eaton-Kortum-Krugman model
        f.write('        \\midrule\n')
        f.write('        \\addlinespace[10pt]\n')
        f.write('        \\multicolumn{6}{l}{\\textbf{Alternative 4: Eaton-Kortum-Krugman model ($\\varphi=1$, $\\nu=0$)}} \\\\\n')
        f.write('        \\midrule\n')
        
        f.write(f'        {country_names[us_idx]} & ')
        f.write(f'{results[us_idx, 0, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 1, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 2, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 3, 2]:.1f}\\% & ')
        f.write(f'{results[us_idx, 4, 2]:.2f}\\% & ')
        f.write(f'{results[us_idx, 5, 2]:.1f}\\% \\\\\n')
        
        f.write('         \\addlinespace[3pt]\n')
        avg_non_US = (E_i[non_us_idx, np.newaxis] * results[non_us_idx, :, 2]).sum(axis=0) / E_i[non_us_idx].sum()
        
        f.write('        non-US (average) & ')
        f.write(f'{avg_non_US[0]:.2f}\\%  & ')
        f.write(f'{avg_non_US[1]:.1f}\\% & ')
        f.write(f'{avg_non_US[2]:.1f}\\% & ')
        f.write(f'{avg_non_US[3]:.1f}\\% & ')
        f.write(f'{avg_non_US[4]:.2f}\\% & ')
        f.write(f'{avg_non_US[5]:.1f}\\% \\\\\n')
        
        # Table closing
        f.write('         \\bottomrule\n')
        f.write('\\end{tabular}\n')
    
    print(f"Table 9 generated: {file_path}")


if __name__ == '__main__':
    # This would be called from main_baseline.py
    pass

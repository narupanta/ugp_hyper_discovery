import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

for model in ['123028_isihara', '142439_nh4']:
    out_dir = f'distillation/distilled_models/20260804T{model}_syn_d0.0001_l0.01_gmr_nolog_standard_interp_uqmodeldisc'
    csv_dir = os.path.join(out_dir, 'output', 'sensitivities')
    
    stats_path = os.path.join(csv_dir, 'total_sobol_indices_statistics_output_0.csv')
    first_stats_path = os.path.join(csv_dir, 'first_sobol_indices_statistics_output_0.csv')
    
    if os.path.exists(stats_path) and os.path.exists(first_stats_path):
        df_tot = pd.read_csv(stats_path).fillna(0.0)
        df_first = pd.read_csv(first_stats_path).fillna(0.0)
        
        param_cols = [c for c in df_tot.columns if c not in ['Unnamed: 0', '']]
        tot_means = df_tot.iloc[0][param_cols].values.astype(float)
        tot_stds = df_tot.iloc[1][param_cols].values.astype(float)
        first_means = df_first.iloc[0][param_cols].values.astype(float)
        first_stds = df_first.iloc[1][param_cols].values.astype(float)
        
        x = np.arange(len(param_cols))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width/2, first_means, width, yerr=first_stds, label='First-Order ($S_1$)', capsize=4, color='#3498db', alpha=0.85)
        ax.bar(x + width/2, tot_means, width, yerr=tot_stds, label='Total-Order ($S_T$)', capsize=4, color='#e74c3c', alpha=0.85)
        
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-5)
        
        ax.axhline(0.01, color='black', linestyle='--', linewidth=1.5, label='Threshold (0.01)')
        ax.set_ylabel('Sobol Sensitivity Index', fontsize=12, fontweight='bold')
        ax.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(param_cols, fontsize=11, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        plt.tight_layout()
        
        sobol_plot_path = os.path.join(out_dir, 'sobol_sensitivity_indices.png')
        plt.savefig(sobol_plot_path, dpi=200)
        plt.close()
        print(f'Updated {sobol_plot_path}')

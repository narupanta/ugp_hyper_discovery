import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_sensitivities(out_dir, prefix, subdir):
    dfs_tot = []
    dfs_first = []
    
    search_dirs = [
        os.path.join(out_dir, "output", subdir),
        os.path.join(out_dir, subdir)
    ]
    
    valid_dir = None
    for d in search_dirs:
        if os.path.exists(d):
            valid_dir = d
            break
            
    if not valid_dir:
        return {}, {}
        
    for i in range(4):
        # Try both with and without prefix
        for pfx in [prefix, ""]:
            p_tot = os.path.join(valid_dir, f"{pfx}total_sobol_indices_statistics_output_{i}.csv")
            p_first = os.path.join(valid_dir, f"{pfx}first_sobol_indices_statistics_output_{i}.csv")
            if os.path.exists(p_tot) and os.path.exists(p_first):
                dfs_tot.append(pd.read_csv(p_tot).fillna(0.0))
                dfs_first.append(pd.read_csv(p_first).fillna(0.0))
                break
            
    if not dfs_tot:
        return {}, {}
        
    param_cols = [c for c in dfs_tot[0].columns if c not in ["Unnamed: 0", ""]]
    
    tot_means_list = np.array([df.iloc[0][param_cols].values.astype(float) for df in dfs_tot])
    first_means_list = np.array([df.iloc[0][param_cols].values.astype(float) for df in dfs_first])
    
    tot_means = np.nanmean(tot_means_list, axis=0)
    first_means = np.nanmean(first_means_list, axis=0)
    
    tot_dict = {col: val for col, val in zip(param_cols, tot_means)}
    first_dict = {col: val for col, val in zip(param_cols, first_means)}
    
    return tot_dict, first_dict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--sobol_threshold", type=float, default=0.001)
    args = parser.parse_args()

    out_dir = args.distilled_dir
    
    is_split = False
    if os.path.exists(os.path.join(out_dir, "output", "dev_sensitivities")) or os.path.exists(os.path.join(out_dir, "dev_sensitivities")):
        is_split = True
        
    tot_means_all = {}
    first_means_all = {}
    param_types = {}
    
    if is_split:
        dev_tot, dev_first = get_sensitivities(out_dir, "dev_", "dev_sensitivities")
        vol_tot, vol_first = get_sensitivities(out_dir, "vol_", "vol_sensitivities")
        
        for k, v in dev_tot.items():
            tot_means_all[k] = v
            first_means_all[k] = dev_first.get(k, 0.0)
            param_types[k] = "dev"
            
        for k, v in vol_tot.items():
            tot_means_all[k] = v
            first_means_all[k] = vol_first.get(k, 0.0)
            param_types[k] = "vol"
    else:
        tot, first = get_sensitivities(out_dir, "", "sensitivities")
        for k, v in tot.items():
            tot_means_all[k] = v
            first_means_all[k] = first.get(k, 0.0)
            param_types[k] = "all"
            
    if not tot_means_all:
        print("No sensitivities found.")
        return
        
    sorted_params = sorted(tot_means_all.keys(), key=lambda k: tot_means_all[k], reverse=True)
    
    sorted_tot_means = np.array([tot_means_all[k] for k in sorted_params])
    sorted_first_means = np.array([first_means_all[k] for k in sorted_params])
    
    denominator = max(np.sum(sorted_first_means), np.sum(sorted_tot_means))
    if denominator == 0:
        denominator = 1.0
        
    est_coverage_frac = np.cumsum(sorted_tot_means) / denominator
    est_coverage_pct = est_coverage_frac * 100.0
    
    x = np.arange(len(sorted_params))
    width = 0.5
    
    fig, ax1 = plt.subplots(figsize=(12, 7))
    
    true_model_name = "isihara"
    parts = os.path.basename(os.path.normpath(args.distilled_dir)).split('_')
    for p in ["isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4", "c20d10d05", "c20_d10_d05", "aniso30", "aniso_30"]:
        if p in parts:
            true_model_name = p
            break
            
    true_params = set()
    if true_model_name == "isihara":
        true_params = {"C10", "C01", "C20", "D1"}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10", "D1"}
    elif true_model_name in ["nh4", "neohookean4"]:
        true_params = {"C10", "D2"}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10", "E", "CL2", "D1"}
    elif true_model_name in ["c20d10d05", "c20_d10_d05"]:
        true_params = {"C10", "D1", "D2"}
    elif true_model_name in ["aniso30", "aniso_30"]:
        true_params = {"C10", "D1", "K1", "K2", "aniso_theta"}
        
    added_gt_legend = False
    
    for i, p in enumerate(sorted_params):
        clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
        
        if clean_p in true_params:
            if not added_gt_legend:
                ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.15, zorder=0, label="Ground Truth Parameter")
                added_gt_legend = True
            else:
                ax1.axvspan(i - 0.5, i + 0.5, color='gray', alpha=0.15, zorder=0)
                
        ptype = param_types[p]
        if ptype == "dev":
            color = "#0072B2" # Okabe-Ito Blue (colorblind friendly)
            label = "Mean Total-Order (Dev)"
        elif ptype == "vol":
            color = "#D55E00" # Okabe-Ito Vermilion/Red-Orange (colorblind friendly)
            label = "Mean Total-Order (Vol)"
        else:
            color = "#0072B2"
            label = "Mean Total-Order"
            
        handles, labels = ax1.get_legend_handles_labels()
        if label not in labels:
            ax1.bar(x[i], sorted_tot_means[i], width, color=color, alpha=0.9, label=label)
        else:
            ax1.bar(x[i], sorted_tot_means[i], width, color=color, alpha=0.9)
            
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1))
    ax1.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")
    
    ax1.set_ylabel('Sobol Sensitivity Index (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_params, fontsize=11, fontweight='bold')
    
    ax2 = ax1.twinx()
    ax2.plot(x, est_coverage_pct, color='black', marker='o', linestyle='-', linewidth=2, markersize=6, label="Estimated Coverage (EC)")
    ax2.set_ylabel('Estimated Coverage (%)', color='black', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 95, 100])
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.axhline(100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(95, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=11, loc='center right', bbox_to_anchor=(1.0, 0.6))
    
    plt.tight_layout()
    plot_name = "combined_sobol_sensitivity_indices.pdf" if is_split else "sobol_sensitivity_indices.pdf"
    sobol_plot_path = os.path.join(out_dir, plot_name)
    fig.savefig(sobol_plot_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Sobol sensitivity plot to {sobol_plot_path}")

if __name__ == "__main__":
    main()

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--sobol_threshold", type=float, default=0.001)
    args = parser.parse_args()

    out_dir = args.distilled_dir
    csv_dir = os.path.join(out_dir, "output", "sensitivities")
    if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
        csv_dir = os.path.join(out_dir, "sensitivities")
        
    stats_path = os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")
    raw_path = os.path.join(csv_dir, "total_sobol_indices_output_0.csv")
    
    if not os.path.exists(stats_path):
        print(f"Error: {stats_path} not found.")
        return
        
    first_stats_path = os.path.join(csv_dir, "first_sobol_indices_statistics_output_0.csv")
    df_tot = pd.read_csv(stats_path).fillna(0.0)
    df_first = pd.read_csv(first_stats_path).fillna(0.0)
    
    param_cols = [c for c in df_tot.columns if c not in ["Unnamed: 0", ""]]
    tot_means = df_tot.iloc[0][param_cols].values.astype(float)
    first_means = df_first.iloc[0][param_cols].values.astype(float)
    
    df_raw = pd.read_csv(raw_path)
    tot_maxs = df_raw[param_cols].max().values.astype(float)
    
    sorted_indices = np.argsort(tot_means)[::-1]
    sorted_param_cols = [param_cols[i] for i in sorted_indices]
    
    sorted_tot_means = tot_means[sorted_indices]
    sorted_tot_maxs = tot_maxs[sorted_indices]
    
    denominator = max(np.sum(first_means), np.sum(tot_means))
    if denominator == 0:
        denominator = 1.0
        
    norm_tot_means = sorted_tot_means / denominator
    est_coverage_frac = np.cumsum(sorted_tot_means) / denominator
    est_coverage_pct = est_coverage_frac * 100.0
    
    x = np.arange(len(sorted_param_cols))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(14, 7))
    
    # 3 bluish-green bars per parameter
    ax1.bar(x - width, sorted_tot_means, width, label="Mean Total-Order", color="#008080", alpha=0.9)  # Teal
    ax1.bar(x, sorted_tot_maxs, width, label="Max Total-Order", color="#20B2AA", alpha=0.9)  # LightSeaGreen
    ax1.bar(x + width, norm_tot_means, width, label="Norm. Total-Order", color="#48D1CC", alpha=0.9)  # MediumTurquoise
    
    ax1.set_yscale('log')
    ax1.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1))
    ax1.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")
    
    ax1.set_ylabel('Sobol Sensitivity Index (Log Scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Material Parameter Sensitivity (Sobol Indices)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(sorted_param_cols, fontsize=11, fontweight='bold')
    # Removed grid
    
    # Black line for EC on right y-axis
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
    sobol_plot_path = os.path.join(out_dir, "sobol_sensitivity_indices.png")
    plt.savefig(sobol_plot_path, dpi=200)
    plt.close()
    print(f"Updated {sobol_plot_path}")

if __name__ == "__main__":
    main()

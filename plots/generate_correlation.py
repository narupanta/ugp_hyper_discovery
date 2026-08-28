import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import gaussian_kde
import os
import argparse
import sys

def main():
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm'
    })
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="gmr_nolog")
    parser.add_argument("--before_sensitivity", action="store_true", help="Load samples generated before sensitivity masking")
    args = parser.parse_args()
    
    out_dir = args.distilled_dir
    
    # Load samples
    flow_path = os.path.join(out_dir, "flow_samples.npy")
    mcmc_path = os.path.join(out_dir, "mcmc_samples_output_0.npy")
    
    suffix_file = "_before_sensitivity.npy" if args.before_sensitivity else ".npy"
    dev_path = os.path.join(out_dir, f"dev_flow_samples{suffix_file}")
    vol_path = os.path.join(out_dir, f"vol_flow_samples{suffix_file}")
    
    if os.path.exists(dev_path) and os.path.exists(vol_path):
        dev = np.load(dev_path)
        vol = np.load(vol_path)
        # Combine dev and vol by randomly pairing or just hstack if same size
        if dev.shape[0] != vol.shape[0]:
            min_len = min(dev.shape[0], vol.shape[0])
            dev = dev[:min_len]
            vol = vol[:min_len]
        samples_np = np.hstack((dev, vol))
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3")
    elif os.path.exists(dev_path):
        dev = np.load(dev_path)
        vol = np.zeros((dev.shape[0], 3))
        samples_np = np.hstack((dev, vol))
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3")
    elif os.path.exists(vol_path):
        vol = np.load(vol_path)
        dev = np.zeros((vol.shape[0], 10))
        samples_np = np.hstack((dev, vol))
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3")
    elif os.path.exists(flow_path):
        samples_np = np.load(flow_path)
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3") if samples_np.shape[1] == 13 else ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
    elif os.path.exists(mcmc_path):
        samples_np = np.load(mcmc_path)
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3") if samples_np.shape[1] == 13 else ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
    else:
        # Fallback to looking for any mcmc_samples_*.npy
        mcmc_files = [f for f in os.listdir(out_dir) if f.startswith("mcmc_samples") and f.endswith(".npy")]
        if not mcmc_files:
            raise FileNotFoundError(f"Could not find flow_samples.npy or mcmc_samples_*.npy in {out_dir}")
        samples_np = np.load(os.path.join(out_dir, mcmc_files[0]))
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3") if samples_np.shape[1] == 13 else ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")

    df = pd.DataFrame(samples_np, columns=full_param_names)
    
    # Load sensitivity CSV to find active parameters
    dfs = []
    # Try combined, or separate dev and vol folders
    search_dirs = [
        os.path.join(out_dir, "output", "sensitivities"),
        os.path.join(out_dir, "sensitivities"),
        os.path.join(out_dir, "output", "dev_sensitivities"),
        os.path.join(out_dir, "dev_sensitivities"),
        os.path.join(out_dir, "output", "vol_sensitivities"),
        os.path.join(out_dir, "vol_sensitivities")
    ]
    
    for s_dir in search_dirs:
        for pfx in ["", "dev_", "vol_"]:
            for i in range(4):
                p = os.path.join(s_dir, f"{pfx}total_sobol_indices_statistics_output_{i}.csv")
                if os.path.exists(p):
                    dfs.append(pd.read_csv(p))
            
    if dfs:
        param_cols = [c for c in dfs[0].columns if c not in ["Unnamed: 0", ""]]
        # Fix param_cols to map $C_{10}$ to C10
        param_cols = [c.replace("$", "").replace("{", "").replace("}", "").replace("_", "") for c in param_cols]
        # In case the columns in CSV are actually using $C_{10}$ etc
        
        means_list = []
        for df_sub in dfs:
            cols = [c for c in df_sub.columns if c not in ["Unnamed: 0", ""]]
            means = df_sub.iloc[0][cols].values.astype(float)
            means_list.append((cols, means))
            
        active_parameter_names = []
        for cols, means in means_list:
            clean_cols = [c.replace("$", "").replace("{", "").replace("}", "").replace("_", "") for c in cols]
            for c, m in zip(clean_cols, means):
                if m >= 1e-4 and c in full_param_names and c not in active_parameter_names:
                    active_parameter_names.append(c)
        if not active_parameter_names:
            active_parameter_names = list(full_param_names)
    else:
        active_parameter_names = list(full_param_names)
        
    df_active = df[active_parameter_names]
    
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.material_models import get_material
    from core.utils import infer_material_model_name
    
    true_model_name = infer_material_model_name(args.distilled_dir)
    true_model = get_material(true_model_name, jit_P=False)
    true_params = {}
    if true_model_name == "isihara":
        true_params = {"C10": true_model.c10, "C01": true_model.c01, "C20": true_model.c20, "D1": true_model.d1}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0]}
    elif true_model_name in ["nh4", "neohookean4"]:
        true_params = {"C10": true_model.dev_params[0], "D2": true_model.vol_params[1]}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10": true_model.dev_params[0], "E": true_model.dev_params[9], "D1": true_model.vol_params[0]}
    elif true_model_name in ["c20d10d05", "c20_d10_d05"]:
        true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0], "D2": true_model.vol_params[1]}
        
    def generate_pairplot(param_names, suffix):
        def to_latex(name):
            if name.startswith("C") and len(name) == 3 and name[1:].isdigit():
                return rf"$C_{{{name[1:]}}}$"
            if name.startswith("D") and len(name) == 2 and name[1:].isdigit():
                return rf"$D_{{{name[1:]}}}$"
            if name == "E":
                return r"$E$"
            return rf"${name}$"

        def get_comp_color(name):
            return "#D55E00" if name.startswith("D") else "#0072B2"

        if len(param_names) > 1:
            print(f"Generating parameter correlation pairplot for {len(param_names)} parameters using matplotlib...")
            df_subset = df[param_names]
            df_sample = df_subset.sample(n=min(1000, len(df_subset)), random_state=42)
            n_params = len(param_names)
            fig, axes = plt.subplots(n_params, n_params, figsize=(n_params*2.5, n_params*2.5))
            for i in range(n_params):
                for j in range(n_params):
                    ax = axes[i, j]
                    col_i = param_names[i]
                    col_j = param_names[j]
                    
                    color_i = get_comp_color(col_i)
                    
                    if i < j:
                        ax.set_visible(False)
                    elif i == j:
                        data = df_sample[col_i].values
                        ax.hist(data, bins=30, color=color_i, alpha=0.7, density=True, edgecolor='white', linewidth=0.5)
                        mean_val = data.mean()
                        true_val = true_params.get(col_i, 0.0)
                        ci_lower = np.percentile(data, 2.5)
                        ci_upper = np.percentile(data, 97.5)
                        
                        ax.axvline(mean_val, color=color_i, linestyle='-', lw=2.5, alpha=1.0)
                        ax.axvline(true_val, color='black', linestyle='--', lw=2.0, alpha=1.0)
                        
                        ax.axvspan(ci_lower, ci_upper, color=color_i, alpha=0.15)
                        
                        ax.text(0.05, 0.95, fr"$\mathtt{{True:}}\ {true_val:.3f}$", transform=ax.transAxes,
                                ha='left', va='top', fontsize=9, color='black',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor='black', lw=1.5))
                                
                        ax.text(0.05, 0.82, fr"$\mathtt{{Mean:}}\ {mean_val:.3f}$", transform=ax.transAxes,
                                ha='left', va='top', fontsize=9, color=color_i,
                                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=color_i, lw=1.5))
                                
                        if i == n_params - 1:
                            ax.set_xlabel(to_latex(col_j), fontsize=12, fontweight='bold')
                        else:
                            ax.set_xticklabels([])
                        ax.set_yticks([])
                        if j == 0 and n_params > 1:
                            ax.set_ylabel(to_latex(col_i), fontsize=12, fontweight='bold')
                    else:
                        ax.scatter(df_sample[col_j], df_sample[col_i], alpha=0.5, s=15, color='gray', edgecolors='none')
                        std_i = df_sample[col_i].std()
                        std_j = df_sample[col_j].std()
                        if std_i == 0 or std_j == 0:
                            corr = 0.0
                        else:
                            corr = df_sample[col_j].corr(df_sample[col_i])
                            if np.isnan(corr): corr = 0.0
                        ax.annotate(f"r = {corr:.3f}", xy=(0.95, 0.95), xycoords='axes fraction', 
                                    ha='right', va='top', fontsize=10, fontweight='bold',
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))
                        if i == n_params - 1:
                            ax.set_xlabel(to_latex(col_j), fontsize=12, fontweight='bold')
                        else:
                            ax.set_xticklabels([])
                        if j == 0:
                            ax.set_ylabel(to_latex(col_i), fontsize=12, fontweight='bold')
                        else:
                            ax.set_yticklabels([])
            plt.tight_layout()
            pairplot_path = os.path.join(out_dir, f"parameter_correlation_{args.material_model}_{suffix}.pdf")
            fig.savefig(pairplot_path, dpi=200, bbox_inches='tight')
            plt.close()
            print(f"Saved parameter correlation pairplot to {pairplot_path}")
        else:
            print(f"Not enough parameters to generate a correlation pairplot for {suffix}.")

    generate_pairplot(list(full_param_names), "all")
    generate_pairplot(active_parameter_names, "active")

    # Generate parameter violin plot
    try:
        print("Generating parameter violin plot...")
        plt.figure(figsize=(max(8, len(full_param_names) * 0.8), 6))
        
        violin_data = []
        violin_positions = []
        labels = []
        
        for i, col in enumerate(full_param_names):
            if col in active_parameter_names:
                violin_data.append(df[col].values)
                violin_positions.append(i)
            labels.append(col)
            
        ax = plt.gca()
        
        def to_latex(name):
            if name.startswith("C") and len(name) == 3 and name[1:].isdigit():
                return rf"$C_{{{name[1:]}}}$"
            if name.startswith("D") and len(name) == 2 and name[1:].isdigit():
                return rf"$D_{{{name[1:]}}}$"
            if name == "E":
                return r"$E$"
            return rf"${name}$"

        def get_comp_color(name):
            return "#D55E00" if name.startswith("D") else "#0072B2"

        for i, col in enumerate(full_param_names):
            if col in active_parameter_names:
                data = df[col].values
                mean_val = np.mean(data)
                ci_lower = np.percentile(data, 2.5)
                ci_upper = np.percentile(data, 97.5)
                true_val = true_params.get(col, 0.0)
                
                color = get_comp_color(col)
                
                # 95% CI interval as a light background bar
                ax.bar(i, ci_upper - ci_lower, bottom=ci_lower, width=0.6, color=color, alpha=0.1, edgecolor='none')
                
                # Mini bar histogram for distribution shape
                counts, bin_edges = np.histogram(data, bins=30, density=True)
                if np.max(counts) > 0:
                    counts = counts / np.max(counts) * 0.4
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                b_height = bin_edges[1] - bin_edges[0]
                ax.barh(bin_centers, counts, height=b_height, left=i - counts/2, color=color, alpha=0.5, edgecolor='none')
                
                # Mean line
                ax.plot([i - 0.35, i + 0.35], [mean_val, mean_val], color=color, lw=3)
                
                # True value line
                ax.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=2, linestyle='--')
                
                # Text boxes for True and Mean stacked at the top of the plot
                ax.text(i, 2.45, fr"$\mathtt{{True:}}\ {true_val:.3f}$", 
                        ha='center', va='top', fontsize=7, color='black',
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor='black', lw=1.0), zorder=10)
                
                ax.text(i, 2.33, fr"$\mathtt{{Mean:}}\ {mean_val:.3f}$", 
                        ha='center', va='top', fontsize=7, color=color,
                        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, lw=1.0), zorder=10)
            else:
                true_val = true_params.get(col, 0.0)
                ax.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=2, linestyle='--')
                
        ax.set_xticks(range(len(full_param_names)))
        ax.set_xticklabels([to_latex(lbl) for lbl in labels], fontsize=12)
        ax.set_ylim([0, 2.5])
        ax.set_ylabel("Material Parameter Value", fontsize=12)
        
        handles, lbls = ax.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        violin_path = os.path.join(out_dir, f"parameter_violin_{args.material_model}.pdf")
        plt.savefig(violin_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter violin plot to {violin_path}")
    except Exception as e:
        print(f"Error generating parameter violin plot: {e}")

if __name__ == "__main__":
    main()

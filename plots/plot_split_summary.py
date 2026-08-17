import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor

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

def generate_standard_modes(num_points=100, max_gamma=2.0):
    gamma = jnp.linspace(0.0, max_gamma, num_points)
    
    F_all = jnp.zeros((6, num_points, 3, 3))
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(1 + gamma, 1.0, 1.0))            
    F_all = F_all.at[1].set(set_F(1 + gamma, 1 + gamma, 1.0))    
    F_all = F_all.at[2].set(set_F(1 + gamma, 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[3].set(set_F(1/(1 + gamma), 1.0, 1.0))       
    F_all = F_all.at[4].set(set_F(1/(1 + gamma), 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))      
    return F_all, gamma

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
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'text.usetex': False
    })
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default=None)
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="isihara")
    parser.add_argument("--distill_target", type=str, default="sef_split")
    parser.add_argument("--sobol_threshold", type=float, default=0.0001)
    args = parser.parse_args()
    
    distilled_dir = args.distilled_dir
    saved_model_dir = args.saved_model_dir
    
    if saved_model_dir is None:
        for fname in ["source_extraction_dir.txt", "dev_source_extraction_dir.txt"]:
            fpath = os.path.join(distilled_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    saved_model_dir = f.read().strip()
                break
        if saved_model_dir is None:
            raise ValueError("saved_model_dir not found.")

    model_folder_name = os.path.basename(os.path.normpath(saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = parts[1] if len(parts) > 1 else "isihara"
    true_model = get_material(true_model_name, jit_P=False)
    
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev, max_dev = jnp.min(dev_z, axis=0), jnp.max(dev_z, axis=0)
    min_vol, max_vol = jnp.min(vol_z, axis=0), jnp.max(vol_z, axis=0)
    
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    # Generate Data
    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=1.0)
    mode_names = ["UT", "ET", "PS", "UC", "EC", "SS"]

    psi_true = jax.vmap(true_model.psi)(F_all)
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    # Distilled Samples
    num_samples = 32
    if args.distill_target == "sef_split":
        dev_samples = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))[:num_samples]
        vol_samples = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))[:num_samples]
        
        def get_distilled_energy_stress_split(theta_dev, theta_vol, F_chunk):
            dev_theta = list(theta_dev) + [0.0, 0.0, 0.0]
            vol_theta = [0.0]*9 + list(theta_vol)
            mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
            mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
            return jax.vmap(mat_dev.psi)(F_chunk) + jax.vmap(mat_vol.psi)(F_chunk)
            
        dist_psi_samples = []
        for mode in range(len(mode_names)):
            s_psi = jax.vmap(lambda td, tv: get_distilled_energy_stress_split(td, tv, F_all[mode]))(dev_samples, vol_samples)
            dist_psi_samples.append(s_psi)

    # Sensitivity
    tot_means_all = {}
    first_means_all = {}
    param_types = {}
    
    is_split = os.path.exists(os.path.join(distilled_dir, "output", "dev_sensitivities")) or os.path.exists(os.path.join(distilled_dir, "dev_sensitivities"))
    if is_split:
        dev_tot, dev_first = get_sensitivities(distilled_dir, "dev_", "dev_sensitivities")
        vol_tot, vol_first = get_sensitivities(distilled_dir, "vol_", "vol_sensitivities")
        for k, v in dev_tot.items():
            tot_means_all[k], first_means_all[k], param_types[k] = v, dev_first.get(k, 0.0), "dev"
        for k, v in vol_tot.items():
            tot_means_all[k], first_means_all[k], param_types[k] = v, vol_first.get(k, 0.0), "vol"
    
    sorted_params = sorted(tot_means_all.keys(), key=lambda k: tot_means_all[k], reverse=True)
    sorted_tot_means = np.array([tot_means_all[k] for k in sorted_params])
    sorted_first_means = np.array([first_means_all[k] for k in sorted_params])
    
    denominator = max(np.sum(sorted_first_means), np.sum(sorted_tot_means), 1.0)
    est_coverage_pct = (np.cumsum(sorted_tot_means) / denominator) * 100.0

    # Violin data
    if is_split:
        dev = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        vol = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))
        min_len = min(dev.shape[0], vol.shape[0])
        samples_np = np.hstack((dev[:min_len], vol[:min_len]))
        full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E", "D1", "D2", "D3")
    df = pd.DataFrame(samples_np, columns=full_param_names)

    # True params dirty extract
    true_val_dict = {}
    if true_model_name in ["c20d10d05", "c20_d10_d05"]:
        true_params_set = {"C10", "D1", "D2"}
        true_val_dict = {"C10": 2.0, "D1": 1.0, "D2": 0.5}
    elif true_model_name in ["nh2", "neohookean"]:
        true_params_set = {"C10", "D1"}
        true_val_dict = {"C10": 0.5, "D1": 1.5}
    elif true_model_name == "isihara":
        true_params_set = {"C10", "C01", "C20", "D1"}
        true_val_dict = {"C10": 0.5, "C01": 1.0, "C20": 1.0, "D1": 1.5}
    elif true_model_name == "gentthomas":
        true_params_set = {"C10", "E", "D1"}
        true_val_dict = {"C10": 0.5, "E": 1.0, "D1": 1.5}
    else:
        true_params_set = {"C10", "D1"}

    # Figure dimensions
    fig_width = 8.27
    h_energy = 8.183 * (0.23 / 0.75)
    h_sens = 8.183 * (0.25 / 0.75)
    h_viol = 8.183
    h_params = 4.0  # Increased vertical height slightly to fix EC label cutoff
    
    # 1. Energy Plots
    h_energy_split = 4.5
    fig_width_energy = 8.5
    fig_energy = plt.figure(figsize=(fig_width_energy, h_energy_split))
    gs_top = fig_energy.add_gridspec(2, 3, wspace=1.1, hspace=0.05)
    dist_color = "#009E73"
    for i, name in enumerate(mode_names):
        row, col = i // 3, i % 3
        ax_psi = fig_energy.add_subplot(gs_top[row, col])
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=1.5, label="Ground Truth", zorder=5)
        gp_psi_lower = psi_dist_mean[i] - 1.96 * jnp.sqrt(psi_dist_var[i])
        gp_psi_upper = psi_dist_mean[i] + 1.96 * jnp.sqrt(psi_dist_var[i])
        ax_psi.fill_between(gamma, gp_psi_lower, gp_psi_upper, color='gray', alpha=0.3, label="GP 95% CI")
        ax_psi.plot(gamma, psi_dist_mean[i], color='gray', lw=1.5, ls='-', label="GP mean", zorder=4)
        
        gp_cov_psi = jnp.mean((psi_true[i] >= gp_psi_lower) & (psi_true[i] <= gp_psi_upper))
        rmse_psi_gp = jnp.sqrt(jnp.mean((psi_dist_mean[i] - psi_true[i]) ** 2))
        ss_tot_psi = jnp.sum((psi_true[i] - jnp.mean(psi_true[i])) ** 2)
        r2_psi_gp = 1 - jnp.sum((psi_true[i] - psi_dist_mean[i]) ** 2) / (ss_tot_psi + 1e-12)
        
        nf_psi_lower = jnp.percentile(dist_psi_samples[i], 2.5, axis=0)
        nf_psi_upper = jnp.percentile(dist_psi_samples[i], 97.5, axis=0)
        dist_psi_mean = dist_psi_samples[i].mean(axis=0)
        
        ax_psi.fill_between(gamma, nf_psi_lower, nf_psi_upper, color=dist_color, alpha=0.15, label="Distilled 95%CI", zorder=2)
        ax_psi.plot(gamma, dist_psi_samples[i].T, color=dist_color, lw=0.6, alpha=0.1, zorder=1)
        ax_psi.plot(gamma, dist_psi_mean, color=dist_color, lw=2.0, label="Distilled Mean", zorder=3)
        
        nf_cov_psi = jnp.mean((psi_true[i] >= nf_psi_lower) & (psi_true[i] <= nf_psi_upper))
        rmse_psi = jnp.sqrt(jnp.mean((dist_psi_mean - psi_true[i]) ** 2))
        r2_psi = 1 - jnp.sum((psi_true[i] - dist_psi_mean) ** 2) / (ss_tot_psi + 1e-12)
        
        if col == 0:
            ax_psi.set_ylabel(r"$\Psi_\mathrm{total}$", fontsize=8)
        if row == 1:
            ax_psi.set_xlabel(r"$\gamma$", fontsize=8)
        ax_psi.set_title(name, fontsize=9, pad=2)
        ax_psi.set_box_aspect(1)
        ax_psi.grid(False)
        ax_psi.tick_params(axis='both', which='major', labelsize=7)
            
        annotation_gp_psi = f"GP\nEC: {gp_cov_psi:.0%}\nRMSE: {rmse_psi_gp:.3f}\n$R^2$: {r2_psi_gp:.3f}"
        annotation_dist_psi = f"Distilled\nEC: {nf_cov_psi:.0%}\nRMSE: {rmse_psi:.3f}\n$R^2$: {r2_psi:.3f}"
        
        ax_psi.annotate(annotation_gp_psi, xy=(1.05, 0.75), xycoords='axes fraction', 
                        ha='left', va='center', fontsize=7, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=0.5, alpha=0.8), zorder=6)
                        
        ax_psi.annotate(annotation_dist_psi, xy=(1.05, 0.25), xycoords='axes fraction', 
                        ha='left', va='center', fontsize=7, clip_on=False,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=dist_color, lw=0.5, alpha=0.8), zorder=6)

    # Global legend for energy plots
    handles, labels = ax_psi.get_legend_handles_labels()
    fig_energy.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01), fontsize=8, frameon=False)
    
    fig_energy.savefig(os.path.join(distilled_dir, f"split_energy_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_energy)

    # 2. Parameters Figure (Sensitivity + Violin)
    fig_params = plt.figure(figsize=(fig_width, h_params))
    gs_params = fig_params.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.4)
    
    ax_sens = fig_params.add_subplot(gs_params[0, 0])
    x_pos = np.arange(len(sorted_params))
    
    gt_label_added = False
    for i, p in enumerate(sorted_params):
        clean_p = p.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
        if clean_p in true_params_set:
            label_gt = "Ground Truth Parameter" if not gt_label_added else ""
            ax_sens.axvspan(i - 0.25, i + 0.25, color='#E0E0E0', alpha=0.8, zorder=1, edgecolor='none', label=label_gt)
            gt_label_added = True
            
        color = "#0072B2" if param_types[p] == "dev" else "#D55E00"
        label = "Mean Total-Order (Dev)" if param_types[p] == "dev" else "Mean Total-Order (Vol)"
        handles, labels = ax_sens.get_legend_handles_labels()
        if label not in labels:
            ax_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3, label=label)
        else:
            ax_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3)
        
    ax_sens.set_yscale('log')
    ax_sens.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1))
    ax_sens.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.5, label=f"Threshold ({args.sobol_threshold})")
    ax_sens.set_ylabel('Sobol Sensitivity', fontsize=8)
    
    ax_sens.set_xticks(x_pos)
    ax_sens.set_xticklabels([]) # Hide for sensitivity since violin shares it
    ax_sens.tick_params(axis='y', labelsize=7)
    
    # Add estimated coverage secondary axis
    ax2 = ax_sens.twinx()
    ax2.plot(x_pos, est_coverage_pct, color='black', marker='o', linestyle='-', linewidth=1.5, markersize=4, label="Estimated Coverage (EC)")
    ax2.set_ylabel('Estimated Coverage (%)', color='black', fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.set_yticks([0, 20, 40, 60, 80, 100])
    ax2.tick_params(axis='y', labelcolor='black', labelsize=7)
    ax2.axhline(100, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    ax2.axhline(95, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    
    lines_1, labels_1 = ax_sens.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax_sens.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=6, loc='center right', bbox_to_anchor=(1.0, 0.6))
    
    plt.setp(ax_sens.get_xticklabels(), visible=False)

    # 3. Violin Plot (Row 3)
    ax_viol = fig_params.add_subplot(gs_params[1, 0], sharex=ax_sens)
    
    for i, p in enumerate(sorted_params):
        clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
        is_active = sorted_tot_means[i] > args.sobol_threshold
        is_true = clean_p in true_params_set
        
        data = df[clean_p].values
        mean_val = np.mean(data)
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        true_val = true_val_dict.get(clean_p, 0.0)
        
        color = get_comp_color(clean_p)
        
        if is_active:
            # 95% CI interval as a light background bar
            ax_viol.bar(i, ci_upper - ci_lower, bottom=ci_lower, width=0.6, color=color, alpha=0.1, edgecolor='none')
            
            # Mini bar histogram
            counts, bin_edges = np.histogram(data, bins=30, density=True)
            if np.max(counts) > 0:
                counts = counts / np.max(counts) * 0.4
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            b_height = bin_edges[1] - bin_edges[0]
            ax_viol.barh(bin_centers, counts, height=b_height, left=i - counts/2, color=color, alpha=0.5, edgecolor='none')
            
            ax_viol.plot([i - 0.35, i + 0.35], [mean_val, mean_val], color=color, lw=2)
            
            ax_viol.text(i, 2.60, fr"${mean_val:.3f}$", 
                    ha='center', va='bottom', fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, lw=0.5), zorder=10, clip_on=False)
        
        if is_true:
            ax_viol.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=1.5, linestyle='--')
            
            ax_viol.text(i, 2.95, fr"${true_val:.3f}$", 
                    ha='center', va='bottom', fontsize=7, color='black',
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor='black', lw=0.5), zorder=10, clip_on=False)
                
    ax_viol.set_xticks(range(len(sorted_params)))
    ax_viol.set_xticklabels(sorted_params, fontsize=9)
    ax_viol.set_ylabel('Parameter Value', fontsize=8)
    ax_viol.set_ylim([0, 2.5])
    ax_viol.tick_params(axis='y', labelsize=7)
    ax_viol.grid(False)
    
    # Row labels for the text boxes
    ax_viol.text(-1.2, 2.60, "Mean", ha='left', va='bottom', fontsize=8, color='black', clip_on=False)
    ax_viol.text(-1.2, 2.95, "True", ha='left', va='bottom', fontsize=8, color='black', clip_on=False)

    # Legend for the parameter plot
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    viol_legend = [
        mpatches.Patch(color='gray', alpha=0.5, label='Density'),
        mlines.Line2D([0], [0], color='gray', lw=2, label='Mean'),
        mpatches.Patch(color='gray', alpha=0.1, label='95% CI'),
        mlines.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Ground Truth')
    ]
    ax_viol.legend(handles=viol_legend, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, fontsize=7, frameon=False)
    
    fig_params.savefig(os.path.join(distilled_dir, f"split_params_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_params)
    
    print(f"Saved split summary plots to {distilled_dir}")

if __name__ == "__main__":
    main()

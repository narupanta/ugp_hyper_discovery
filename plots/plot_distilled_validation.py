import argparse
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams.update({
    "mathtext.fontset": "cm",
    "font.family": "serif",
    "axes.formatter.use_mathtext": True
})
import numpy as np
import os

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default=None)
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default="isihara", choices=["ogden", "gmr", "gmr_log", "gmr_nolog", "isihara"])
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    args = parser.parse_args()
    
    distilled_dir = args.distilled_dir
    saved_model_dir = args.saved_model_dir
    
    if saved_model_dir is None:
        source_file = os.path.join(distilled_dir, "source_extraction_dir.txt")
        dev_source_file = os.path.join(distilled_dir, "dev_source_extraction_dir.txt")
        if os.path.exists(source_file):
            with open(source_file, "r") as f:
                saved_model_dir = f.read().strip()
        elif os.path.exists(dev_source_file):
            with open(dev_source_file, "r") as f:
                saved_model_dir = f.read().strip()
        else:
            raise ValueError(f"saved_model_dir must be provided if {source_file} does not exist.")
    
    # 1. Load True Model dynamically from saved_model_dir folder name
    model_folder_name = os.path.basename(os.path.normpath(saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = parts[1] if len(parts) > 1 else "isihara"
    true_model = get_material(true_model_name, jit_P=False)
    
    # 2. Load GP Model
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    # 3. Load Distilled Samples
    if args.distill_target == "sef_split":
        # For sef_split, we load dev and vol samples separately. We'll get num_samples from dev_samples.
        tmp_samples = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        num_samples = min(32, tmp_samples.shape[0])
        selected_samples = None # We will load them fully later
    else:
        samples = np.load(os.path.join(distilled_dir, "flow_samples.npy"))
        num_samples = min(32, samples.shape[0])
        selected_samples = samples[:num_samples]
    
    # 4. Generate Data
    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=1.0)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # Calculate exact interpolation / extrapolation transitions for each mode
    extractor = IsotropicFeatureExtractor()
    true_min_dev = min_dev - 1e-4
    true_max_dev = max_dev + 1e-4
    true_min_vol = min_vol - 1e-4
    true_max_vol = max_vol + 1e-4
    
    transitions = []
    for mode in range(len(mode_names)):
        dev_m, vol_m = jax.vmap(extractor.extract)(F_all[mode])
        in_bounds = ((dev_m[:, 0] >= true_min_dev[0]) & (dev_m[:, 0] <= true_max_dev[0]) &
                     (dev_m[:, 1] >= true_min_dev[1]) & (dev_m[:, 1] <= true_max_dev[1]) &
                     (vol_m[:, 0] >= true_min_vol[0]) & (vol_m[:, 0] <= true_max_vol[0]))
        if not jnp.all(in_bounds):
            idx = int(jnp.argmax(~in_bounds))
            trans_g = float(gamma[idx])
            if idx == 0:
                trans_g = float(gamma[1])
        else:
            trans_g = float(gamma.max())
        transitions.append(trans_g)

    # 5. Evaluate True and GP
    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    P_dist_mean = [learned_gp.piola_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    P_dist_var = [learned_gp.piola_dist(F_all[mode]).var for mode in range(len(mode_names))]
    
    dev_psi_dist_mean = [learned_gp.dev_psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    dev_psi_dist_var = [learned_gp.dev_psi_dist(F_all[mode]).var for mode in range(len(mode_names))]
    vol_psi_dist_mean = [learned_gp.vol_psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    vol_psi_dist_var = [learned_gp.vol_psi_dist(F_all[mode]).var for mode in range(len(mode_names))]
    
    if args.distill_target == "sef_split":
        # Load samples for both components
        dev_samples = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))[:num_samples]
        vol_samples = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))[:num_samples]
        
        def get_distilled_energy_stress_split(theta_dev, theta_vol, F_chunk):
            dev_theta = list(theta_dev) + [0.0, 0.0, 0.0]
            vol_theta = [0.0]*9 + list(theta_vol)
            mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
            mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
            return jax.vmap(mat_dev.psi)(F_chunk), jax.vmap(mat_vol.psi)(F_chunk), jax.vmap(mat_dev.P)(F_chunk) + jax.vmap(mat_vol.P)(F_chunk)
            
        dist_psi_dev_samples, dist_psi_vol_samples, dist_psi_samples, dist_p_samples = [], [], [], []
        for mode in range(len(mode_names)):
            mode_F = F_all[mode]
            s_psi_dev, s_psi_vol, s_p = jax.vmap(lambda td, tv: get_distilled_energy_stress_split(td, tv, mode_F))(dev_samples, vol_samples)
            dist_psi_dev_samples.append(s_psi_dev)
            dist_psi_vol_samples.append(s_psi_vol)
            dist_psi_samples.append(s_psi_dev + s_psi_vol)
            dist_p_samples.append(s_p)
    else:
        def get_distilled_energy_stress(theta, F_chunk):
            if args.material_model == "ogden":
                mu = theta[:3]
                alpha = theta[3:6]
                vol = theta[6:9]
                mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
            elif args.material_model in ["gmr", "gmr_log", "gmr_nolog"]:
                if len(theta) == 13: # 9 dev + 1 log + 3 vol
                    dev = theta[:10]
                    vol = theta[10:13]
                elif len(theta) >= 14:
                    dev = theta[:11]
                    vol = theta[11:14]
                else: # 9 dev + 3 vol (no log term)
                    dev = theta[:9]
                    vol = theta[9:12]
                mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
            elif args.material_model == "isihara":
                mat = get_material("isihara", c10=theta[0], c01=theta[1], c20=theta[2], d1=theta[3], jit_P=False)
            return jax.vmap(mat.psi)(F_chunk), jax.vmap(mat.P)(F_chunk)
            
        dist_psi_samples = []
        dist_p_samples = []
        for mode in range(len(mode_names)):
            mode_F = F_all[mode]
            s_psi, s_p = jax.vmap(lambda t: get_distilled_energy_stress(t, mode_F))(selected_samples)
            dist_psi_samples.append(s_psi)
            dist_p_samples.append(s_p)
        
    # 7. Plotting Setup
    fig_width = 8
    fig_height = 12
    
    def generate_energy_plot(samples_list, true_psi_list, title_prefix, save_name, gp_mean_list=None, gp_var_list=None, ylabel=r"Strain Energy ($\Psi$)"):
        fig_psi, axes_psi = plt.subplots(3, 2, figsize=(fig_width, fig_height), sharex=True)
        
        for i, name in enumerate(mode_names):
            row, col = i // 2, i % 2
            ax_psi = axes_psi[row, col]
            
            # 1. Plot ground truth
            ax_psi.plot(gamma, true_psi_list[i], 'k--', lw=1.5, label="Ground Truth", zorder=5)
            
            # 2. Plot GP posterior if provided
            if gp_mean_list is not None and gp_var_list is not None:
                gp_psi_lower = gp_mean_list[i] - 1.96 * jnp.sqrt(gp_var_list[i])
                gp_psi_upper = gp_mean_list[i] + 1.96 * jnp.sqrt(gp_var_list[i])
                gp_cov_psi = jnp.mean((true_psi_list[i] >= gp_psi_lower) & (true_psi_list[i] <= gp_psi_upper))
                
                rmse_psi_gp = jnp.sqrt(jnp.mean((gp_mean_list[i] - true_psi_list[i]) ** 2))
                ss_tot_psi = jnp.sum((true_psi_list[i] - jnp.mean(true_psi_list[i])) ** 2)
                r2_psi_gp = 1 - jnp.sum((true_psi_list[i] - gp_mean_list[i]) ** 2) / (ss_tot_psi + 1e-12)
                
                ax_psi.fill_between(gamma, gp_psi_lower, gp_psi_upper, color='gray', alpha=0.3, label="GP Posterior (95% CI)")
                ax_psi.plot(gamma, gp_mean_list[i], color='gray', lw=1.5, ls='-', label="GP Mean", zorder=4)
            else:
                gp_cov_psi = 0.0
                rmse_psi_gp = 0.0
                r2_psi_gp = 0.0
            
            # Since GP metrics aren't separated in plotting, we just plot true and distilled samples for dev/vol
            nf_psi_lower = jnp.percentile(samples_list[i], 2.5, axis=0)
            nf_psi_upper = jnp.percentile(samples_list[i], 97.5, axis=0)
            nf_cov_psi = jnp.mean((true_psi_list[i] >= nf_psi_lower) & (true_psi_list[i] <= nf_psi_upper))
            
            dist_psi_mean = samples_list[i].mean(axis=0)
            rmse_psi = jnp.sqrt(jnp.mean((dist_psi_mean - true_psi_list[i]) ** 2))
            ss_tot_psi = jnp.sum((true_psi_list[i] - jnp.mean(true_psi_list[i])) ** 2)
            r2_psi = 1 - jnp.sum((true_psi_list[i] - dist_psi_mean) ** 2) / (ss_tot_psi + 1e-12)
            
            ax_psi.plot(gamma, samples_list[i].T, color="blue", lw=0.6, alpha=0.35, zorder=2)
            ax_psi.plot([], [], color="blue", lw=2.0, label=f"Distilled Samples")
            
            annotation_dist_psi = (
                "Distilled Samples\n" +
                fr"$\mathrm{{EC}}_{{95\%}}$: {nf_cov_psi:.1%}" + "\n" + 
                f"RMSE: {rmse_psi:.4f}\n" +
                fr"$R^2$: {r2_psi:.4f}"
            )
            
            if gp_mean_list is not None and gp_var_list is not None:
                annotation_gp_psi = (
                    "GP Extraction\n" +
                    fr"$\mathrm{{EC}}_{{95\%}}$: {gp_cov_psi:.1%}" + "\n" + 
                    f"RMSE: {rmse_psi_gp:.4f}\n" +
                    fr"$R^2$: {r2_psi_gp:.4f}"
                )
                ax_psi.annotate(annotation_gp_psi, xy=(0.02, 0.95), xycoords='axes fraction', 
                                ha='left', va='top', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1.5, alpha=0.8), zorder=6)
                
                ax_psi.annotate(annotation_dist_psi, xy=(0.42, 0.95), xycoords='axes fraction', 
                                ha='left', va='top', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=1.5, alpha=0.8), zorder=6)
            else:
                ax_psi.annotate(annotation_dist_psi, xy=(0.02, 0.95), xycoords='axes fraction', 
                                ha='left', va='top', fontsize=9, fontweight='bold',
                                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=1.5, alpha=0.8), zorder=6)

            y_min_psi, y_max_psi = jnp.min(true_psi_list[i]), jnp.max(true_psi_list[i])
            pad_psi = (y_max_psi - y_min_psi) * 0.1 if y_max_psi != y_min_psi else 1.0
            ax_psi.set_ylim(y_min_psi - pad_psi, y_max_psi + pad_psi)
            ax_psi.set_xlim(0, gamma.max())
            ax_psi.set_title(f"($m={i+1}$) {name}", fontsize=11)
            if col == 0:
                ax_psi.set_ylabel(ylabel, fontsize=11)
            if row == 2:
                ax_psi.set_xlabel(r"Stretch Measure ($\gamma$)", fontsize=11)
            else:
                ax_psi.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)
                
            ax_psi.grid(False)
            ax_psi.set_box_aspect(1)
            ax_psi.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax_psi.tick_params(axis='both', labelsize=10)

        handles_psi, labels_psi = axes_psi[0, 0].get_legend_handles_labels()
        fig_psi.legend(handles_psi, labels_psi, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10, framealpha=1.0)
        fig_psi.tight_layout(rect=[0, 0.05, 1, 1])
        save_file = os.path.join(distilled_dir, save_name)
        fig_psi.savefig(save_file, bbox_inches='tight', dpi=200)
        plt.close(fig_psi)
        print(f"Validation {title_prefix} plot saved to: {save_file}")

    if args.distill_target == "sef_split":
        psi_true_dev = []
        psi_true_vol = []
        for F_mode in F_all:
            if hasattr(true_model, 'psi_dev') and hasattr(true_model, 'psi_vol'):
                psi_true_dev.append(jax.vmap(true_model.psi_dev)(F_mode))
                psi_true_vol.append(jax.vmap(true_model.psi_vol)(F_mode))
            elif hasattr(true_model, 'dev_params') and hasattr(true_model, 'vol_params'):
                td = list(true_model.dev_params)
                tv = list(true_model.vol_params)
                t_dev = get_material(true_model_name, dev_params=td, vol_params=[0,0,0], jit_P=False)
                t_vol = get_material(true_model_name, dev_params=[0]*9, vol_params=tv, jit_P=False)
                psi_true_dev.append(jax.vmap(t_dev.psi)(F_mode))
                psi_true_vol.append(jax.vmap(t_vol.psi)(F_mode))
            else:
                # Fallback to total energy if model cannot be split
                psi_true_dev.append(jax.vmap(true_model.psi)(F_mode))
                psi_true_vol.append(jax.vmap(true_model.psi)(F_mode))
        generate_energy_plot(dist_psi_dev_samples, psi_true_dev, "DEV Energy", f"distilled_validation_energy_dev_{args.material_model}.pdf", dev_psi_dist_mean, dev_psi_dist_var, ylabel=r"Deviatoric SEF ($\Psi_{\mathrm{dev}}$)")
        generate_energy_plot(dist_psi_vol_samples, psi_true_vol, "VOL Energy", f"distilled_validation_energy_vol_{args.material_model}.pdf", vol_psi_dist_mean, vol_psi_dist_var, ylabel=r"Volumetric SEF ($\Psi_{\mathrm{vol}}$)")
        generate_energy_plot(dist_psi_samples, psi_true, "TOTAL Energy", f"distilled_validation_energy_total_{args.material_model}.pdf", psi_dist_mean, psi_dist_var)
    
    fig_psi, axes_psi = plt.subplots(3, 2, figsize=(fig_width, fig_height), sharex=True)
    fig_p, axes_p = plt.subplots(3, 2, figsize=(fig_width, fig_height), sharex=True)

    for i, name in enumerate(mode_names):
        row = i // 2
        col = i % 2
        
        ax_psi = axes_psi[row, col]
        ax_p = axes_p[row, col]
        
        if name == "Pure Shear":
            idx_comp = (1, 1); label_P = r"$P_{22}$"
        elif name == "Simple Shear":
            idx_comp = (0, 1); label_P = r"$P_{12}$"
        else:
            idx_comp = (0, 0); label_P = r"$P_{11}$"

        # --- ENERGY PLOT (ax_psi) ---
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=1.5, label="Ground Truth", zorder=5)
        
        gp_psi_lower = psi_dist_mean[i] - 1.96 * jnp.sqrt(psi_dist_var[i])
        gp_psi_upper = psi_dist_mean[i] + 1.96 * jnp.sqrt(psi_dist_var[i])
        gp_cov_psi = jnp.mean((psi_true[i] >= gp_psi_lower) & (psi_true[i] <= gp_psi_upper))
        
        nf_psi_lower = jnp.percentile(dist_psi_samples[i], 2.5, axis=0)
        nf_psi_upper = jnp.percentile(dist_psi_samples[i], 97.5, axis=0)
        nf_cov_psi = jnp.mean((psi_true[i] >= nf_psi_lower) & (psi_true[i] <= nf_psi_upper))
        
        dist_psi_mean = dist_psi_samples[i].mean(axis=0)
        rmse_psi = jnp.sqrt(jnp.mean((dist_psi_mean - psi_true[i]) ** 2))
        ss_tot_psi = jnp.sum((psi_true[i] - jnp.mean(psi_true[i])) ** 2)
        r2_psi = 1 - jnp.sum((psi_true[i] - dist_psi_mean) ** 2) / (ss_tot_psi + 1e-12)
        
        rmse_psi_gp = jnp.sqrt(jnp.mean((psi_dist_mean[i] - psi_true[i]) ** 2))
        r2_psi_gp = 1 - jnp.sum((psi_true[i] - psi_dist_mean[i]) ** 2) / (ss_tot_psi + 1e-12)
        
        ax_psi.plot(gamma, dist_psi_samples[i].T, color="blue", lw=0.6, alpha=0.35, zorder=2)
        ax_psi.plot([], [], color="blue", lw=2.0, label=f"Distilled Samples ({args.material_model.upper()})")
        ax_psi.plot(gamma, psi_dist_mean[i], color="gray", lw=1.5, label="GP Mean", zorder=4)
        ax_psi.fill_between(gamma, gp_psi_lower, gp_psi_upper, color="gray", alpha=0.3, zorder=3, label="GP Posterior (95% CI)")
        
        annotation_gp_psi = (
            "GP Extraction\n" +
            fr"$\mathrm{{EC}}_{{95\%}}$: {gp_cov_psi:.1%}" + "\n" + 
            f"RMSE: {rmse_psi_gp:.4f}\n" +
            fr"$R^2$: {r2_psi_gp:.4f}"
        )
        annotation_dist_psi = (
            "Distilled Samples\n" +
            fr"$\mathrm{{EC}}_{{95\%}}$: {nf_cov_psi:.1%}" + "\n" + 
            f"RMSE: {rmse_psi:.4f}\n" +
            fr"$R^2$: {r2_psi:.4f}"
        )
        ax_psi.annotate(annotation_gp_psi, xy=(0.02, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1.5, alpha=0.8), zorder=6)
        ax_psi.annotate(annotation_dist_psi, xy=(0.42, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=1.5, alpha=0.8), zorder=6)
        
        y_min, y_max = jnp.min(psi_true[i]), jnp.max(psi_true[i])
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())
        ax_psi.set_title(f"($m={i+1}$) {name}", fontsize=11)
        if col == 0:
            ax_psi.set_ylabel(r"$\Psi$", fontsize=11)

        # --- STRESS PLOT (ax_p) ---
        p_true_comp = P_true[i, :, idx_comp[0], idx_comp[1]]
        p_mean_comp = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(P_dist_var[i][:, idx_comp[0], idx_comp[1]])
        p_samples_comp = dist_p_samples[i][:, :, idx_comp[0], idx_comp[1]]

        ax_p.plot(gamma, p_true_comp, 'k--', lw=1.5, label="Ground Truth", zorder=5)
        
        gp_p_lower = p_mean_comp - 1.96 * p_std_comp
        gp_p_upper = p_mean_comp + 1.96 * p_std_comp
        gp_cov_p = jnp.mean((p_true_comp >= gp_p_lower) & (p_true_comp <= gp_p_upper))
        
        nf_p_lower = jnp.percentile(p_samples_comp, 2.5, axis=0)
        nf_p_upper = jnp.percentile(p_samples_comp, 97.5, axis=0)
        nf_cov_p = jnp.mean((p_true_comp >= nf_p_lower) & (p_true_comp <= nf_p_upper))
        
        dist_p_mean = p_samples_comp.mean(axis=0)
        rmse_p = jnp.sqrt(jnp.mean((dist_p_mean - p_true_comp) ** 2))
        ss_tot_p = jnp.sum((p_true_comp - jnp.mean(p_true_comp)) ** 2)
        r2_p = 1 - jnp.sum((p_true_comp - dist_p_mean) ** 2) / (ss_tot_p + 1e-12)
        
        rmse_p_gp = jnp.sqrt(jnp.mean((p_mean_comp - p_true_comp) ** 2))
        r2_p_gp = 1 - jnp.sum((p_true_comp - p_mean_comp) ** 2) / (ss_tot_p + 1e-12)
        
        ax_p.plot(gamma, p_samples_comp.T, color="blue", lw=0.6, alpha=0.35, zorder=2)
        ax_p.plot([], [], color="blue", lw=2.0, label=f"Distilled Samples ({args.material_model.upper()})")
        ax_p.plot(gamma, p_mean_comp, color="gray", lw=1.5, label="GP Mean", zorder=4)
        ax_p.fill_between(gamma, gp_p_lower, gp_p_upper, color="gray", alpha=0.3, zorder=3, label="GP Posterior (95% CI)")
                         
        annotation_gp_p = (
            "GP Extraction\n" +
            fr"$\mathrm{{EC}}_{{95\%}}$: {gp_cov_p:.1%}" + "\n" + 
            f"RMSE: {rmse_p_gp:.4f}\n" +
            fr"$R^2$: {r2_p_gp:.4f}"
        )
        annotation_dist_p = (
            "Distilled Samples\n" +
            fr"$\mathrm{{EC}}_{{95\%}}$: {nf_cov_p:.1%}" + "\n" + 
            f"RMSE: {rmse_p:.4f}\n" +
            fr"$R^2$: {r2_p:.4f}"
        )
        ax_p.annotate(annotation_gp_p, xy=(0.02, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", lw=1.5, alpha=0.8), zorder=6)
        ax_p.annotate(annotation_dist_p, xy=(0.42, 0.95), xycoords='axes fraction', 
                        ha='left', va='top', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="blue", lw=1.5, alpha=0.8), zorder=6)

        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())
        ax_p.set_title(f"($m={i+1}$) {name}", fontsize=11)
        if col == 0:
            ax_p.set_ylabel(label_P, fontsize=11)
        
        trans_g = transitions[i]
        max_g = float(gamma.max())
        
        # Format both axes
        for ax in [ax_psi, ax_p]:
            if trans_g < max_g:
                ax.axvline(x=trans_g, color='black', linestyle=':', lw=1.5, alpha=0.7, zorder=4, label="Data Boundary" if i == 0 else "")
                ax.axvspan(trans_g, max_g, color='lightgrey', alpha=0.3, zorder=1)
                y_mid = (ax.get_ylim()[0] + ax.get_ylim()[1]) / 2.0
                ax.text(trans_g + 0.02, y_mid, fr"$\gamma_c = {trans_g:.2f}$", color='black', fontsize=9, va='center', ha='left', fontweight='bold', rotation=-90, zorder=7)
                
            ax.grid(False)
            ax.set_box_aspect(1)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
            ax.tick_params(axis='both', labelsize=10)
            
            # X-axis label only on bottom row (row == 2 for a 3x2 grid)
            if row == 2:
                ax.set_xlabel(r"Stretch Measure ($\gamma$)", fontsize=11)
            else:
                ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)

    if args.distill_target != "sef_split":
        handles_psi, labels_psi = axes_psi[0, 0].get_legend_handles_labels()
        fig_psi.legend(handles_psi, labels_psi, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10, framealpha=1.0)
        fig_psi.tight_layout(rect=[0, 0.05, 1, 1])
        save_file_psi = os.path.join(distilled_dir, f"distilled_validation_energy_{args.material_model}.pdf")
        fig_psi.savefig(save_file_psi, bbox_inches='tight', dpi=200)
        plt.close(fig_psi)
        print(f"Validation Energy plot saved to: {save_file_psi}")
    else:
        plt.close(fig_psi) # We already saved 3 custom energy plots

    # Save Stress Figure
    handles_p, labels_p = axes_p[0, 0].get_legend_handles_labels()
    fig_p.legend(handles_p, labels_p, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=10, framealpha=1.0)
    fig_p.tight_layout(rect=[0, 0.05, 1, 1])
    save_file_p = os.path.join(distilled_dir, f"distilled_validation_stress_{args.material_model}.pdf")
    fig_p.savefig(save_file_p, bbox_inches='tight', dpi=200)
    plt.close(fig_p)
    print(f"Validation Stress plot saved to: {save_file_p}")

if __name__ == "__main__":
    main()

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
from core.utils import infer_material_model_name
from scipy.spatial import ConvexHull

def to_latex(name):
    if name.startswith("C") and len(name) == 3 and name[1:].isdigit():
        return rf"$C_{{{name[1:]}}}$"
    if name.startswith("D") and len(name) == 2 and name[1:].isdigit():
        return rf"$D_{{{name[1:]}}}$"
    if name == "E":
        return r"$E$"
    return rf"${name}$"

def get_comp_color(name):
    if name.startswith("D"):
        return "#D55E00"
    elif name.startswith("C4") or name.startswith("C6"):
        return "#CC79A7"
    return "#0072B2"

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
    parser.add_argument("--material_model", type=str, required=True)
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

    true_model_name = infer_material_model_name(saved_model_dir)
    true_model = get_material(true_model_name, jit_P=False)
    
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:3] if I_z.shape[1] > 3 else I_z[:, 2:]
    aniso_z = I_z[:, 3:] if I_z.shape[1] > 3 else None

    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    min_aniso = jnp.min(aniso_z, axis=0) if aniso_z is not None else None
    max_aniso = jnp.max(aniso_z, axis=0) if aniso_z is not None else None

    feature_extractor = None
    if aniso_z is not None:
        from core.features import AnisotropicFeatureExtractor
        if hasattr(true_model, 'a1') and hasattr(true_model, 'a2'):
            a0 = np.array(true_model.a1)
            a1 = np.array(true_model.a2)
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        elif hasattr(true_model, 'a0'):
            a0 = np.array(true_model.a0)
            feature_extractor = AnisotropicFeatureExtractor(a0)
        elif aniso_z.shape[1] == 4:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            a1 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        else:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)
    
    obs_path = os.path.join(saved_model_dir, "I_obs_all.npy")
    if os.path.exists(obs_path):
        I_obs_all = jnp.load(obs_path)
        I_obs_all = I_obs_all.reshape(-1, I_obs_all.shape[-1])
        dev_obs = I_obs_all[:, :2]
        vol_obs = I_obs_all[:, 2:3]
    else:
        dev_obs = dev_z
        vol_obs = vol_z
        I_obs_all = I_z
        
    limit_min_vol, limit_max_vol = jnp.min(vol_obs, axis=0), jnp.max(vol_obs, axis=0)
    dev_hull = ConvexHull(np.array(dev_obs))
    hull_eqs = dev_hull.equations
    dev_tol = 0.001
    vol_tol = 0.001

    feature_extractor = None
    if aniso_z is not None:
        from core.features import AnisotropicFeatureExtractor
        if aniso_z.shape[1] == 4:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            a1 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        else:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)

    import json
    metadata_path = os.path.join(saved_model_dir, "metadata.json")
    cov_mode = "diag"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            cov_mode = json.load(f).get("covariance_mode", "diag")
            
    learned_gp = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode
    )
    
    # Generate Data

    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=1.0)
    mode_names = ["UT", "ET", "PS", "UC", "EC", "SS"]
    
    extractor = feature_extractor if feature_extractor is not None else IsotropicFeatureExtractor()

    psi_true = jax.vmap(true_model.psi)(F_all)
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    # Distilled Samples
    num_samples = 32
    has_aniso = os.path.exists(os.path.join(distilled_dir, "aniso_flow_samples.npy"))
    
    if args.distill_target == "sef_split":
        dev_samples = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))[:num_samples]
        vol_samples = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))[:num_samples]
        aniso_samples = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy"))[:num_samples] if has_aniso else None
        
        def psi_aniso_single(theta_aniso, F_single):
            if F_single.shape == (2, 2):
                F_s = jnp.array([[F_single[0, 0], F_single[0, 1], 0.0],
                                 [F_single[1, 0], F_single[1, 1], 0.0],
                                 [0.0, 0.0, 1.0]])
            else:
                F_s = F_single
            from core.utils import C_func, I3_func
            C = C_func(F_s)
            I3_safe = jnp.clip(I3_func(C), 1.0e-8, 1.0e8)
            C_bar = (I3_safe**(-1/3)) * C
            if hasattr(feature_extractor, "a1") and feature_extractor.a1 is not None:
                a1 = feature_extractor.a0
                a2 = feature_extractor.a1
            else:
                a1 = feature_extractor.a0 if feature_extractor is not None else jnp.array([jnp.cos(jnp.pi/4), jnp.sin(jnp.pi/4), 0.0])
                a2 = jnp.array([jnp.cos(-jnp.pi/4), jnp.sin(-jnp.pi/4), 0.0])
            I4_bar_1 = jnp.einsum('i,ij,j->', a1, C_bar, a1)
            I4_bar_2 = jnp.einsum('i,ij,j->', a2, C_bar, a2)
            I4_m1 = I4_bar_1 - 1.0
            I6_m1 = I4_bar_2 - 1.0
            if len(theta_aniso) == 6:
                C42, C43, C44, C62, C63, C64 = theta_aniso[:6]
                return (C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4 +
                        C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4)
            else:
                ta = list(theta_aniso) + [0.0] * (8 - len(theta_aniso))
                C42, C44, k1, k2, C62, C64, k3, k4 = ta[:8]
                exp_arg1 = jnp.clip(k2 * I4_m1**2, -30.0, 30.0)
                exp_arg2 = jnp.clip(k4 * I6_m1**2, -30.0, 30.0)
                return (C42 * I4_m1**2 + C44 * I4_m1**4 + k1 * (jnp.exp(exp_arg1) - 1.0) +
                        C62 * I6_m1**2 + C64 * I6_m1**4 + k3 * (jnp.exp(exp_arg2) - 1.0))

        def get_distilled_energy_stress_split(theta_dev, theta_vol, F_chunk):
            dev_theta = list(theta_dev) + [0.0, 0.0, 0.0]
            vol_theta = [0.0]*9 + list(theta_vol)
            mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
            mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
            return jax.vmap(mat_dev.psi)(F_chunk) + jax.vmap(mat_vol.psi)(F_chunk)

        def get_distilled_energy_stress_split_3(theta_dev, theta_vol, theta_aniso, F_chunk):
            dev_theta = list(theta_dev) + [0.0, 0.0, 0.0]
            vol_theta = [0.0]*9 + list(theta_vol)
            mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
            mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
            s_psi_aniso = jax.vmap(lambda f: psi_aniso_single(theta_aniso, f))(F_chunk)
            return jax.vmap(mat_dev.psi)(F_chunk) + jax.vmap(mat_vol.psi)(F_chunk) + s_psi_aniso
            
        dist_psi_samples = []
        for mode in range(len(mode_names)):
            if has_aniso:
                s_psi = jax.vmap(lambda td, tv, ta: get_distilled_energy_stress_split_3(td, tv, ta, F_all[mode]))(dev_samples, vol_samples, aniso_samples)
            else:
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
        aniso_tot, aniso_first = get_sensitivities(distilled_dir, "aniso_", "aniso_sensitivities")
        for k, v in dev_tot.items():
            tot_means_all[k], first_means_all[k], param_types[k] = v, dev_first.get(k, 0.0), "dev"
        for k, v in vol_tot.items():
            tot_means_all[k], first_means_all[k], param_types[k] = v, vol_first.get(k, 0.0), "vol"
        for k, v in aniso_tot.items():
            tot_means_all[k], first_means_all[k], param_types[k] = v, aniso_first.get(k, 0.0), "aniso"
    
    sorted_params = sorted(tot_means_all.keys(), key=lambda k: tot_means_all[k], reverse=True)
    sorted_tot_means = np.array([tot_means_all[k] for k in sorted_params])
    sorted_first_means = np.array([first_means_all[k] for k in sorted_params])
    
    denominator = max(np.sum(sorted_first_means), np.sum(sorted_tot_means), 1.0)
    est_coverage_pct = (np.cumsum(sorted_tot_means) / denominator) * 100.0
    
    # Violin data
    if is_split:
        dev = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        vol = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))
        all_samples = [dev, vol]
        all_dev_names = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]
        all_vol_names = ["D1", "D2", "D3"]
        full_param_names = all_dev_names[:dev.shape[1]] + all_vol_names[:vol.shape[1]]
        
        if has_aniso:
            aniso = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy"))
            all_samples.append(aniso)
            if aniso.shape[1] == 6:
                all_aniso_names = ["C42", "C43", "C44", "C62", "C63", "C64"]
            else:
                all_aniso_names = ["C42", "C44", "k1", "k2", "C62", "C64", "k3", "k4"]
            full_param_names += all_aniso_names[:aniso.shape[1]]
            
        min_len = min(s.shape[0] for s in all_samples)
        samples_np = np.hstack([s[:min_len] for s in all_samples])
        df = pd.DataFrame(samples_np, columns=full_param_names)

    # Calculate cumulative RMSE
    rmse_history = []
    if args.distill_target == "sef_split":
        for k in range(1, len(sorted_params) + 1):
            active_params_k = sorted_params[:k]
            
            # Construct mean parameters
            theta_dev = np.zeros(9)
            theta_vol = np.zeros(3)
            theta_aniso = np.zeros(6 if has_aniso and aniso.shape[1] == 6 else 8)
            
            for p in active_params_k:
                clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
                if clean_p in df.columns:
                    mean_val = df[clean_p].values.mean()
                    if param_types.get(p) == "dev":
                        if clean_p in all_dev_names:
                            idx = all_dev_names.index(clean_p)
                            if idx < 9:
                                theta_dev[idx] = mean_val
                    elif param_types.get(p) == "vol":
                        if clean_p in all_vol_names:
                            idx = all_vol_names.index(clean_p)
                            if idx < 3:
                                theta_vol[idx] = mean_val
                    elif param_types.get(p) == "aniso" and has_aniso:
                        if clean_p in all_aniso_names:
                            idx = all_aniso_names.index(clean_p)
                            if idx < 8:
                                theta_aniso[idx] = mean_val
                    
            # Compute RMSE across all modes
            total_sq_err = 0.0
            total_pts = 0
            for mode in range(len(mode_names)):
                if has_aniso:
                    s_psi = get_distilled_energy_stress_split_3(theta_dev, theta_vol, theta_aniso, F_all[mode])
                else:
                    s_psi = get_distilled_energy_stress_split(theta_dev, theta_vol, F_all[mode])
                total_sq_err += np.sum((s_psi - psi_true[mode])**2)
                total_pts += len(s_psi)
            rmse = np.sqrt(total_sq_err / total_pts)
            rmse_history.append(float(rmse))
    else:
        rmse_history = [0.0] * len(sorted_params)

    # Try reading ground truth parameters directly from recipe config YAML
    recipe_file = f"configs/recipes/{true_model_name}.yaml"
    recipe_data = {}
    if os.path.exists(recipe_file):
        try:
            import yaml
            with open(recipe_file, 'r') as rf:
                recipe_data = yaml.safe_load(rf).get('material_params', {})
        except Exception:
            recipe_data = {}

    true_val_dict = {}
    if true_model_name in ["ortho45", "symnonortho60"]:
        true_params_set = {"C10", "D1", "C42", "C62"}
        c10 = recipe_data.get('dev_params', [0.5])[0]
        d1 = recipe_data.get('vol_params', [1.0])[0]
        aniso_p = recipe_data.get('aniso_params', [0.7, 0.0, 0.9])
        c42 = aniso_p[0] if len(aniso_p) > 0 else 0.7
        c62 = aniso_p[2] if len(aniso_p) > 2 else 0.9
        true_val_dict = {"C10": c10, "D1": d1, "C42": c42, "C62": c62}
    elif true_model_name == "aniso30":
        true_params_set = {"C10", "D1", "C42"}
        c10 = recipe_data.get('dev_params', [0.5])[0]
        d1 = recipe_data.get('vol_params', [1.0])[0]
        aniso_p = recipe_data.get('aniso_params', [0.7])
        c42 = aniso_p[0] if len(aniso_p) > 0 else 0.7
        true_val_dict = {"C10": c10, "D1": d1, "C42": c42}
    elif true_model_name in ["c20d10d05", "c20_d10_d05"]:
        true_params_set = {"C10", "D1", "D2"}
        c10 = recipe_data.get('dev_params', [2.0])[0]
        vol_p = recipe_data.get('vol_params', [1.0, 0.5])
        d1 = vol_p[0] if len(vol_p) > 0 else 1.0
        d2 = vol_p[1] if len(vol_p) > 1 else 0.5
        true_val_dict = {"C10": c10, "D1": d1, "D2": d2}
    elif true_model_name in ["nh2", "neohookean2", "nh"]:
        true_params_set = {"C10", "D1"}
        c10 = recipe_data.get('dev_params', [0.5])[0]
        d1 = recipe_data.get('vol_params', [1.5])[0]
        true_val_dict = {"C10": c10, "D1": d1}
    elif true_model_name == "isihara":
        true_params_set = {"C10", "C01", "C20", "D1"}
        dev_p = recipe_data.get('dev_params', [0.5, 1.0, 1.0])
        c10 = dev_p[0] if len(dev_p) > 0 else 0.5
        c01 = dev_p[1] if len(dev_p) > 1 else 1.0
        c20 = dev_p[2] if len(dev_p) > 2 else 1.0
        d1 = recipe_data.get('vol_params', [1.5])[0]
        true_val_dict = {"C10": c10, "C01": c01, "C20": c20, "D1": d1}
    elif true_model_name == "gentthomas":
        true_params_set = {"C10", "E", "D1"}
        dev_p = recipe_data.get('dev_params', [0.5, 1.0])
        c10 = dev_p[0] if len(dev_p) > 0 else 0.5
        e_param = dev_p[1] if len(dev_p) > 1 else 1.0
        d1 = recipe_data.get('vol_params', [1.5])[0]
        true_val_dict = {"C10": c10, "E": e_param, "D1": d1}
    else:
        true_params_set = {"C10", "D1"}
        c10 = recipe_data.get('dev_params', [0.5])[0]
        d1 = recipe_data.get('vol_params', [1.5])[0]
        true_val_dict = {"C10": c10, "D1": d1}


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
    
    mode_limits_dev = {}
    mode_limits_vol = {}
    
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
        
        # Interpolation Limit boundary
        feats_ext = jax.vmap(extractor.extract)(F_all[i])
        dev_I, vol_J = feats_ext[0], feats_ext[1]
        inside_vol = (vol_J[:, 0] >= limit_min_vol[0] - vol_tol) & (vol_J[:, 0] <= limit_max_vol[0] + vol_tol)
        dev_I_np = np.array(dev_I)
        inside_dev = np.all(dev_I_np @ hull_eqs[:, :-1].T + hull_eqs[:, -1] <= dev_tol, axis=1)
        
        # Determine dev limits
        crossings_dev = np.where(np.diff(inside_dev.astype(int)) != 0)[0]
        limits_dev = []
        for idx in crossings_dev:
            limits_dev.append((float(gamma[idx]), dev_I_np[idx]))
        mode_limits_dev[name] = {
            'crossings': limits_dev,
            'always_out': len(limits_dev) == 0 and not inside_dev[0]
        }
            
        # Determine vol limits
        crossings_vol = np.where(np.diff(inside_vol.astype(int)) != 0)[0]
        limits_vol = []
        for idx in crossings_vol:
            limits_vol.append((float(gamma[idx]), float(vol_J[idx, 0])))
        mode_limits_vol[name] = {
            'crossings': limits_vol,
            'always_out': len(limits_vol) == 0 and not inside_vol[0]
        }

        inside_mask = inside_vol & inside_dev
        
        crossings_mask = np.where(np.diff(inside_mask.astype(int)) != 0)[0]
        gamma_boundaries = [float(gamma[0])] + [float(gamma[idx]) for idx in crossings_mask] + [float(gamma[-1])]
        state_is_out = not inside_mask[0]
        for k in range(len(gamma_boundaries) - 1):
            if state_is_out:
                ax_psi.axvspan(gamma_boundaries[k], gamma_boundaries[k+1], color='#E69F00', alpha=0.15, zorder=-1, label='Extrapolation')
            state_is_out = not state_is_out
            
        for idx in crossings_mask:
            ax_psi.axvline(float(gamma[idx]), color='#E69F00', linestyle=':', linewidth=1.5, zorder=0)

    # Global legend for energy plots
    handles, labels = ax_psi.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_energy.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=8, frameon=False)
    
    fig_energy.savefig(os.path.join(distilled_dir, f"split_energy_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    plt.close(fig_energy)

    # 2. Parameters Figure (Sensitivity + Violin)
    h_params = 5.0
    fig_params = plt.figure(figsize=(fig_width, h_params))
    gs_params = fig_params.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.65)
    
    ax_sens = fig_params.add_subplot(gs_params[0, 0])
    x_pos = np.arange(len(sorted_params))
    
    gt_label_added = False
    for i, p in enumerate(sorted_params):
        clean_p = p.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
        if clean_p in true_params_set:
            label_gt = "Ground Truth Parameter" if not gt_label_added else ""
            ax_sens.axvspan(i - 0.25, i + 0.25, color='#E0E0E0', alpha=0.8, zorder=1, label=label_gt)
            gt_label_added = True
            
        if param_types.get(p) == "dev":
            color = "#0072B2"
            label = r"$\bar{S}_{\mathrm{T,d}}$"
        elif param_types.get(p) == "vol":
            color = "#D55E00"
            label = r"$\bar{S}_{\mathrm{T,v}}$"
        else:
            color = "#CC79A7"
            label = r"$\bar{S}_{\mathrm{T,a}}$"
            
        handles, labels = ax_sens.get_legend_handles_labels()
        if label not in labels:
            ax_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3, label=label)
        else:
            ax_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3)
        
    ax_sens.set_yscale('log')
    ax_sens.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1), top=10.0)
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
    by_label_sens = dict(zip(labels_1 + labels_2, lines_1 + lines_2))
    ax_sens.legend(by_label_sens.values(), by_label_sens.keys(), fontsize=6.5, loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=5, frameon=False)
    
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
            
            ax_viol.text(i, 1.05, fr"${mean_val:.3f}$", transform=ax_viol.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6.5, color=color,
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor=color, lw=0.5), zorder=10, clip_on=False)
        
        if is_true:
            ax_viol.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=1.5, linestyle='--')
            
            ax_viol.text(i, 1.18, fr"${true_val:.3f}$", transform=ax_viol.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=6.5, color='black',
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor='black', lw=0.5), zorder=10, clip_on=False)
                
    ax_viol.set_xticks(range(len(sorted_params)))
    ax_viol.set_xticklabels(sorted_params, fontsize=9)
    ax_viol.set_ylabel('Parameter Value', fontsize=8)
    ax_viol.set_ylim([0, 2.1])
    ax_viol.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_viol.tick_params(axis='y', labelsize=7)
    ax_viol.grid(False)
    
    # Row labels for the text boxes placed cleanly outside ax_viol above the top spine
    ax_viol.text(-0.025, 1.05, "Mean", transform=ax_viol.transAxes, ha='right', va='bottom', fontsize=7.5, fontweight='bold', color='black', clip_on=False)
    ax_viol.text(-0.025, 1.18, "True", transform=ax_viol.transAxes, ha='right', va='bottom', fontsize=7.5, fontweight='bold', color='black', clip_on=False)

    # Legend for the parameter plot
    import matplotlib.patches as mpatches
    import matplotlib.lines as mlines
    
    viol_legend = [
        mpatches.Patch(color='gray', alpha=0.5, label='Density'),
        mlines.Line2D([0], [0], color='gray', lw=2, label='Mean'),
        mpatches.Patch(color='gray', alpha=0.1, label='95% CI'),
        mlines.Line2D([0], [0], color='black', lw=1.5, linestyle='--', label='Ground Truth')
    ]
    ax_viol.legend(handles=viol_legend, loc='upper center', bbox_to_anchor=(0.5, -0.22), ncol=4, fontsize=7, frameon=False)
    
    fig_params.savefig(os.path.join(distilled_dir, f"split_params_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    fig_params.savefig(os.path.join(distilled_dir, f"split_params_{true_model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_params)
    
    # 4. Deviatoric and Volumetric Space Combined Plot
    fig_space, (ax_dev, ax_vol) = plt.subplots(1, 2, figsize=(12, 5))
    
    # --- Deviatoric Space ---
    # Plot tight boundary by connecting outermost datapoints (Convex Hull)
    dev_hull_pts = np.array(dev_obs)[dev_hull.vertices]
    dev_hull_pts = np.vstack((dev_hull_pts, dev_hull_pts[0]))
    ax_dev.plot(dev_hull_pts[:, 0], dev_hull_pts[:, 1], 'k--', lw=1.5, label='Interpolation Boundary', zorder=4)
    ax_dev.fill(dev_hull_pts[:, 0], dev_hull_pts[:, 1], color='gray', alpha=0.1, zorder=1)
    
    # Plot training dataset
    ax_dev.scatter(I_obs_all[:, 0], I_obs_all[:, 1], color='#1f77b4', marker='.', s=10, alpha=0.3, label='Training Dataset', zorder=2)
    
    # Plot inducing points
    ax_dev.scatter(dev_z[:, 0], dev_z[:, 1], color='black', marker='X', s=15, linewidths=0.2, label='Inducing Points', zorder=5)
    
    # Plot mode trajectories and collect interception text
    mode_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    extrap_texts_dev = []
    
    for i, name in enumerate(mode_names):
        feats_ext = jax.vmap(extractor.extract)(F_all[i])
        dev_I, vol_J = feats_ext[0], feats_ext[1]
        ax_dev.plot(dev_I[:, 0], dev_I[:, 1], color=mode_colors[i], lw=1.5, label=f'{name} Trajectory')
        
        dev_info = mode_limits_dev[name]
        if dev_info['always_out']:
            extrap_texts_dev.append(fr"$\gamma^{{(\mathrm{{{name}}})}}$ (Extrap)")
        else:
            for g_dev, p_dev in dev_info['crossings']:
                subscript = r"\mathrm{min}" if name in ["UC", "EC"] else r"\mathrm{max}"
                extrap_texts_dev.append(fr"$\gamma_{{{subscript}}}^{{(\mathrm{{{name}}})}} = {g_dev:.2f}$")
                
    # Add boundary intercepts text box outside
    ax_dev.text(1.02, 0.5, "Boundary Intercepts:\n" + "\n".join(extrap_texts_dev), 
                transform=ax_dev.transAxes, va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
                
    # Start point
    ax_dev.scatter([3.0], [3.0], color='black', marker='o', facecolors='none', s=50, label='Undeformed (3, 3)', zorder=5)
    
    ax_dev.set_xlabel(r"$\bar{I}_1$", fontsize=10)
    ax_dev.set_ylabel(r"$\bar{I}_2$", fontsize=10)
    ax_dev.set_title("Deviatoric Invariant Space & Interpolation Limits", fontsize=11)
    ax_dev.grid(True, linestyle=':', alpha=0.6)
    
    # --- Volumetric Space ---
    # Plot interpolation bounds
    ax_vol.axhspan(limit_min_vol[0], limit_max_vol[0], color='gray', alpha=0.15, label='Interpolation Boundary (Vol)', zorder=1)
    ax_vol.axhline(limit_min_vol[0], color='k', linestyle='--', lw=1.5, zorder=2)
    ax_vol.axhline(limit_max_vol[0], color='k', linestyle='--', lw=1.5, zorder=2)
    
    # Plot inducing points as faint lines
    for idx, vz in enumerate(vol_z):
        lbl = 'Inducing Points (J)' if idx == 0 else ""
        ax_vol.axhline(vz[0], color='black', alpha=0.5, ls=':', lw=1, label=lbl, zorder=3)
        
    # Plot mode trajectories and collect interception text
    extrap_texts_vol = []
    
    for i, name in enumerate(mode_names):
        feats_ext = jax.vmap(extractor.extract)(F_all[i])
        vol_J = feats_ext[1]
        ax_vol.plot(gamma, vol_J[:, 0], color=mode_colors[i], lw=2.0, zorder=4)
        
        vol_info = mode_limits_vol[name]
        if vol_info['always_out']:
            extrap_texts_vol.append(fr"$\gamma^{{(\mathrm{{{name}}})}}$ (Extrap)")
        else:
            for g_vol, p_vol in vol_info['crossings']:
                subscript = r"\mathrm{min}" if name in ["UC", "EC"] else r"\mathrm{max}"
                extrap_texts_vol.append(fr"$\gamma_{{{subscript}}}^{{(\mathrm{{{name}}})}} = {g_vol:.2f}$")
                
    # Add boundary intercepts text box outside
    ax_vol.text(1.02, 0.5, "Boundary Intercepts:\n" + "\n".join(extrap_texts_vol), 
                transform=ax_vol.transAxes, va='center', fontsize=8, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.8))
        
    ax_vol.scatter([0.0], [1.0], color='black', marker='o', facecolors='none', s=50, label='Undeformed (J=1)', zorder=5)
    
    ax_vol.set_xlabel(r"Deformation ($\gamma$)", fontsize=10)
    ax_vol.set_ylabel(r"Volumetric Invariant ($J$)", fontsize=10)
    ax_vol.set_title("Volumetric Invariant Space & Interpolation Limits", fontsize=11)
    ax_vol.grid(True, linestyle=':', alpha=0.6)
    
    # Merge legends and place at bottom
    handles_dev, labels_dev = ax_dev.get_legend_handles_labels()
    handles_vol, labels_vol = ax_vol.get_legend_handles_labels()
    
    # Deduplicate labels
    by_label = dict(zip(labels_dev + labels_vol, handles_dev + handles_vol))
    fig_space.legend(by_label.values(), by_label.keys(), loc='upper center', bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8, frameon=True)
    
    # Adjust layout to make room for bottom legend and right text boxes
    fig_space.subplots_adjust(bottom=0.25, right=0.85, wspace=0.6)
    
    space_path = os.path.join(distilled_dir, f"split_invariant_spaces_{true_model_name}.pdf")
    fig_space.savefig(space_path, dpi=300, bbox_inches='tight')
    plt.close(fig_space)
    
    print(f"Saved split summary plots to {distilled_dir}")

if __name__ == "__main__":
    main()

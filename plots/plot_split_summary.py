import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from plots.theme import apply_style, save_figure
from plots.sensitivity import load_sobol_csv
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor
from core.utils import (
    infer_material_model_name,
    generate_standard_deformation_modes as generate_standard_modes,
    compute_invariants_np,
    load_f3x3_from_distilled,
)
from scipy.spatial import ConvexHull

def to_latex(name):
    clean = str(name).replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    if clean.startswith("C") and len(clean) == 3 and clean[1:].isdigit():
        return rf"$C_{{{clean[1:]}}}$"
    if clean.startswith("D") and len(clean) == 2 and clean[1:].isdigit():
        return rf"$D_{{{clean[1:]}}}$"
    if clean == "E":
        return r"$E$"
    return rf"${clean}$"

def get_comp_color(name):
    clean = name.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    dev_palette = {
        "C10": "#1f77b4",  # deep classic blue
        "C01": "#4ba3e3",  # sky blue
        "C20": "#084081",  # dark navy blue
        "C11": "#009999",  # ocean teal / cyan
        "C02": "#7bccc4",  # soft aquamarine / light teal
        "C30": "#2b5c8f",  # slate steel blue
        "C21": "#41b6c4",  # medium cyan-blue
        "C12": "#225ea8",  # royal blue
        "C03": "#a6bddb",  # soft ice blue
        "E":   "#02818a",  # rich deep teal
    }
    vol_palette = {
        "D1": "#d95f02",   # vivid rust orange
        "D2": "#fdbb84",   # warm amber / peach
        "D3": "#7f2704",   # deep dark burnt orange
    }
    if clean in dev_palette:
        return dev_palette[clean]
    elif clean in vol_palette:
        return vol_palette[clean]
    elif clean.startswith("C4") or clean.startswith("C6"):
        return "#CC79A7"
    elif clean.startswith("D"):
        return "#d95f02"
    return "#1f77b4"

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
    apply_style()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default=None)
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, default=None)
    parser.add_argument("--distill_target", type=str, default="sef_split")
    parser.add_argument("--sobol_threshold", type=float, default=0.0001)
    args = parser.parse_args()
    
    distilled_dir = os.path.abspath(args.distilled_dir)
    if not os.path.exists(os.path.join(distilled_dir, "dev_flow_samples.npy")) and os.path.isdir(os.path.join(distilled_dir, "distilled")):
        distilled_dir = os.path.join(distilled_dir, "distilled")
    if args.material_model is None:
        args.material_model = infer_material_model_name(distilled_dir)
    true_model_name = args.material_model
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

    from core.material_models import get_material_from_dir
    try:
        true_model = get_material_from_dir(saved_model_dir, jit_P=False)
    except FileNotFoundError:
        true_model = get_material_from_dir(distilled_dir, jit_P=False)
    true_model_name = infer_material_model_name(saved_model_dir)
    
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
        a0_val = getattr(true_model, "a0", None)
        if a0_val is None:
            a0_val = getattr(true_model, "a1", None)
        
        a1_val = getattr(true_model, "a1", None) if getattr(true_model, "a0", None) is not None else getattr(true_model, "a2", None)
        
        if a0_val is not None and a1_val is not None:
            feature_extractor = AnisotropicFeatureExtractor(np.array(a0_val), a1=np.array(a1_val))
        elif a0_val is not None:
            feature_extractor = AnisotropicFeatureExtractor(np.array(a0_val))
        elif getattr(gp_params, "raw_aniso_theta_mean", None) is not None:
            raw_th = gp_params.raw_aniso_theta_mean
            theta = float(np.pi * (1.0 / (1.0 + np.exp(-raw_th)) - 0.5))
            a0 = np.array([np.cos(theta), np.sin(theta), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)
        elif os.path.exists(os.path.join(saved_model_dir, "metadata.json")):
            with open(os.path.join(saved_model_dir, "metadata.json")) as mf:
                m_data = json.load(mf)
                if "a0" in m_data:
                    a0 = np.array(m_data["a0"])
                    a1 = np.array(m_data["a1"]) if "a1" in m_data else None
                    feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
                elif aniso_z.shape[1] == 4:
                    a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
                    a1 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0])
                    feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
                else:
                    theta = np.pi / 4.0
                    a0 = np.array([np.cos(theta), np.sin(theta), 0.0])
                    feature_extractor = AnisotropicFeatureExtractor(a0)
        elif aniso_z.shape[1] == 4:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            a1 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        else:
            theta = np.pi / 4.0
            a0 = np.array([np.cos(theta), np.sin(theta), 0.0])
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

    import json
    metadata_path = os.path.join(saved_model_dir, "metadata.json")
    cov_mode = "diag"
    constraint_lengthscale = 1
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            _meta = json.load(f)
            cov_mode = _meta.get("covariance_mode", "diag")
            constraint_lengthscale = _meta.get("constraint_lengthscale", 1)
            
    learned_gp = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode,
        constraint_lengthscale=constraint_lengthscale
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
    
    all_dev_names = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]
    all_vol_names = ["D1", "D2", "D3"]
    all_aniso_names = ["C42", "C43", "C44", "C62", "C63", "C64"]
    
    if args.distill_target == "sef_split":
        dev_raw = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        vol_raw = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))
        aniso_raw = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy")) if has_aniso else None
        
        # Pad filtered-out parameters with 0.0
        dev_samples = np.zeros((min(num_samples, dev_raw.shape[0]), 10))
        dev_samples[:, :dev_raw.shape[1]] = dev_raw[:num_samples, :10]
        
        vol_samples = np.zeros((min(num_samples, vol_raw.shape[0]), 3))
        vol_samples[:, :vol_raw.shape[1]] = vol_raw[:num_samples, :3]
        
        if has_aniso and aniso_raw is not None:
            aniso_samples = np.zeros((min(num_samples, aniso_raw.shape[0]), 6))
            aniso_samples[:, :aniso_raw.shape[1]] = aniso_raw[:num_samples, :6]
        else:
            aniso_samples = None
        
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
            ta = list(theta_aniso) + [0.0] * max(0, 6 - len(theta_aniso))
            C42, C43, C44, C62, C63, C64 = ta[:6]
            return (C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4 +
                    C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4)

        def get_distilled_energy_stress_split(theta_dev, theta_vol, F_chunk):
            dev_theta = list(theta_dev) + [0.0] * max(0, 10 - len(theta_dev))
            vol_theta = list(theta_vol) + [0.0] * max(0, 3 - len(theta_vol))
            mat_dev = get_material("gmr", dev_params=dev_theta[:10], vol_params=[0.0, 0.0, 0.0], jit_P=False)
            mat_vol = get_material("gmr", dev_params=[0.0]*10, vol_params=vol_theta[:3], jit_P=False)
            return jax.vmap(mat_dev.psi)(F_chunk) + jax.vmap(mat_vol.psi)(F_chunk)

        def get_distilled_energy_stress_split_3(theta_dev, theta_vol, theta_aniso, F_chunk):
            dev_theta = list(theta_dev) + [0.0] * max(0, 10 - len(theta_dev))
            vol_theta = list(theta_vol) + [0.0] * max(0, 3 - len(theta_vol))
            mat_dev = get_material("gmr", dev_params=dev_theta[:10], vol_params=[0.0, 0.0, 0.0], jit_P=False)
            mat_vol = get_material("gmr", dev_params=[0.0]*10, vol_params=vol_theta[:3], jit_P=False)
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
        aniso = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy")) if has_aniso else None
        
        min_len = min(dev.shape[0], vol.shape[0])
        if has_aniso and aniso is not None:
            min_len = min(min_len, aniso.shape[0])
            
        data_dict = {}
        for idx, name in enumerate(all_dev_names):
            if idx < dev.shape[1]:
                data_dict[name] = dev[:min_len, idx]
            else:
                data_dict[name] = np.zeros(min_len)
        for idx, name in enumerate(all_vol_names):
            if idx < vol.shape[1]:
                data_dict[name] = vol[:min_len, idx]
            else:
                data_dict[name] = np.zeros(min_len)
        if has_aniso and aniso is not None:
            for idx, name in enumerate(all_aniso_names):
                if idx < aniso.shape[1]:
                    data_dict[name] = aniso[:min_len, idx]
                else:
                    data_dict[name] = np.zeros(min_len)
        df = pd.DataFrame(data_dict)

    # Calculate cumulative RMSE
    rmse_history = []
    if args.distill_target == "sef_split":
        for k in range(1, len(sorted_params) + 1):
            active_params_k = sorted_params[:k]
            
            # Construct mean parameters
            theta_dev = np.zeros(10)
            theta_vol = np.zeros(3)
            theta_aniso = np.zeros(6)
            
            for p in active_params_k:
                clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
                if clean_p in df.columns:
                    mean_val = float(df[clean_p].values.mean())
                    if param_types.get(p) == "dev":
                        if clean_p in all_dev_names:
                            idx = all_dev_names.index(clean_p)
                            if idx < 10:
                                theta_dev[idx] = mean_val
                    elif param_types.get(p) == "vol":
                        if clean_p in all_vol_names:
                            idx = all_vol_names.index(clean_p)
                            if idx < 3:
                                theta_vol[idx] = mean_val
                    elif param_types.get(p) == "aniso" and has_aniso:
                        if clean_p in all_aniso_names:
                            idx = all_aniso_names.index(clean_p)
                            if idx < 6:
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

    # Dynamic ground truth parameters from true_model
    true_val_dict = {}
    true_params_set = set()
    if hasattr(true_model, "dev_params") and true_model.dev_params is not None:
        for name, val in zip(all_dev_names, true_model.dev_params):
            if abs(val) > 1e-12:
                true_params_set.add(name)
                true_val_dict[name] = float(val)
    if hasattr(true_model, "vol_params") and true_model.vol_params is not None:
        for name, val in zip(all_vol_names, true_model.vol_params):
            if abs(val) > 1e-12:
                true_params_set.add(name)
                true_val_dict[name] = float(val)
    if hasattr(true_model, "aniso_params") and true_model.aniso_params is not None:
        for name, val in zip(all_aniso_names, true_model.aniso_params):
            if abs(val) > 1e-12:
                true_params_set.add(name)
                true_val_dict[name] = float(val)


    # Figure dimensions
    fig_width = 8.27
    h_energy = 8.183 * (0.23 / 0.75)
    h_sens = 8.183 * (0.25 / 0.75)
    h_viol = 8.183
    h_params = 4.0  # Increased vertical height slightly to fix EC label cutoff
    
    # 1. Energy Plots (3x2 grid layout: 3 rows, 2 cols with square plots)
    fig_width_energy = 3.65
    h_energy_split = 4.85
    fig_energy = plt.figure(figsize=(fig_width_energy, h_energy_split))
    gs_top = fig_energy.add_gridspec(3, 2, hspace=0.30, wspace=0.14, top=0.96, bottom=0.13, left=0.13, right=0.97)
    dist_color = "#009E73"
    
    mode_limits_dev = {}
    mode_limits_vol = {}
    
    for i, name in enumerate(mode_names):
        row, col = divmod(i, 2)
        ax_psi = fig_energy.add_subplot(gs_top[row, col])
        ax_psi.set_box_aspect(1)
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=1.2, label="Ground Truth", zorder=5)
        gp_psi_lower = psi_dist_mean[i] - 1.96 * jnp.sqrt(psi_dist_var[i])
        gp_psi_upper = psi_dist_mean[i] + 1.96 * jnp.sqrt(psi_dist_var[i])
        ax_psi.fill_between(gamma, gp_psi_lower, gp_psi_upper, color='gray', alpha=0.3, label="GP 95% CI")
        ax_psi.plot(gamma, psi_dist_mean[i], color='gray', lw=1.2, ls='-', label="GP mean", zorder=4)
        
        gp_cov_psi = jnp.mean((psi_true[i] >= gp_psi_lower) & (psi_true[i] <= gp_psi_upper))
        rmse_psi_gp = jnp.sqrt(jnp.mean((psi_dist_mean[i] - psi_true[i]) ** 2))
        ss_tot_psi = jnp.sum((psi_true[i] - jnp.mean(psi_true[i])) ** 2)
        r2_psi_gp = 1 - jnp.sum((psi_true[i] - psi_dist_mean[i]) ** 2) / (ss_tot_psi + 1e-12)
        
        nf_psi_lower = jnp.percentile(dist_psi_samples[i], 2.5, axis=0)
        nf_psi_upper = jnp.percentile(dist_psi_samples[i], 97.5, axis=0)
        dist_psi_mean = dist_psi_samples[i].mean(axis=0)
        
        ax_psi.fill_between(gamma, nf_psi_lower, nf_psi_upper, color=dist_color, alpha=0.15, label="Distilled 95%CI", zorder=2)
        ax_psi.plot(gamma, dist_psi_samples[i].T, color=dist_color, lw=0.4, alpha=0.1, zorder=1)
        ax_psi.plot(gamma, dist_psi_mean, color=dist_color, lw=1.5, label="Distilled Mean", zorder=3)
        
        nf_cov_psi = jnp.mean((psi_true[i] >= nf_psi_lower) & (psi_true[i] <= nf_psi_upper))
        rmse_psi = jnp.sqrt(jnp.mean((dist_psi_mean - psi_true[i]) ** 2))
        r2_psi = 1 - jnp.sum((psi_true[i] - dist_psi_mean) ** 2) / (ss_tot_psi + 1e-12)
        
        if col == 0:
            ax_psi.set_ylabel(r"$\Psi$", fontsize=7.5, labelpad=2)
        if row == 2:
            ax_psi.set_xlabel(r"$\gamma$", fontsize=7.5, labelpad=2)
        else:
            ax_psi.tick_params(axis='x', labelbottom=False)
            
        mode_full_names = {
            "UT": "Uniaxial Tension",
            "ET": "Equibiaxial Tension",
            "PS": "Pure Shear",
            "UC": "Uniaxial Compression",
            "EC": "Equibiaxial Compression",
            "SS": "Simple Shear",
        }
        title_text = mode_full_names.get(name, name)
        ax_psi.set_title(title_text, fontsize=6.8, pad=2, fontweight='normal')
        ax_psi.grid(False)
        ax_psi.tick_params(axis='both', which='major', labelsize=6.0, pad=1)
        
        # Clean ylim margins
        cur_ylim = ax_psi.get_ylim()
        y_span = cur_ylim[1] - cur_ylim[0]
        ax_psi.set_ylim(bottom=-0.04 * y_span, top=cur_ylim[1] + 0.05 * y_span)
            
        # Top-left r^2 box for distilled model
        ax_psi.text(0.06, 0.92, rf"$r^2 = {r2_psi:.3f}$", transform=ax_psi.transAxes,
                    ha='left', va='top', fontsize=5.8,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="#bbbbbb", lw=0.4, alpha=0.90), zorder=6)
        
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
            is_entry = (not inside_dev[idx]) and inside_dev[idx + 1]
            limits_dev.append((float(gamma[idx]), dev_I_np[idx], is_entry))
        mode_limits_dev[name] = {
            'crossings': limits_dev,
            'always_out': len(limits_dev) == 0 and not inside_dev[0]
        }
            
        # Determine vol limits
        crossings_vol = np.where(np.diff(inside_vol.astype(int)) != 0)[0]
        limits_vol = []
        for idx in crossings_vol:
            is_entry = (not inside_vol[idx]) and inside_vol[idx + 1]
            limits_vol.append((float(gamma[idx]), float(vol_J[idx, 0]), is_entry))
        mode_limits_vol[name] = {
            'crossings': limits_vol,
            'always_out': len(limits_vol) == 0 and not inside_vol[0]
        }

        inside_mask = inside_vol & inside_dev
        
        crossings_mask = np.where(np.diff(inside_mask.astype(int)) != 0)[0]
        gamma_boundaries = [float(gamma[0])] + [float(gamma[idx]) for idx in crossings_mask] + [float(gamma[-1])]
        for k in range(len(gamma_boundaries) - 1):
            if k == 0:
                is_inside = bool(inside_mask[0])
            else:
                is_inside = bool(inside_mask[crossings_mask[k - 1] + 1])
            if not is_inside:
                ax_psi.axvspan(gamma_boundaries[k], gamma_boundaries[k+1], color='#E69F00', alpha=0.15, zorder=-1, label='Extrapolation')
            
        for idx in crossings_mask:
            ax_psi.axvline(float(gamma[idx]), color='#E69F00', linestyle=':', linewidth=1.2, zorder=0)

    # Global legend for energy plots placed in bottom margin below x-axis labels
    handles, labels = ax_psi.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig_energy.legend(by_label.values(), by_label.keys(), loc='lower center', ncol=3,
                      bbox_to_anchor=(0.5, 0.01), fontsize=5.8, frameon=False, handlelength=1.2, handletextpad=0.3, columnspacing=0.8)
    
    fig_energy.savefig(os.path.join(distilled_dir, f"split_energy_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    fig_energy.savefig(os.path.join(distilled_dir, f"split_energy_{true_model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_energy)
    # Save SEF metrics across modes to validation_metrics.json
    try:
        val_json_path = os.path.join(distilled_dir, "validation_metrics.json")
        vdata = {}
        if os.path.exists(val_json_path):
            with open(val_json_path, "r") as f:
                vdata = json.load(f)
        if "sef" not in vdata:
            vdata["sef"] = {}
        
        # Calculate overall averages
        gp_rmses = []
        gp_covs = []
        gp_r2s = []
        dist_rmses = []
        dist_covs = []
        dist_r2s = []
        for i in range(len(mode_names)):
            gp_l = psi_dist_mean[i] - 1.96 * jnp.sqrt(psi_dist_var[i])
            gp_u = psi_dist_mean[i] + 1.96 * jnp.sqrt(psi_dist_var[i])
            gp_covs.append(float(jnp.mean((psi_true[i] >= gp_l) & (psi_true[i] <= gp_u)) * 100.0))
            gp_rmses.append(float(jnp.sqrt(jnp.mean((psi_dist_mean[i] - psi_true[i]) ** 2))))
            ss_tot = jnp.sum((psi_true[i] - jnp.mean(psi_true[i])) ** 2)
            gp_r2s.append(float(1 - jnp.sum((psi_true[i] - psi_dist_mean[i]) ** 2) / (ss_tot + 1e-12)))

            nf_l = jnp.percentile(dist_psi_samples[i], 2.5, axis=0)
            nf_u = jnp.percentile(dist_psi_samples[i], 97.5, axis=0)
            dist_m = dist_psi_samples[i].mean(axis=0)
            dist_covs.append(float(jnp.mean((psi_true[i] >= nf_l) & (psi_true[i] <= nf_u)) * 100.0))
            dist_rmses.append(float(jnp.sqrt(jnp.mean((dist_m - psi_true[i]) ** 2))))
            dist_r2s.append(float(1 - jnp.sum((psi_true[i] - dist_m) ** 2) / (ss_tot + 1e-12)))

        vdata["sef"]["gp"] = {
            "total": {
                "rmse": float(np.mean(gp_rmses)),
                "coverage": float(np.mean(gp_covs)),
                "r2": float(np.mean(gp_r2s))
            }
        }
        vdata["sef"]["dist"] = {
            "total": {
                "rmse": float(np.mean(dist_rmses)),
                "coverage": float(np.mean(dist_covs)),
                "r2": float(np.mean(dist_r2s))
            }
        }
        with open(val_json_path, "w") as f:
            json.dump(vdata, f, indent=4)
        
        # Also sync to fem_validation and seed_dir if they exist
        for parent_sub in ["fem_validation", ".."]:
            sync_p = os.path.join(distilled_dir, parent_sub, "validation_metrics.json")
            if os.path.exists(os.path.dirname(sync_p)):
                with open(sync_p, "w") as f:
                    json.dump(vdata, f, indent=4)
    except Exception as e:
        print(f"Warning: Failed saving SEF metrics to validation_metrics.json: {e}")

    # Parse disabled parameters from config if available
    import yaml
    disabled_params_names = set()
    try:
        with open(os.path.join(distilled_dir, "recipe_config.yaml"), "r") as f:
            recipe = yaml.safe_load(f)
            for k, names_list in [("dev_params_disabled", all_dev_names), 
                                  ("vol_params_disabled", all_vol_names), 
                                  ("aniso_params_disabled", all_aniso_names)]:
                disabled = recipe.get(k, [])
                if disabled is not None:
                    for idx in (disabled if isinstance(disabled, list) else [disabled]):
                        if idx < len(names_list):
                            disabled_params_names.add(names_list[idx].replace("$", "").replace("{", "").replace("}", "").replace("_", ""))
    except Exception:
        pass

    # 2. Unified Parameters & Invariant Sensitivity Figure (Left: Sensitivity + Violin, Right: 3 Invariant Plots + Legend)
    fig_width_params = 7.8
    h_params = 3.5
    fig_params = plt.figure(figsize=(fig_width_params, h_params))
    gs_master = fig_params.add_gridspec(1, 2, width_ratios=[0.76, 0.24], wspace=0.22, top=0.96, bottom=0.08, left=0.07, right=0.91)
    
    gs_left = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_master[0, 0], height_ratios=[1, 1.0], hspace=0.14)
    gs_right = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs_master[0, 1], hspace=0.22)
    
    ax_sens = fig_params.add_subplot(gs_left[0, 0])
    x_pos = np.arange(len(sorted_params))
    
    gt_label_added = False
    disabled_label_added = False
    for i, p in enumerate(sorted_params):
        clean_p = p.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
        if clean_p in true_params_set:
            label_gt = "Ground Truth" if not gt_label_added else ""
            ax_sens.axvspan(i - 0.25, i + 0.25, color='#E0E0E0', alpha=0.8, zorder=1, label=label_gt)
            gt_label_added = True
            
        if clean_p in disabled_params_names:
            label_dis = "Disabled" if not disabled_label_added else ""
            ax_sens.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.15, zorder=2, label=label_dis)
            disabled_label_added = True
            
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
    ax_sens.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1), top=12.0)
    ax_sens.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.2, label=f"Threshold ({args.sobol_threshold})")
    ax_sens.set_ylabel('Sobol Sensitivity', fontsize=6.8, labelpad=2)
    ax_sens.grid(False)
    
    ax_sens.set_xticks(x_pos)
    ax_sens.set_xticklabels([]) # Hide for sensitivity since violin shares it
    ax_sens.tick_params(axis='y', labelsize=5.8, pad=1)
    
    lines_1, labels_1 = ax_sens.get_legend_handles_labels()
    by_label_sens = dict(zip(labels_1, lines_1))
    
    # Legend placed inside ax_sens
    desired_sens_order = ["Ground Truth", f"Threshold ({args.sobol_threshold})", r"$\bar{S}_{\mathrm{T,d}}$", r"$\bar{S}_{\mathrm{T,v}}$", r"$\bar{S}_{\mathrm{T,a}}$", "Disabled"]
    sens_handles = [by_label_sens[k] for k in desired_sens_order if k in by_label_sens]
    sens_labels = [k for k in desired_sens_order if k in by_label_sens]
    
    ax_sens.legend(sens_handles, sens_labels, fontsize=5.0,
                   loc='upper right', bbox_to_anchor=(0.98, 0.96), ncol=3, frameon=True,
                   facecolor='white', framealpha=0.9, edgecolor='#cccccc',
                   handlelength=1.0, handletextpad=0.25, columnspacing=0.5, borderpad=0.25, labelspacing=0.25)
    
    plt.setp(ax_sens.get_xticklabels(), visible=False)

    # 3. Violin Plot (Row 2, Left Column)
    ax_viol = fig_params.add_subplot(gs_left[1, 0], sharex=ax_sens)

    for i, p in enumerate(sorted_params):
        clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
        is_active = sorted_tot_means[i] > args.sobol_threshold
        is_true = clean_p in true_params_set
        
        data = df[clean_p].values
        mean_val = np.mean(data)
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        true_val = true_val_dict.get(clean_p, 0.0)
        
        if param_types.get(p) == "dev":
            color = "#0072B2"
        elif param_types.get(p) == "vol":
            color = "#D55E00"
        else:
            color = "#CC79A7"
        
        if clean_p in disabled_params_names:
            ax_viol.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.15, zorder=0)
            ax_viol.plot(i, 0.0, marker='x', color='red', markersize=4, zorder=10)
        
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
            
            ax_viol.plot([i - 0.35, i + 0.35], [mean_val, mean_val], color=color, lw=1.5)
        
        if is_true:
            ax_viol.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=1.2, linestyle='--')
                
    ax_viol.set_xticks(range(len(sorted_params)))
    ax_viol.set_xticklabels(sorted_params, fontsize=6.2)
    ax_viol.tick_params(axis='x', pad=1)
    ax_viol.set_ylabel('Parameter Value', fontsize=6.8, labelpad=2)
    ax_viol.set_ylim([0, 2.05])
    ax_viol.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_viol.tick_params(axis='y', labelsize=5.8, pad=1)
    ax_viol.grid(False)

    viol_legend = [
        mpatches.Patch(color='gray', alpha=0.5, label='Density'),
        mlines.Line2D([0], [0], color='gray', lw=1.3, label='Mean'),
        mpatches.Patch(color='gray', alpha=0.12, label='95% CI'),
        mlines.Line2D([0], [0], color='black', lw=1.1, linestyle='--', label='Ground Truth')
    ]
    ax_viol.legend(handles=viol_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=2, fontsize=5.2, frameon=True,
                   facecolor='white', framealpha=0.85, edgecolor='none',
                   handlelength=1.0, handletextpad=0.25, columnspacing=0.5, borderpad=0.25, labelspacing=0.25)

    # 4. Invariant-dependent Sobol Sensitivity (Right Column: 3 subplots)
    ax_inv1 = fig_params.add_subplot(gs_right[0, 0])
    ax_inv2 = fig_params.add_subplot(gs_right[1, 0])
    ax_inv3 = fig_params.add_subplot(gs_right[2, 0])
    for ax_sub in [ax_inv1, ax_inv2, ax_inv3]:
        ax_sub.set_box_aspect(1)
    
    f3x3 = load_f3x3_from_distilled(distilled_dir)
    if f3x3 is not None:
        I1_bar, I2_bar, J = compute_invariants_np(f3x3)
        n_points = len(f3x3)

        df_dev_inv, _ = load_sobol_csv(distilled_dir, "dev_sensitivities")
        if df_dev_inv is None:
            df_dev_inv, _ = load_sobol_csv(distilled_dir, "sensitivities")
        df_vol_inv, _ = load_sobol_csv(distilled_dir, "vol_sensitivities")

        if df_dev_inv is not None and df_vol_inv is not None:
            df_dev_inv = df_dev_inv.iloc[:n_points]
            df_vol_inv = df_vol_inv.iloc[:n_points]

            exclude_cols = ['test cases', 'Unnamed: 0']
            dev_inv_cols = [c for c in df_dev_inv.columns if c not in exclude_cols and not c.startswith('Unnamed')]
            vol_inv_cols = [c for c in df_vol_inv.columns if c not in exclude_cols and not c.startswith('Unnamed')]

            active_dev_inv = [p for p in dev_inv_cols if df_dev_inv[p].max() > args.sobol_threshold]
            active_vol_inv = [p for p in vol_inv_cols if df_vol_inv[p].max() > args.sobol_threshold]

            if not active_dev_inv and dev_inv_cols:
                active_dev_inv = [dev_inv_cols[0]]
            if not active_vol_inv and vol_inv_cols:
                active_vol_inv = [vol_inv_cols[0]]

            # Panel 1: ST,d vs I1_bar
            for p in active_dev_inv:
                col = get_comp_color(p)
                ax_inv1.scatter(I1_bar, df_dev_inv[p].values, color=col, alpha=0.65, s=8, edgecolors='none')
            ax_inv1.set_xlabel(r"$\bar{I}_1$", fontsize=6.8, labelpad=1)
            ax_inv1.set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=7.0, labelpad=1)
            ax_inv1.set_ylim(-0.05, 1.05)
            ax_inv1.set_yticks([0.0, 0.5, 1.0])
            ax_inv1.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
            ax_inv1.grid(False)

            # Panel 2: ST,d vs I2_bar
            for p in active_dev_inv:
                col = get_comp_color(p)
                ax_inv2.scatter(I2_bar, df_dev_inv[p].values, color=col, alpha=0.65, s=8, edgecolors='none')
            ax_inv2.set_xlabel(r"$\bar{I}_2$", fontsize=6.8, labelpad=1)
            ax_inv2.set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=7.0, labelpad=1)
            ax_inv2.set_ylim(-0.05, 1.05)
            ax_inv2.set_yticks([0.0, 0.5, 1.0])
            ax_inv2.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
            ax_inv2.grid(False)

            # Panel 3: ST,v vs J
            for p in active_vol_inv:
                col = get_comp_color(p)
                ax_inv3.scatter(J, df_vol_inv[p].values, color=col, alpha=0.65, s=8, edgecolors='none')
            ax_inv3.set_xlabel(r"$J$", fontsize=6.8, labelpad=1)
            ax_inv3.set_ylabel(r"$S_{\mathrm{T,v}}$", fontsize=7.0, labelpad=1)
            ax_inv3.set_ylim(-0.05, 1.05)
            ax_inv3.set_yticks([0.0, 0.5, 1.0])
            ax_inv3.tick_params(axis='both', which='major', labelsize=5.5, pad=1)
            ax_inv3.grid(False)

            # Unified right-side column legend for invariant plots
            all_inv_active = list(dict.fromkeys(active_dev_inv + active_vol_inv))
            inv_handles = []
            inv_labels = []
            for p in all_inv_active:
                col = get_comp_color(p)
                inv_handles.append(mlines.Line2D([], [], color=col, marker='o', linestyle='none', markersize=4))
                inv_labels.append(to_latex(p))

            ax_inv2.legend(handles=inv_handles, labels=inv_labels, loc='center left', bbox_to_anchor=(1.05, 0.5),
                           ncol=1, fontsize=6.2, frameon=False, handletextpad=0.2, labelspacing=0.5)

            # --- Also generate a clean, standalone 1x3 horizontal publication figure for Invariant Sensitivities ---
            fig_inv_horiz, axes_inv_h = plt.subplots(1, 3, figsize=(6.8, 2.3), constrained_layout=True)
            for ax_h in axes_inv_h:
                ax_h.set_box_aspect(1)
                ax_h.grid(False)
                ax_h.set_ylim(-0.05, 1.05)
                ax_h.set_yticks([0.0, 0.5, 1.0])
                ax_h.tick_params(axis='both', which='major', labelsize=7.0, pad=1)

            for p in active_dev_inv:
                col = get_comp_color(p)
                axes_inv_h[0].scatter(I1_bar, df_dev_inv[p].values, color=col, alpha=0.65, s=10, edgecolors='none')
                axes_inv_h[1].scatter(I2_bar, df_dev_inv[p].values, color=col, alpha=0.65, s=10, edgecolors='none')
            for p in active_vol_inv:
                col = get_comp_color(p)
                axes_inv_h[2].scatter(J, df_vol_inv[p].values, color=col, alpha=0.65, s=10, edgecolors='none')

            axes_inv_h[0].set_xlabel(r"$\bar{I}_1$", fontsize=8.5, labelpad=2)
            axes_inv_h[0].set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=8.5, labelpad=2)
            axes_inv_h[1].set_xlabel(r"$\bar{I}_2$", fontsize=8.5, labelpad=2)
            axes_inv_h[1].set_ylabel(r"$S_{\mathrm{T,d}}$", fontsize=8.5, labelpad=2)
            axes_inv_h[2].set_xlabel(r"$J$", fontsize=8.5, labelpad=2)
            axes_inv_h[2].set_ylabel(r"$S_{\mathrm{T,v}}$", fontsize=8.5, labelpad=2)

            inv_h_handles = []
            inv_h_labels = []
            for p in all_inv_active:
                col = get_comp_color(p)
                inv_h_handles.append(mlines.Line2D([], [], color=col, marker='o', linestyle='none', markersize=5))
                inv_h_labels.append(to_latex(p))

            fig_inv_horiz.legend(handles=inv_h_handles, labels=inv_h_labels, loc='lower center',
                                 bbox_to_anchor=(0.5, -0.12), ncol=len(all_inv_active),
                                 fontsize=7.5, frameon=False, handletextpad=0.2, columnspacing=1.0)

            fig_inv_horiz.savefig(os.path.join(distilled_dir, f"invariant_sensitivities_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
            fig_inv_horiz.savefig(os.path.join(distilled_dir, f"invariant_sensitivities_{true_model_name}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig_inv_horiz)

    # Also generate a clean, standalone 2-row publication figure for Discovery & Violins only (without right column)
    fig_standalone_params, (ax_s_sens, ax_s_viol) = plt.subplots(2, 1, figsize=(5.8, 3.4), sharex=True,
                                                                  gridspec_kw={'height_ratios': [1, 1.0], 'hspace': 0.12})
    
    # Clone ax_sens bars into ax_s_sens
    gt_label_added = False
    disabled_label_added = False
    for i, p in enumerate(sorted_params):
        clean_p = p.replace('$', '').replace('{', '').replace('}', '').replace('_', '')
        if clean_p in true_params_set:
            label_gt = "Ground Truth" if not gt_label_added else ""
            ax_s_sens.axvspan(i - 0.25, i + 0.25, color='#E0E0E0', alpha=0.8, zorder=1, label=label_gt)
            gt_label_added = True
        if clean_p in disabled_params_names:
            label_dis = "Disabled" if not disabled_label_added else ""
            ax_s_sens.axvspan(i - 0.4, i + 0.4, color='red', alpha=0.15, zorder=2, label=label_dis)
            disabled_label_added = True
        color = "#0072B2" if param_types.get(p) == "dev" else ("#D55E00" if param_types.get(p) == "vol" else "#CC79A7")
        label = r"$\bar{S}_{\mathrm{T,d}}$" if param_types.get(p) == "dev" else (r"$\bar{S}_{\mathrm{T,v}}$" if param_types.get(p) == "vol" else r"$\bar{S}_{\mathrm{T,a}}$")
        h_list, l_list = ax_s_sens.get_legend_handles_labels()
        if label not in l_list:
            ax_s_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3, label=label)
        else:
            ax_s_sens.bar(x_pos[i], sorted_tot_means[i], width=0.5, color=color, alpha=0.9, zorder=3)
            
    ax_s_sens.set_yscale('log')
    ax_s_sens.set_ylim(bottom=max(1e-5, args.sobol_threshold * 0.1), top=12.0)
    ax_s_sens.axhline(args.sobol_threshold, color='black', linestyle='--', linewidth=1.2, label=f"Threshold ({args.sobol_threshold})")
    ax_s_sens.set_ylabel('Sobol Sensitivity', fontsize=7.5, labelpad=2)
    ax_s_sens.grid(False)
    ax_s_sens.tick_params(axis='y', labelsize=6.8, pad=1)
    
    l1_s, lab1_s = ax_s_sens.get_legend_handles_labels()
    by_label_s = dict(zip(lab1_s, l1_s))
    order_s = ["Ground Truth", f"Threshold ({args.sobol_threshold})", r"$\bar{S}_{\mathrm{T,d}}$", r"$\bar{S}_{\mathrm{T,v}}$", r"$\bar{S}_{\mathrm{T,a}}$", "Disabled"]
    ax_s_sens.legend([by_label_s[k] for k in order_s if k in by_label_s],
                     [k for k in order_s if k in by_label_s], fontsize=5.8,
                     loc='upper right', bbox_to_anchor=(0.98, 0.96), ncol=3, frameon=True,
                     facecolor='white', framealpha=0.9, edgecolor='#cccccc',
                     handlelength=1.0, handletextpad=0.25, columnspacing=0.6, borderpad=0.25, labelspacing=0.25)
    plt.setp(ax_s_sens.get_xticklabels(), visible=False)

    # Clone violins into ax_s_viol
    for i, p in enumerate(sorted_params):
        clean_p = p.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
        is_active = sorted_tot_means[i] > args.sobol_threshold
        is_true = clean_p in true_params_set
        data = df[clean_p].values
        mean_val = np.mean(data)
        ci_lower = np.percentile(data, 2.5)
        ci_upper = np.percentile(data, 97.5)
        true_val = true_val_dict.get(clean_p, 0.0)
        color = "#0072B2" if param_types.get(p) == "dev" else ("#D55E00" if param_types.get(p) == "vol" else "#CC79A7")

        if is_active:
            ax_s_viol.axvspan(i - 0.35, i + 0.35, ymin=ci_lower/2.05, ymax=ci_upper/2.05, color=color, alpha=0.15, edgecolor='none')
            counts, bin_edges = np.histogram(data, bins=25)
            if np.max(counts) > 0:
                counts = counts / np.max(counts) * 0.4
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            b_height = bin_edges[1] - bin_edges[0]
            ax_s_viol.barh(bin_centers, counts, height=b_height, left=i - counts/2, color=color, alpha=0.5, edgecolor='none')
            ax_s_viol.plot([i - 0.35, i + 0.35], [mean_val, mean_val], color=color, lw=1.5)
        if is_true:
            ax_s_viol.plot([i - 0.35, i + 0.35], [true_val, true_val], color='black', lw=1.2, linestyle='--')

    ax_s_viol.set_xticks(range(len(sorted_params)))
    ax_s_viol.set_xticklabels(sorted_params, fontsize=7.2)
    ax_s_viol.tick_params(axis='x', pad=1)
    ax_s_viol.set_ylabel('Parameter Value', fontsize=7.5, labelpad=2)
    ax_s_viol.set_ylim([0, 2.05])
    ax_s_viol.set_yticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax_s_viol.tick_params(axis='y', labelsize=6.8, pad=1)
    ax_s_viol.grid(False)
    ax_s_viol.legend(handles=viol_legend, loc='upper right', bbox_to_anchor=(0.98, 0.98), ncol=2, fontsize=6.0, frameon=True,
                     facecolor='white', framealpha=0.85, edgecolor='none',
                     handlelength=1.0, handletextpad=0.25, columnspacing=0.6, borderpad=0.25, labelspacing=0.25)

    fig_standalone_params.tight_layout()
    fig_standalone_params.savefig(os.path.join(distilled_dir, f"parameter_identification_{true_model_name}.pdf"), dpi=300, bbox_inches='tight')
    fig_standalone_params.savefig(os.path.join(distilled_dir, f"parameter_identification_{true_model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_standalone_params)
    
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
            for g_dev, p_dev, is_entry in dev_info['crossings']:
                subscript = r"\mathrm{min}" if is_entry else r"\mathrm{max}"
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
            for g_vol, p_vol, is_entry in vol_info['crossings']:
                subscript = r"\mathrm{min}" if is_entry else r"\mathrm{max}"
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
    fig_space.savefig(os.path.join(distilled_dir, f"split_invariant_spaces_{true_model_name}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig_space)
    
    print(f"Saved split summary plots to {distilled_dir}")

if __name__ == "__main__":
    main()

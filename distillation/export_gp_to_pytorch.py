import sys
import os
import argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.utils import fto3x3, farthest_point_sampling
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor

def generate_standard_modes(num_points=32, max_gamma=1.0):
    gamma = np.linspace(0.0, max_gamma, num_points)
    
    F_all = np.zeros((6, num_points, 2, 2))
    def set_F(f11, f22, f12=0.0):
        arr = np.zeros((num_points, 2, 2))
        arr[:, 0, 0] = f11
        arr[:, 1, 1] = f22
        arr[:, 0, 1] = f12
        return arr

    F_all[0] = set_F(1 + gamma, 1.0)            
    F_all[1] = set_F(1 + gamma, 1 + gamma)    
    F_all[2] = set_F(1 + gamma, 1/(1 + gamma)) 
    F_all[3] = set_F(1/(1 + gamma), 1.0)       
    F_all[4] = set_F(1/(1 + gamma), 1/(1 + gamma)) 
    F_all[5] = set_F(1.0, 1.0, f12=gamma)
    
    # We want to return a flat array (192, 2, 2)
    return F_all.reshape(-1, 2, 2)

def generate_standard_modes_interp(num_points=32, max_search_gamma=1.0, min_dev=None, max_dev=None, min_vol=None, max_vol=None):
    search_points = 10000
    gamma_search = np.linspace(0.0, max_search_gamma, search_points)
    
    def get_mode_F(mode_idx, g_arr):
        n = len(g_arr)
        arr = np.zeros((n, 2, 2))
        arr[:, 0, 0] = 1.0
        arr[:, 1, 1] = 1.0
        if mode_idx == 0:
            arr[:, 0, 0] = 1 + g_arr
        elif mode_idx == 1:
            arr[:, 0, 0] = 1 + g_arr
            arr[:, 1, 1] = 1 + g_arr
        elif mode_idx == 2:
            arr[:, 0, 0] = 1 + g_arr
            arr[:, 1, 1] = 1.0 / (1 + g_arr)
        elif mode_idx == 3:
            arr[:, 0, 0] = 1.0 / (1 + g_arr)
        elif mode_idx == 4:
            arr[:, 0, 0] = 1.0 / (1 + g_arr)
            arr[:, 1, 1] = 1.0 / (1 + g_arr)
        elif mode_idx == 5:
            arr[:, 0, 1] = g_arr
        return arr

    F_sampled = np.zeros((6, num_points, 2, 2))
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]
                  
    true_min_dev = np.array(min_dev) - 1e-4
    true_max_dev = np.array(max_dev) + 1e-4
    true_min_vol = np.array(min_vol) - 1e-4
    true_max_vol = np.array(max_vol) + 1e-4
    
    extractor = IsotropicFeatureExtractor()
    
    print(f"\n--- Dynamically determining interpolation transition points (gamma in [0, {max_search_gamma}]) ---")
    for i in range(6):
        F_search_2x2 = get_mode_F(i, gamma_search)
        F_search_3x3 = np.zeros((search_points, 3, 3))
        F_search_3x3[:, :2, :2] = F_search_2x2
        F_search_3x3[:, 2, 2] = 1.0
        
        dev_m, vol_m = jax.vmap(extractor.extract)(jnp.array(F_search_3x3))
        dev_m, vol_m = np.array(dev_m), np.array(vol_m)
        
        in_bounds_dev0 = (dev_m[:, 0] >= true_min_dev[0]) & (dev_m[:, 0] <= true_max_dev[0])
        in_bounds_dev1 = (dev_m[:, 1] >= true_min_dev[1]) & (dev_m[:, 1] <= true_max_dev[1])
        in_bounds_vol = (vol_m[:, 0] >= true_min_vol[0]) & (vol_m[:, 0] <= true_max_vol[0])
        in_bounds = in_bounds_dev0 & in_bounds_dev1 & in_bounds_vol
        
        if not np.all(in_bounds):
            exit_idx = np.argmax(~in_bounds)
            trans_g = gamma_search[exit_idx]
            if exit_idx == 0:
                trans_g = gamma_search[1]
            print(f"Mode {i} ({mode_names[i]}): Interpolation region ends at gamma = {trans_g:.4f}")
        else:
            trans_g = max_search_gamma
            print(f"Mode {i} ({mode_names[i]}): Entirely within interpolation up to gamma = {trans_g:.4f}")
            
        gamma_mode = np.linspace(0.0, trans_g, num_points)
        F_sampled[i] = get_mode_F(i, gamma_mode)
        
    return F_sampled.reshape(-1, 2, 2)

def invariants(f):
    F = fto3x3(f)
    C = F.T @ F
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    J = jnp.linalg.det(F)
    return jnp.array([I1, I2, J])

jax.config.update("jax_enable_x64", True)

def get_F_from_invariants(I1_bar, I2_bar, J):
    coeffs = [1.0, -I1_bar, I2_bar, -1.0]
    roots = np.roots(coeffs)
    lambda_sq = np.real(roots)
    lambda_sq = np.maximum(lambda_sq, 1e-8)
    lambdas = np.sqrt(lambda_sq) * (J**(1/3))
    return np.diag(lambdas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--max_gamma", type=float, default=0.8)
    parser.add_argument("--sample_mode", type=str, default="dataset_f", choices=["standard", "standard_interp", "dataset_f", "dataset_all", "inducing_points"], help="Sample deformations from standard modes (with or without interpolation clipping), extraction dataset with FPS, all extraction dataset points, or directly from inducing points.")
    parser.add_argument("--num_points", type=int, default=192, help="Number of points to evaluate GP over.")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"], help="Distillation target mode: solely Strain Energy Function (sef), joint SEF + Piola stress (sef_stress), joint SEF + Cauchy stress (sef_cauchy), or separate DEV and VOL energy (sef_split).")
    parser.add_argument("--export_subfolder", type=str, default="", help="Custom output subfolder for exported PyTorch matrices.")
    args = parser.parse_args()

    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(args.saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    if I_z.shape[1] > 3:
        vol_z = I_z[:, 2:3]
        aniso_z = I_z[:, 3:]
    elif I_z.shape[1] == 3:
        vol_z = I_z[:, 2:3]
        aniso_z = None
    else:
        vol_z = I_z[:, 2:]
        aniso_z = None

    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    min_aniso = jnp.min(aniso_z, axis=0) if aniso_z is not None else None
    max_aniso = jnp.max(aniso_z, axis=0) if aniso_z is not None else None

    model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = "isihara"
    if len(parts) > 1 and parts[1] in ["ortho45", "symnonortho60", "aniso30", "isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4", "c20d10d05", "c20_d10_d05"]:
        true_model_name = parts[1]
    else:
        for p in ["ortho45", "symnonortho60", "aniso30", "isihara", "nh", "neohookean2", "nh2", "gentthomas", "nh4", "neohookean4", "c20d10d05", "c20_d10_d05"]:
            if p in parts or p in model_folder_name.lower():
                true_model_name = p
                break
    from core.material_models import get_material
    try:
        true_model = get_material(true_model_name, jit_P=False)
    except Exception:
        true_model = None

    feature_extractor = None
    if aniso_z is not None:
        if true_model is not None and hasattr(true_model, 'a1') and hasattr(true_model, 'a2'):
            a0 = np.array(true_model.a1)
            a1 = np.array(true_model.a2)
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        elif true_model is not None and hasattr(true_model, 'a0'):
            a0 = np.array(true_model.a0)
            feature_extractor = AnisotropicFeatureExtractor(a0)
        elif getattr(gp_params, "raw_aniso_theta_mean", None) is not None:
            raw_th = gp_params.raw_aniso_theta_mean
            theta = float(np.pi * (1.0 / (1.0 + np.exp(-raw_th)) - 0.5))
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

    
    import json
    metadata_path = os.path.join(args.saved_model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            cov_mode = json.load(f).get("covariance_mode", "diag")
    else:
        cov_mode = "full" if gp_params.raw_dev_u_var.ndim == 2 else "diag"
        
    gp_model = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode
    )

    
    # Try to load the dataset for background plotting and dataset_* modes
    dataset_F_flat_2x2 = None
    try:
        model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
        parent_folder_name = os.path.basename(os.path.dirname(os.path.normpath(args.saved_model_dir)))
        
        parts = model_folder_name.split('_')
        if len(parts) < 6 and '_' in parent_folder_name:
            parts = parent_folder_name.split('_')

        if len(parts) >= 4:
            ugp_model_name = parts[1]
            disp_noise = parts[2]
            load_noise = parts[3]
            
            prep_dataset_path = None
            for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
                if os.path.exists(search_dir):
                    for fname in os.listdir(search_dir):
                        if fname.startswith(f"{ugp_model_name}_{disp_noise}_{load_noise}") and fname.endswith(".npz"):
                            prep_dataset_path = os.path.join(search_dir, fname)
                            break
                if prep_dataset_path is not None:
                    break
                
            if prep_dataset_path is not None:
                prep_data = np.load(prep_dataset_path, allow_pickle=True)
                F_all_steps_2x2 = prep_data["F"]
                
                log_file = os.path.join(args.saved_model_dir, "optimization_log.txt")
                load_steps = None
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as lf:
                        first_line = lf.readline()
                        if "[" in first_line and "]" in first_line:
                            steps_str = first_line.split("]")[0].split("[")[1].strip()
                            if steps_str:
                                load_steps = [int(x.strip()) for x in steps_str.split(",") if x.strip().isdigit()]
                
                if load_steps and len(load_steps) > 0 and max(load_steps) < F_all_steps_2x2.shape[0]:
                    F_train_full_2x2 = F_all_steps_2x2[load_steps]
                else:
                    default_steps = [2, 10, 20]
                    valid_steps = [s for s in default_steps if s < F_all_steps_2x2.shape[0]]
                    F_train_full_2x2 = F_all_steps_2x2[valid_steps] if len(valid_steps) > 0 else F_all_steps_2x2
                    
                dataset_F_flat_2x2 = F_train_full_2x2.reshape(-1, 2, 2)
    except Exception as e:
        print(f"Could not load background dataset for plotting: {e}")

    # Generate points
    if args.sample_mode in ["dataset_f", "dataset_all"]:
        if dataset_F_flat_2x2 is None:
            raise ValueError(f"Dataset loading failed, cannot use {args.sample_mode}.")
        F_flat_2x2 = dataset_F_flat_2x2
        
        if args.sample_mode == "dataset_all":
            print(f"Using exactly ALL {len(F_flat_2x2)} observed deformation points from extraction load steps (no FPS!).")
            f3x3_flat_2x2 = F_flat_2x2
            export_subfolder = "pytorch_export_dataset_all"
        else:
            print(f"Applying Farthest Point Sampling (FPS) over {len(F_flat_2x2)} observed deformations...")
            pts = jnp.array(F_flat_2x2.reshape(-1, 4), dtype=jnp.float64)
            if len(F_flat_2x2) <= args.num_points:
                indices = np.arange(len(F_flat_2x2))
            else:
                indices = np.array(farthest_point_sampling(pts, args.num_points))
                
            f3x3_flat_2x2 = F_flat_2x2[indices]
            export_subfolder = "pytorch_export_dataset_f"
            print(f"Sampled {len(indices)} deformations directly from extraction dataset via Farthest Point Sampling.")
    elif args.sample_mode == "standard_interp":
        print(f"Generating standard deformation modes strictly within GP interpolation bounds (up to gamma = {args.max_gamma})...")
        f3x3_flat_2x2 = generate_standard_modes_interp(num_points=max(1, args.num_points // 6), max_search_gamma=args.max_gamma, min_dev=min_dev, max_dev=max_dev, min_vol=min_vol, max_vol=max_vol)
        export_subfolder = "pytorch_export_standard_interp"
    elif args.sample_mode == "inducing_points":
        print(f"Generating F directly from the {len(I_z)} GP inducing points...")
        f3x3_list = []
        for i in range(len(I_z)):
            I1_bar = I_z[i, 0]
            I2_bar = I_z[i, 1]
            J = I_z[i, 2]
            f3x3_list.append(get_F_from_invariants(I1_bar, I2_bar, J))
        f3x3_flat = np.stack(f3x3_list)
        export_subfolder = "pytorch_export_inducing_points"
    else:
        f3x3_flat_2x2 = generate_standard_modes(num_points=max(1, args.num_points // 6), max_gamma=args.max_gamma)
        export_subfolder = f"pytorch_export_standard_g{args.max_gamma}" if args.max_gamma != 0.8 else "pytorch_export"
    
    if args.export_subfolder:
        export_subfolder = args.export_subfolder
    elif args.distill_target in ["sef_stress", "sef_cauchy"]:
        export_subfolder = f"{export_subfolder}_{args.distill_target}"
    
    if args.sample_mode != "inducing_points":
        # Pad to 3x3 Plane Strain!
        f3x3_flat = np.zeros((f3x3_flat_2x2.shape[0], 3, 3))
        for i in range(f3x3_flat_2x2.shape[0]):
            f3x3_flat[i, :2, :2] = f3x3_flat_2x2[i]
            f3x3_flat[i, 2, 2] = 1.0
            
    f3x3_flat = jnp.array(f3x3_flat)
    
    if args.distill_target == "sef":
        mean_psi = gp_model.psi_gp_mean(f3x3_flat)
        cov_psi = gp_model.psi_joint_cov(f3x3_flat)
        mean_psi = np.array(mean_psi)
        cov_psi = np.array(cov_psi)
        batch_size = mean_psi.shape[0]
        
        # Rebuild perfectly smooth positive-definite matrix
        cov_psi = 0.5 * (cov_psi + cov_psi.T)
        w, v = np.linalg.eigh(cov_psi)
        w = np.clip(w, a_min=1e-8, a_max=None)
        cov_psi = v @ np.diag(w) @ v.T
    elif args.distill_target == "sef_split":
        feats = jax.vmap(gp_model.feature_extractor.extract)(f3x3_flat)
        dev_feats, vol_feats = feats[0], feats[1]
        
        mean_dev = np.array(gp_model.dev_gp_mean(dev_feats))
        cov_dev = np.array(gp_model.dev_psi_joint_cov(f3x3_flat))
        
        mean_vol = np.array(gp_model.vol_gp_mean(vol_feats))
        cov_vol = np.array(gp_model.vol_psi_joint_cov(f3x3_flat))
        
        cov_dev = 0.5 * (cov_dev + cov_dev.T)
        w, v = np.linalg.eigh(cov_dev)
        w = np.clip(w, a_min=1e-8, a_max=None)
        cov_dev = v @ np.diag(w) @ v.T

        cov_vol = 0.5 * (cov_vol + cov_vol.T)
        w, v = np.linalg.eigh(cov_vol)
        w = np.clip(w, a_min=1e-8, a_max=None)
        cov_vol = v @ np.diag(w) @ v.T

        if gp_model.is_anisotropic:
            aniso_feats = feats[2]
            mean_aniso = np.array(gp_model.aniso_gp_mean(aniso_feats))
            cov_aniso = np.array(gp_model.aniso_psi_joint_cov(f3x3_flat))
            cov_aniso = 0.5 * (cov_aniso + cov_aniso.T)
            w, v = np.linalg.eigh(cov_aniso)
            w = np.clip(w, a_min=1e-8, a_max=None)
            cov_aniso = v @ np.diag(w) @ v.T
    elif args.distill_target in ["sef_stress", "sef_cauchy"]:
        print(f"Drawing 2048 GP Pathwise realizations for joint SEF + {args.distill_target.upper()} covariance estimation over {f3x3_flat.shape[0]} points...")
        keys = jax.random.split(jax.random.PRNGKey(42), 2048)
        
        def sample_joint(key):
            path_psi = gp_model.get_path_psi_fn(key)
            psi_val = jax.vmap(path_psi)(f3x3_flat)
            piola_val = jax.vmap(jax.grad(path_psi))(f3x3_flat)
            if args.distill_target == "sef_cauchy":
                J_val = jnp.linalg.det(f3x3_flat).reshape(-1, 1, 1)
                stress_val = (piola_val @ f3x3_flat.transpose(0, 2, 1)) / J_val
            else:
                stress_val = piola_val
            p00 = stress_val[:, 0, 0]
            p11 = stress_val[:, 1, 1]
            p01 = stress_val[:, 0, 1]
            return jnp.stack([psi_val, p00, p11, p01], axis=0).reshape(-1)
            
        sample_matrix = np.array(jax.jit(jax.vmap(sample_joint))(keys), dtype=np.float64)
        mean_psi = np.mean(sample_matrix, axis=0)
        cov_psi = np.cov(sample_matrix, rowvar=False)
        batch_size = mean_psi.shape[0]
        cov_psi = cov_psi + 1e-6 * np.eye(batch_size)

    # Save to disk
    out_dir = os.path.join(args.saved_model_dir, export_subfolder)
    os.makedirs(out_dir, exist_ok=True)
    
    if args.distill_target == "sef_split":
        np.save(os.path.join(out_dir, "mean_dev.npy"), np.array(mean_dev))
        np.save(os.path.join(out_dir, "cov_dev.npy"), np.array(cov_dev))
        np.save(os.path.join(out_dir, "mean_vol.npy"), np.array(mean_vol))
        np.save(os.path.join(out_dir, "cov_vol.npy"), np.array(cov_vol))
        if gp_model.is_anisotropic:
            np.save(os.path.join(out_dir, "mean_aniso.npy"), np.array(mean_aniso))
            np.save(os.path.join(out_dir, "cov_aniso.npy"), np.array(cov_aniso))
        np.save(os.path.join(out_dir, "f3x3.npy"), np.array(f3x3_flat))
        print(f"Exported GP Target Mean and Cov for DEV, VOL (and ANISO if present) to {out_dir}")

    else:
        np.save(os.path.join(out_dir, "mean_psi.npy"), np.array(mean_psi))
        np.save(os.path.join(out_dir, "cov_psi.npy"), np.array(cov_psi))
        np.save(os.path.join(out_dir, "f3x3.npy"), np.array(f3x3_flat))
        print(f"Exported GP Target Mean ({mean_psi.shape}) and Cov ({cov_psi.shape}) to {out_dir}")
    
    # Automatically generate the GP sample plot if it's the SEF target
    if args.distill_target == "sef":
        import subprocess
        print(f"Automatically generating GP sample visualizations for {out_dir}...")
        try:
            model_name = os.path.basename(os.path.normpath(args.saved_model_dir)).split("_")[1]
            subprocess.run(["python3", "plots/plot_gp_samples.py", "--export_dir", out_dir, "--model_name", model_name], check=True)
        except Exception as e:
            print(f"Failed to automatically plot GP samples: {e}")

    # Plot GP Posterior vs Ground Truth per component
    if true_model is not None:
        try:
            from core.plotter import plot_energy_decomposition_validation
            print(f"Generating energy decomposition validation plot (GP Posterior vs Ground Truth) for {out_dir}...")
            plot_energy_decomposition_validation(gp_model, true_model, out_dir)
            print("Successfully saved energy_decomposition.pdf in export directory.")
        except Exception as e:
            print(f"Failed to plot energy decomposition validation: {e}")

    # === ADDITIONAL EXPORT PLOTS ===
    import matplotlib.pyplot as plt
    print("Generating export summary plots (Invariant Space & Energy Distributions)...")
    try:
        extractor = IsotropicFeatureExtractor()
        dev_feat, vol_feat = jax.vmap(extractor.extract)(f3x3_flat)
        dev_feat, vol_feat = np.array(dev_feat), np.array(vol_feat)
        
        # 1. Invariant Space Plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        if dataset_F_flat_2x2 is not None:
            ds_f3x3 = np.zeros((dataset_F_flat_2x2.shape[0], 3, 3))
            ds_f3x3[:, :2, :2] = dataset_F_flat_2x2
            ds_f3x3[:, 2, 2] = 1.0
            ds_dev, ds_vol = jax.vmap(extractor.extract)(jnp.array(ds_f3x3))
            ds_dev, ds_vol = np.array(ds_dev), np.array(ds_vol)
            
            ds_i1_m3 = ds_dev[:, 0] - 3.0
            ds_i2_m3 = ds_dev[:, 1] - 3.0
            ds_j_m1_sq = (ds_vol[:, 0] - 1.0)**2
            
            axes[0].scatter(ds_i1_m3, ds_i2_m3, c='gray', alpha=0.3, s=10, label='Extraction Dataset', marker='s')
            axes[1].scatter(ds_i1_m3, ds_j_m1_sq, c='gray', alpha=0.3, s=10, label='Extraction Dataset', marker='s')
            axes[2].scatter(ds_i2_m3, ds_j_m1_sq, c='gray', alpha=0.3, s=10, label='Extraction Dataset', marker='s')
            
        i1_m3 = dev_feat[:, 0] - 3.0
        i2_m3 = dev_feat[:, 1] - 3.0
        j_m1_sq = (vol_feat[:, 0] - 1.0)**2
        
        axes[0].scatter(i1_m3, i2_m3, c='red', s=25, label=f'Export Points ({args.sample_mode})', zorder=5, marker='x')
        axes[1].scatter(i1_m3, j_m1_sq, c='red', s=25, label=f'Export Points ({args.sample_mode})', zorder=5, marker='x')
        axes[2].scatter(i2_m3, j_m1_sq, c='red', s=25, label=f'Export Points ({args.sample_mode})', zorder=5, marker='x')
        
        axes[0].set_xlabel("$\\bar{I}_1 - 3$")
        axes[0].set_ylabel("$\\bar{I}_2 - 3$")
        axes[0].set_title("Deviatoric Space")
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        axes[1].set_xlabel("$\\bar{I}_1 - 3$")
        axes[1].set_ylabel("$(J - 1)^2$")
        axes[1].set_title("$\\bar{I}_1$ vs Volumetric")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        axes[2].set_xlabel("$\\bar{I}_2 - 3$")
        axes[2].set_ylabel("$(J - 1)^2$")
        axes[2].set_title("$\\bar{I}_2$ vs Volumetric")
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "export_invariant_space.pdf"), dpi=150)
        plt.close(fig)
        
        # 2. Distribution Plot
        feats_all = jax.vmap(gp_model.feature_extractor.extract)(f3x3_flat)
        dev_feat, vol_feat = feats_all[0], feats_all[1]
        
        dev_psi_mean = np.array(jax.vmap(gp_model.dev_gp_mean)(dev_feat))
        vol_psi_mean = np.array(jax.vmap(gp_model.vol_gp_mean)(vol_feat))
        
        if gp_model.is_anisotropic:
            aniso_feat = feats_all[2]
            aniso_psi_mean = np.array(jax.vmap(gp_model.aniso_gp_mean)(aniso_feat))
            total_psi_mean = dev_psi_mean + vol_psi_mean + aniso_psi_mean
            
            fig, axes = plt.subplots(1, 4, figsize=(20, 4))
            
            axes[0].hist(total_psi_mean, bins=30, color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_title("Total Mean Energy Distribution")
            axes[0].set_xlabel("Strain Energy (SEF)")
            axes[0].set_ylabel("Count")
            
            axes[1].hist(dev_psi_mean, bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[1].set_title("Deviatoric Mean Energy Distribution")
            axes[1].set_xlabel("Deviatoric Energy")
            
            axes[2].hist(vol_psi_mean, bins=30, color='green', alpha=0.7, edgecolor='black')
            axes[2].set_title("Volumetric Mean Energy Distribution")
            axes[2].set_xlabel("Volumetric Energy")

            axes[3].hist(aniso_psi_mean, bins=30, color='orange', alpha=0.7, edgecolor='black')
            axes[3].set_title("Anisotropic Mean Energy Distribution")
            axes[3].set_xlabel("Anisotropic Energy")
        else:
            total_psi_mean = dev_psi_mean + vol_psi_mean
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            
            axes[0].hist(total_psi_mean, bins=30, color='blue', alpha=0.7, edgecolor='black')
            axes[0].set_title("Total Mean Energy Distribution")
            axes[0].set_xlabel("Strain Energy (SEF)")
            axes[0].set_ylabel("Count")
            
            axes[1].hist(dev_psi_mean, bins=30, color='purple', alpha=0.7, edgecolor='black')
            axes[1].set_title("Deviatoric Mean Energy Distribution")
            axes[1].set_xlabel("Deviatoric Energy")
            
            axes[2].hist(vol_psi_mean, bins=30, color='green', alpha=0.7, edgecolor='black')
            axes[2].set_title("Volumetric Mean Energy Distribution")
            axes[2].set_xlabel("Volumetric Energy")
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "export_energy_distribution.pdf"), dpi=150)
        plt.close(fig)
        print("Successfully saved invariant space and energy distribution plots.")

    except Exception as e:
        print(f"Failed to generate extra export plots: {e}")

if __name__ == "__main__":
    main()

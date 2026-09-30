import sys
import os
import argparse
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.utils import fto3x3, farthest_point_sampling, stratified_high_strain_fps, uniform_energy_fps, compute_invariants_np, infer_material_model_name
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
import json
import datetime

def verify_and_report_export(out_dir, components_dict, f3x3_flat, sample_mode="dataset_f", distill_target="sef_split"):
    """
    Verifies exported GP posterior arrays for PyTorch distillation:
    1. Checks for NaNs and Infs in F, mean, and covariance matrices.
    2. Checks covariance matrix symmetry: max|Sigma - Sigma^T| < 1e-4.
    3. Checks strict positive-definiteness: min(eigvalsh(Sigma)) > 0.
    4. Computes and saves precomputed Cholesky factor L = chol(Sigma) as L_{name}.npy.
    5. Writes export_verification_report.json and export_verification_report.txt.
    6. If any check fails, logs full report and raises RuntimeError immediately to halt pipeline execution.
    """
    errors = []
    component_stats = {}
    all_healthy = True

    f_arr = np.asarray(f3x3_flat)
    if np.isnan(f_arr).any() or np.isinf(f_arr).any():
        errors.append("Deformation gradient tensor f3x3 contains NaN or Inf values")
        all_healthy = False

    for comp_name, comp_data in components_dict.items():
        mean = np.asarray(comp_data["mean"])
        cov = np.asarray(comp_data["cov"])
        comp_errors = []

        # 1. Finite check
        has_nan_mean = bool(np.isnan(mean).any() or np.isinf(mean).any())
        has_nan_cov = bool(np.isnan(cov).any() or np.isinf(cov).any())
        if has_nan_mean:
            comp_errors.append("Mean vector contains NaN or Inf values")
        if has_nan_cov:
            comp_errors.append("Covariance matrix contains NaN or Inf values")

        # 2. Symmetry check
        sym_res = float(np.max(np.abs(cov - cov.T)))
        if sym_res > 1e-4:
            comp_errors.append(f"Covariance asymmetry residual {sym_res:.2e} exceeds tolerance 1e-4")

        # 3. Eigenvalue and positive-definiteness check
        try:
            cov_sym = 0.5 * (cov + cov.T)
            eigvals = np.linalg.eigvalsh(cov_sym)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            cond_num = float(max_eig / min_eig) if min_eig > 0 else float("inf")
            if min_eig <= 0.0:
                comp_errors.append(f"Covariance is not positive definite: min eigenvalue = {min_eig:.3e} <= 0")
        except Exception as e:
            min_eig, max_eig, cond_num = None, None, None
            comp_errors.append(f"Eigenvalue calculation failed: {e}")

        # 4. Cholesky factorization and caching
        cholesky_success = False
        L_filename = f"L_{comp_name}.npy"
        L_path = os.path.join(out_dir, L_filename)
        try:
            cov_sym = 0.5 * (cov + cov.T)
            L = np.linalg.cholesky(cov_sym)
            np.save(L_path, L)
            cholesky_success = True
        except Exception as e:
            comp_errors.append(f"Cholesky factorization failed: {e}")

        comp_healthy = (len(comp_errors) == 0)
        if not comp_healthy:
            all_healthy = False
            for err in comp_errors:
                errors.append(f"[{comp_name}] {err}")

        component_stats[comp_name] = {
            "shape": list(cov.shape),
            "min_eigenvalue": min_eig,
            "max_eigenvalue": max_eig,
            "condition_number": cond_num,
            "symmetry_residual": sym_res,
            "has_nan_or_inf": has_nan_mean or has_nan_cov,
            "cholesky_success": cholesky_success,
            "cholesky_file": L_filename if cholesky_success else None,
            "healthy": comp_healthy,
            "errors": comp_errors
        }

    # 5. Build and save verification reports
    report = {
        "status": "PASSED" if all_healthy else "FAILED",
        "overall_healthy": all_healthy,
        "timestamp": datetime.datetime.now().isoformat(),
        "export_dir": os.path.abspath(out_dir),
        "sample_mode": sample_mode,
        "distill_target": distill_target,
        "num_points": int(f_arr.shape[0]),
        "components": component_stats,
        "errors": errors
    }

    json_path = os.path.join(out_dir, "export_verification_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    txt_path = os.path.join(out_dir, "export_verification_report.txt")
    with open(txt_path, "w") as f:
        f.write("=" * 65 + "\n")
        f.write(f"GP EXPORT VERIFICATION REPORT: {report['status']}\n")
        f.write("=" * 65 + "\n")
        f.write(f"Timestamp:      {report['timestamp']}\n")
        f.write(f"Export Dir:     {report['export_dir']}\n")
        f.write(f"Sample Mode:    {sample_mode} (N={report['num_points']})\n")
        f.write(f"Distill Target: {distill_target}\n")
        f.write("-" * 65 + "\n")
        for cname, cinfo in component_stats.items():
            f.write(f"Component: {cname.upper()}\n")
            f.write(f"  Shape:             {cinfo['shape']}\n")
            min_e = f"{cinfo['min_eigenvalue']:.3e}" if cinfo['min_eigenvalue'] is not None else "N/A"
            max_e = f"{cinfo['max_eigenvalue']:.3e}" if cinfo['max_eigenvalue'] is not None else "N/A"
            c_num = f"{cinfo['condition_number']:.2e}" if cinfo['condition_number'] is not None else "N/A"
            f.write(f"  Min Eigenvalue:    {min_e}\n")
            f.write(f"  Max Eigenvalue:    {max_e}\n")
            f.write(f"  Condition Number:  {c_num}\n")
            f.write(f"  Symmetry Residual: {cinfo['symmetry_residual']:.2e}\n")
            f.write(f"  Cholesky Success:  {cinfo['cholesky_success']}\n")
            f.write(f"  Cholesky Cache:    {cinfo['cholesky_file']}\n")
            f.write(f"  Healthy:           {cinfo['healthy']}\n")
            if cinfo['errors']:
                f.write(f"  Errors:            {cinfo['errors']}\n")
            f.write("-" * 65 + "\n")
        if errors:
            f.write("ERRORS DETECTED:\n")
            for err in errors:
                f.write(f"  ❌ {err}\n")
        else:
            f.write("✅ All components verified successfully. Matrix health confirmed.\n")
        f.write("=" * 65 + "\n")

    if all_healthy:
        print(f"\n[EXPORT VERIFICATION] ✅ Status: PASSED for {out_dir}")
        for cname, cinfo in component_stats.items():
            print(f"  - [{cname.upper()}] min_eig: {cinfo['min_eigenvalue']:.3e}, cond: {cinfo['condition_number']:.2e} -> Precomputed Cholesky saved to {cinfo['cholesky_file']}")
        print(f"  - Verification report written to: {json_path}\n")
    else:
        print("\n" + "!" * 75)
        print(f"❌ [EXPORT VERIFICATION FAILED] Corrupt/non-positive-definite covariance in {out_dir}!")
        for err in errors:
            print(f"   -> {err}")
        print(f"Full verification report saved to: {json_path}")
        print("!" * 75 + "\n")
        raise RuntimeError(f"Export verification failed: {errors}. Aborting execution immediately.")

    return report

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
    F_all[2] = set_F(1 + gamma, 1 / (1 + gamma))
    F_all[3] = set_F(1 / (1 + gamma), 1.0)
    F_all[4] = set_F(1 / (1 + gamma), 1 / (1 + gamma))
    F_all[5] = set_F(1.0, 1.0, f12=gamma)
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

def get_F_from_invariants(I1_bar, I2_bar, J):
    coeffs = [1.0, -I1_bar, I2_bar, -1.0]
    roots = np.roots(coeffs)
    lambda_sq = np.real(roots)
    lambda_sq = np.maximum(lambda_sq, 1e-8)
    lambdas = np.sqrt(lambda_sq) * (J ** (1 / 3))
    return np.diag(lambdas)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--max_gamma", type=float, default=0.8, help="Retained for CLI backward compatibility.")
    parser.add_argument("--sample_mode", type=str, default="dataset_f", choices=["dataset_f", "dataset_f_stratified", "dataset_f_uniform_energy", "dataset_all", "standard", "standard_interp", "inducing_points"], help="Sample deformations from extraction dataset (standard FPS, stratified high-strain FPS, uniform energy spectrum sampling, or all points), standard modes, or inducing points.")
    parser.add_argument("--stratified_power", type=float, default=1.5, help="Exponent for weighting high-strain bins in stratified FPS.")
    parser.add_argument("--num_points", type=int, default=192, help="Number of points to evaluate GP over.")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"], help="Distillation target mode: solely Strain Energy Function (sef), joint SEF + Piola stress (sef_stress), joint SEF + Cauchy stress (sef_cauchy), or separate DEV and VOL energy (sef_split).")
    parser.add_argument("--export_subfolder", type=str, default="", help="Custom output subfolder for exported PyTorch matrices.")
    parser.add_argument("--dataset_path", type=str, default="", help="Explicit path to precomputed dataset npz file.")
    args = parser.parse_args()

    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    valid_keys = set(GPRawParams._fields)
    filtered_params = {k: v for k, v in best_params_dict.items() if k in valid_keys}
    gp_params = GPRawParams(**filtered_params)
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

    from core.material_models import get_material_from_dir
    try:
        true_model = get_material_from_dir(args.saved_model_dir, jit_P=False)
    except Exception:
        true_model = None
    try:
        true_model_name = infer_material_model_name(args.saved_model_dir)
    except Exception:
        true_model_name = "experimental"

    import json
    metadata_path = os.path.join(args.saved_model_dir, "metadata.json")
    meta_dict = {}
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            meta_dict = json.load(f)
            cov_mode = meta_dict.get("covariance_mode", "diag")
    else:
        cov_mode = "full" if gp_params.raw_dev_u_var.ndim == 2 else "diag"

    pos_var_mean = meta_dict.get("pos_var_mean", 1)
    augmented_var_dist = meta_dict.get("augmented_var_dist", 1)

    feature_extractor = None
    if aniso_z is not None:
        if true_model is not None and getattr(true_model, 'a0', None) is not None:
            a0 = np.array(true_model.a0)
            a1 = np.array(true_model.a1) if getattr(true_model, 'a1', None) is not None else None
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        elif getattr(gp_params, "raw_aniso_theta_mean", None) is not None:
            raw_th = gp_params.raw_aniso_theta_mean
            theta = float(np.pi * (1.0 / (1.0 + np.exp(-raw_th)) - 0.5))
            a0 = np.array([np.cos(theta), np.sin(theta), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)
        elif "a0" in meta_dict:
            a0 = np.array(meta_dict["a0"])
            a1 = np.array(meta_dict["a1"]) if "a1" in meta_dict else None
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        elif aniso_z.shape[1] == 4:
            a0 = np.array([np.cos(np.pi / 4.0), np.sin(np.pi / 4.0), 0.0])
            a1 = np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
        else:
            theta = np.pi / 4.0
            a0 = np.array([np.cos(theta), np.sin(theta), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)
        
    constraint_lengthscale = meta_dict.get("constraint_lengthscale", 1)

    gp_model = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode,
        constraint_lengthscale=constraint_lengthscale
    )

    
    # Try to load the dataset for background plotting and dataset_* modes
    dataset_F_flat_2x2 = None
    try:
        saved_dir_abs = os.path.abspath(args.saved_model_dir)
        all_parts = saved_dir_abs.split(os.sep)

        ugp_model_name = infer_material_model_name(args.saved_model_dir)
        disp_noise = "0.0001"
        load_noise = "0.01"
        
        for p in reversed(all_parts):
            subparts = p.split('_')
            for sp in subparts:
                if sp.startswith("d") and sp[1:].replace('.', '', 1).isdigit():
                    disp_noise = sp[1:]
                elif sp.startswith("l") and sp[1:].replace('.', '', 1).isdigit():
                    load_noise = sp[1:]
                elif sp.replace('.', '', 1).isdigit() and sp in subparts:
                    if disp_noise == "0.0001" and float(sp) < 0.005:
                        disp_noise = sp
            
        prep_dataset_path = None
        load_steps = None
        if args.dataset_path and os.path.exists(args.dataset_path):
            prep_dataset_path = os.path.abspath(args.dataset_path)
            print(f"[EXPORT] Using explicit dataset path: {prep_dataset_path}")
        else:
            # Check config.json in saved_model_dir or parent
            for cdir in [args.saved_model_dir, os.path.dirname(os.path.abspath(args.saved_model_dir))]:
                cfg_file = os.path.join(cdir, "config.json")
                if os.path.exists(cfg_file):
                    try:
                        with open(cfg_file, "r") as cf:
                            cfg_dict = json.load(cf)
                            if "train_load_steps_indices" in cfg_dict and load_steps is None:
                                load_steps = cfg_dict["train_load_steps_indices"]
                            cfg_dsp = cfg_dict.get("dataset_path")
                            if cfg_dsp:
                                repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
                                candidates = [
                                    cfg_dsp if os.path.isabs(cfg_dsp) else os.path.join(repo_root, cfg_dsp),
                                    os.path.abspath(cfg_dsp)
                                ]
                                for cand in candidates:
                                    if os.path.exists(cand):
                                        prep_dataset_path = cand
                                        print(f"[EXPORT] Found dataset path from config.json: {prep_dataset_path}")
                                        break
                    except Exception as e:
                        print(f"[EXPORT] Error reading config.json: {e}")
                if prep_dataset_path is not None:
                    break

            if prep_dataset_path is None:
                meta_path = os.path.join(args.saved_model_dir, "metadata.json")
                seed_val = None
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as mf:
                            mdata = json.load(mf)
                            seed_val = mdata.get("seed")
                            meta_dsp = mdata.get("dataset_path")
                            if meta_dsp and os.path.exists(meta_dsp):
                                prep_dataset_path = meta_dsp
                                print(f"[EXPORT] Found dataset path from metadata.json: {prep_dataset_path}")
                    except Exception as e:
                        print(f"[EXPORT] Error reading metadata.json: {e}")

            if prep_dataset_path is None:
                # Also check config.yaml in saved_model_dir or parent
                for cdir in [args.saved_model_dir, os.path.dirname(os.path.abspath(args.saved_model_dir))]:
                    yaml_file = os.path.join(cdir, "config.yaml")
                    if os.path.exists(yaml_file):
                        try:
                            import yaml
                            with open(yaml_file, "r") as yf:
                                ycfg = yaml.safe_load(yf) or {}
                                y_dsp = ycfg.get("dataset_path")
                                if y_dsp and os.path.exists(y_dsp):
                                    prep_dataset_path = y_dsp
                                    print(f"[EXPORT] Found dataset path from config.yaml: {prep_dataset_path}")
                                    break
                                m_name = ycfg.get("material_model_name", ugp_model_name)
                                d_n = ycfg.get("disp_noise", disp_noise)
                                l_n = ycfg.get("load_noise", load_noise)
                                t_l = ycfg.get("target_load_true_top")
                                asym = ycfg.get("asym_factor")
                                geom = ycfg.get("geometry_train", "block")
                                if t_l is not None and asym is not None:
                                    candidates = []
                                    if seed_val is not None:
                                        candidates.append(f"{m_name}_{d_n}_{l_n}_{t_l}_{asym}_{geom}_{seed_val}.npz")
                                    candidates.append(f"{m_name}_{d_n}_{l_n}_{t_l}_{asym}_{geom}.npz")
                                    for sdir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
                                        for cand in candidates:
                                            cand_path = os.path.join(sdir, cand)
                                            if os.path.exists(cand_path):
                                                prep_dataset_path = cand_path
                                                print(f"[EXPORT] Found matching dataset from config.yaml specs: {prep_dataset_path}")
                                                break
                                        if prep_dataset_path is not None:
                                            break
                        except Exception as e:
                            print(f"[EXPORT] Error reading config.yaml: {e}")
                    if prep_dataset_path is not None:
                        break

            if prep_dataset_path is None:
                for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
                    if os.path.exists(search_dir):
                        if seed_val is not None:
                            for fname in sorted(os.listdir(search_dir)):
                                if fname.endswith(f"_{seed_val}.npz") and (fname.startswith(f"{ugp_model_name}_{disp_noise}_{load_noise}") or fname.startswith(f"{ugp_model_name}_")):
                                    prep_dataset_path = os.path.join(search_dir, fname)
                                    break
                        if prep_dataset_path is None:
                            for fname in sorted(os.listdir(search_dir)):
                                if (fname.startswith(f"{ugp_model_name}_{disp_noise}_{load_noise}") or fname.startswith(f"{ugp_model_name}_")) and fname.endswith(".npz"):
                                    prep_dataset_path = os.path.join(search_dir, fname)
                                    break
                    if prep_dataset_path is not None:
                        break
            if prep_dataset_path is not None:
                print(f"[EXPORT] Found matching dataset: {prep_dataset_path}")
        if prep_dataset_path is not None:
            prep_data = np.load(prep_dataset_path, allow_pickle=True)
            F_all_steps_2x2 = prep_data["F"]
            F_all_steps_3d = prep_data.get("F_3d", None)
            
            if load_steps is None:
                log_file = os.path.join(args.saved_model_dir, "optimization_log.txt")
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as lf:
                        first_line = lf.readline()
                        if "[" in first_line and "]" in first_line:
                            steps_str = first_line.split("]")[0].split("[")[1].strip()
                            if steps_str:
                                load_steps = [int(x.strip()) for x in steps_str.split(",") if x.strip().isdigit()]
            
            if load_steps and len(load_steps) > 0 and max(load_steps) < F_all_steps_2x2.shape[0]:
                F_train_full_2x2 = F_all_steps_2x2[load_steps]
                F_train_full_3d = F_all_steps_3d[load_steps] if F_all_steps_3d is not None else None
            else:
                default_steps = [2, 10, 20]
                valid_steps = [s for s in default_steps if s < F_all_steps_2x2.shape[0]]
                F_train_full_2x2 = F_all_steps_2x2[valid_steps] if len(valid_steps) > 0 else F_all_steps_2x2
                F_train_full_3d = F_all_steps_3d[valid_steps] if (F_all_steps_3d is not None and len(valid_steps) > 0) else F_all_steps_3d
                
            dataset_F_flat_2x2 = F_train_full_2x2.reshape(-1, 2, 2)
            dataset_F_flat_3d = F_train_full_3d.reshape(-1, 3, 3) if F_train_full_3d is not None else None
    except Exception as e:
        print(f"Could not load background dataset for plotting: {e}")

    f3x3_flat = None

    # Generate points
    if args.sample_mode in ["dataset_f", "dataset_f_stratified", "dataset_f_uniform_energy", "dataset_all"]:
        if dataset_F_flat_2x2 is None:
            raise ValueError(f"Dataset loading failed, cannot use sample_mode '{args.sample_mode}'.")
        F_flat_2x2 = dataset_F_flat_2x2

        if args.sample_mode == "dataset_all":
            print(f"Using exactly ALL {len(F_flat_2x2)} observed deformation points from extraction load steps (no FPS!).")
            f3x3_flat_2x2 = F_flat_2x2
            if dataset_F_flat_3d is not None:
                f3x3_flat = dataset_F_flat_3d
            default_export_subfolder = "pytorch_export_dataset_all"
        elif args.sample_mode == "dataset_f_stratified":
            print(f"Applying Stratified High-Strain FPS over {len(F_flat_2x2)} observed deformations (power={args.stratified_power})...")
            if dataset_F_flat_3d is not None:
                _, i2_all, _ = compute_invariants_np(dataset_F_flat_3d)
                strain_metric = i2_all - 3.0
            else:
                f3_temp = np.zeros((F_flat_2x2.shape[0], 3, 3))
                f3_temp[:, :2, :2] = F_flat_2x2
                f3_temp[:, 2, 2] = 1.0
                _, i2_all, _ = compute_invariants_np(f3_temp)
                strain_metric = i2_all - 3.0

            pts = jnp.array(F_flat_2x2.reshape(-1, 4), dtype=jnp.float64)
            indices = stratified_high_strain_fps(pts, strain_metric, args.num_points, n_bins=8, power=args.stratified_power)

            f3x3_flat_2x2 = F_flat_2x2[indices]
            if dataset_F_flat_3d is not None:
                f3x3_flat = dataset_F_flat_3d[indices]
            default_export_subfolder = f"pytorch_export_{args.sample_mode}_n{args.num_points}"
            print(f"Sampled {len(indices)} deformations directly from extraction dataset via Stratified High-Strain FPS.")
        elif args.sample_mode == "dataset_f_uniform_energy":
            print(f"Applying Uniform Energy Spectrum Sampling over {len(F_flat_2x2)} observed deformations...")
            if dataset_F_flat_3d is not None:
                f3_cand = dataset_F_flat_3d
            else:
                f3_cand = np.zeros((F_flat_2x2.shape[0], 3, 3))
                f3_cand[:, :2, :2] = F_flat_2x2
                f3_cand[:, 2, 2] = 1.0

            # Evaluate GP posterior mean energy across all candidate observed deformations
            if args.distill_target == "sef_split":
                feats = jax.vmap(gp_model.feature_extractor.extract)(jnp.array(f3_cand))
                energy_metric = np.array(gp_model.dev_gp_mean(feats[0]))
            else:
                energy_metric = np.array(gp_model.psi_gp_mean(jnp.array(f3_cand)))

            pts = jnp.array(F_flat_2x2.reshape(-1, 4), dtype=jnp.float64)
            indices = uniform_energy_fps(pts, energy_metric, args.num_points, n_bins=8)

            f3x3_flat_2x2 = F_flat_2x2[indices]
            if dataset_F_flat_3d is not None:
                f3x3_flat = dataset_F_flat_3d[indices]
            default_export_subfolder = f"pytorch_export_{args.sample_mode}_n{args.num_points}"
            print(f"Sampled {len(indices)} deformations directly from extraction dataset via Uniform Energy Spectrum Sampling.")
        else:  # "dataset_f"
            print(f"Applying Farthest Point Sampling (FPS) over {len(F_flat_2x2)} observed deformations...")
            pts = jnp.array(F_flat_2x2.reshape(-1, 4), dtype=jnp.float64)
            if len(F_flat_2x2) <= args.num_points:
                indices = np.arange(len(F_flat_2x2))
            else:
                indices = np.array(farthest_point_sampling(pts, args.num_points))

            f3x3_flat_2x2 = F_flat_2x2[indices]
            if dataset_F_flat_3d is not None:
                f3x3_flat = dataset_F_flat_3d[indices]
            default_export_subfolder = f"pytorch_export_dataset_f_n{args.num_points}"
            print(f"Sampled {len(indices)} deformations directly from extraction dataset via Farthest Point Sampling.")
    elif args.sample_mode == "standard_interp":
        print(f"Generating standard deformation modes strictly within GP interpolation bounds (up to gamma = {args.max_gamma})...")
        f3x3_flat_2x2 = generate_standard_modes_interp(num_points=max(1, args.num_points // 6), max_search_gamma=args.max_gamma, min_dev=min_dev, max_dev=max_dev, min_vol=min_vol, max_vol=max_vol)
        default_export_subfolder = "pytorch_export_standard_interp"
    elif args.sample_mode == "inducing_points":
        print(f"Generating F directly from the {len(I_z)} GP inducing points...")
        f3x3_list = []
        for i in range(len(I_z)):
            I1_bar = I_z[i, 0]
            I2_bar = I_z[i, 1]
            J = I_z[i, 2]
            f3x3_list.append(get_F_from_invariants(I1_bar, I2_bar, J))
        f3x3_flat = np.stack(f3x3_list)
        default_export_subfolder = "pytorch_export_inducing_points"
    else:  # "standard"
        f3x3_flat_2x2 = generate_standard_modes(num_points=max(1, args.num_points // 6), max_gamma=args.max_gamma)
        default_export_subfolder = f"pytorch_export_standard_g{args.max_gamma}" if args.max_gamma != 0.8 else "pytorch_export"

    if args.export_subfolder:
        export_subfolder = args.export_subfolder
    else:
        export_subfolder = default_export_subfolder
        if args.distill_target in ["sef_stress", "sef_cauchy"]:
            export_subfolder = f"{export_subfolder}_{args.distill_target}"

    if f3x3_flat is None:
        # Pad to 3x3 Plane Strain
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

        components_dict = {
            "dev": {"mean": mean_dev, "cov": cov_dev},
            "vol": {"mean": mean_vol, "cov": cov_vol},
        }
        if gp_model.is_anisotropic:
            components_dict["aniso"] = {"mean": mean_aniso, "cov": cov_aniso}

    else:
        np.save(os.path.join(out_dir, "mean_psi.npy"), np.array(mean_psi))
        np.save(os.path.join(out_dir, "cov_psi.npy"), np.array(cov_psi))
        np.save(os.path.join(out_dir, "f3x3.npy"), np.array(f3x3_flat))
        print(f"Exported GP Target Mean ({mean_psi.shape}) and Cov ({cov_psi.shape}) to {out_dir}")

        components_dict = {
            "psi": {"mean": mean_psi, "cov": cov_psi}
        }

    # Verify exported matrices, precompute Cholesky factors, write reports, and halt if error occurs
    verify_and_report_export(
        out_dir=out_dir,
        components_dict=components_dict,
        f3x3_flat=np.array(f3x3_flat),
        sample_mode=args.sample_mode,
        distill_target=args.distill_target
    )
    
    # Automatically generate the GP sample plot if it's the SEF target
    if args.distill_target == "sef":
        import subprocess
        print(f"Automatically generating GP sample visualizations for {out_dir}...")
        try:
            model_name = true_model_name
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
        
        if dataset_F_flat_3d is not None:
            ds_f3x3 = dataset_F_flat_3d
        elif dataset_F_flat_2x2 is not None:
            ds_f3x3 = np.zeros((dataset_F_flat_2x2.shape[0], 3, 3))
            ds_f3x3[:, :2, :2] = dataset_F_flat_2x2
            ds_f3x3[:, 2, 2] = 1.0
        else:
            ds_f3x3 = None

        if ds_f3x3 is not None:
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
        fig.savefig(os.path.join(out_dir, "export_invariant_space.png"), dpi=150)
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
        fig.savefig(os.path.join(out_dir, "export_energy_distribution.png"), dpi=150)
        plt.close(fig)
        print("Successfully saved invariant space and energy distribution plots.")

    except Exception as e:
        print(f"Failed to generate extra export plots: {e}")

if __name__ == "__main__":
    main()

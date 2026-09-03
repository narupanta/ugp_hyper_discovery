import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.utils import infer_material_model_name

def find_dataset_path(saved_model_dir, true_model_name):
    saved_dir_abs = os.path.abspath(saved_model_dir)
    all_parts = saved_dir_abs.split(os.sep)

    disp_noise = "0.0001"
    load_noise = "0.01"
    geometry = "block"
    
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
            elif sp in ["block", "holes", "cross"]:
                geometry = sp

    for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
        if os.path.exists(search_dir):
            for fname in os.listdir(search_dir):
                if (fname.startswith(f"{true_model_name}_{disp_noise}_{load_noise}") or fname.startswith(f"{true_model_name}_")) and geometry in fname and fname.endswith(".npz"):
                    return os.path.join(search_dir, fname)
    return None

def parse_train_load_steps(saved_model_dir):
    log_file = os.path.join(saved_model_dir, "optimization_log.txt")
    if os.path.exists(log_file):
        try:
            with open(log_file, "r", encoding="utf-8") as lf:
                first_line = lf.readline()
                if "[" in first_line and "]" in first_line:
                    steps_str = first_line.split("]")[0].split("[")[1].strip()
                    if steps_str:
                        return [int(x.strip()) for x in steps_str.split(",") if x.strip().isdigit()]
        except Exception:
            pass
    return [0, 3, 6]

def main():
    sys.stdout.reconfigure(line_buffering=True)
    plt.rcParams.update({
        'font.family': 'serif',
        'mathtext.fontset': 'cm',
        'text.usetex': False
    })

    parser = argparse.ArgumentParser(description="Plot distilled model and GP energy predictions on validation load steps.")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distillation output directory")
    parser.add_argument("--saved_model_dir", type=str, default=None, help="Path to extraction output directory")
    parser.add_argument("--material_model", type=str, default="gmr", choices=["ogden", "gmr", "gmr_log", "gmr_nolog", "isihara", "gmr_aniso", "aniso_gmr"])
    parser.add_argument("--distill_target", type=str, default="sef_split", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    parser.add_argument("--val_load_steps", type=int, nargs="+", default=None, help="Validation load steps (e.g. 7 8 9)")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of posterior parameter samples to use")
    args = parser.parse_args()

    distilled_dir = os.path.abspath(args.distilled_dir)
    if args.material_model == "gmr" and os.path.exists(os.path.join(distilled_dir, "aniso_flow_samples.npy")):
        args.material_model = "gmr_aniso"
    saved_model_dir = args.saved_model_dir

    if saved_model_dir is None:
        for fname in ["source_extraction_dir.txt", "dev_source_extraction_dir.txt"]:
            fpath = os.path.join(distilled_dir, fname)
            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    saved_model_dir = f.read().strip()
                break
        if saved_model_dir is None:
            cand = os.path.join(os.path.dirname(distilled_dir), "extraction")
            if os.path.exists(cand):
                saved_model_dir = cand
            else:
                raise ValueError("saved_model_dir must be provided or available in source_extraction_dir.txt.")

    saved_model_dir = os.path.abspath(saved_model_dir)
    true_model_name = infer_material_model_name(saved_model_dir)
    from core.material_models import get_material_from_dir
    try:
        true_model = get_material_from_dir(saved_model_dir, jit_P=False)
    except FileNotFoundError:
        true_model = get_material_from_dir(distilled_dir, jit_P=False)

    # 1. Load Dataset F field
    dataset_path = find_dataset_path(saved_model_dir, true_model_name)
    if dataset_path is None or not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Could not locate dataset .npz file for {true_model_name} from {saved_model_dir}")

    prep_data = np.load(dataset_path, allow_pickle=True)
    F_all_steps_2x2 = prep_data["F"]  # [num_steps, num_elements, 2, 2]
    total_steps = F_all_steps_2x2.shape[0]

    # 2. Determine validation load steps
    train_steps = parse_train_load_steps(saved_model_dir)
    if args.val_load_steps is not None and len(args.val_load_steps) > 0:
        val_steps = [s for s in args.val_load_steps if s < total_steps]
    else:
        max_train = max(train_steps) if len(train_steps) > 0 else 0
        unseen_later = [s for s in range(max_train + 1, total_steps)]
        if len(unseen_later) > 0:
            val_steps = unseen_later
        else:
            val_steps = [s for s in range(total_steps) if s not in train_steps]
            if len(val_steps) == 0:
                val_steps = [total_steps - 1]

    print(f"Dataset: {dataset_path}")
    print(f"True Model: {true_model_name}")
    print(f"Training Load Steps: {train_steps}")
    print(f"Evaluating Validation Load Steps: {val_steps} (Total points per step: {F_all_steps_2x2.shape[1]})")

    # 3. Load GP Model
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))

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

    metadata_path = os.path.join(saved_model_dir, "metadata.json")
    cov_mode = "diag"
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            cov_mode = json.load(f).get("covariance_mode", "diag")

    feature_extractor = None
    if aniso_z is not None:
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
        elif os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                meta_dict = json.load(f)
                if "a0" in meta_dict:
                    a0 = np.array(meta_dict["a0"])
                    a1 = np.array(meta_dict["a1"]) if "a1" in meta_dict else None
                    feature_extractor = AnisotropicFeatureExtractor(a0, a1=a1)
                else:
                    a0 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0])
                    feature_extractor = AnisotropicFeatureExtractor(a0)
        else:
            a0 = np.array([np.cos(np.pi/4), np.sin(np.pi/4), 0.0])
            feature_extractor = AnisotropicFeatureExtractor(a0)

    learned_gp = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode
    )

    # 4. Load distilled parameter samples
    has_aniso = os.path.exists(os.path.join(distilled_dir, "aniso_flow_samples.npy"))
    
    if args.distill_target == "sef_split":
        dev_raw = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        vol_raw = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))
        aniso_raw = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy")) if has_aniso else None
        
        n_eval_samples = min(args.num_samples, dev_raw.shape[0], vol_raw.shape[0])
        dev_samples = np.zeros((n_eval_samples, 10))
        dev_samples[:, :dev_raw.shape[1]] = dev_raw[:n_eval_samples, :10]
        
        vol_samples = np.zeros((n_eval_samples, 3))
        vol_samples[:, :vol_raw.shape[1]] = vol_raw[:n_eval_samples, :3]

        from core.utils import C_func, I1_func, I2_func, I3_func

        def psi_dev_formula(p, F_3d):
            C = C_func(F_3d)
            I1 = I1_func(C)
            I2 = I2_func(C)
            I3 = I3_func(C)
            I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
            i1_dev = I3_safe ** (-1 / 3) * I1
            i2_dev = I3_safe ** (-2 / 3) * I2
            i1_m3 = i1_dev - 3.0
            i2_m3 = i2_dev - 3.0
            return (
                p[0] * i1_m3 +
                p[1] * i2_m3 +
                p[2] * i1_m3**2 +
                p[3] * i1_m3 * i2_m3 +
                p[4] * i2_m3**2 +
                p[5] * i1_m3**3 +
                p[6] * (i1_m3**2) * i2_m3 +
                p[7] * i1_m3 * (i2_m3**2) +
                p[8] * i2_m3**3 +
                p[9] * jnp.log(jnp.maximum(i2_dev / 3.0, 1e-8))
            )

        def psi_vol_formula(p, F_3d):
            C = C_func(F_3d)
            I3 = I3_func(C)
            I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)
            J = jnp.sqrt(I3_safe)
            J_m1 = J - 1.0
            return (
                p[0] * J_m1**2 +
                p[1] * J_m1**4 +
                p[2] * J_m1**6
            )

        if has_aniso and aniso_raw is not None:
            aniso_samples = np.zeros((n_eval_samples, 6))
            aniso_samples[:, :aniso_raw.shape[1]] = aniso_raw[:n_eval_samples, :6]
            
            a1_val = getattr(true_model, "a0", None)
            if a1_val is None:
                a1_val = getattr(true_model, "a1", [1.0, 0.0, 0.0])
            a1 = jnp.array(a1_val)
            
            a2_val = getattr(true_model, "a1", None) if getattr(true_model, "a0", None) is not None else getattr(true_model, "a2", None)
            if a2_val is None:
                a2_val = [0.0, 1.0, 0.0]
            a2 = jnp.array(a2_val)
            
            def psi_aniso_single(theta_a, F_mat):
                J = jnp.linalg.det(F_mat)
                F_bar = F_mat * (J ** (-1.0 / 3.0))
                C_bar = F_bar.T @ F_bar
                I4_bar_1 = jnp.einsum('i,ij,j->', a1, C_bar, a1)
                I4_bar_2 = jnp.einsum('i,ij,j->', a2, C_bar, a2)
                I4_m1 = I4_bar_1 - 1.0
                I6_m1 = I4_bar_2 - 1.0
                return (theta_a[0] * I4_m1**2 + theta_a[1] * I4_m1**3 + theta_a[2] * I4_m1**4 +
                        theta_a[3] * I6_m1**2 + theta_a[4] * I6_m1**3 + theta_a[5] * I6_m1**4)
        else:
            psi_aniso_single = None

        eval_dev_batch = jax.jit(jax.vmap(lambda p, f_chunk: jax.vmap(lambda f: psi_dev_formula(p, f))(f_chunk), in_axes=(0, None)))
        eval_vol_batch = jax.jit(jax.vmap(lambda p, f_chunk: jax.vmap(lambda f: psi_vol_formula(p, f))(f_chunk), in_axes=(0, None)))
        if has_aniso:
            eval_aniso_batch = jax.jit(jax.vmap(lambda p, f_chunk: jax.vmap(lambda f: psi_aniso_single(p, f))(f_chunk), in_axes=(0, None)))
    else:
        samples = np.load(os.path.join(distilled_dir, "flow_samples.npy"))
        n_eval_samples = min(args.num_samples, samples.shape[0])
        samples = samples[:n_eval_samples]

        def eval_distilled_psi_single(theta, F_chunk):
            if args.material_model == "ogden":
                mat = get_material("ogden", mu_params=theta[:3], alpha_params=theta[3:6], vol_params=theta[6:9], jit_P=False)
            elif args.material_model in ["gmr", "gmr_log", "gmr_nolog"]:
                if len(theta) >= 13:
                    mat = get_material("gmr", dev_params=theta[:10], vol_params=theta[10:13], jit_P=False)
                else:
                    mat = get_material("gmr", dev_params=theta[:9], vol_params=theta[9:12], jit_P=False)
            elif args.material_model == "isihara":
                mat = get_material("isihara", c10=theta[0], c01=theta[1], c20=theta[2], d1=theta[3], jit_P=False)
            return jax.vmap(mat.psi)(F_chunk)

        eval_all_samples_fn = jax.jit(jax.vmap(eval_distilled_psi_single, in_axes=(0, None)))

    # 5. Evaluate each validation load step
    components = ["dev", "vol"]
    if has_aniso:
        components.append("aniso")
    components.append("total")

    data_by_comp = {comp: {"true": [], "dist_mean": [], "dist_std": [], "dist_q025": [], "dist_q975": [], "gp_mean": [], "gp_std": []} for comp in components}

    dev_samples_jnp = jnp.array(dev_samples) if args.distill_target == "sef_split" else None
    vol_samples_jnp = jnp.array(vol_samples) if args.distill_target == "sef_split" else None
    aniso_samples_jnp = jnp.array(aniso_samples) if (args.distill_target == "sef_split" and has_aniso) else None

    for step_idx in val_steps:
        print(f"Evaluating load step {step_idx}...", flush=True)
        F_step_2x2 = F_all_steps_2x2[step_idx]
        n_elems = F_step_2x2.shape[0]
        
        # Promote to 3x3 plane strain
        F_step_3x3 = np.zeros((n_elems, 3, 3), dtype=np.float64)
        F_step_3x3[:, :2, :2] = F_step_2x2
        F_step_3x3[:, 2, 2] = 1.0
        F_step_3x3_jnp = jnp.array(F_step_3x3)

        if args.distill_target == "sef_split":
            # 1. Deviatoric
            psi_dev_true = np.array(jax.vmap(true_model.psi_dev)(F_step_3x3_jnp))
            dev_samples_step = np.array(eval_dev_batch(dev_samples_jnp, F_step_3x3_jnp))
            gp_dev_dist = learned_gp.dev_psi_dist(F_step_3x3_jnp)
            
            data_by_comp["dev"]["true"].append(psi_dev_true)
            data_by_comp["dev"]["dist_mean"].append(np.mean(dev_samples_step, axis=0))
            data_by_comp["dev"]["dist_std"].append(np.std(dev_samples_step, axis=0))
            data_by_comp["dev"]["dist_q025"].append(np.percentile(dev_samples_step, 2.5, axis=0))
            data_by_comp["dev"]["dist_q975"].append(np.percentile(dev_samples_step, 97.5, axis=0))
            data_by_comp["dev"]["gp_mean"].append(np.array(gp_dev_dist.mean))
            data_by_comp["dev"]["gp_std"].append(np.sqrt(np.maximum(1e-12, np.array(gp_dev_dist.var))))

            # 2. Volumetric
            psi_vol_true = np.array(jax.vmap(true_model.psi_vol)(F_step_3x3_jnp))
            vol_samples_step = np.array(eval_vol_batch(vol_samples_jnp, F_step_3x3_jnp))
            gp_vol_dist = learned_gp.vol_psi_dist(F_step_3x3_jnp)

            data_by_comp["vol"]["true"].append(psi_vol_true)
            data_by_comp["vol"]["dist_mean"].append(np.mean(vol_samples_step, axis=0))
            data_by_comp["vol"]["dist_std"].append(np.std(vol_samples_step, axis=0))
            data_by_comp["vol"]["dist_q025"].append(np.percentile(vol_samples_step, 2.5, axis=0))
            data_by_comp["vol"]["dist_q975"].append(np.percentile(vol_samples_step, 97.5, axis=0))
            data_by_comp["vol"]["gp_mean"].append(np.array(gp_vol_dist.mean))
            data_by_comp["vol"]["gp_std"].append(np.sqrt(np.maximum(1e-12, np.array(gp_vol_dist.var))))

            # 3. Anisotropic (if active)
            if has_aniso:
                psi_aniso_true = np.array(jax.vmap(true_model.psi_aniso)(F_step_3x3_jnp))
                aniso_samples_step = np.array(eval_aniso_batch(aniso_samples_jnp, F_step_3x3_jnp))
                gp_aniso_dist = learned_gp.aniso_psi_dist(F_step_3x3_jnp) if hasattr(learned_gp, "aniso_psi_dist") and learned_gp.is_anisotropic else None

                data_by_comp["aniso"]["true"].append(psi_aniso_true)
                data_by_comp["aniso"]["dist_mean"].append(np.mean(aniso_samples_step, axis=0))
                data_by_comp["aniso"]["dist_std"].append(np.std(aniso_samples_step, axis=0))
                data_by_comp["aniso"]["dist_q025"].append(np.percentile(aniso_samples_step, 2.5, axis=0))
                data_by_comp["aniso"]["dist_q975"].append(np.percentile(aniso_samples_step, 97.5, axis=0))
                if gp_aniso_dist is not None:
                    data_by_comp["aniso"]["gp_mean"].append(np.array(gp_aniso_dist.mean))
                    data_by_comp["aniso"]["gp_std"].append(np.sqrt(np.maximum(1e-12, np.array(gp_aniso_dist.var))))
                else:
                    data_by_comp["aniso"]["gp_mean"].append(np.zeros_like(psi_aniso_true))
                    data_by_comp["aniso"]["gp_std"].append(np.zeros_like(psi_aniso_true))

                total_samples_step = dev_samples_step + vol_samples_step + aniso_samples_step
                psi_total_true = psi_dev_true + psi_vol_true + psi_aniso_true
            else:
                total_samples_step = dev_samples_step + vol_samples_step
                psi_total_true = psi_dev_true + psi_vol_true

            gp_total_dist = learned_gp.psi_dist(F_step_3x3_jnp)
            data_by_comp["total"]["true"].append(psi_total_true)
            data_by_comp["total"]["dist_mean"].append(np.mean(total_samples_step, axis=0))
            data_by_comp["total"]["dist_std"].append(np.std(total_samples_step, axis=0))
            data_by_comp["total"]["dist_q025"].append(np.percentile(total_samples_step, 2.5, axis=0))
            data_by_comp["total"]["dist_q975"].append(np.percentile(total_samples_step, 97.5, axis=0))
            data_by_comp["total"]["gp_mean"].append(np.array(gp_total_dist.mean))
            data_by_comp["total"]["gp_std"].append(np.sqrt(np.maximum(1e-12, np.array(gp_total_dist.var))))

        else:
            psi_total_true = np.array(jax.vmap(true_model.psi)(F_step_3x3_jnp))
            total_samples_step = np.array(eval_all_samples_fn(jnp.array(samples), F_step_3x3_jnp))
            gp_total_dist = learned_gp.psi_dist(F_step_3x3_jnp)

            data_by_comp["total"]["true"].append(psi_total_true)
            data_by_comp["total"]["dist_mean"].append(np.mean(total_samples_step, axis=0))
            data_by_comp["total"]["dist_std"].append(np.std(total_samples_step, axis=0))
            data_by_comp["total"]["dist_q025"].append(np.percentile(total_samples_step, 2.5, axis=0))
            data_by_comp["total"]["dist_q975"].append(np.percentile(total_samples_step, 97.5, axis=0))
            data_by_comp["total"]["gp_mean"].append(np.array(gp_total_dist.mean))
            data_by_comp["total"]["gp_std"].append(np.sqrt(np.maximum(1e-12, np.array(gp_total_dist.var))))

    # 6. Concatenate validation metrics per component
    metrics_by_comp = {}
    for comp in components:
        all_true = np.concatenate(data_by_comp[comp]["true"])
        all_dist_mean = np.concatenate(data_by_comp[comp]["dist_mean"])
        all_dist_std = np.concatenate(data_by_comp[comp]["dist_std"])
        all_dist_q025 = np.concatenate(data_by_comp[comp]["dist_q025"])
        all_dist_q975 = np.concatenate(data_by_comp[comp]["dist_q975"])
        all_gp_mean = np.concatenate(data_by_comp[comp]["gp_mean"])
        all_gp_std = np.concatenate(data_by_comp[comp]["gp_std"])

        r2_dist = float(r2_score(all_true, all_dist_mean))
        rmse_dist = float(np.sqrt(np.mean((all_true - all_dist_mean)**2)))
        in_ci_dist = (all_true >= all_dist_q025) & (all_true <= all_dist_q975)
        coverage_dist = float(np.mean(in_ci_dist) * 100.0)

        r2_gp = float(r2_score(all_true, all_gp_mean))
        rmse_gp = float(np.sqrt(np.mean((all_true - all_gp_mean)**2)))
        in_ci_gp = (all_true >= (all_gp_mean - 1.96 * all_gp_std)) & (all_true <= (all_gp_mean + 1.96 * all_gp_std))
        coverage_gp = float(np.mean(in_ci_gp) * 100.0)

        metrics_by_comp[comp] = {
            "all_true": all_true, "all_dist_mean": all_dist_mean, "all_dist_std": all_dist_std,
            "all_dist_q025": all_dist_q025, "all_dist_q975": all_dist_q975,
            "all_gp_mean": all_gp_mean, "all_gp_std": all_gp_std,
            "r2_dist": r2_dist, "rmse_dist": rmse_dist, "coverage_dist": coverage_dist,
            "r2_gp": r2_gp, "rmse_gp": rmse_gp, "coverage_gp": coverage_gp
        }

        print(f"\n--- Validation Parity Metrics: [{comp.upper()}] (Steps: {val_steps}) ---", flush=True)
        print(f"Distilled Model: R² = {r2_dist:.4f}, RMSE = {rmse_dist:.4f}, EC(95% CI) = {coverage_dist:.1f}%", flush=True)
        print(f"GP Baseline:     R² = {r2_gp:.4f}, RMSE = {rmse_gp:.4f}, EC(95% CI) = {coverage_gp:.1f}%", flush=True)

    # 7. Generate Multi-Panel Split Energy Parity Plot
    n_plots = len(components)
    fig_split, axes_split = plt.subplots(1, n_plots, figsize=(5.5 * n_plots, 5.2))
    if n_plots == 1:
        axes_split = [axes_split]

    comp_titles = {
        "dev": r"Deviatoric Energy $\Psi_{\mathrm{dev}}$",
        "vol": r"Volumetric Energy $\Psi_{\mathrm{vol}}$",
        "aniso": r"Anisotropic Energy $\Psi_{\mathrm{aniso}}$",
        "total": r"Total Strain Energy $\Psi_{\mathrm{total}}$"
    }

    comp_colors = {
        "dev": "#0072B2",
        "vol": "#D55E00",
        "aniso": "#CC79A7",
        "total": "#009E73"
    }

    for ax, comp in zip(axes_split, components):
        m = metrics_by_comp[comp]
        color = comp_colors.get(comp, "#0072B2")

        # 1) GP Prediction
        ax.errorbar(
            x=m["all_true"], y=m["all_gp_mean"], yerr=1.96 * m["all_gp_std"],
            fmt='x', color='#7f8c8d', ecolor='#bdc3c7', elinewidth=0.5, capsize=0, alpha=0.4,
            markersize=3.0, label=r"GP Baseline ($95\%$ CI)", zorder=2
        )

        # 2) Distilled Model Prediction
        y_err_low = np.maximum(0.0, m["all_dist_mean"] - m["all_dist_q025"])
        y_err_high = np.maximum(0.0, m["all_dist_q975"] - m["all_dist_mean"])
        ax.errorbar(
            x=m["all_true"], y=m["all_dist_mean"], yerr=[y_err_low, y_err_high],
            fmt='o', color=color, ecolor=color, elinewidth=0.5, capsize=0, alpha=0.45,
            markersize=2.8, label=f"Distilled (95% CI)", zorder=3
        )

        # Diagonal line
        min_v = min(float(m["all_true"].min()), float(m["all_dist_mean"].min()), float(m["all_gp_mean"].min()))
        max_v = max(float(m["all_true"].max()), float(m["all_dist_mean"].max()), float(m["all_gp_mean"].max()))
        pad = max((max_v - min_v) * 0.05, 1e-4)
        l_min, l_max = max(0.0, min_v - pad), max_v + pad
        ax.plot([l_min, l_max], [l_min, l_max], 'k--', lw=1.5, zorder=5, label="Parity ($y = x$)")

        # Info Box
        info_txt = (
            r"$\mathbf{Distilled}$" + "\n"
            + f"$R^2$: {m['r2_dist']:.4f}\n"
            + rf"$\mathrm{{RMSE}}$: {m['rmse_dist']:.4f}" + "\n"
            + rf"$\mathrm{{EC}}$: {m['coverage_dist']:.1f}%" + "\n\n"
            + r"$\mathbf{GP\ Baseline}$" + "\n"
            + f"$R^2$: {m['r2_gp']:.4f}\n"
            + rf"$\mathrm{{RMSE}}$: {m['rmse_gp']:.4f}" + "\n"
            + rf"$\mathrm{{EC}}$: {m['coverage_gp']:.1f}%"
        )
        ax.text(
            0.05, 0.95, info_txt, transform=ax.transAxes, verticalalignment='top',
            fontsize=8.5, bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.9, lw=0.8)
        )

        ax.set_title(comp_titles.get(comp, comp.upper()), fontsize=11, fontweight='bold', pad=8)
        ax.set_xlabel(r"Ground Truth $\Psi_{\mathrm{true}}$", fontsize=10)
        if comp == components[0]:
            ax.set_ylabel(r"Predicted $\hat{\Psi}$", fontsize=10)
        ax.set_xlim([l_min, l_max])
        ax.set_ylim([l_min, l_max])
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig_split.suptitle(f"Validation Energy Parity & Uncertainty: Distilled ({args.material_model.upper()}) vs Ground Truth ({true_model_name})\nValidation Load Steps: {val_steps}", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_split_pdf = os.path.join(distilled_dir, f"distilled_validation_r2_split_energy_{args.material_model}.pdf")
    out_split_png = os.path.join(distilled_dir, f"distilled_validation_r2_split_energy_{args.material_model}.png")
    fig_split.savefig(out_split_pdf, dpi=300, bbox_inches='tight')
    fig_split.savefig(out_split_png, dpi=300, bbox_inches='tight')
    plt.close(fig_split)

    # 8. Also generate individual Total Energy Parity Plot
    m_tot = metrics_by_comp["total"]
    fig_tot, ax_tot = plt.subplots(figsize=(7.0, 6.5))

    ax_tot.errorbar(
        x=m_tot["all_true"], y=m_tot["all_gp_mean"], yerr=1.96 * m_tot["all_gp_std"],
        fmt='x', color='#7f8c8d', ecolor='#bdc3c7', elinewidth=0.6, capsize=0, alpha=0.45,
        markersize=3.5, markeredgewidth=1.0, label=r"GP Baseline ($95\%$ CI)", zorder=2
    )

    y_err_low = np.maximum(0.0, m_tot["all_dist_mean"] - m_tot["all_dist_q025"])
    y_err_high = np.maximum(0.0, m_tot["all_dist_q975"] - m_tot["all_dist_mean"])
    ax_tot.errorbar(
        x=m_tot["all_true"], y=m_tot["all_dist_mean"], yerr=[y_err_low, y_err_high],
        fmt='o', color="#009E73", ecolor="#5dade2", elinewidth=0.6, capsize=0, alpha=0.5,
        markersize=3.0, markeredgewidth=0, label=f"Distilled {args.material_model.upper()} (95% CI)", zorder=3
    )

    min_axis = min(float(m_tot["all_dist_mean"].min()), float(m_tot["all_true"].min()), float(m_tot["all_gp_mean"].min()))
    max_axis = max(float(m_tot["all_dist_mean"].max()), float(m_tot["all_true"].max()), float(m_tot["all_gp_mean"].max()))
    padding = (max_axis - min_axis) * 0.05
    line_min = max(0.0, min_axis - padding)
    line_max = max_axis + padding
    ax_tot.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1.8, zorder=5, label="Parity ($y = x$)")

    metrics_box = (
        r"$\mathbf{Distilled\ " + args.material_model.upper() + r"\ Validation}$" + "\n"
        f"Validation Steps: {val_steps}\n"
        f"$R^2$: {m_tot['r2_dist']:.4f}\n"
        + rf"$\mathrm{{RMSE}}$: {m_tot['rmse_dist']:.4f}" + "\n"
        + rf"$\mathrm{{EC}}\ (95\%\ \mathrm{{CI}})$: {m_tot['coverage_dist']:.1f}%" + "\n\n"
        + r"$\mathbf{GP\ Baseline}$" + "\n"
        f"$R^2$: {m_tot['r2_gp']:.4f}\n"
        + rf"$\mathrm{{RMSE}}$: {m_tot['rmse_gp']:.4f}" + "\n"
        + rf"$\mathrm{{EC}}\ (95\%\ \mathrm{{CI}})$: {m_tot['coverage_gp']:.1f}%"
    )
    ax_tot.text(
        0.05, 0.95, metrics_box, transform=ax_tot.transAxes, verticalalignment='top',
        fontsize=10.0, bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#7f8c8d', alpha=0.92, lw=1.2)
    )

    ax_tot.set_xlabel(r"True Strain Energy Density $\Psi_{\mathrm{true}}\ (\mathrm{" + true_model_name + r"})$", fontsize=11)
    ax_tot.set_ylabel(r"Predicted Strain Energy Density $\hat{\Psi}$", fontsize=11)
    ax_tot.set_title(f"Validation Energy Parity & Uncertainty\nDistilled {args.material_model.upper()} vs Ground Truth ({true_model_name})", fontsize=12, fontweight='bold', pad=10)
    ax_tot.set_xlim([line_min, line_max])
    ax_tot.set_ylim([line_min, line_max])
    ax_tot.grid(True, linestyle=':', alpha=0.6)
    ax_tot.legend(loc='lower right', fontsize=9.5, framealpha=0.9)
    plt.tight_layout()

    out_tot_pdf = os.path.join(distilled_dir, f"distilled_validation_r2_energy_{args.material_model}.pdf")
    out_tot_png = os.path.join(distilled_dir, f"distilled_validation_r2_energy_{args.material_model}.png")
    fig_tot.savefig(out_tot_pdf, dpi=300, bbox_inches='tight')
    fig_tot.savefig(out_tot_png, dpi=300, bbox_inches='tight')
    plt.close(fig_tot)

    # 9. Generate Domain Spatial Heatmaps (Mean & Std across cells for Total, Dev, Vol, Aniso)
    if "mesh_pos" in prep_data and "cells" in prep_data:
        import matplotlib.tri as mtri
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        mesh_pos = prep_data["mesh_pos"]
        cells = prep_data["cells"]
        u_all = prep_data["u"] if "u" in prep_data else None

        rep_step = val_steps[-1]
        rep_idx_in_val = val_steps.index(rep_step)

        if u_all is not None:
            coords_rep = mesh_pos + u_all[rep_step]
        else:
            coords_rep = mesh_pos

        triangulation = mtri.Triangulation(coords_rep[:, 0], coords_rep[:, 1], cells)

        # Domain columns: Total, Dev, Vol, (Aniso)
        domain_cols = ["total", "dev", "vol"]
        if has_aniso:
            domain_cols.append("aniso")

        col_titles = {
            "total": r"$\mathbf{Total\ Energy\ \Psi_{\mathrm{total}}}$",
            "dev": r"$\mathbf{Deviatoric\ \Psi_{\mathrm{dev}}}$",
            "vol": r"$\mathbf{Volumetric\ \Psi_{\mathrm{vol}}}$",
            "aniso": r"$\mathbf{Anisotropic\ \Psi_{\mathrm{aniso}}}$"
        }

        def plot_domain_figure(source_type="distilled"):
            is_dist = (source_type == "distilled")
            fig_dom, axes_dom = plt.subplots(2, len(domain_cols), figsize=(4.6 * len(domain_cols), 7.2))
            
            for j, comp in enumerate(domain_cols):
                if is_dist:
                    mean_arr = data_by_comp[comp]["dist_mean"][rep_idx_in_val]
                    std_arr = data_by_comp[comp]["dist_std"][rep_idx_in_val]
                else:
                    mean_arr = data_by_comp[comp]["gp_mean"][rep_idx_in_val]
                    std_arr = data_by_comp[comp]["gp_std"][rep_idx_in_val]

                # Row 0: Mean Energy
                ax_mean = axes_dom[0, j]
                im_mean = ax_mean.tripcolor(triangulation, facecolors=mean_arr, cmap='viridis', edgecolors='none')
                ax_mean.set_title(col_titles.get(comp, comp.upper()), fontsize=11, fontweight='bold', pad=6)
                ax_mean.set_aspect('equal')
                ax_mean.axis('off')
                
                div_mean = make_axes_locatable(ax_mean)
                cax_mean = div_mean.append_axes("right", size="5%", pad=0.08)
                cb_mean = fig_dom.colorbar(im_mean, cax=cax_mean)
                cb_mean.ax.tick_params(labelsize=8)

                # Row 1: Std Energy (Uncertainty)
                ax_std = axes_dom[1, j]
                im_std = ax_std.tripcolor(triangulation, facecolors=std_arr, cmap='magma', edgecolors='none')
                ax_std.set_aspect('equal')
                ax_std.axis('off')
                
                div_std = make_axes_locatable(ax_std)
                cax_std = div_std.append_axes("right", size="5%", pad=0.08)
                cb_std = fig_dom.colorbar(im_std, cax=cax_std)
                cb_std.ax.tick_params(labelsize=8)

            # Row labels
            axes_dom[0, 0].text(-0.08, 0.5, "Mean Energy\n$\mathbf{\mu(\Psi)}$", transform=axes_dom[0, 0].transAxes,
                                fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)
            axes_dom[1, 0].text(-0.08, 0.5, "Uncertainty (Std)\n$\mathbf{\sigma(\Psi)}$", transform=axes_dom[1, 0].transAxes,
                                fontsize=11, fontweight='bold', va='center', ha='right', rotation=90)

            title_prefix = f"Distilled Model ({args.material_model.upper()})" if is_dist else "Extracted GP Baseline"
            fig_dom.suptitle(f"{title_prefix} - Domain Energy & Uncertainty Distribution\nValidation Load Step: {rep_step} | Ground Truth: {true_model_name}", fontsize=13, fontweight='bold', y=0.98)
            plt.tight_layout()

            prefix = "distilled" if is_dist else "extracted"
            out_dom_pdf = os.path.join(distilled_dir, f"{prefix}_validation_domain_energy_{args.material_model}.pdf")
            out_dom_png = os.path.join(distilled_dir, f"{prefix}_validation_domain_energy_{args.material_model}.png")
            fig_dom.savefig(out_dom_pdf, dpi=300, bbox_inches='tight')
            fig_dom.savefig(out_dom_png, dpi=300, bbox_inches='tight')
            plt.close(fig_dom)
            return out_dom_pdf, out_dom_png

        out_dist_dom_pdf, out_dist_dom_png = plot_domain_figure("distilled")
        out_gp_dom_pdf, out_gp_dom_png = plot_domain_figure("extracted")
        print(f"Saved distilled domain energy plot to:\n  - {out_dist_dom_pdf}\n  - {out_dist_dom_png}", flush=True)
        print(f"Saved extracted domain energy plot to:\n  - {out_gp_dom_pdf}\n  - {out_gp_dom_png}", flush=True)

    print(f"\nSaved validation parity plots to:\n  - {out_split_pdf}\n  - {out_split_png}\n  - {out_tot_pdf}\n  - {out_tot_png}", flush=True)

    # 10. Save Validation Metrics to JSON
    val_json_path = os.path.join(distilled_dir, f"validation_metrics_{args.material_model}.json")
    val_summary = {}
    for comp, m in metrics_by_comp.items():
        val_summary[comp] = {
            "r2_dist": m["r2_dist"],
            "rmse_dist": m["rmse_dist"],
            "coverage_dist": m["coverage_dist"],
            "r2_gp": m["r2_gp"],
            "rmse_gp": m["rmse_gp"],
            "coverage_gp": m["coverage_gp"]
        }
    with open(val_json_path, "w") as f:
        json.dump(val_summary, f, indent=4)
    print(f"Saved validation metrics JSON to: {val_json_path}", flush=True)

if __name__ == "__main__":
    main()

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

    for search_dir in ["dataset/preprocessed/syn_f", "dataset/precomputed_vfm"]:
        if os.path.exists(search_dir):
            for fname in os.listdir(search_dir):
                if (fname.startswith(f"{true_model_name}_{disp_noise}_{load_noise}") or fname.startswith(f"{true_model_name}_")) and fname.endswith(".npz"):
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
    true_model = get_material(true_model_name, jit_P=False)

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
        if true_model is not None and hasattr(true_model, 'a1') and hasattr(true_model, 'a2'):
            feature_extractor = AnisotropicFeatureExtractor(np.array(true_model.a1), a1=np.array(true_model.a2))
        elif true_model is not None and hasattr(true_model, 'a0'):
            feature_extractor = AnisotropicFeatureExtractor(np.array(true_model.a0))

    learned_gp = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feature_extractor,
        aniso_z=aniso_z, min_aniso=min_aniso, max_aniso=max_aniso,
        covariance_mode=cov_mode
    )

    # 4. Load distilled parameter samples
    has_aniso = os.path.exists(os.path.join(distilled_dir, "aniso_flow_samples.npy"))
    
    if args.distill_target == "sef_split":
        dev_samples = np.load(os.path.join(distilled_dir, "dev_flow_samples.npy"))
        vol_samples = np.load(os.path.join(distilled_dir, "vol_flow_samples.npy"))
        
        n_eval_samples = min(args.num_samples, dev_samples.shape[0], vol_samples.shape[0])
        dev_samples = dev_samples[:n_eval_samples]
        vol_samples = vol_samples[:n_eval_samples]

        if has_aniso:
            aniso_samples = np.load(os.path.join(distilled_dir, "aniso_flow_samples.npy"))[:n_eval_samples]
            
            def eval_distilled_psi_sample(td, tv, ta, F_chunk):
                dev_theta = list(td) + [0.0, 0.0, 0.0]
                vol_theta = [0.0] * 9 + list(tv)
                mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
                mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
                
                a1 = jnp.array(getattr(true_model, "a1", getattr(true_model, "a0", [1.0, 0.0, 0.0])))
                a2 = jnp.array(getattr(true_model, "a2", [0.0, 1.0, 0.0]))
                
                def psi_aniso_single(theta_a, F_mat):
                    J = jnp.linalg.det(F_mat)
                    F_bar = F_mat * (J ** (-1.0 / 3.0))
                    C_bar = F_bar.T @ F_bar
                    I4_bar_1 = jnp.einsum('i,ij,j->', a1, C_bar, a1)
                    I4_bar_2 = jnp.einsum('i,ij,j->', a2, C_bar, a2)
                    I4_m1 = I4_bar_1 - 1.0
                    I6_m1 = I4_bar_2 - 1.0
                    if len(theta_a) == 6:
                        C42, C43, C44, C62, C63, C64 = theta_a[:6]
                        return (C42 * I4_m1**2 + C43 * I4_m1**3 + C44 * I4_m1**4 +
                                C62 * I6_m1**2 + C63 * I6_m1**3 + C64 * I6_m1**4)
                    else:
                        ta_full = list(theta_a) + [0.0] * (8 - len(theta_a))
                        C42, C44, k1, k2, C62, C64, k3, k4 = ta_full[:8]
                        exp_arg1 = jnp.clip(k2 * I4_m1**2, -30.0, 30.0)
                        exp_arg2 = jnp.clip(k4 * I6_m1**2, -30.0, 30.0)
                        return (C42 * I4_m1**2 + C44 * I4_m1**4 + k1 * (jnp.exp(exp_arg1) - 1.0) +
                                C62 * I6_m1**2 + C64 * I6_m1**4 + k3 * (jnp.exp(exp_arg2) - 1.0))
                            
                psi_dev = jax.vmap(mat_dev.psi)(F_chunk)
                psi_vol = jax.vmap(mat_vol.psi)(F_chunk)
                psi_aniso = jax.vmap(lambda f: psi_aniso_single(ta, f))(F_chunk)
                return psi_dev + psi_vol + psi_aniso

            eval_all_samples_fn = jax.jit(jax.vmap(eval_distilled_psi_sample, in_axes=(0, 0, 0, None)))
        else:
            def eval_distilled_psi_sample(td, tv, F_chunk):
                dev_theta = list(td) + [0.0, 0.0, 0.0]
                vol_theta = [0.0] * 9 + list(tv)
                mat_dev = get_material("gmr", dev_params=dev_theta[:9], vol_params=dev_theta[9:12], jit_P=False)
                mat_vol = get_material("gmr", dev_params=vol_theta[:9], vol_params=vol_theta[9:12], jit_P=False)
                return jax.vmap(mat_dev.psi)(F_chunk) + jax.vmap(mat_vol.psi)(F_chunk)

            eval_all_samples_fn = jax.jit(jax.vmap(eval_distilled_psi_sample, in_axes=(0, 0, None)))
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
    all_true_list = []
    all_dist_mean_list = []
    all_dist_std_list = []
    all_dist_q025_list = []
    all_dist_q975_list = []
    all_gp_mean_list = []
    all_gp_std_list = []

    for step_idx in val_steps:
        F_step_2x2 = F_all_steps_2x2[step_idx]
        n_elems = F_step_2x2.shape[0]
        
        # Promote to 3x3 plane strain
        F_step_3x3 = np.zeros((n_elems, 3, 3), dtype=np.float64)
        F_step_3x3[:, :2, :2] = F_step_2x2
        F_step_3x3[:, 2, 2] = 1.0
        F_step_3x3_jnp = jnp.array(F_step_3x3)

        # True energy
        psi_true = np.array(jax.vmap(true_model.psi)(F_step_3x3_jnp))

        # Distilled predictive distribution
        if args.distill_target == "sef_split":
            if has_aniso:
                psi_samples_step = np.array(eval_all_samples_fn(jnp.array(dev_samples), jnp.array(vol_samples), jnp.array(aniso_samples), F_step_3x3_jnp))
            else:
                psi_samples_step = np.array(eval_all_samples_fn(jnp.array(dev_samples), jnp.array(vol_samples), F_step_3x3_jnp))
        else:
            psi_samples_step = np.array(eval_all_samples_fn(jnp.array(samples), F_step_3x3_jnp))

        psi_dist_mean = np.mean(psi_samples_step, axis=0)
        psi_dist_std = np.std(psi_samples_step, axis=0)
        psi_dist_q025 = np.percentile(psi_samples_step, 2.5, axis=0)
        psi_dist_q975 = np.percentile(psi_samples_step, 97.5, axis=0)

        # GP predictive distribution
        gp_dist_step = learned_gp.psi_dist(F_step_3x3_jnp)
        gp_mean_step = np.array(gp_dist_step.mean)
        gp_var_step = np.array(gp_dist_step.var)
        gp_std_step = np.sqrt(np.maximum(1e-12, gp_var_step))

        all_true_list.append(psi_true)
        all_dist_mean_list.append(psi_dist_mean)
        all_dist_std_list.append(psi_dist_std)
        all_dist_q025_list.append(psi_dist_q025)
        all_dist_q975_list.append(psi_dist_q975)
        all_gp_mean_list.append(gp_mean_step)
        all_gp_std_list.append(gp_std_step)

    all_true = np.concatenate(all_true_list)
    all_dist_mean = np.concatenate(all_dist_mean_list)
    all_dist_std = np.concatenate(all_dist_std_list)
    all_dist_q025 = np.concatenate(all_dist_q025_list)
    all_dist_q975 = np.concatenate(all_dist_q975_list)
    all_gp_mean = np.concatenate(all_gp_mean_list)
    all_gp_std = np.concatenate(all_gp_std_list)

    # 6. Compute global validation metrics
    r2_dist = float(r2_score(all_true, all_dist_mean))
    rmse_dist = float(np.sqrt(np.mean((all_true - all_dist_mean)**2)))
    in_ci_dist = (all_true >= all_dist_q025) & (all_true <= all_dist_q975)
    coverage_dist = float(np.mean(in_ci_dist) * 100.0)

    r2_gp = float(r2_score(all_true, all_gp_mean))
    rmse_gp = float(np.sqrt(np.mean((all_true - all_gp_mean)**2)))
    in_ci_gp = (all_true >= (all_gp_mean - 1.96 * all_gp_std)) & (all_true <= (all_gp_mean + 1.96 * all_gp_std))
    coverage_gp = float(np.mean(in_ci_gp) * 100.0)

    print(f"\n--- Validation Energy Parity Metrics (Steps: {val_steps}) ---")
    print(f"Total Validation Points: {len(all_true)}")
    print(f"Distilled Model ({args.material_model.upper()}):")
    print(f"  R²: {r2_dist:.4f}")
    print(f"  RMSE: {rmse_dist:.4f}")
    print(f"  EC (95% CI): {coverage_dist:.1f}%")
    print(f"GP Model:")
    print(f"  R²: {r2_gp:.4f}")
    print(f"  RMSE: {rmse_gp:.4f}")
    print(f"  EC (95% CI): {coverage_gp:.1f}%")

    # 7. Plotting: X-axis = True Energy, Y-axis = Predicted Energy (with vertical error bars)
    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # 1) GP Prediction with Uncertainty Error Bars (Grey x marks)
    ax.errorbar(
        x=all_true,
        y=all_gp_mean,
        yerr=1.96 * all_gp_std,
        fmt='x',
        color='#7f8c8d',
        ecolor='#bdc3c7',
        elinewidth=0.6,
        capsize=0,
        alpha=0.45,
        markersize=3.5,
        markeredgewidth=1.0,
        label=r"GP Baseline ($95\%$ CI)",
        zorder=2
    )

    # 2) Distilled Model Prediction (Unified color with error bars)
    dist_color = "#0072B2"
    y_err_low = np.maximum(0.0, all_dist_mean - all_dist_q025)
    y_err_high = np.maximum(0.0, all_dist_q975 - all_dist_mean)

    ax.errorbar(
        x=all_true,
        y=all_dist_mean,
        yerr=[y_err_low, y_err_high],
        fmt='o',
        color=dist_color,
        ecolor='#5dade2',
        elinewidth=0.6,
        capsize=0,
        alpha=0.5,
        markersize=3.0,
        markeredgewidth=0,
        label=f"Distilled {args.material_model.upper()} (95% CI)",
        zorder=3
    )

    # Diagonal parity line y = x
    min_axis = min(float(all_dist_mean.min()), float(all_true.min()), float(all_gp_mean.min()))
    max_axis = max(float(all_dist_mean.max()), float(all_true.max()), float(all_gp_mean.max()))
    padding = (max_axis - min_axis) * 0.05
    line_min = max(0.0, min_axis - padding)
    line_max = max_axis + padding

    ax.plot([line_min, line_max], [line_min, line_max], 'k--', lw=1.8, zorder=5, label="Parity ($y = x$)")

    # Metric box top left
    metrics_box = (
        r"$\mathbf{Distilled\ " + args.material_model.upper() + r"\ Validation}$" + "\n"
        f"Validation Steps: {val_steps}\n"
        f"$R^2$: {r2_dist:.4f}\n"
        + rf"$\mathrm{{RMSE}}$: {rmse_dist:.4f}" + "\n"
        + rf"$\mathrm{{EC}}\ (95\%\ \mathrm{{CI}})$: {coverage_dist:.1f}%" + "\n\n"
        + r"$\mathbf{GP\ Baseline}$" + "\n"
        f"$R^2$: {r2_gp:.4f}\n"
        + rf"$\mathrm{{RMSE}}$: {rmse_gp:.4f}" + "\n"
        + rf"$\mathrm{{EC}}\ (95\%\ \mathrm{{CI}})$: {coverage_gp:.1f}%"
    )
    ax.text(
        0.05, 0.95, metrics_box,
        transform=ax.transAxes,
        verticalalignment='top',
        fontsize=10.0,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#7f8c8d', alpha=0.92, lw=1.2)
    )

    ax.set_xlabel(r"True Strain Energy Density $\Psi_{\mathrm{true}}\ (\mathrm{" + true_model_name + r"})$", fontsize=11)
    ax.set_ylabel(r"Predicted Strain Energy Density $\hat{\Psi}$", fontsize=11)
    ax.set_title(f"Validation Energy Parity & Uncertainty\nDistilled {args.material_model.upper()} vs Ground Truth ({true_model_name})", fontsize=12, fontweight='bold', pad=10)
    
    ax.set_xlim([line_min, line_max])
    ax.set_ylim([line_min, line_max])
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=9.5, framealpha=0.9)

    plt.tight_layout()

    out_pdf = os.path.join(distilled_dir, f"distilled_validation_r2_energy_{args.material_model}.pdf")
    out_png = os.path.join(distilled_dir, f"distilled_validation_r2_energy_{args.material_model}.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches='tight')
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved validation energy parity plot to:\n  - {out_pdf}\n  - {out_png}")

if __name__ == "__main__":
    main()

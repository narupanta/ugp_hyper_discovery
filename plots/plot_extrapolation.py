"""
plots/plot_extrapolation.py: Consolidated Extrapolation & Loading Modes Visualization

Evaluates and visualizes GP predictions in interpolation and extrapolation regimes
against true physics across configurable deformation modes (tension, compression, shear).
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor, AnisotropicFeatureExtractor
from core.utils import generate_standard_deformation_modes as generate_standard_modes, infer_material_model_name
from plots.theme import apply_style, save_figure, MODE_NAMES, MODE_LABELS_P, CURVE_STYLES
from plots.training import _compute_regime_transitions


def load_gp_model(saved_model_dir: str):
    """Loads SparseHyperelasticityGP model from extraction directory."""
    best_params_path = os.path.join(saved_model_dir, "best_params.npy")
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(f"Missing best_params.npy in {saved_model_dir}")

    best_params_dict = np.load(best_params_path, allow_pickle=True).item()
    valid_fields = set(GPRawParams._fields)
    filtered_params = {k: v for k, v in best_params_dict.items() if k in valid_fields}
    gp_params = GPRawParams(**filtered_params)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))

    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:3] if I_z.shape[1] == 3 else I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)

    is_aniso = "aniso30" in os.path.basename(saved_model_dir) or "aniso" in os.path.basename(saved_model_dir)
    feat_extractor = AnisotropicFeatureExtractor() if is_aniso else IsotropicFeatureExtractor()

    return SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=feat_extractor
    )


def main():
    apply_style()
    parser = argparse.ArgumentParser(description="Extrapolation and loading modes validation plot")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to saved GP model directory")
    parser.add_argument("--true_model", type=str, default=None, help="Material model name (e.g., isihara, nh2, gentthomas)")
    parser.add_argument("--max_gamma", type=float, default=2.0, help="Maximum strain gamma for extrapolation")
    parser.add_argument("--modes", type=str, default="all", help="'all', 'tension', 'compression', or comma-separated mode indices (0-5)")
    parser.add_argument("--output_name", type=str, default="extrapolation_analysis.pdf", help="Output file name")
    args = parser.parse_args()

    saved_model_dir = os.path.abspath(args.model_dir)
    model_name = args.true_model or infer_material_model_name(saved_model_dir)
    try:
        from core.material_models import get_material_from_dir
        true_model = get_material_from_dir(saved_model_dir, jit_P=False)
    except Exception:
        true_model = get_material(model_name, jit_P=False)
    learned_gp = load_gp_model(saved_model_dir)

    # Resolve modes
    all_mode_names = MODE_NAMES
    if args.modes == "all":
        selected_mode_indices = list(range(6))
    elif args.modes == "tension":
        selected_mode_indices = [0, 1, 2]
    elif args.modes == "compression":
        selected_mode_indices = [3, 4]
    else:
        selected_mode_indices = [int(x.strip()) for x in args.modes.split(",")]

    num_modes = len(selected_mode_indices)

    # Generate standard modes up to max_gamma
    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=args.max_gamma)

    # Compute or load transitions
    trans_path = os.path.join(saved_model_dir, "extrapolation_transitions.npy")
    if os.path.exists(trans_path):
        transitions = np.load(trans_path)
    else:
        transitions, _, _ = _compute_regime_transitions(learned_gp, F_all, gamma)

    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)

    psi_dist_mean = [learned_gp.psi_dist(F_all[m]).mean for m in range(6)]
    psi_dist_var = [learned_gp.psi_dist(F_all[m]).var for m in range(6)]
    P_dist_mean = [learned_gp.piola_dist(F_all[m]).mean for m in range(6)]
    P_dist_var = [learned_gp.piola_dist(F_all[m]).var for m in range(6)]

    fig, axes = plt.subplots(num_modes, 2, figsize=(12, 4.0 * num_modes), squeeze=False)
    fig.suptitle("GP Validation: Interpolation vs. Extrapolation Regimes", fontsize=16, y=1.01)

    for row_idx, mode_idx in enumerate(selected_mode_indices):
        name = all_mode_names[mode_idx]
        if name == "Pure Shear":
            idx_comp = (1, 1)
        elif name == "Simple Shear":
            idx_comp = (0, 1)
        else:
            idx_comp = (0, 0)
        label_p = MODE_LABELS_P[mode_idx]
        trans_g = float(transitions[mode_idx])

        # Column 0: Energy
        ax_psi = axes[row_idx, 0]
        ax_psi.plot(gamma, psi_true[mode_idx], **CURVE_STYLES["gt"])
        ax_psi.plot(gamma, psi_dist_mean[mode_idx], **CURVE_STYLES["gp"])
        ax_psi.fill_between(
            gamma,
            psi_dist_mean[mode_idx] - 1.96 * jnp.sqrt(psi_dist_var[mode_idx]),
            psi_dist_mean[mode_idx] + 1.96 * jnp.sqrt(psi_dist_var[mode_idx]),
            **CURVE_STYLES["gp_ci"]
        )

        ax_psi.axvspan(0, min(trans_g, args.max_gamma), color='green', alpha=0.10, label="Interpolation" if row_idx == 0 else "")
        if trans_g < args.max_gamma:
            ax_psi.axvspan(trans_g, args.max_gamma, color='red', alpha=0.10, label="Extrapolation" if row_idx == 0 else "")
            ax_psi.axvline(trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8)

        y_min, y_max = jnp.min(psi_true[mode_idx]), jnp.max(psi_true[mode_idx])
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())
        ax_psi.set_title(f"{name}: Energy ($\\Psi$)")
        ax_psi.set_xlabel(r"Stretch Measure ($\gamma$)")
        ax_psi.set_ylabel(r"$\Psi$")
        ax_psi.grid(True, alpha=0.25)
        if row_idx == 0:
            ax_psi.legend(loc="upper left")

        # Column 1: Stress
        ax_p = axes[row_idx, 1]
        p_true_comp = P_true[mode_idx, :, idx_comp[0], idx_comp[1]]
        p_mean_comp = P_dist_mean[mode_idx][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(P_dist_var[mode_idx][:, idx_comp[0], idx_comp[1]])

        ax_p.plot(gamma, p_true_comp, **CURVE_STYLES["gt"])
        ax_p.plot(gamma, p_mean_comp, **CURVE_STYLES["gp"])
        ax_p.fill_between(
            gamma,
            p_mean_comp - 1.96 * p_std_comp,
            p_mean_comp + 1.96 * p_std_comp,
            **CURVE_STYLES["gp_ci"]
        )

        ax_p.axvspan(0, min(trans_g, args.max_gamma), color='green', alpha=0.10, label="Interpolation" if row_idx == 0 else "")
        if trans_g < args.max_gamma:
            ax_p.axvspan(trans_g, args.max_gamma, color='red', alpha=0.10, label="Extrapolation" if row_idx == 0 else "")
            ax_p.axvline(trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8)

        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())
        ax_p.set_title(f"{name}: Stress ({label_p})")
        ax_p.set_xlabel(r"Stretch Measure ($\gamma$)")
        ax_p.set_ylabel(label_p)
        ax_p.grid(True, alpha=0.25)

    plt.tight_layout()
    out_pdf = os.path.join(saved_model_dir, args.output_name)
    base_name = os.path.splitext(args.output_name)[0]
    out_png = os.path.join(saved_model_dir, f"{base_name}.png")
    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)
    print(f"Extrapolation plot saved to: {out_pdf} and {out_png}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plots/plot_failed_fem_samples.py

Checks if failed_samples.npz (or failed_samples_*.npz) exists and is non-empty.
If empty or not found, safely skips.
If non-empty, plots:
  1. Strain energy predictions across the 6 standard deformation modes
     (UT, ET, PS, UC, EC, SS), comparing failed samples against the Ground Truth
     and converged/successful FEM sample distribution envelope.
  2. Material parameter distributions of failed vs successful samples.
"""

import os
import glob
import argparse
from typing import Optional, Dict, List, Any, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from core.utils import (
    generate_standard_deformation_modes,
    load_model_config,
    C_func,
    I1_func,
    I2_func,
    I3_func
)
from core.material_models import get_material_from_config

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "text.usetex": False
})


def eval_psi_3d(F_3d, p, a0=None, a1=None):
    """
    Evaluates hyperelastic strain energy density scalar psi(F)
    for isotropic and anisotropic models given parameter vector p.
    """
    C = C_func(F_3d)
    I1 = I1_func(C)
    I2 = I2_func(C)
    I3 = I3_func(C)
    I3_safe = jnp.clip(I3, 1.0e-8, 1.0e8)

    i1_dev = I3_safe ** (-1 / 3) * I1
    i2_dev = I3_safe ** (-2 / 3) * I2
    J = jnp.sqrt(I3_safe)
    i1_m3 = i1_dev - 3.0
    i2_m3 = i2_dev - 3.0
    J_m1 = J - 1.0

    dev_p = p[:10]
    vol_p = p[10:13]

    w_dev = (
        dev_p[0] * i1_m3 + dev_p[1] * i2_m3 + dev_p[2] * i1_m3**2 +
        dev_p[3] * i1_m3 * i2_m3 + dev_p[4] * i2_m3**2 + dev_p[5] * i1_m3**3 +
        dev_p[6] * (i1_m3**2) * i2_m3 + dev_p[7] * i1_m3 * (i2_m3**2) +
        dev_p[8] * i2_m3**3 + dev_p[9] * jnp.log(jnp.maximum(i2_dev / 3.0, 1e-8))
    )
    w_vol = vol_p[0] * J_m1**2 + vol_p[1] * J_m1**4 + vol_p[2] * J_m1**6

    w_aniso = 0.0
    if a0 is not None and len(p) > 13:
        C_bar = (I3_safe ** (-1 / 3))[..., None, None] * C
        I4_bar = jnp.einsum('i,...ij,j->...', a0, C_bar, a0)
        I4_m1 = I4_bar - 1.0
        aniso_p = p[13:19]
        w_aniso = aniso_p[0] * I4_m1**2 + aniso_p[1] * I4_m1**3 + aniso_p[2] * I4_m1**4
        if a1 is not None and len(p) > 16:
            I6_bar = jnp.einsum('i,...ij,j->...', a1, C_bar, a1)
            I6_m1 = I6_bar - 1.0
            w_aniso += aniso_p[3] * I6_m1**2 + aniso_p[4] * I6_m1**3 + aniso_p[5] * I6_m1**4

    return w_dev + w_vol + w_aniso


def load_failed_samples(folder_path):
    """
    Finds and aggregates all failed_samples*.npz files in folder_path.
    Returns:
        failed_samples: ndarray of shape (M, n_params) or None
        failed_reasons: ndarray of shape (M,) or None
    """
    pattern = os.path.join(folder_path, "failed_samples*.npz")
    cand_files = sorted(glob.glob(pattern))
    if not cand_files:
        return None, None

    all_samples = []
    all_reasons = []
    for f in cand_files:
        try:
            d = np.load(f, allow_pickle=True)
            if "failed_samples" in d and len(d["failed_samples"]) > 0:
                all_samples.append(d["failed_samples"])
                if "failed_reasons" in d:
                    all_reasons.append(d["failed_reasons"])
                else:
                    all_reasons.append(["Unknown error"] * len(d["failed_samples"]))
        except Exception as e:
            print(f"[WARN] Could not load failed samples file {f}: {e}")

    if not all_samples:
        return None, None

    samples_arr = np.concatenate(all_samples, axis=0)
    reasons_arr = np.concatenate(all_reasons, axis=0)
    return samples_arr, reasons_arr


def plot_failed_samples_energy_modes(
    failed_samples: np.ndarray,
    success_samples: Optional[np.ndarray],
    cfg: dict,
    save_path: str,
    stress_mode: str = "plane_strain",
    max_gamma: float = 1.0,
    num_points: int = 100
):
    """
    Plots strain energy density for failed material parameters across the 6 standard deformation modes.
    """
    # Try to load Ground Truth material model
    true_mat = None
    try:
        true_mat = get_material_from_config(cfg)
    except Exception as e:
        pass

    # Generate standard deformation modes
    F_all, gamma = generate_standard_deformation_modes(
        num_points=num_points,
        max_gamma=max_gamma,
        stress_mode=stress_mode,
        material_model=true_mat
    )
    # F_all shape: (6, num_points, 3, 3)

    # Compute Ground Truth energy
    psi_gt = None
    if true_mat is not None:
        try:
            psi_gt = jax.vmap(jax.vmap(true_mat.psi))(F_all) # shape: (6, num_points)
        except Exception as e:
            print(f"[WARN] Could not evaluate Ground Truth energy: {e}")

    # Fiber vectors if anisotropic
    a0 = getattr(true_mat, "a0", None)
    a1 = getattr(true_mat, "a1", None)

    # Vectorized energy evaluation across deformation modes
    @jax.jit
    def calc_modes_psi(p_arr):
        # p_arr: (N, 13 or 19)
        def single_p_psi(p):
            return jax.vmap(lambda F_mode: jax.vmap(lambda F: eval_psi_3d(F, p, a0, a1))(F_mode))(F_all)
        return jax.vmap(single_p_psi)(p_arr)

    # Compute successful samples envelope (if available)
    success_envelope = None
    if success_samples is not None and len(success_samples) > 0:
        sub_success = success_samples[:min(256, len(success_samples))]
        p_sub = jnp.zeros((len(sub_success), 19))
        p_sub = p_sub.at[:, :sub_success.shape[1]].set(sub_success)
        psi_succ = np.array(calc_modes_psi(p_sub)) # (N_succ, 6, num_points)
        # Percentiles
        q025 = np.percentile(psi_succ, 2.5, axis=0)
        q975 = np.percentile(psi_succ, 97.5, axis=0)
        mean_succ = np.mean(psi_succ, axis=0)
        success_envelope = (q025, q975, mean_succ)

    # Compute failed samples energy
    p_fail = jnp.zeros((len(failed_samples), 19))
    p_fail = p_fail.at[:, :failed_samples.shape[1]].set(failed_samples)
    psi_fail = np.array(calc_modes_psi(p_fail)) # (N_fail, 6, num_points)

    # Plot 2x3 or 6x1 subplots
    mode_names = [
        "(1) Uniaxial Tension (UT)",
        "(2) Equibiaxial Tension (ET)",
        "(3) Pure Shear (PS)",
        "(4) Uniaxial Compression (UC)",
        "(5) Equibiaxial Compression (EC)",
        "(6) Simple Shear (SS)"
    ]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    axes = axes.flatten()

    for m in range(6):
        ax = axes[m]
        # 1. Ground truth
        if psi_gt is not None:
            ax.plot(gamma, psi_gt[m], 'k--', lw=2.2, label="Ground Truth", zorder=5)

        # 2. Successful envelope
        if success_envelope is not None:
            q025, q975, mean_succ = success_envelope
            ax.fill_between(gamma, q025[m], q975[m], color='#2ca02c', alpha=0.25, label="Converged 95% CI", zorder=2)
            ax.plot(gamma, mean_succ[m], color='#2ca02c', lw=1.8, ls=':', label="Converged Mean", zorder=3)

        # 3. Failed sample curves
        n_fail_plot = min(15, len(failed_samples))
        for s_idx in range(n_fail_plot):
            lbl = "Failed FEM Realizations" if s_idx == 0 else None
            ax.plot(gamma, psi_fail[s_idx, m], color='#d62728', alpha=0.75, lw=1.5, ls='-', label=lbl, zorder=4)

        ax.set_title(mode_names[m], fontsize=12, fontweight='bold')
        ax.set_xlabel(r"Strain / Stretch Increment $\gamma$", fontsize=11)
        ax.set_ylabel(r"Strain Energy $\Psi$ [$\mathrm{J/m^3}$]", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--')
        
        # Adaptive y-limits: don't let extreme blowups squash the GT and converged envelope
        if success_envelope is not None:
            ref_max = max(np.max(success_envelope[1][m]), np.max(psi_gt[m]) if psi_gt is not None else 1.0)
            ref_min = min(0.0, np.min(success_envelope[0][m]))
            y_span = ref_max - ref_min
            ax.set_ylim([ref_min - 0.1 * y_span, ref_max + 1.5 * y_span])
        
        if m == 0:
            ax.legend(loc='upper left', fontsize=9.5, framealpha=0.9)

    fig.suptitle(rf"Energy Profiles of Failed FEM Parameter Realizations ($N_\mathrm{{fail}} = {len(failed_samples)}$)",
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()

    out_pdf = os.path.join(save_path, "failed_samples_energy_modes.pdf")
    out_png = os.path.join(save_path, "failed_samples_energy_modes.png")
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Saved failed samples energy modes plot:\n   {out_pdf}\n   {out_png}")


def plot_failed_samples_parameters(
    failed_samples: np.ndarray,
    success_samples: Optional[np.ndarray],
    true_dict: dict,
    save_path: str
):
    """
    Plots the parameter values of failed samples against successful/converged sample distributions.
    """
    ref_samples = success_samples if success_samples is not None else failed_samples
    
    # Active parameter indices
    active_indices = [
        i for i in range(ref_samples.shape[1])
        if np.std(ref_samples[:, i]) > 1e-5 or np.abs(np.mean(ref_samples[:, i])) > 1e-4 or
           (failed_samples is not None and (np.std(failed_samples[:, i]) > 1e-5 or np.abs(np.mean(failed_samples[:, i])) > 1e-4))
    ]

    param_names = []
    param_labels = []
    for idx in active_indices:
        if idx < 10:
            param_names.append(f"dev_p{idx}")
            if idx == 0:
                param_labels.append(r"$C_{10}$")
            elif idx == 1:
                param_labels.append(r"$C_{01}$")
            elif idx == 9:
                param_labels.append(r"$E$ (Log Term)")
            else:
                param_labels.append(rf"$\theta_{{\mathrm{{dev}}, {idx}}}$")
        elif idx < 13:
            v_idx = idx - 10
            param_names.append(f"vol_p{v_idx}")
            if v_idx == 0:
                param_labels.append(r"$D_1$ (Bulk)")
            else:
                param_labels.append(rf"$\theta_{{\mathrm{{vol}}, {v_idx}}}$")
        else:
            a_idx = idx - 13
            param_names.append(f"aniso_p{a_idx}")
            param_labels.append(rf"$\theta_{{\mathrm{{aniso}}, {a_idx}}}$")

    n_params = len(active_indices)
    if n_params == 0:
        print("[INFO] No active parameters found for failed sample distribution plot.")
        return

    fig, axes = plt.subplots(1, n_params, figsize=(4.8 * n_params, 4.2))
    if n_params == 1:
        axes = [axes]

    for i, (idx, name, lbl) in enumerate(zip(active_indices, param_names, param_labels)):
        ax = axes[i]
        
        # 1. Success histogram
        if success_samples is not None and len(success_samples) > 0:
            succ_vals = success_samples[:, idx]
            ax.hist(succ_vals, bins=25, density=True, alpha=0.35, color='#1f77b4', edgecolor='black', lw=0.6, label="Converged Posterior")
            ax.axvline(np.mean(succ_vals), color='navy', linestyle='-', lw=1.8, label=rf"Conv Mean: {np.mean(succ_vals):.4f}")

        # 2. Failed values
        fail_vals = failed_samples[:, idx]
        # Strip/jitter plot of failed points along the bottom
        y_scatter = np.zeros_like(fail_vals)
        ax.scatter(fail_vals, y_scatter, color='#d62728', marker='x', s=60, lw=2.0, zorder=6, label=f"Failed ({len(fail_vals)})")
        
        # Also plot vertical lines or rug plot for failed samples
        for fv in fail_vals:
            ax.axvline(fv, color='#d62728', linestyle='--', alpha=0.6, lw=1.2)

        # 3. Ground truth if known
        if name in true_dict:
            true_v = true_dict[name]
            ax.axvline(true_v, color='black', linestyle=':', lw=2.2, label=rf"True: {true_v:.4f}")

        ax.set_title(lbl, fontsize=13, fontweight='bold')
        ax.set_xlabel("Parameter Value", fontsize=11)
        ax.set_ylabel("Probability Density", fontsize=11)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc='best', fontsize=9.0, framealpha=0.9)

    fig.suptitle(rf"Parameter Locations of Non-Convergent FEM Realizations ($N_\mathrm{{fail}} = {len(failed_samples)}$)",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    out_pdf = os.path.join(save_path, "failed_samples_parameters.pdf")
    out_png = os.path.join(save_path, "failed_samples_parameters.png")
    fig.savefig(out_pdf, bbox_inches='tight')
    fig.savefig(out_png, bbox_inches='tight', dpi=300)
    plt.close(fig)
    print(f"✅ Saved failed samples parameter distribution plot:\n   {out_pdf}\n   {out_png}")


def check_and_plot_failed_samples(folder_path: str, save_path: Optional[str] = None):
    """
    Main function call:
    Checks if failed_samples is present in folder_path.
    If empty or non-existent, skips gracefully.
    If non-empty, generates the 6 deformation modes energy plot and parameter distribution plot.
    """
    target_dir = os.path.abspath(folder_path)
    failed_samples, failed_reasons = load_failed_samples(target_dir)

    if failed_samples is None or len(failed_samples) == 0:
        print(f"[INFO] No failed FEM parameter samples found in {target_dir}. Skipping failed sample plotting.")
        return False

    print(f"[INFO] Found {len(failed_samples)} non-convergent FEM realizations in {target_dir}. Generating diagnostic plots...")
    out_dir = save_path or target_dir
    os.makedirs(out_dir, exist_ok=True)

    # Attempt to load successful samples
    succ_file = os.path.join(target_dir, "fem_distilled_samples.npz")
    success_samples = None
    stress_mode = "plane_strain"
    if os.path.exists(succ_file):
        try:
            d_succ = np.load(succ_file, allow_pickle=True)
            success_samples = d_succ.get("selected_samples", None)
            stress_mode = str(d_succ.get("stress_mode", "plane_strain"))
        except Exception:
            pass

    # Attempt to load experiment config and true parameters
    seed_dir = target_dir
    while seed_dir != "/" and not os.path.basename(seed_dir).isdigit():
        seed_dir = os.path.dirname(seed_dir)
    exp_dir = os.path.dirname(seed_dir) if os.path.basename(seed_dir).isdigit() else target_dir

    cfg = {}
    try:
        cfg = load_model_config(target_dir)
    except Exception:
        pass

    true_dict = {}
    mat_p = cfg.get("material_params", {})
    dev_p = mat_p.get("dev_params", [])
    vol_p = mat_p.get("vol_params", [])
    for i, v in enumerate(dev_p):
        true_dict[f"dev_p{i}"] = float(v)
    for i, v in enumerate(vol_p):
        true_dict[f"vol_p{i}"] = float(v)

    # 1. Plot energy across 6 deformation modes
    plot_failed_samples_energy_modes(
        failed_samples=failed_samples,
        success_samples=success_samples,
        cfg=cfg,
        save_path=out_dir,
        stress_mode=stress_mode
    )

    # 2. Plot parameter distribution comparison
    plot_failed_samples_parameters(
        failed_samples=failed_samples,
        success_samples=success_samples,
        true_dict=true_dict,
        save_path=out_dir
    )

    return True


def main():
    parser = argparse.ArgumentParser(description="Check and plot failed FEM parameter realizations.")
    parser.add_argument(
        "--folder", type=str, required=True,
        help="Path to directory containing failed_samples.npz (or fem_validation/{block,holes})"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Custom output directory for plots"
    )
    args = parser.parse_args()

    # If folder contains block or holes, check them too
    subdirs = ["block", "holes"]
    checked_any = False
    for sub in subdirs:
        sub_p = os.path.join(args.folder, sub)
        if os.path.exists(sub_p):
            check_and_plot_failed_samples(sub_p, args.output_dir or sub_p)
            checked_any = True

    if not checked_any:
        check_and_plot_failed_samples(args.folder, args.output_dir)


if __name__ == "__main__":
    main()

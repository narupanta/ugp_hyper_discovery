#!/usr/bin/env python3
"""
plots/plot_beta_candidates_reaction_force.py

Evaluates predicted reaction forces per load step for all beta candidates in:
results/.../1/extracted_candidates/

For each beta candidate:
1. Loads the GP posterior (best_params.npy, I_z.npy, metadata.json).
2. Evaluates N=256 pathwise GP realizations across all 20 load steps on both
   observed displacement u_exp and clean displacement u_true.
3. Computes exact ground truth reaction forces R_true and compares against noisy observations R_obs.
4. Calculates R^2, RMSE, and Empirical Coverage (EC) for Rx, Ry, and combined.
5. Saves 2-panel breakdown plot (reaction_force_breakdown.png/.pdf) inside each candidate folder.
6. Generates a global multi-panel comparison plot and a metric comparison curve vs beta.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jr

from core.utils import deformation_gradient_element, transformation_jacobian
from core.dataclass import GPRawParams
from core.model import SparseHyperelasticityGP
from plots.theme import apply_style, save_figure
from plots.plot_reaction_force_distilled import format_sci, piola_stress_2d


def run_beta_reaction_force_evaluation(
    candidates_dir: str,
    fem_data_file: str,
    n_samples: int = 256,
    batch_size: int = 16,
    artifact_dir: str = None
):
    apply_style()
    cand_path = Path(candidates_dir).resolve()
    print(f"=== Evaluating Reaction Forces across Beta Candidates in: {cand_path} ===")

    # 1. Load FEM mesh and displacement data
    data = np.load(fem_data_file, allow_pickle=True)
    coords = jnp.array(data["node_coords"])
    cells = jnp.array(data["cells"])
    node_type = np.array(data["node_type"])
    stress_mode = str(data["stress_mode"]).lower()
    n_nodes = coords.shape[0]

    coords_elems = coords[cells]
    J = transformation_jacobian(coords_elems)
    dA = 0.5 * jnp.abs(jnp.linalg.det(J))
    is_fix_x = jnp.array(node_type[:, 1] == 1)
    is_fix_y = jnp.array(node_type[:, 2] == 1)

    u_exp = jnp.array(data["u_exp"])
    u_true = jnp.array(data["u_true"])
    loads_obs = np.array(data["loads"])
    n_steps = u_exp.shape[0]
    steps_arr = np.arange(n_steps)

    lam3_exp = jnp.array(data["lam3"]) if "lam3" in data else None
    lam3_true = jnp.array(data["lam3_true"]) if "lam3_true" in data else None

    # Precompute F_3d for u_exp and u_true
    def build_f3d(u_all, lam3_all):
        F_list = []
        dNdX_out = None
        for s in range(n_steps):
            disp_elems = u_all[s][cells]
            F_cells, dNdX = deformation_gradient_element(coords_elems, disp_elems)
            F_3d = jnp.eye(3, dtype=jnp.float64)[None, :, :].repeat(F_cells.shape[0], axis=0).at[:, :2, :2].set(F_cells)
            if stress_mode == "plane_stress" and lam3_all is not None:
                F_3d = F_3d.at[:, 2, 2].set(lam3_all[s])
            F_list.append(F_3d)
            if dNdX_out is None:
                dNdX_out = dNdX
        return jnp.stack(F_list, axis=0), dNdX_out

    F_3d_exp, dNdX = build_f3d(u_exp, lam3_exp)
    F_3d_true, _ = build_f3d(u_true, lam3_true)

    # 2. Compute exact ground truth reaction forces
    p_gt = jnp.array([0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5, 0.0])
    rx_gt, ry_gt = [], []
    for s in range(n_steps):
        disp_elems = u_true[s][cells]
        F_cells, _ = deformation_gradient_element(coords_elems, disp_elems)
        P_cells = jax.vmap(lambda f: piola_stress_2d(f, p_gt, stress_mode=stress_mode))(F_cells)
        f_elem = jnp.einsum('cij,cnj->cni', P_cells, dNdX) * dA[:, None, None]
        f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
        for a in range(3):
            f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
        rx_gt.append(-float(jnp.sum(f_int[is_fix_x, 0])))
        ry_gt.append(-float(jnp.sum(f_int[is_fix_y, 1])))
    rx_gt = np.array(rx_gt)
    ry_gt = np.array(ry_gt)

    # 3. Discover all beta candidates
    subdirs = sorted([d for d in cand_path.iterdir() if d.is_dir() and d.name.startswith("beta_")],
                     key=lambda p: float(p.name.replace("beta_", "")))
    print(f"Found {len(subdirs)} candidates: {[d.name for d in subdirs]}")

    results = []

    keys = jr.split(jr.PRNGKey(42), n_samples)

    for cand_dir in subdirs:
        beta_str = cand_dir.name.replace("beta_", "")
        beta_val = float(beta_str)
        print(f"\n--- Processing Candidate beta = {beta_val} ({cand_dir.name}) ---")

        # Load GP parameters
        best_params_dict = np.load(cand_dir / "best_params.npy", allow_pickle=True).item()
        valid_fields = set(GPRawParams._fields)
        gp_params = GPRawParams(**{k: v for k, v in best_params_dict.items() if k in valid_fields})
        I_z = jnp.load(cand_dir / "I_z.npy")
        dev_z = I_z[:, :2]
        vol_z = I_z[:, 2:3]
        min_dev = jnp.min(dev_z, axis=0)
        min_vol = jnp.min(vol_z, axis=0)
        max_dev = jnp.max(dev_z, axis=0)
        max_vol = jnp.max(vol_z, axis=0)

        metadata_path = cand_dir / "metadata.json"
        cov_mode = "diag"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                cov_mode = json.load(f).get("covariance_mode", "diag")

        gp = SparseHyperelasticityGP(
            gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
            beta=beta_val, covariance_mode=cov_mode
        )

        def eval_one_sample(k, F_3d_seq):
            def eval_step(F_step):
                P_3d = jax.vmap(lambda f: gp.piola(f, k))(F_step)
                P_2d = P_3d[:, :2, :2]
                f_elem = jnp.einsum('cij,cnj->cni', P_2d, dNdX) * dA[:, None, None]
                f_int = jnp.zeros((n_nodes, 2), dtype=jnp.float64)
                for a in range(3):
                    f_int = f_int.at[cells[:, a]].add(f_elem[:, a])
                rx = -jnp.sum(jnp.where(is_fix_x[:, None], f_int, 0.0)[:, 0])
                ry = -jnp.sum(jnp.where(is_fix_y[:, None], f_int, 0.0)[:, 1])
                return rx, ry
            return jax.vmap(eval_step)(F_3d_seq)

        @jax.jit
        def eval_batch_exp(b_keys):
            return jax.vmap(lambda k: eval_one_sample(k, F_3d_exp))(b_keys)

        @jax.jit
        def eval_batch_true(b_keys):
            return jax.vmap(lambda k: eval_one_sample(k, F_3d_true))(b_keys)

        rx_exp_list, ry_exp_list = [], []
        rx_true_list, ry_true_list = [], []

        for b in range(0, n_samples, batch_size):
            b_keys = keys[b:b + batch_size]
            rx_eb, ry_eb = eval_batch_exp(b_keys)
            rx_tb, ry_tb = eval_batch_true(b_keys)
            rx_exp_list.append(np.array(rx_eb))
            ry_exp_list.append(np.array(ry_eb))
            rx_true_list.append(np.array(rx_tb))
            ry_true_list.append(np.array(ry_tb))

        rx_exp = np.concatenate(rx_exp_list, axis=0)   # (n_samples, 20)
        ry_exp = np.concatenate(ry_exp_list, axis=0)   # (n_samples, 20)
        rx_true = np.concatenate(rx_true_list, axis=0) # (n_samples, 20)
        ry_true = np.concatenate(ry_true_list, axis=0) # (n_samples, 20)

        # Calculate statistics
        mu_rx_exp, std_rx_exp = np.mean(rx_exp, axis=0), np.std(rx_exp, axis=0)
        mu_ry_exp, std_ry_exp = np.mean(ry_exp, axis=0), np.std(ry_exp, axis=0)
        mu_rx_true, std_rx_true = np.mean(rx_true, axis=0), np.std(rx_true, axis=0)
        mu_ry_true, std_ry_true = np.mean(ry_true, axis=0), np.std(ry_true, axis=0)

        ci_low_xe, ci_high_xe = mu_rx_exp - 1.96 * std_rx_exp, mu_rx_exp + 1.96 * std_rx_exp
        ci_low_ye, ci_high_ye = mu_ry_exp - 1.96 * std_ry_exp, mu_ry_exp + 1.96 * std_ry_exp
        ci_low_xt, ci_high_xt = mu_rx_true - 1.96 * std_rx_true, mu_rx_true + 1.96 * std_rx_true
        ci_low_yt, ci_high_yt = mu_ry_true - 1.96 * std_ry_true, mu_ry_true + 1.96 * std_ry_true

        # Coverage evaluation against observed noisy loads
        cov_xe_obs = (loads_obs[:, 0] >= ci_low_xe) & (loads_obs[:, 0] <= ci_high_xe)
        cov_ye_obs = (loads_obs[:, 1] >= ci_low_ye) & (loads_obs[:, 1] <= ci_high_ye)
        ec_xe_obs = float(np.mean(cov_xe_obs) * 100.0)
        ec_ye_obs = float(np.mean(cov_ye_obs) * 100.0)
        tot_ec_obs = float((np.sum(cov_xe_obs) + np.sum(cov_ye_obs)) / (2 * n_steps) * 100.0)

        # Coverage evaluation against exact ground truth loads
        cov_xe_true = (rx_gt >= ci_low_xe) & (rx_gt <= ci_high_xe)
        cov_ye_true = (ry_gt >= ci_low_ye) & (ry_gt <= ci_high_ye)
        ec_xe_true = float(np.mean(cov_xe_true) * 100.0)
        ec_ye_true = float(np.mean(cov_ye_true) * 100.0)
        tot_ec_true = float((np.sum(cov_xe_true) + np.sum(cov_ye_true)) / (2 * n_steps) * 100.0)

        # RMSE and R^2 (evaluated on observed loads)
        rmse_rx = float(np.sqrt(np.mean((loads_obs[:, 0] - mu_rx_exp)**2)))
        rmse_ry = float(np.sqrt(np.mean((loads_obs[:, 1] - mu_ry_exp)**2)))
        rmse_tot = float(np.sqrt(0.5 * (rmse_rx**2 + rmse_ry**2)))

        ss_tot_x = np.sum((loads_obs[:, 0] - np.mean(loads_obs[:, 0]))**2)
        ss_res_x = np.sum((loads_obs[:, 0] - mu_rx_exp)**2)
        r2_rx = float(1.0 - ss_res_x / (ss_tot_x + 1e-12))

        ss_tot_y = np.sum((loads_obs[:, 1] - np.mean(loads_obs[:, 1]))**2)
        ss_res_y = np.sum((loads_obs[:, 1] - mu_ry_exp)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12))

        ss_tot_all = np.sum((loads_obs.flatten() - np.mean(loads_obs.flatten()))**2)
        ss_res_all = np.sum((loads_obs[:, 0] - mu_rx_exp)**2) + np.sum((loads_obs[:, 1] - mu_ry_exp)**2)
        r2_tot = float(1.0 - ss_res_all / (ss_tot_all + 1e-12))

        # Save cache
        cache_out = cand_dir / "reaction_forces_eval.npz"
        np.savez_compressed(
            cache_out,
            rx_exp=rx_exp, ry_exp=ry_exp,
            rx_true=rx_true, ry_true=ry_true,
            rx_gt=rx_gt, ry_gt=ry_gt,
            loads_obs=loads_obs,
            beta=beta_val
        )

        cand_metrics = {
            "beta": beta_val,
            "name": cand_dir.name,
            "r2_rx": r2_rx,
            "r2_ry": r2_ry,
            "r2_tot": r2_tot,
            "rmse_rx": rmse_rx,
            "rmse_ry": rmse_ry,
            "rmse_tot": rmse_tot,
            "ec_rx_obs": ec_xe_obs,
            "ec_ry_obs": ec_ye_obs,
            "tot_ec_obs": tot_ec_obs,
            "ec_rx_true": ec_xe_true,
            "ec_ry_true": ec_ye_true,
            "tot_ec_true": tot_ec_true,
            "mu_rx_exp": mu_rx_exp,
            "std_rx_exp": std_rx_exp,
            "mu_ry_exp": mu_ry_exp,
            "std_ry_exp": std_ry_exp,
        }
        results.append(cand_metrics)

        print(f"  R^2: total={r2_tot:.4f} (Rx={r2_rx:.4f}, Ry={r2_ry:.4f})")
        print(f"  RMSE: total={rmse_tot:.4f} (Rx={rmse_rx:.4f}, Ry={rmse_ry:.4f})")
        print(f"  EC (Obs): total={tot_ec_obs:.1f}% (Rx={ec_xe_obs:.1f}%, Ry={ec_ye_obs:.1f}%)")

        # -------------------------------------------------------------
        # Generate 2-Panel Breakdown Plot for this candidate
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.2), dpi=300)

        # Panel (a): Shear Reaction Force Rx
        ax_x = axes[0]
        ax_x.plot(steps_arr, rx_gt, 'k-d', lw=2.0, ms=6, label=r"Ground Truth $R_{\mathrm{true}, x}$", zorder=4)
        ax_x.plot(steps_arr, loads_obs[:, 0], 's', color='#08306b', mec='black', mew=1.5, ms=7,
                  label=r"Observed $R_{\mathrm{obs}, x}$ (Noisy)", zorder=5)
        ax_x.plot(steps_arr, mu_rx_true, ':', color='#41b6c4', lw=2.2, label=r"Model on True Disp $\mathbf{u}_{\mathrm{true}}$ (Mean)")
        ax_x.fill_between(steps_arr, ci_low_xt, ci_high_xt, color='#41b6c4', alpha=0.25, label=r"$95\%$ CI on $\mathbf{u}_{\mathrm{true}}$")
        ax_x.plot(steps_arr, mu_rx_exp, '--', color='#1f77b4', lw=2.2, label=r"Model on Obs Disp $\mathbf{u}_{\mathrm{exp}}$ (Mean)")
        ax_x.fill_between(steps_arr, ci_low_xe, ci_high_xe, color='#1f77b4', alpha=0.25, label=r"$95\%$ CI on $\mathbf{u}_{\mathrm{exp}}$")

        ax_x.set_title(rf"(a) Shear Reaction Force $R_x$ ($\beta = {beta_val}$)", fontsize=13, pad=10)
        ax_x.set_xlabel("Load Step (0 to 19)", fontsize=12)
        ax_x.set_ylabel(r"Shear Reaction Force $R_x$", fontsize=12)
        ax_x.set_xticks(steps_arr)
        ax_x.grid(True, linestyle="--", alpha=0.35)
        ax_x.legend(loc="upper left", fontsize=9.2, framealpha=0.92)

        box_x = (
            f"$r^2_{{R_x}} = {r2_rx:.4f}$\n"
            f"RMSE$_{{R_x}} = {format_sci(rmse_rx)}$\n"
            f"EC (Obs) = {ec_xe_obs:.1f}% ({int(np.sum(cov_xe_obs))}/{n_steps})\n"
            f"EC (True) = {ec_xe_true:.1f}% ({int(np.sum(cov_xe_true))}/{n_steps})"
        )
        ax_x.text(0.96, 0.05, box_x, transform=ax_x.transAxes, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=9.2)

        # Panel (b): Tensile Reaction Force Ry
        ax_y = axes[1]
        ax_y.plot(steps_arr, ry_gt, 'k-d', lw=2.0, ms=6, label=r"Ground Truth $R_{\mathrm{true}, y}$", zorder=4)
        ax_y.plot(steps_arr, loads_obs[:, 1], 's', color='#00441b', mec='black', mew=1.5, ms=7,
                  label=r"Observed $R_{\mathrm{obs}, y}$ (Noisy)", zorder=5)
        ax_y.plot(steps_arr, mu_ry_true, ':', color='#74c476', lw=2.2, label=r"Model on True Disp $\mathbf{u}_{\mathrm{true}}$ (Mean)")
        ax_y.fill_between(steps_arr, ci_low_yt, ci_high_yt, color='#74c476', alpha=0.25, label=r"$95\%$ CI on $\mathbf{u}_{\mathrm{true}}$")
        ax_y.plot(steps_arr, mu_ry_exp, '--', color='#2ca02c', lw=2.2, label=r"Model on Obs Disp $\mathbf{u}_{\mathrm{exp}}$ (Mean)")
        ax_y.fill_between(steps_arr, ci_low_ye, ci_high_ye, color='#2ca02c', alpha=0.25, label=r"$95\%$ CI on $\mathbf{u}_{\mathrm{exp}}$")

        ax_y.set_title(rf"(b) Tensile Reaction Force $R_y$ ($\beta = {beta_val}$)", fontsize=13, pad=10)
        ax_y.set_xlabel("Load Step (0 to 19)", fontsize=12)
        ax_y.set_ylabel(r"Tensile Reaction Force $R_y$", fontsize=12)
        ax_y.set_xticks(steps_arr)
        ax_y.grid(True, linestyle="--", alpha=0.35)
        ax_y.legend(loc="upper left", fontsize=9.2, framealpha=0.92)

        box_y = (
            f"$r^2_{{R_y}} = {r2_ry:.4f}$\n"
            f"RMSE$_{{R_y}} = {format_sci(rmse_ry)}$\n"
            f"EC (Obs) = {ec_ye_obs:.1f}% ({int(np.sum(cov_ye_obs))}/{n_steps})\n"
            f"EC (True) = {ec_ye_true:.1f}% ({int(np.sum(cov_ye_true))}/{n_steps})"
        )
        ax_y.text(0.96, 0.05, box_y, transform=ax_y.transAxes, verticalalignment='bottom', horizontalalignment='right',
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'), fontsize=9.2)

        plt.tight_layout()
        plot_out = cand_dir / "reaction_force_block_all_steps_breakdown.pdf"
        save_figure(fig, str(plot_out), make_png=True)
        plt.close(fig)

        if artifact_dir:
            art_png = Path(artifact_dir) / f"reaction_force_breakdown_beta_{beta_val}.png"
            art_pdf = Path(artifact_dir) / f"reaction_force_breakdown_beta_{beta_val}.pdf"
            src_png = cand_dir / "reaction_force_block_all_steps_breakdown.png"
            if src_png.exists():
                import shutil
                shutil.copy(src_png, art_png)
                shutil.copy(plot_out, art_pdf)

    # -------------------------------------------------------------
    # 4. Generate Comparative Progression Figure Across All Betas
    # -------------------------------------------------------------
    print("\n--- Generating Consolidated Multi-Beta Comparison Plots ---")
    betas = [r["beta"] for r in results]
    r2_vals = [r["r2_tot"] for r in results]
    rmse_vals = [r["rmse_tot"] for r in results]
    ec_obs_vals = [r["tot_ec_obs"] for r in results]
    ec_true_vals = [r["tot_ec_true"] for r in results]

    # (a) Metrics vs Beta curve
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.8), dpi=300)

    # Subplot 1: R^2 vs Beta
    axes[0].plot(betas, r2_vals, 'o-', color='#1f77b4', lw=2.2, ms=6)
    axes[0].set_xscale('log')
    axes[0].set_xlabel(r"Regularization Weight $\beta$", fontsize=11)
    axes[0].set_ylabel(r"Total $R^2$ (Reaction Force)", fontsize=11)
    axes[0].set_title(r"Accuracy ($R^2$ vs. $\beta$)", fontsize=12)
    axes[0].grid(True, linestyle="--", alpha=0.35)

    # Subplot 2: RMSE vs Beta
    axes[1].plot(betas, rmse_vals, 's-', color='#d62728', lw=2.2, ms=6)
    axes[1].set_xscale('log')
    axes[1].set_xlabel(r"Regularization Weight $\beta$", fontsize=11)
    axes[1].set_ylabel(r"RMSE (N/mm)", fontsize=11)
    axes[1].set_title(r"Force Residuals (RMSE vs. $\beta$)", fontsize=12)
    axes[1].grid(True, linestyle="--", alpha=0.35)

    # Subplot 3: Empirical Coverage vs Beta
    axes[2].plot(betas, ec_obs_vals, '^-', color='#2ca02c', lw=2.2, ms=6, label="Observed Force")
    axes[2].plot(betas, ec_true_vals, 'd--', color='#17becf', lw=2.0, ms=5, label="Ground Truth Force")
    axes[2].axhline(95.0, color='gray', linestyle=':', lw=1.5, label="Nominal 95% Target")
    axes[2].set_xscale('log')
    axes[2].set_xlabel(r"Regularization Weight $\beta$", fontsize=11)
    axes[2].set_ylabel("Empirical Coverage (%)", fontsize=11)
    axes[2].set_title(r"Uncertainty Calibration (EC vs. $\beta$)", fontsize=12)
    axes[2].set_ylim(-5, 105)
    axes[2].grid(True, linestyle="--", alpha=0.35)
    axes[2].legend(loc="lower right", fontsize=9.5)

    plt.tight_layout()
    metrics_plot_out = cand_path / "reaction_force_metrics_vs_beta.pdf"
    save_figure(fig, str(metrics_plot_out), make_png=True)
    plt.close(fig)

    # (b) 3x3 Grid of All Betas (Shear and Tensile)
    fig, axes = plt.subplots(3, 3, figsize=(18.0, 14.0), dpi=300)
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        b_val = r["beta"]
        mu_x = r["mu_rx_exp"]
        std_x = r["std_rx_exp"]
        mu_y = r["mu_ry_exp"]
        std_y = r["std_ry_exp"]

        # Plot observed points
        ax.plot(steps_arr, loads_obs[:, 0], 's', color='#08306b', ms=4, label=r"$R_{\mathrm{obs}, x}$")
        ax.plot(steps_arr, loads_obs[:, 1], 's', color='#00441b', ms=4, label=r"$R_{\mathrm{obs}, y}$")

        # Plot model predictions
        ax.plot(steps_arr, mu_x, '--', color='#1f77b4', lw=1.6, label=r"GP $R_x$")
        ax.fill_between(steps_arr, mu_x - 1.96 * std_x, mu_x + 1.96 * std_x, color='#1f77b4', alpha=0.2)

        ax.plot(steps_arr, mu_y, '--', color='#2ca02c', lw=1.6, label=r"GP $R_y$")
        ax.fill_between(steps_arr, mu_y - 1.96 * std_y, mu_y + 1.96 * std_y, color='#2ca02c', alpha=0.2)

        ax.set_title(rf"$\beta = {b_val}$ ($R^2$: {r['r2_tot']:.3f}, EC: {r['tot_ec_obs']:.1f}%)", fontsize=11)
        ax.set_xlabel("Load Step", fontsize=9)
        ax.set_ylabel("Force", fontsize=9)
        ax.grid(True, linestyle="--", alpha=0.25)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=7.8)

    plt.tight_layout()
    grid_plot_out = cand_path / "all_betas_reaction_forces_grid.pdf"
    save_figure(fig, str(grid_plot_out), make_png=True)
    plt.close(fig)

    if artifact_dir:
        import shutil
        shutil.copy(cand_path / "reaction_force_metrics_vs_beta.png", Path(artifact_dir) / "reaction_force_metrics_vs_beta.png")
        shutil.copy(cand_path / "reaction_force_metrics_vs_beta.pdf", Path(artifact_dir) / "reaction_force_metrics_vs_beta.pdf")
        shutil.copy(cand_path / "all_betas_reaction_forces_grid.png", Path(artifact_dir) / "all_betas_reaction_forces_grid.png")
        shutil.copy(cand_path / "all_betas_reaction_forces_grid.pdf", Path(artifact_dir) / "all_betas_reaction_forces_grid.pdf")

    # 5. Save summary JSON
    summary_out = cand_path / "reaction_force_beta_comparison.json"
    with open(summary_out, "w") as f:
        # Filter non-serializable ndarrays
        clean_res = []
        for r in results:
            clean_res.append({k: v for k, v in r.items() if not isinstance(v, np.ndarray)})
        json.dump(clean_res, f, indent=4)

    print(f"\n🎉 Successfully finished evaluation! Saved summary to {summary_out}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_dir", type=str, default="results/20261002T123703_isihara_0.0005_0.05_1.0_0.5_5_0.01_isotropic_block/1/extracted_candidates")
    parser.add_argument("--fem_data_file", type=str, default="results/20261001T223754_isihara_0.0005_0.05_1.0_0.5_5_1.0_isotropic_block/1/fem_validation/block/fem_distilled_samples.npz")
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--artifact_dir", type=str, default="/root/.gemini/antigravity/brain/737a06df-42ad-49d1-9b57-e67225b038cf")
    args = parser.parse_args()

    run_beta_reaction_force_evaluation(
        candidates_dir=args.candidates_dir,
        fem_data_file=args.fem_data_file,
        n_samples=args.n_samples,
        artifact_dir=args.artifact_dir
    )

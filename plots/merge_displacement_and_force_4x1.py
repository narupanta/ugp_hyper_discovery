"""
merge_displacement_and_force_4x1.py

Generates publication-quality vertical 4x1 and matched 4x2 side-by-side figures
merging the 2D displacement field analysis (magnitude contour, nodal RMSE contour,
parity plot) and the reaction force validation plot into unified layouts.

Features:
- Exact matched subplot sizes and row alignments across Block and Holes domains.
- Compact design sized to fit ~25% of an A4 page width per column.
- Reaction force trajectories plotted from Step 1 to 20 for both geometries.
- Clean lowercase r^2 notation across all displacement and reaction force metrics.
- Produces individual 4x1 figures (Block, Holes) and pre-composited 4x2 side-by-side figures.
"""

import os
import shutil
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import r2_score

from plots.theme import apply_style
from plots.plot_reaction_force_distilled import (
    compute_distilled_reaction_forces,
    format_sci,
    get_experiment_test_steps
)
from core.conformal import apply_conformal_band


def load_cached_reaction_forces(data_file: str, cache_file: str):
    """Loads reaction force realizations from cache if available, else computes and caches them."""
    if os.path.exists(cache_file):
        try:
            cached = np.load(cache_file, allow_pickle=True)
            res = {
                "rx_all": cached["rx_all"],
                "ry_all": cached["ry_all"],
                "loads": cached["loads"],
                "n_steps": int(cached["n_steps"])
            }
            if "rx_gt" in cached:
                res["rx_gt"] = cached["rx_gt"]
            if "ry_gt" in cached:
                res["ry_gt"] = cached["ry_gt"]
            if "rx_true_all" in cached:
                res["rx_true_all"] = cached["rx_true_all"]
            if "ry_true_all" in cached:
                res["ry_true_all"] = cached["ry_true_all"]
            return res
        except Exception:
            pass

    res = compute_distilled_reaction_forces(data_file)
    try:
        save_dict = {
            "rx_all": res["rx_all"],
            "ry_all": res["ry_all"],
            "loads": res["loads"],
            "n_steps": res["n_steps"]
        }
        if "rx_gt" in res:
            save_dict["rx_gt"] = res["rx_gt"]
        if "ry_gt" in res:
            save_dict["ry_gt"] = res["ry_gt"]
        if "rx_true_all" in res:
            save_dict["rx_true_all"] = res["rx_true_all"]
        if "ry_true_all" in res:
            save_dict["ry_true_all"] = res["ry_true_all"]
        np.savez_compressed(cache_file, **save_dict)
    except Exception:
        pass

    return res


def extract_plot_data(
    geom: str,
    data_file: str,
    calib_disp_file: str,
    calib_force_file: str,
    reaction_cache_file: str,
    is_conformal: bool = False,
    alpha: float = 0.05,
    use_clean: bool = False
):
    """Extracts and prepares displacement and reaction force validation data."""
    is_block = (geom.lower() == "block")

    # 1. FEM Simulation & Displacement Data
    data = np.load(data_file, allow_pickle=True)
    coords = np.array(data["node_coords"])
    cells = np.array(data["cells"])
    n_samples = data["u_pred"].shape[0]
    n_steps_avail = data["u_pred"].shape[1]

    if use_clean:
        if "u_true" in data:
            u_ref_all = np.array(data["u_true"])
        elif "u_exp" in data:
            u_ref_all = np.array(data["u_exp"])
        else:
            u_ref_all = np.mean(data["u_pred"], axis=0)
    else:
        if "u_exp" in data:
            u_ref_all = np.array(data["u_exp"])
        elif "u_true" in data:
            u_ref_all = np.array(data["u_true"])
        else:
            u_ref_all = np.mean(data["u_pred"], axis=0)

    test_steps = get_experiment_test_steps(data_file, n_steps_avail)
    valid_test_steps = [s for s in test_steps if s < n_steps_avail]
    if not valid_test_steps:
        valid_test_steps = [n_steps_avail - 1]

    step_eval = valid_test_steps[-1]
    u_obs_step = u_ref_all[step_eval]
    u_pred_samples_step = data["u_pred"][:, step_eval]
    u_pred_mean_step = np.mean(u_pred_samples_step, axis=0)

    coords_obs = coords + u_obs_step
    coords_pred = coords + u_pred_mean_step

    def get_mag(u): return np.linalg.norm(u, axis=1)
    mag_obs = get_mag(u_obs_step)
    mag_pred = get_mag(u_pred_mean_step)
    error_step = np.sqrt(np.mean((u_obs_step - u_pred_mean_step)**2, axis=-1))

    vmin_disp = min(mag_obs.min(), mag_pred.min())
    vmax_disp = max(mag_obs.max(), mag_pred.max())

    # 2. Conformal Calibration Parameters
    q_disp = 1.0
    if os.path.exists(calib_disp_file):
        try:
            with open(calib_disp_file, "r") as f:
                c_disp = json.load(f)
            q_disp = float(c_disp.get("q_disp", 1.0))
        except Exception:
            q_disp = 1.0

    q_force = 1.0
    if os.path.exists(calib_force_file):
        try:
            with open(calib_force_file, "r") as f:
                c_force = json.load(f)
            q_force = float(c_force.get("q_force", 1.0))
        except Exception:
            q_force = 1.0

    # 3. Parity & Coverage Data across Test Steps
    u_obs_test = u_ref_all[valid_test_steps].reshape(-1, 2)
    u_pred_test_samples = data["u_pred"][:, valid_test_steps].reshape(n_samples, -1, 2)
    u_pred_test_mean = np.mean(u_pred_test_samples, axis=0)

    if is_conformal:
        _, u_test_lower, u_test_upper = apply_conformal_band(u_pred_test_samples, q_disp)
    else:
        u_test_lower = np.quantile(u_pred_test_samples, alpha / 2.0, axis=0)
        u_test_upper = np.quantile(u_pred_test_samples, 1.0 - alpha / 2.0, axis=0)

    ux_obs, uy_obs = u_obs_test[:, 0], u_obs_test[:, 1]
    ux_pred, uy_pred = u_pred_test_mean[:, 0], u_pred_test_mean[:, 1]
    ux_low, ux_high = u_test_lower[:, 0], u_test_upper[:, 0]
    uy_low, uy_high = u_test_lower[:, 1], u_test_upper[:, 1]

    cov_x = float(np.mean((ux_obs >= ux_low) & (ux_obs <= ux_high)) * 100.0)
    cov_y = float(np.mean((uy_obs >= uy_low) & (uy_obs <= uy_high)) * 100.0)
    cov_xy = float(np.mean(np.concatenate([
        (ux_obs >= ux_low) & (ux_obs <= ux_high),
        (uy_obs >= uy_low) & (uy_obs <= uy_high)
    ])) * 100.0)

    r2_x = float(r2_score(ux_obs, ux_pred))
    r2_y = float(r2_score(uy_obs, uy_pred))
    rmse_x = float(np.sqrt(np.mean((ux_pred - ux_obs)**2)))
    rmse_y = float(np.sqrt(np.mean((uy_pred - uy_obs)**2)))

    ux_err = [np.maximum(0.0, ux_pred - ux_low), np.maximum(0.0, ux_high - ux_pred)]
    uy_err = [np.maximum(0.0, uy_pred - uy_low), np.maximum(0.0, uy_high - uy_pred)]

    # 4. Reaction Force Data
    rf_data = load_cached_reaction_forces(data_file, reaction_cache_file)
    loads = np.array(rf_data["loads"])
    rx_all = np.array(rf_data["rx_all"])
    ry_all = np.array(rf_data["ry_all"])
    n_rf_steps = rf_data["n_steps"]

    return {
        "geom": geom, "is_block": is_block, "is_conformal": is_conformal, "use_clean": use_clean,
        "coords_obs": coords_obs, "coords_pred": coords_pred, "cells": cells,
        "mag_obs": mag_obs, "mag_pred": mag_pred, "error_step": error_step,
        "vmin_disp": vmin_disp, "vmax_disp": vmax_disp,
        "q_disp": q_disp, "q_force": q_force,
        "ux_obs": ux_obs, "uy_obs": uy_obs, "ux_pred": ux_pred, "uy_pred": uy_pred,
        "ux_err": ux_err, "uy_err": uy_err, "u_obs_test": u_obs_test, "u_pred_test_mean": u_pred_test_mean,
        "cov_x": cov_x, "cov_y": cov_y, "cov_xy": cov_xy,
        "r2_x": r2_x, "r2_y": r2_y, "rmse_x": rmse_x, "rmse_y": rmse_y,
        "loads": loads, "rx_all": rx_all, "ry_all": ry_all, "n_rf_steps": n_rf_steps,
        "rx_gt": rf_data.get("rx_gt", None), "ry_gt": rf_data.get("ry_gt", None),
        "rx_true_all": rf_data.get("rx_true_all", None), "ry_true_all": rf_data.get("ry_true_all", None),
        "alpha": alpha
    }


def populate_axes_column(fig, axes_col, d, show_ylabels: bool = True):
    """Fills a vertical column of 4 axes (magnitude, error, parity, reaction force) with data."""
    is_block = d["is_block"]
    is_conformal = d["is_conformal"]
    domain_label = "Block" if is_block else "Holes"

    # =============================================================
    # PANEL 1: Predicted Displacement Magnitude Field
    # =============================================================
    ax1 = axes_col[0]
    tri_obs = tri.Triangulation(d["coords_obs"][:, 0], d["coords_obs"][:, 1], d["cells"])
    tri_pred = tri.Triangulation(d["coords_pred"][:, 0], d["coords_pred"][:, 1], d["cells"])
    ax1.tripcolor(tri_obs, d["mag_obs"], cmap="Blues", alpha=0.30, vmin=d["vmin_disp"], vmax=d["vmax_disp"], zorder=1)
    ax1.triplot(tri_obs, color="#444444", linestyle=":", linewidth=0.65, alpha=0.75, zorder=2)
    im1 = ax1.tripcolor(tri_pred, d["mag_pred"], cmap="Blues", alpha=0.88, vmin=d["vmin_disp"], vmax=d["vmax_disp"], zorder=3)
    ax1.triplot(tri_pred, color="#002b4d", linestyle="-", linewidth=0.35, alpha=0.5, zorder=4)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(rf"(a) $\|\mathbf{{u}}_{{\mathrm{{pred}}}}\|$ ({domain_label})", fontsize=11.2, pad=6)

    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.08)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cbar1.set_label(r"$\|\mathbf{u}_{\mathrm{pred}}\|$", fontsize=10.0, fontweight="bold")
    cbar1.ax.tick_params(labelsize=8.8)
    cbar1.locator = ticker.MaxNLocator(nbins=4)
    cbar1.update_ticks()

    obs_line = mlines.Line2D([], [], color='#444444', linestyle=':', linewidth=1.3, label=r'Observed ($\mathbf{u}_{\mathrm{obs}}$)')
    pred_line = mlines.Line2D([], [], color='#002b4d', linestyle='-', linewidth=1.3, label=r'Predicted ($\mathbf{u}_{\mathrm{pred}}$)')
    leg_x = 0.38 if is_block else 0.30
    ax1.legend(handles=[obs_line, pred_line], loc='lower center', bbox_to_anchor=(leg_x, -0.12),
               ncol=2, frameon=False, fontsize=7.8, handlelength=1.3, borderpad=0.1, columnspacing=0.8)

    # =============================================================
    # PANEL 2: Nodal Displacement RMSE
    # =============================================================
    ax2 = axes_col[1]
    max_err = np.max(d["error_step"])
    exp_err = int(np.floor(np.log10(max_err))) if max_err > 0 else 0
    scale_err = 10.0**exp_err
    err_scaled = d["error_step"] / scale_err
    cbar2_label = rf"$\mathrm{{RMSE}}\ [\times 10^{{{exp_err}}}]$"
    im2 = ax2.tripcolor(tri_pred, err_scaled, cmap="Oranges")
    ax2.triplot(tri_pred, color="#8c2d04", linestyle="-", linewidth=0.35, alpha=0.35, zorder=2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title(rf"(b) Displacement Error ({domain_label})", fontsize=11.2, pad=6)

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.08)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation="vertical")
    cbar2.set_label(cbar2_label, fontsize=10.0, fontweight="bold")
    cbar2.ax.tick_params(labelsize=8.8)
    cbar2.locator = ticker.MaxNLocator(nbins=4)
    cbar2.update_ticks()

    # =============================================================
    # PANEL 3: Displacement Parity & Coverage
    # =============================================================
    ax3 = axes_col[2]
    ax3.errorbar(
        d["ux_obs"], d["ux_pred"], yerr=d["ux_err"], fmt="x", color="#0072B2", ecolor="#0072B2",
        alpha=0.45, label=rf"$u_x$ ({d['cov_x']:.1f}%)", markersize=4.2, capsize=0, elinewidth=0.8
    )
    ax3.errorbar(
        d["uy_obs"], d["uy_pred"], yerr=d["uy_err"], fmt="o", color="#D55E00", ecolor="#D55E00",
        alpha=0.45, label=rf"$u_y$ ({d['cov_y']:.1f}%)", markersize=3.8, capsize=0, elinewidth=0.8
    )
    all_vals = np.concatenate([d["u_obs_test"].flatten(), d["u_pred_test_mean"].flatten()])
    lims = [all_vals.min(), all_vals.max()]
    ax3.plot(lims, lims, "k--", linewidth=1.2, label="Isoline", zorder=5)
    ax3.set_xlabel(r"$u_{\mathrm{obs}}$", fontsize=10.5, labelpad=3)
    if show_ylabels:
        ax3.set_ylabel(r"$u_{\mathrm{pred}}$", fontsize=10.5, labelpad=3)
    t_pref = f"[Conformal $Q={d['q_disp']:.2f}$]" if is_conformal else "[Uncalibrated]"
    ax3.set_title(rf"(c) {t_pref} Parity ({domain_label})", fontsize=11.2, pad=6)
    ax3.tick_params(axis="both", which="major", labelsize=8.8)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=7.6)
    ax3.xaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax3.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax3.set_box_aspect(1)

    rmse_x_str = format_sci(d["rmse_x"], 2) if d["rmse_x"] < 1e-3 else f"{d['rmse_x']:.4f}"
    rmse_y_str = format_sci(d["rmse_y"], 2) if d["rmse_y"] < 1e-3 else f"{d['rmse_y']:.4f}"
    parity_stats = (
        rf"$95\%\;\mathrm{{EC}}_{{u}} = \mathbf{{{d['cov_xy']:.1f}\%}}$" + "\n"
        rf"$r^2_{{u_x}} = {d['r2_x']:.4f},\; r^2_{{u_y}} = {d['r2_y']:.4f}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{u_x}} = {rmse_x_str}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{u_y}} = {rmse_y_str}$"
    )
    ax3.text(
        0.04, 0.96, parity_stats, transform=ax3.transAxes, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
        fontsize=7.6
    )

    # =============================================================
    # PANEL 4: Reaction Force Validation (Steps 0 to 19)
    # =============================================================
    ax4 = axes_col[3]
    rf_title_prefix = "[Conformal]" if is_conformal else "[Uncalibrated]"
    steps_disp = np.arange(0, d["n_rf_steps"])
    ax4.set_xticks([0, 5, 10, 15, d["n_rf_steps"] - 1])
    ax4.set_xticks(np.arange(0, d["n_rf_steps"]), minor=True)
    ax4.set_xlim(-0.6, d["n_rf_steps"] - 0.4)
    ax4.set_xlabel(f"Load Step (0 to {d['n_rf_steps'] - 1})", fontsize=10.5)
    ax4.tick_params(axis="both", which="major", labelsize=8.8)
    ax4.tick_params(axis="x", which="minor", length=2.5)
    ax4.grid(True, alpha=0.25, linestyle="--")

    q_force = d["q_force"]
    loads = d["loads"]
    rx_all = d["rx_all"]
    ry_all = d["ry_all"]
    n_rf_steps = d["n_rf_steps"]
    alpha = d["alpha"]

    if is_block:
        r_obs_x = loads[:, 0]
        r_obs_y = loads[:, 1]
        mu_rx = np.mean(rx_all, axis=0)
        mu_ry = np.mean(ry_all, axis=0)
        std_rx = np.std(rx_all, axis=0)
        std_ry = np.std(ry_all, axis=0)

        if is_conformal:
            low_rx = mu_rx - q_force * std_rx
            high_rx = mu_rx + q_force * std_rx
            low_ry = mu_ry - q_force * std_ry
            high_ry = mu_ry + q_force * std_ry
            lbl_rx = r"Conf. $R_x$"
            lbl_ry = r"Conf. $R_y$"
        else:
            low_rx, high_rx = np.quantile(rx_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
            low_ry, high_ry = np.quantile(ry_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
            lbl_rx = r"Dist. $R_x$"
            lbl_ry = r"Dist. $R_y$"

        err_x = np.vstack([mu_rx - low_rx, high_rx - mu_rx])
        err_y = np.vstack([mu_ry - low_ry, high_ry - mu_ry])

        has_gt = ("rx_gt" in d and d["rx_gt"] is not None and "ry_gt" in d and d["ry_gt"] is not None)
        if has_gt:
            r_gt_x = d["rx_gt"]
            r_gt_y = d["ry_gt"]
            ax4.plot(steps_disp, r_gt_x, color="#084594", lw=1.2, linestyle="-.", marker="d", markersize=2.8, alpha=0.85, label=r"True $R_{\mathrm{true}, x}$")
            ax4.plot(steps_disp, r_gt_y, color="#006d2c", lw=1.2, linestyle=":", marker="^", markersize=2.8, alpha=0.85, label=r"True $R_{\mathrm{true}, y}$")

        ax4.plot(steps_disp, mu_rx, color='#1f77b4', lw=1.3, linestyle='--', alpha=0.7)
        ax4.fill_between(steps_disp, low_rx, high_rx, color='#1f77b4', alpha=0.15)
        ax4.errorbar(
            steps_disp, mu_rx, yerr=err_x, fmt='o',
            color='#1f77b4', ecolor='#1f77b4', elinewidth=1.3, capsize=2.5, markersize=3.5, label=lbl_rx
        )
        ax4.scatter(
            steps_disp, r_obs_x, color='#084594', edgecolors='black',
            marker='s', s=24, zorder=5, label=r"Obs. $R_{\mathrm{obs}, x}$"
        )

        ax4.plot(steps_disp, mu_ry, color='#2ca02c', lw=1.3, linestyle='--', alpha=0.7)
        ax4.fill_between(steps_disp, low_ry, high_ry, color='#2ca02c', alpha=0.15)
        ax4.errorbar(
            steps_disp, mu_ry, yerr=err_y, fmt='o',
            color='#2ca02c', ecolor='#2ca02c', elinewidth=1.3, capsize=2.5, markersize=3.5, label=lbl_ry
        )
        ax4.scatter(
            steps_disp, r_obs_y, color='#006d2c', edgecolors='black',
            marker='s', s=24, zorder=5, label=r"Obs. $R_{\mathrm{obs}, y}$"
        )

        all_y = [r_obs_x, r_obs_y, low_rx, high_rx, low_ry, high_ry]
        if has_gt:
            all_y.extend([r_gt_x, r_gt_y])
        flat_y = np.concatenate([np.asarray(v).flatten() for v in all_y])
        y_min, y_max = float(np.nanmin(flat_y)), float(np.nanmax(flat_y))
        pad = max(0.08 * (y_max - y_min), 0.1)
        ax4.set_ylim(min(y_min - 0.05 * (y_max - y_min), -0.05), y_max + pad)

        if show_ylabels:
            ax4.set_ylabel("Reaction Force", fontsize=10.5)
        ax4.set_title(rf"(d) {rf_title_prefix} Reaction Force (Block)", fontsize=11.2, pad=6)

        handles, labels = ax4.get_legend_handles_labels()
        ax4.legend(
            loc="upper left", fontsize=6.8, framealpha=0.92, edgecolor='#cccccc',
            ncol=2, columnspacing=0.5
        )

        hit_x = (r_obs_x >= low_rx) & (r_obs_x <= high_rx)
        hit_y = (r_obs_y >= low_ry) & (r_obs_y <= high_ry)
        ec_rx = float(np.mean(hit_x) * 100.0)
        ec_ry = float(np.mean(hit_y) * 100.0)
        total_hits = int(np.sum(hit_x) + np.sum(hit_y))
        total_count = 2 * n_rf_steps
        total_ec = float(total_hits / total_count * 100.0)

        ss_tot_x = np.sum((r_obs_x - np.mean(r_obs_x))**2)
        ss_res_x = np.sum((r_obs_x - mu_rx)**2)
        r2_rx = float(1.0 - ss_res_x / (ss_tot_x + 1e-12)) if ss_tot_x > 1e-12 else 1.0
        rmse_rx = float(np.sqrt(np.mean((r_obs_x - mu_rx)**2)))

        ss_tot_y = np.sum((r_obs_y - np.mean(r_obs_y))**2)
        ss_res_y = np.sum((r_obs_y - mu_ry)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12)) if ss_tot_y > 1e-12 else 1.0
        rmse_ry = float(np.sqrt(np.mean((r_obs_y - mu_ry)**2)))

        header_title = rf"$\mathbf{{{rf_title_prefix}\ Metrics}}\ (Q_{{0.95}}={q_force:.2f})$:" if is_conformal else rf"$\mathbf{{{rf_title_prefix}\ Metrics}}$:"
        rf_box_text = (
            header_title + "\n"
            rf"$r^2_{{R_x}}: {r2_rx:.4f} \mid \mathrm{{EC}}: {ec_rx:.0f}\%$" + "\n"
            rf"$\mathrm{{RMSE}}_{{R_x}} = {format_sci(rmse_rx, 2)}$" + "\n"
            rf"$r^2_{{R_y}}: {r2_ry:.4f} \mid \mathrm{{EC}}: {ec_ry:.0f}\%$" + "\n"
            rf"$\mathrm{{RMSE}}_{{R_y}} = {format_sci(rmse_ry, 2)}$" + "\n"
            rf"$\mathrm{{Total\ EC}}: {total_hits}/{total_count} \ ({total_ec:.1f}\%)$"
        )
        ax4.text(
            0.98, 0.04, rf_box_text, transform=ax4.transAxes,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
            fontsize=7.0
        )

    else:
        r_obs_y = loads[:, 1]
        mu_ry = np.mean(ry_all, axis=0)
        std_ry = np.std(ry_all, axis=0)

        if is_conformal:
            low_ry = mu_ry - q_force * std_ry
            high_ry = mu_ry + q_force * std_ry
            band_lbl = rf"Conf. 95% ($Q={q_force:.2f}$)"
            pts_lbl = r"Conf. $R_y$"
        else:
            low_ry, high_ry = np.quantile(ry_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
            band_lbl = r"Dist. 95% CI"
            pts_lbl = r"Dist. $R_y$"

        err_y = np.vstack([mu_ry - low_ry, high_ry - mu_ry])

        has_gt_y = ("ry_gt" in d and d["ry_gt"] is not None)
        if has_gt_y:
            r_gt_y = d["ry_gt"]
            ax4.plot(steps_disp, r_gt_y, color='#006d2c', lw=1.2, linestyle='-.', marker='d', markersize=2.8, alpha=0.85, label=r"True $R_{\mathrm{true}, y}$")

        ax4.plot(steps_disp, mu_ry, color='#2ca02c', lw=1.3, linestyle='--', alpha=0.7)
        ax4.fill_between(steps_disp, low_ry, high_ry, color='#2ca02c', alpha=0.15, label=band_lbl)
        ax4.errorbar(
            steps_disp, mu_ry, yerr=err_y, fmt='o',
            color='#2ca02c', ecolor='#2ca02c', elinewidth=1.3, capsize=2.5, markersize=3.5, label=pts_lbl
        )
        ax4.scatter(
            steps_disp, r_obs_y, color='#006d2c', edgecolors='black',
            marker='s', s=24, zorder=5, label=r"Obs. $R_{\mathrm{obs}, y}$"
        )

        all_y = [r_obs_y, low_ry, high_ry]
        if has_gt_y:
            all_y.append(r_gt_y)
        flat_y = np.concatenate([np.asarray(v).flatten() for v in all_y])
        y_min, y_max = float(np.nanmin(flat_y)), float(np.nanmax(flat_y))
        pad = max(0.08 * (y_max - y_min), 0.1)
        ax4.set_ylim(min(y_min - 0.05 * (y_max - y_min), -0.05), y_max + pad)
        if show_ylabels:
            ax4.set_ylabel(r"Reaction Force $R_y$", fontsize=10.5)
        ax4.set_title(rf"(d) {rf_title_prefix} Reaction Force (Holes)", fontsize=11.2, pad=6)
        ax4.legend(loc="upper left", fontsize=7.4, framealpha=0.92, edgecolor='#cccccc')

        hit_y = (r_obs_y >= low_ry) & (r_obs_y <= high_ry)
        ec_ry = float(np.mean(hit_y) * 100.0)
        ss_tot_y = np.sum((r_obs_y - np.mean(r_obs_y))**2)
        ss_res_y = np.sum((r_obs_y - mu_ry)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12))
        rmse_ry = float(np.sqrt(np.mean((r_obs_y - mu_ry)**2)))

        header_title = rf"$\mathbf{{{rf_title_prefix}\ Metrics}}\ (Q_{{0.95}}={q_force:.2f})$:" if is_conformal else rf"$\mathbf{{{rf_title_prefix}\ Metrics}}$:"
        rf_box_text = (
            header_title + "\n"
            rf"$r^2_{{R_y}}: {r2_ry:.4f}$" + "\n"
            rf"$\mathrm{{RMSE}}_{{R_y}} = {format_sci(rmse_ry, 2)}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_y}}: {int(np.sum(hit_y))}/{n_rf_steps} \ ({ec_ry:.1f}\%)$"
        )
        ax4.text(
            0.98, 0.04, rf_box_text, transform=ax4.transAxes,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.32", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
            fontsize=7.2
        )


def create_merged_4x1_figure(d: dict, save_path: str, fig_w: float = 3.6, fig_h: float = 13.8, make_png: bool = True):
    """Renders a single 4x1 vertical figure with exact matched canvas sizing."""
    apply_style()
    geom = d["geom"]
    is_conformal = d["is_conformal"]
    conf_tag = "conformal" if is_conformal else "raw"
    base_name = f"merged_4x1_disp_force_{conf_tag}_{geom}"

    fig, axes = plt.subplots(
        nrows=4, ncols=1,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.22, 1.22, 1.05, 1.08]}
    )
    fig.subplots_adjust(left=0.17, right=0.88, top=0.965, bottom=0.045, hspace=0.32)

    populate_axes_column(fig, axes, d, show_ylabels=True)

    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, f"{base_name}.pdf")
    out_png = os.path.join(save_path, f"{base_name}.png")

    plt.savefig(out_pdf)
    if make_png:
        plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Generated {base_name}: {out_pdf} and {out_png}")

    artifact_dir = Path("/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03")
    if artifact_dir.exists():
        try:
            shutil.copy2(out_png, artifact_dir / f"{base_name}.png")
            shutil.copy2(out_pdf, artifact_dir / f"{base_name}.pdf")
        except Exception:
            pass

    return out_png


def create_side_by_side_figure(
    d_block: dict,
    d_holes: dict,
    save_path: str,
    fig_w: float = 7.2,
    fig_h: float = 13.8,
    make_png: bool = True
):
    """Renders a 4x2 side-by-side figure with Block on left and Holes on right."""
    apply_style()
    is_conformal = d_block["is_conformal"]
    conf_tag = "conformal" if is_conformal else "raw"
    base_name = f"merged_side_by_side_{conf_tag}"

    fig, axes = plt.subplots(
        nrows=4, ncols=2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.22, 1.22, 1.05, 1.08]}
    )
    fig.subplots_adjust(left=0.09, right=0.95, top=0.965, bottom=0.045, hspace=0.32, wspace=0.28)

    populate_axes_column(fig, [axes[i, 0] for i in range(4)], d_block, show_ylabels=True)
    populate_axes_column(fig, [axes[i, 1] for i in range(4)], d_holes, show_ylabels=True)

    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, f"{base_name}.pdf")
    out_png = os.path.join(save_path, f"{base_name}.png")

    plt.savefig(out_pdf)
    if make_png:
        plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Generated {base_name}: {out_pdf} and {out_png}")

    artifact_dir = Path("/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03")
    if artifact_dir.exists():
        try:
            shutil.copy2(out_png, artifact_dir / f"{base_name}.png")
            shutil.copy2(out_pdf, artifact_dir / f"{base_name}.pdf")
        except Exception:
            pass

    return out_png


def main():
    parser = argparse.ArgumentParser(description="Merge displacement and reaction force into vertical 4x1 and 4x2 side-by-side figures")
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="/home/mmdiscovery/shared/results/20260925T123618_nh2_1e-05_0.01_1.0_0.5_8_1.0_isotropic_block",
        help="Path to experiment root directory"
    )
    parser.add_argument("--seed", type=int, default=14, help="Seed directory to load validation results from")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save final merged plots")
    parser.add_argument("--make_png", action="store_true", default=True, help="Save PNG along with PDF")

    args = parser.parse_args()
    exp_dir = Path(args.exp_dir)
    seed = args.seed
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    block_val_dir = exp_dir / str(seed) / "fem_validation" / "block"
    holes_val_dir = exp_dir / str(seed) / "fem_validation" / "holes"

    for is_conf in [False, True]:
        conf_tag = "Conformal" if is_conf else "Raw"
        print(f"\n==================================================")
        print(f"Extracting Validation Data ({conf_tag})")
        print(f"==================================================")
        d_block = extract_plot_data(
            geom="block",
            data_file=str(block_val_dir / "fem_distilled_samples.npz"),
            calib_disp_file=str(block_val_dir / "conformal_calibration_metrics.json"),
            calib_force_file=str(block_val_dir / "conformal_force_calibration.json"),
            reaction_cache_file=str(block_val_dir / "reaction_forces_cache_block.npz"),
            is_conformal=is_conf
        )
        d_holes = extract_plot_data(
            geom="holes",
            data_file=str(holes_val_dir / "fem_distilled_samples.npz"),
            calib_disp_file=str(block_val_dir / "conformal_calibration_metrics.json"),
            calib_force_file=str(block_val_dir / "conformal_force_calibration.json"),
            reaction_cache_file=str(holes_val_dir / "reaction_forces_cache_holes.npz"),
            is_conformal=is_conf
        )

        print(f"Generating 4x1 Figure for Block ({conf_tag})...")
        create_merged_4x1_figure(d_block, str(out_dir), make_png=args.make_png)

        print(f"Generating 4x1 Figure for Holes ({conf_tag})...")
        create_merged_4x1_figure(d_holes, str(out_dir), make_png=args.make_png)

        print(f"Generating 4x2 Side-by-Side Figure ({conf_tag})...")
        create_side_by_side_figure(d_block, d_holes, str(out_dir), make_png=args.make_png)


if __name__ == "__main__":
    main()

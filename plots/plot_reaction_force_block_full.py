import os
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import yaml

from plots.theme import apply_style, save_figure
from plots.plot_reaction_force_distilled import format_sci

apply_style()

def plot_full_reaction_forces(
    exp_dir: str = None,
    seed: int = 1,
    alpha: float = 0.05,
    make_png: bool = True,
    save_path: str = None,
    cache_file: str = None
):
    """
    Reruns and generates reaction force figures for block geometry across all 20 load steps (0 to 19):
    1. reaction_force_distilled_block: Unified plot showing both Rx and Ry across steps 0-19 with both Obs and True.
    2. reaction_force_conformal_block: Conformal calibrated version across steps 0-19.
    3. reaction_force_block_all_steps_breakdown_obs: 2-panel figure for Observed Displacements,
       incorporating reaction noise intervals on load cell observations and total predictive uncertainty.
    4. reaction_force_block_all_steps_breakdown_true: 2-panel figure for Clean Ground-Truth Displacements.
    5. reaction_force_block_all_steps_breakdown: Combined 2-panel breakdown figure for direct reference.
    """
    apply_style()
    if save_path is not None:
        block_dir = Path(save_path)
        val_dir = Path(exp_dir) if exp_dir else block_dir
    else:
        val_dir = Path(exp_dir)
        if (val_dir / str(seed) / "fem_validation" / "block").exists():
            block_dir = val_dir / str(seed) / "fem_validation" / "block"
        elif (val_dir / "fem_validation" / "block").exists():
            block_dir = val_dir / "fem_validation" / "block"
        elif (val_dir / "fem_distilled_samples.npz").exists():
            block_dir = val_dir
        else:
            raise FileNotFoundError(f"Could not locate block fem_validation directory in {exp_dir}")

    os.makedirs(block_dir, exist_ok=True)
    if cache_file is not None and Path(cache_file).exists():
        cache_path = Path(cache_file)
    else:
        cache_path = block_dir / "reaction_forces_cache_block.npz"
        if not cache_path.exists():
            cache_path = block_dir / "rf_all_steps_eval.npz"
        if not cache_path.exists():
            raise FileNotFoundError(f"No reaction force cache found in {block_dir}")

    data = np.load(cache_path, allow_pickle=True)
    rx_all = data["rx_all"] if "rx_all" in data else data["rx_exp"] # (1024, 20)
    ry_all = data["ry_all"] if "ry_all" in data else data["ry_exp"] # (1024, 20)
    rx_true_all = data["rx_true_all"] if "rx_true_all" in data else (data["rx_true"] if "rx_true" in data else rx_all)
    ry_true_all = data["ry_true_all"] if "ry_true_all" in data else (data["ry_true"] if "ry_true" in data else ry_all)
    rx_gt = data["rx_gt"] # (20,)
    ry_gt = data["ry_gt"] # (20,)
    loads = data["loads"] if "loads" in data else data["loads_obs"] # (20, 2)
    n_steps = rx_all.shape[1]

    # Load experiment configuration to obtain load_noise
    load_noise = 0.02
    for parent_cand in [val_dir, val_dir.parent, val_dir.parent.parent]:
        for cfg_name in ["config.yaml", "config.json"]:
            cand_p = parent_cand / cfg_name
            if cand_p.exists():
                try:
                    with open(cand_p, "r") as cf:
                        cd = yaml.safe_load(cf)
                    if cd and "load_noise" in cd:
                        load_noise = float(cd["load_noise"])
                        break
                except Exception:
                    pass

    # Load conformal scale Q_force if available
    calib_force_file = block_dir / "conformal_force_calibration.json"
    q_force = 1.0
    if calib_force_file.exists():
        try:
            with open(calib_force_file, "r") as f:
                c_data = json.load(f)
            q_force = float(c_data.get("q_force", 1.0))
        except Exception:
            q_force = 1.0

    steps = np.arange(n_steps) # 0 to 19 (start at 0)
    r_obs_x = loads[:, 0]
    r_obs_y = loads[:, 1]

    # Statistics evaluated on observed displacement
    mu_rx = np.mean(rx_all, axis=0)
    mu_ry = np.mean(ry_all, axis=0)
    std_rx = np.std(rx_all, axis=0)
    std_ry = np.std(ry_all, axis=0)

    # 95% Quantiles (Epistemic Model Credible Interval)
    q_low_x, q_high_x = np.quantile(rx_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    q_low_y, q_high_y = np.quantile(ry_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)

    # Reaction Noise intervals (Load cell measurement uncertainty)
    # sigma_noise = load_noise * |R_gt|
    sigma_noise_x = load_noise * np.abs(rx_gt)
    sigma_noise_y = load_noise * np.abs(ry_gt)

    # Total Predictive Variance and Bounds: Var_tot = Var_model + Var_noise
    sigma_tot_x = np.sqrt(std_rx**2 + sigma_noise_x**2)
    sigma_tot_y = np.sqrt(std_ry**2 + sigma_noise_y**2)
    tot_low_x = mu_rx - 1.96 * sigma_tot_x
    tot_high_x = mu_rx + 1.96 * sigma_tot_x
    tot_low_y = mu_ry - 1.96 * sigma_tot_y
    tot_high_y = mu_ry + 1.96 * sigma_tot_y

    # Conformal bounds
    conf_low_x = mu_rx - q_force * std_rx
    conf_high_x = mu_rx + q_force * std_rx
    conf_low_y = mu_ry - q_force * std_ry
    conf_high_y = mu_ry + q_force * std_ry

    # Statistics evaluated on true displacement
    mu_rx_true = np.mean(rx_true_all, axis=0)
    mu_ry_true = np.mean(ry_true_all, axis=0)
    std_rx_true = np.std(rx_true_all, axis=0)
    std_ry_true = np.std(ry_true_all, axis=0)
    q_low_x_true, q_high_x_true = np.quantile(rx_true_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)
    q_low_y_true, q_high_y_true = np.quantile(ry_true_all, [alpha / 2.0, 1.0 - alpha / 2.0], axis=0)

    def calc_stats(low_x, high_x, low_y, high_y, mu_x=mu_rx, mu_y=mu_ry, target_obs_x=r_obs_x, target_obs_y=r_obs_y):
        hit_obs_x = (target_obs_x >= low_x) & (target_obs_x <= high_x)
        hit_obs_y = (target_obs_y >= low_y) & (target_obs_y <= high_y)
        ec_obs_x = float(np.mean(hit_obs_x) * 100.0)
        ec_obs_y = float(np.mean(hit_obs_y) * 100.0)
        tot_obs_hits = int(np.sum(hit_obs_x) + np.sum(hit_obs_y))
        tot_obs_ec = float(tot_obs_hits / (2 * n_steps) * 100.0)

        hit_gt_x = (rx_gt >= low_x) & (rx_gt <= high_x)
        hit_gt_y = (ry_gt >= low_y) & (ry_gt <= high_y)
        ec_gt_x = float(np.mean(hit_gt_x) * 100.0)
        ec_gt_y = float(np.mean(hit_gt_y) * 100.0)
        tot_gt_hits = int(np.sum(hit_gt_x) + np.sum(hit_gt_y))
        tot_gt_ec = float(tot_gt_hits / (2 * n_steps) * 100.0)

        ss_tot_x = np.sum((target_obs_x - np.mean(target_obs_x))**2)
        ss_res_x = np.sum((target_obs_x - mu_x)**2)
        r2_x = float(1.0 - ss_res_x / (ss_tot_x + 1e-12)) if ss_tot_x > 1e-12 else 1.0
        rmse_x = float(np.sqrt(np.mean((target_obs_x - mu_x)**2)))

        ss_tot_y = np.sum((target_obs_y - np.mean(target_obs_y))**2)
        ss_res_y = np.sum((target_obs_y - mu_y)**2)
        r2_y = float(1.0 - ss_res_y / (ss_tot_y + 1e-12)) if ss_tot_y > 1e-12 else 1.0
        rmse_y = float(np.sqrt(np.mean((target_obs_y - mu_y)**2)))

        return {
            "hit_obs_x": hit_obs_x, "hit_obs_y": hit_obs_y, "ec_obs_x": ec_obs_x, "ec_obs_y": ec_obs_y,
            "tot_obs_hits": tot_obs_hits, "tot_obs_ec": tot_obs_ec,
            "hit_gt_x": hit_gt_x, "hit_gt_y": hit_gt_y, "ec_gt_x": ec_gt_x, "ec_gt_y": ec_gt_y,
            "tot_gt_hits": tot_gt_hits, "tot_gt_ec": tot_gt_ec,
            "r2_x": r2_x, "rmse_x": rmse_x, "r2_y": r2_y, "rmse_y": rmse_y
        }

    raw_stats = calc_stats(q_low_x, q_high_x, q_low_y, q_high_y)
    tot_pred_stats = calc_stats(tot_low_x, tot_high_x, tot_low_y, tot_high_y)
    conf_stats = calc_stats(conf_low_x, conf_high_x, conf_low_y, conf_high_y)
    true_stats = calc_stats(q_low_x_true, q_high_x_true, q_low_y_true, q_high_y_true,
                            mu_x=mu_rx_true, mu_y=mu_ry_true, target_obs_x=rx_gt, target_obs_y=ry_gt)

    # =========================================================================
    # FIGURE 1 & 2: Unified Reaction Force vs Load Step (0 to 19)
    # =========================================================================
    def render_unified_figure(low_x, high_x, low_y, high_y, stats, filename, title_prefix, is_conformal=False):
        fig, ax = plt.subplots(figsize=(9.2, 6.4))
        err_x = np.vstack([mu_rx - low_x, high_x - mu_rx])
        err_y = np.vstack([mu_ry - low_y, high_y - mu_ry])

        lbl_pred_x = rf"Conformal $R_x$ ($Q={q_force:.2f}$)" if is_conformal else r"Distilled $R_x$ (Mean $\pm$ 95% CI)"
        lbl_pred_y = rf"Conformal $R_y$ ($Q={q_force:.2f}$)" if is_conformal else r"Distilled $R_y$ (Mean $\pm$ 95% CI)"

        # Ground truth lines
        ax.plot(steps, rx_gt, color='#084594', linestyle='-.', lw=1.6, marker='d', markersize=4.8, alpha=0.85, label=r"True $R_{\mathrm{true}, x}$ (Exact)")
        ax.plot(steps, ry_gt, color='#006d2c', linestyle=':', lw=1.6, marker='^', markersize=4.8, alpha=0.85, label=r"True $R_{\mathrm{true}, y}$ (Exact)")

        # Observed measurements (noisy)
        ax.scatter(steps, r_obs_x, color='#084594', edgecolors='black', marker='s', s=45, zorder=5, label=r"Observed $R_{\mathrm{obs}, x}$ (Load Cell)")
        ax.scatter(steps, r_obs_y, color='#006d2c', edgecolors='black', marker='s', s=45, zorder=5, label=r"Observed $R_{\mathrm{obs}, y}$ (Load Cell)")

        # Model Predictions (Mean line + shaded credible ribbon + error bars)
        ax.plot(steps, mu_rx, color='#1f77b4', lw=1.6, linestyle='--', alpha=0.8)
        ax.fill_between(steps, low_x, high_x, color='#1f77b4', alpha=0.15)
        ax.errorbar(steps, mu_rx, yerr=err_x, fmt='o', color='#1f77b4', ecolor='#1f77b4',
                    elinewidth=1.4, capsize=3.0, capthick=1.2, markersize=4.5, label=lbl_pred_x)

        ax.plot(steps, mu_ry, color='#2ca02c', lw=1.6, linestyle='--', alpha=0.8)
        ax.fill_between(steps, low_y, high_y, color='#2ca02c', alpha=0.15)
        ax.errorbar(steps, mu_ry, yerr=err_y, fmt='o', color='#2ca02c', ecolor='#2ca02c',
                    elinewidth=1.4, capsize=3.0, capthick=1.2, markersize=4.5, label=lbl_pred_y)

        # Dynamic Y-limits with proper margin
        all_vals = np.concatenate([r_obs_x, r_obs_y, rx_gt, ry_gt, low_x, high_x, low_y, high_y])
        y_min, y_max = float(np.min(all_vals)), float(np.max(all_vals))
        pad = max(0.08 * (y_max - y_min), 0.2)
        ax.set_ylim(min(y_min - 0.05 * (y_max - y_min), -0.05), y_max + pad)

        ax.set_xticks(steps)
        ax.set_xticklabels([f"{s}" for s in steps], fontsize=10.5)
        ax.set_xlim(-0.6, n_steps - 0.4)
        ax.set_xlabel("Load Step (0 to 19)", fontsize=12.5)
        ax.set_ylabel("Reaction Force", fontsize=12.5)
        ax.set_title(rf"{title_prefix} Reaction Force vs. Load Step (Block, Steps 0 to 19)", fontsize=13.5)
        ax.grid(True, alpha=0.25, linestyle='--')
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, edgecolor='#cccccc', ncol=2, columnspacing=0.6)

        q_info = rf" \mid Q_{{0.95}}: {q_force:.2f}" if is_conformal else ""
        metrics_text = (
            rf"$\mathbf{{{title_prefix}\ Metrics\ (95\%\ CI){q_info}:}}$" + "\n"
            rf"$r^2_{{R_x}}: {stats['r2_x']:.4f} \mid \mathrm{{RMSE}}_{{R_x}}: {format_sci(stats['rmse_x'])}$" + "\n"
            rf"$r^2_{{R_y}}: {stats['r2_y']:.4f} \mid \mathrm{{RMSE}}_{{R_y}}: {format_sci(stats['rmse_y'])}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_x}}\ (\mathrm{{obs}}): {int(np.sum(stats['hit_obs_x']))}/{n_steps}\ ({stats['ec_obs_x']:.0f}\%) \mid (\mathrm{{true}}): {int(np.sum(stats['hit_gt_x']))}/{n_steps}\ ({stats['ec_gt_x']:.0f}\%)$" + "\n"
            rf"$\mathrm{{EC}}_{{R_y}}\ (\mathrm{{obs}}): {int(np.sum(stats['hit_obs_y']))}/{n_steps}\ ({stats['ec_obs_y']:.0f}\%) \mid (\mathrm{{true}}): {int(np.sum(stats['hit_gt_y']))}/{n_steps}\ ({stats['ec_gt_y']:.0f}\%)$" + "\n"
            rf"$\mathrm{{Total\ EC}}\ (\mathrm{{obs}}): \mathbf{{{stats['tot_obs_ec']:.1f}\%}} \mid (\mathrm{{true}}): \mathbf{{{stats['tot_gt_ec']:.1f}\%}}$"
        )
        ax.text(0.98, 0.04, metrics_text, transform=ax.transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.45', facecolor='white', alpha=0.92, edgecolor='#cccccc'),
                fontsize=8.5)

        plt.tight_layout()
        out_pdf = block_dir / f"{filename}.pdf"
        save_figure(fig, str(out_pdf), make_png=make_png)
        plt.close(fig)
        print(f"✅ Saved {filename}: {out_pdf}")

    render_unified_figure(q_low_x, q_high_x, q_low_y, q_high_y, raw_stats, "reaction_force_distilled_block", "[Uncalibrated]")
    render_unified_figure(conf_low_x, conf_high_x, conf_low_y, conf_high_y, conf_stats, "reaction_force_conformal_block", "[Conformal]", is_conformal=True)

    # =========================================================================
    # FIGURE 3: OBSERVED DISPLACEMENT BREAKDOWN (with Reaction Noise Intervals)
    # =========================================================================
    fig_obs, axes_obs = plt.subplots(nrows=1, ncols=2, figsize=(13.2, 5.8), sharex=True)

    # --- PANEL 1: Rx (Observed) ---
    ax = axes_obs[0]
    ax.plot(steps, rx_gt, color='#000000', linestyle='-', lw=1.8, marker='d', markersize=5.2, label=r"Ground Truth $R_{\mathrm{true}, x}$")

    # Load cell observation with measurement noise error bars
    ax.errorbar(
        steps, r_obs_x, yerr=1.96 * sigma_noise_x, fmt='s', color='#084594', ecolor='#4682b4',
        elinewidth=1.6, capsize=3.5, capthick=1.2, markersize=5.5, zorder=5,
        label=rf"Observed $R_{{\mathrm{{obs}}, x}}$ ($\pm 1.96\sigma_{{\mathrm{{noise}}}}$)"
    )

    # Total Predictive Interval (Model Epistemic + Reaction Noise)
    ax.fill_between(
        steps, tot_low_x, tot_high_x, color='#a6cee3', alpha=0.35,
        label=r"Total Predictive Band (Model $\pm$ Reaction Noise)"
    )
    ax.plot(steps, tot_low_x, color='#1f77b4', linestyle=':', lw=1.1, alpha=0.7)
    ax.plot(steps, tot_high_x, color='#1f77b4', linestyle=':', lw=1.1, alpha=0.7)

    # Model on Obs Disp (Mean & Epistemic Credible Interval)
    ax.plot(steps, mu_rx, color='#1f77b4', lw=1.9, linestyle='--', label=r"Model Mean on Obs Disp $\mathbf{u}_{\mathrm{exp}}$")
    ax.fill_between(steps, q_low_x, q_high_x, color='#1f77b4', alpha=0.28, label=r"Model Epistemic 95% CI")
    ax.errorbar(steps, mu_rx, yerr=np.vstack([mu_rx - q_low_x, q_high_x - mu_rx]), fmt='none', ecolor='#1f77b4', elinewidth=1.2, capsize=2.5)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    y_max_x = max(np.max(r_obs_x + 1.96 * sigma_noise_x), np.max(tot_high_x)) * 1.12
    ax.set_ylim(-0.15, y_max_x)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Shear Reaction Force $R_x$ (N)", fontsize=12)
    ax.set_title(r"(a) Shear Reaction Force $R_x$ [Observed Displacements]", fontsize=12.5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92, edgecolor='#cccccc')

    stats_obs_rx = (
        rf"$r^2_{{R_x}} = {raw_stats['r2_x']:.4f} \mid \mathrm{{RMSE}} = {format_sci(raw_stats['rmse_x'])}$" + "\n"
        rf"$\mathrm{{EC\ (Obs\ in\ Model\ CI)}} = {raw_stats['ec_obs_x']:.1f}\%\ ({int(np.sum(raw_stats['hit_obs_x']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (Obs\ in\ Total\ Band)}} = {tot_pred_stats['ec_obs_x']:.1f}\%\ ({int(np.sum(tot_pred_stats['hit_obs_x']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (GT\ in\ Model\ CI)}} = {raw_stats['ec_gt_x']:.1f}\%\ ({int(np.sum(raw_stats['hit_gt_x']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_obs_rx, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.2)

    # --- PANEL 2: Ry (Observed) ---
    ax = axes_obs[1]
    ax.plot(steps, ry_gt, color='#000000', linestyle='-', lw=1.8, marker='d', markersize=5.2, label=r"Ground Truth $R_{\mathrm{true}, y}$")

    # Load cell observation with measurement noise error bars
    ax.errorbar(
        steps, r_obs_y, yerr=1.96 * sigma_noise_y, fmt='s', color='#006d2c', ecolor='#2ca02c',
        elinewidth=1.6, capsize=3.5, capthick=1.2, markersize=5.5, zorder=5,
        label=rf"Observed $R_{{\mathrm{{obs}}, y}}$ ($\pm 1.96\sigma_{{\mathrm{{noise}}}}$)"
    )

    # Total Predictive Interval (Model Epistemic + Reaction Noise)
    ax.fill_between(
        steps, tot_low_y, tot_high_y, color='#b2df8a', alpha=0.35,
        label=r"Total Predictive Band (Model $\pm$ Reaction Noise)"
    )
    ax.plot(steps, tot_low_y, color='#2ca02c', linestyle=':', lw=1.1, alpha=0.7)
    ax.plot(steps, tot_high_y, color='#2ca02c', linestyle=':', lw=1.1, alpha=0.7)

    # Model on Obs Disp (Mean & Epistemic Credible Interval)
    ax.plot(steps, mu_ry, color='#2ca02c', lw=1.9, linestyle='--', label=r"Model Mean on Obs Disp $\mathbf{u}_{\mathrm{exp}}$")
    ax.fill_between(steps, q_low_y, q_high_y, color='#2ca02c', alpha=0.28, label=r"Model Epistemic 95% CI")
    ax.errorbar(steps, mu_ry, yerr=np.vstack([mu_ry - q_low_y, q_high_y - mu_ry]), fmt='none', ecolor='#2ca02c', elinewidth=1.2, capsize=2.5)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    y_max_y = max(np.max(r_obs_y + 1.96 * sigma_noise_y), np.max(tot_high_y)) * 1.12
    ax.set_ylim(-0.15, y_max_y)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Tensile Reaction Force $R_y$ (N)", fontsize=12)
    ax.set_title(r"(b) Tensile Reaction Force $R_y$ [Observed Displacements]", fontsize=12.5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92, edgecolor='#cccccc')

    stats_obs_ry = (
        rf"$r^2_{{R_y}} = {raw_stats['r2_y']:.4f} \mid \mathrm{{RMSE}} = {format_sci(raw_stats['rmse_y'])}$" + "\n"
        rf"$\mathrm{{EC\ (Obs\ in\ Model\ CI)}} = {raw_stats['ec_obs_y']:.1f}\%\ ({int(np.sum(raw_stats['hit_obs_y']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (Obs\ in\ Total\ Band)}} = {tot_pred_stats['ec_obs_y']:.1f}\%\ ({int(np.sum(tot_pred_stats['hit_obs_y']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (GT\ in\ Model\ CI)}} = {raw_stats['ec_gt_y']:.1f}\%\ ({int(np.sum(raw_stats['hit_gt_y']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_obs_ry, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.2)

    plt.tight_layout()
    out_obs_pdf = block_dir / "reaction_force_block_all_steps_breakdown_obs.pdf"
    save_figure(fig_obs, str(out_obs_pdf), make_png=make_png)
    plt.close(fig_obs)
    print(f"✅ Saved reaction_force_block_all_steps_breakdown_obs: {out_obs_pdf}")

    # =========================================================================
    # FIGURE 4: TRUE DISPLACEMENT BREAKDOWN (Clean Ground-Truth Kinematics)
    # =========================================================================
    fig_true, axes_true = plt.subplots(nrows=1, ncols=2, figsize=(13.2, 5.8), sharex=True)

    # --- PANEL 1: Rx (True) ---
    ax = axes_true[0]
    ax.plot(steps, rx_gt, color='#000000', linestyle='-', lw=2.0, marker='d', markersize=5.5, label=r"Ground Truth $R_{\mathrm{true}, x}$ (Exact)")
    ax.plot(steps, mu_rx_true, color='#17becf', lw=1.9, linestyle='--', label=r"Model Mean on True Disp $\mathbf{u}_{\mathrm{true}}$")
    ax.fill_between(steps, q_low_x_true, q_high_x_true, color='#17becf', alpha=0.25, label=r"Model 95% Credible Interval")
    ax.errorbar(steps, mu_rx_true, yerr=np.vstack([mu_rx_true - q_low_x_true, q_high_x_true - mu_rx_true]),
                fmt='o', color='#17becf', ecolor='#17becf', elinewidth=1.4, capsize=3.0, capthick=1.2, markersize=4.8)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    ax.set_ylim(-0.15, max(np.max(rx_gt), np.max(q_high_x_true)) * 1.12)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Shear Reaction Force $R_x$ (N)", fontsize=12)
    ax.set_title(r"(a) Shear Reaction Force $R_x$ [True Displacements]", fontsize=12.5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, edgecolor='#cccccc')

    stats_true_rx = (
        rf"$r^2_{{R_x}} = {true_stats['r2_x']:.4f}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{R_x}} = {format_sci(true_stats['rmse_x'])}$" + "\n"
        rf"$\mathrm{{EC\ (GT\ in\ Model\ CI)}} = {true_stats['ec_gt_x']:.1f}\%\ ({int(np.sum(true_stats['hit_gt_x']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_true_rx, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.5)

    # --- PANEL 2: Ry (True) ---
    ax = axes_true[1]
    ax.plot(steps, ry_gt, color='#000000', linestyle='-', lw=2.0, marker='d', markersize=5.5, label=r"Ground Truth $R_{\mathrm{true}, y}$ (Exact)")
    ax.plot(steps, mu_ry_true, color='#2ca02c', lw=1.9, linestyle='--', label=r"Model Mean on True Disp $\mathbf{u}_{\mathrm{true}}$")
    ax.fill_between(steps, q_low_y_true, q_high_y_true, color='#2ca02c', alpha=0.25, label=r"Model 95% Credible Interval")
    ax.errorbar(steps, mu_ry_true, yerr=np.vstack([mu_ry_true - q_low_y_true, q_high_y_true - mu_ry_true]),
                fmt='o', color='#2ca02c', ecolor='#2ca02c', elinewidth=1.4, capsize=3.0, capthick=1.2, markersize=4.8)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    ax.set_ylim(-0.15, max(np.max(ry_gt), np.max(q_high_y_true)) * 1.12)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Tensile Reaction Force $R_y$ (N)", fontsize=12)
    ax.set_title(r"(b) Tensile Reaction Force $R_y$ [True Displacements]", fontsize=12.5)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.92, edgecolor='#cccccc')

    stats_true_ry = (
        rf"$r^2_{{R_y}} = {true_stats['r2_y']:.4f}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{R_y}} = {format_sci(true_stats['rmse_y'])}$" + "\n"
        rf"$\mathrm{{EC\ (GT\ in\ Model\ CI)}} = {true_stats['ec_gt_y']:.1f}\%\ ({int(np.sum(true_stats['hit_gt_y']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_true_ry, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.5)

    plt.tight_layout()
    out_true_pdf = block_dir / "reaction_force_block_all_steps_breakdown_true.pdf"
    save_figure(fig_true, str(out_true_pdf), make_png=make_png)
    plt.close(fig_true)
    print(f"✅ Saved reaction_force_block_all_steps_breakdown_true: {out_true_pdf}")

    # =========================================================================
    # FIGURE 5: COMBINED BREAKDOWN FIGURE (Preserved for compatibility)
    # =========================================================================
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(13.0, 5.6), sharex=True)

    # --- PANEL 1: Rx ---
    ax = axes[0]
    ax.plot(steps, rx_gt, color='#000000', linestyle='-', lw=1.8, marker='d', markersize=5.5, label=r"Ground Truth $R_{\mathrm{true}, x}$")
    ax.scatter(steps, r_obs_x, color='#084594', edgecolors='black', marker='s', s=50, zorder=5, label=r"Observed $R_{\mathrm{obs}, x}$ (Noisy)")
    ax.plot(steps, mu_rx_true, color='#17becf', lw=1.5, linestyle=':', label=r"Model on True Disp $\mathbf{u}_{\mathrm{true}}$ (Mean)")
    ax.fill_between(steps, q_low_x_true, q_high_x_true, color='#17becf', alpha=0.18, label=r"95% CI on $\mathbf{u}_{\mathrm{true}}$")
    ax.plot(steps, mu_rx, color='#1f77b4', lw=1.8, linestyle='--', label=r"Model on Obs Disp $\mathbf{u}_{\mathrm{exp}}$ (Mean)")
    ax.fill_between(steps, q_low_x, q_high_x, color='#1f77b4', alpha=0.22, label=r"95% CI on $\mathbf{u}_{\mathrm{exp}}$")
    ax.errorbar(steps, mu_rx, yerr=np.vstack([mu_rx - q_low_x, q_high_x - mu_rx]), fmt='none', ecolor='#1f77b4', elinewidth=1.2, capsize=2.5)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    ax.set_ylim(-0.1, max(np.max(r_obs_x), np.max(q_high_x)) * 1.12)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Shear Reaction Force $R_x$", fontsize=12)
    ax.set_title(r"(a) Reaction Force $R_x$ (Full Steps 0–19)", fontsize=13)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.92, edgecolor='#cccccc')

    stats_rx = (
        rf"$r^2_{{R_x}} = {raw_stats['r2_x']:.4f}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{R_x}} = {format_sci(raw_stats['rmse_x'])}$" + "\n"
        rf"$\mathrm{{EC\ (Obs)}} = {raw_stats['ec_obs_x']:.1f}\% \ ({int(np.sum(raw_stats['hit_obs_x']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (True)}} = {raw_stats['ec_gt_x']:.1f}\% \ ({int(np.sum(raw_stats['hit_gt_x']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_rx, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.2)

    # --- PANEL 2: Ry ---
    ax = axes[1]
    ax.plot(steps, ry_gt, color='#000000', linestyle='-', lw=1.8, marker='d', markersize=5.5, label=r"Ground Truth $R_{\mathrm{true}, y}$")
    ax.scatter(steps, r_obs_y, color='#006d2c', edgecolors='black', marker='s', s=50, zorder=5, label=r"Observed $R_{\mathrm{obs}, y}$ (Noisy)")
    ax.plot(steps, mu_ry_true, color='#85e085', lw=1.5, linestyle=':', label=r"Model on True Disp $\mathbf{u}_{\mathrm{true}}$ (Mean)")
    ax.fill_between(steps, q_low_y_true, q_high_y_true, color='#85e085', alpha=0.18, label=r"95% CI on $\mathbf{u}_{\mathrm{true}}$")
    ax.plot(steps, mu_ry, color='#2ca02c', lw=1.8, linestyle='--', label=r"Model on Obs Disp $\mathbf{u}_{\mathrm{exp}}$ (Mean)")
    ax.fill_between(steps, q_low_y, q_high_y, color='#2ca02c', alpha=0.22, label=r"95% CI on $\mathbf{u}_{\mathrm{exp}}$")
    ax.errorbar(steps, mu_ry, yerr=np.vstack([mu_ry - q_low_y, q_high_y - mu_ry]), fmt='none', ecolor='#2ca02c', elinewidth=1.2, capsize=2.5)

    ax.set_xticks(steps)
    ax.set_xticklabels([f"{s}" for s in steps], fontsize=9.5)
    ax.set_xlim(-0.6, n_steps - 0.4)
    ax.set_ylim(-0.1, max(np.max(r_obs_y), np.max(q_high_y)) * 1.12)
    ax.set_xlabel("Load Step (0 to 19)", fontsize=12)
    ax.set_ylabel(r"Tensile Reaction Force $R_y$", fontsize=12)
    ax.set_title(r"(b) Reaction Force $R_y$ (Full Steps 0–19)", fontsize=13)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc="upper left", fontsize=8.2, framealpha=0.92, edgecolor='#cccccc')

    stats_ry = (
        rf"$r^2_{{R_y}} = {raw_stats['r2_y']:.4f}$" + "\n"
        rf"$\mathrm{{RMSE}}_{{R_y}} = {format_sci(raw_stats['rmse_y'])}$" + "\n"
        rf"$\mathrm{{EC\ (Obs)}} = {raw_stats['ec_obs_y']:.1f}\% \ ({int(np.sum(raw_stats['hit_obs_y']))}/{n_steps})$" + "\n"
        rf"$\mathrm{{EC\ (True)}} = {raw_stats['ec_gt_y']:.1f}\% \ ({int(np.sum(raw_stats['hit_gt_y']))}/{n_steps})$"
    )
    ax.text(0.96, 0.04, stats_ry, transform=ax.transAxes, va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.92, edgecolor='#cccccc'), fontsize=8.2)

    plt.tight_layout()
    out_breakdown_pdf = block_dir / "reaction_force_block_all_steps_breakdown.pdf"
    save_figure(fig, str(out_breakdown_pdf), make_png=make_png)
    plt.close(fig)
    print(f"✅ Saved reaction_force_block_all_steps_breakdown: {out_breakdown_pdf}")

    # Copy to artifact directories
    artifact_dirs = [
        Path("/root/.gemini/antigravity/brain/737a06df-42ad-49d1-9b57-e67225b038cf"),
        Path("/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03")
    ]
    file_bases = [
        "reaction_force_distilled_block",
        "reaction_force_conformal_block",
        "reaction_force_block_all_steps_breakdown",
        "reaction_force_block_all_steps_breakdown_obs",
        "reaction_force_block_all_steps_breakdown_true"
    ]
    for ad in artifact_dirs:
        if ad.exists():
            for fbase in file_bases:
                src_png = block_dir / f"{fbase}.png"
                if src_png.exists():
                    import shutil
                    shutil.copy2(src_png, ad / f"{fbase}.png")
                    if (block_dir / f"{fbase}.pdf").exists():
                        shutil.copy2(block_dir / f"{fbase}.pdf", ad / f"{fbase}.pdf")

    return {
        "raw_stats": raw_stats,
        "tot_pred_stats": tot_pred_stats,
        "true_stats": true_stats
    }

def main():
    parser = argparse.ArgumentParser(description="Generate full loadstep reaction force figures (Steps 0 to 19)")
    parser.add_argument("--exp_dir", type=str, default=None, help="Experiment directory")
    parser.add_argument("--seed", type=int, default=1, help="Seed index")
    parser.add_argument("--save_path", type=str, default=None, help="Output directory to save plots")
    parser.add_argument("--cache_file", type=str, default=None, help="Path to reaction_forces_cache_block.npz")
    args = parser.parse_args()

    plot_full_reaction_forces(
        exp_dir=args.exp_dir,
        seed=args.seed,
        save_path=args.save_path,
        cache_file=args.cache_file
    )

if __name__ == "__main__":
    main()

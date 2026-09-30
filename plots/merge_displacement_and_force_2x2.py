import os
import sys
import json
import shutil
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
from plots.plot_reaction_force_distilled import format_sci
from plots.merge_displacement_and_force_4x1 import extract_plot_data, load_cached_reaction_forces

apply_style()

def create_merged_2x2_figure(
    d: dict,
    save_path: str,
    fig_w: float = 6.4,
    fig_h: float = 5.8,
    make_png: bool = True
):
    """
    Renders a 2x2 grid figure:
      [0, 0]: Predicted Displacement Magnitude |u_pred|
      [0, 1]: Displacement Error (RMSE)
      [1, 0]: Parity plot
      [1, 1]: Reaction Force validation
    """
    apply_style()
    geom = d["geom"]
    is_block = d["is_block"]
    is_conformal = d["is_conformal"]
    use_clean = d.get("use_clean", False)
    clean_tag = "_clean" if use_clean else ""
    conf_tag = "conformal" if is_conformal else "raw"
    base_name = f"merged_2x2_disp_force_{conf_tag}{clean_tag}_{geom}"

    fig, axes = plt.subplots(
        nrows=2, ncols=2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.0]}
    )
    # generous spacing for clear legends, labels, and ticks
    fig.subplots_adjust(left=0.10, right=0.92, top=0.93, bottom=0.10, hspace=0.30, wspace=0.32)

    # =============================================================
    # TOP-LEFT [0, 0]: Predicted Displacement Magnitude Field
    # =============================================================
    ax1 = axes[0, 0]
    tri_obs = tri.Triangulation(d["coords_obs"][:, 0], d["coords_obs"][:, 1], d["cells"])
    tri_pred = tri.Triangulation(d["coords_pred"][:, 0], d["coords_pred"][:, 1], d["cells"])
    ax1.tripcolor(tri_obs, d["mag_obs"], cmap="Blues", alpha=0.30, vmin=d["vmin_disp"], vmax=d["vmax_disp"], zorder=1)
    ax1.triplot(tri_obs, color="#444444", linestyle=":", linewidth=0.65, alpha=0.75, zorder=2)
    im1 = ax1.tripcolor(tri_pred, d["mag_pred"], cmap="Blues", alpha=0.88, vmin=d["vmin_disp"], vmax=d["vmax_disp"], zorder=3)
    ax1.triplot(tri_pred, color="#002b4d", linestyle="-", linewidth=0.35, alpha=0.5, zorder=4)
    ax1.set_aspect("equal")
    ax1.axis("off")
    ax1.set_title(r"$\|\mathbf{u}_{\mathrm{pred}}\|$", fontsize=11.2, pad=6)

    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.08)
    cbar1 = fig.colorbar(im1, cax=cax1, orientation="vertical")
    cbar1.set_label(r"$\|\mathbf{u}_{\mathrm{pred}}\|$", fontsize=9.5, fontweight="bold")
    cbar1.ax.tick_params(labelsize=8.5)
    cbar1.locator = ticker.MaxNLocator(nbins=4)
    cbar1.update_ticks()

    obs_label = r"Clean ($\mathbf{u}_{\mathrm{clean}}$)" if use_clean else r"Obs. ($\mathbf{u}_{\mathrm{obs}}$)"
    obs_line = mlines.Line2D([], [], color="#444444", linestyle=":", linewidth=1.3, label=obs_label)
    pred_line = mlines.Line2D([], [], color="#002b4d", linestyle="-", linewidth=1.3, label=r"Pred. ($\mathbf{u}_{\mathrm{pred}}$)")
    leg_x = 0.40 if is_block else 0.32
    ax1.legend(
        handles=[obs_line, pred_line], loc="lower center", bbox_to_anchor=(leg_x, -0.14),
        ncol=2, frameon=False, fontsize=7.6, handlelength=1.2, borderpad=0.1, columnspacing=0.8
    )

    # =============================================================
    # TOP-RIGHT [0, 1]: Displacement Error (RMSE)
    # =============================================================
    ax2 = axes[0, 1]
    max_err = np.max(d["error_step"])
    exp_err = int(np.floor(np.log10(max_err))) if max_err > 0 else 0
    scale_err = 10.0**exp_err
    err_scaled = d["error_step"] / scale_err
    cbar2_label = rf"$\mathrm{{RMSE}}\ [\times 10^{{{exp_err}}}]$"
    im2 = ax2.tripcolor(tri_pred, err_scaled, cmap="Oranges")
    ax2.triplot(tri_pred, color="#8c2d04", linestyle="-", linewidth=0.35, alpha=0.35, zorder=2)
    ax2.set_aspect("equal")
    ax2.axis("off")
    ax2.set_title("Displacement Error", fontsize=11.2, pad=6)

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.08)
    cbar2 = fig.colorbar(im2, cax=cax2, orientation="vertical")
    cbar2.set_label(cbar2_label, fontsize=9.5, fontweight="bold")
    cbar2.ax.tick_params(labelsize=8.5)
    cbar2.locator = ticker.MaxNLocator(nbins=4)
    cbar2.update_ticks()

    # =============================================================
    # BOTTOM-LEFT [1, 0]: Displacement Parity & Coverage
    # =============================================================
    ax3 = axes[1, 0]
    ax3.errorbar(
        d["ux_obs"], d["ux_pred"], yerr=d["ux_err"], fmt="x", color="#0072B2", ecolor="#0072B2",
        alpha=0.45, label=rf"$u_x$ ({d['cov_x']:.1f}%)", markersize=4.0, capsize=0, elinewidth=0.8
    )
    ax3.errorbar(
        d["uy_obs"], d["uy_pred"], yerr=d["uy_err"], fmt="o", color="#D55E00", ecolor="#D55E00",
        alpha=0.45, label=rf"$u_y$ ({d['cov_y']:.1f}%)", markersize=3.6, capsize=0, elinewidth=0.8
    )
    all_vals = np.concatenate([d["u_obs_test"].flatten(), d["u_pred_test_mean"].flatten()])
    lims = [all_vals.min(), all_vals.max()]
    ax3.plot(lims, lims, "k--", linewidth=1.2, label="Isoline", zorder=5)
    xlabel_str = r"$u_{\mathrm{clean}}$" if use_clean else r"$u_{\mathrm{obs}}$"
    ax3.set_xlabel(xlabel_str, fontsize=10.0, labelpad=3)
    ax3.set_ylabel(r"$u_{\mathrm{pred}}$", fontsize=10.0, labelpad=3)
    t_pref = f"[Conformal $Q={d['q_disp']:.2f}$] " if is_conformal else ""
    ax3.set_title(rf"{t_pref}Parity", fontsize=11.2, pad=6)
    ax3.tick_params(axis="both", which="major", labelsize=8.5)
    ax3.grid(True, linestyle=":", alpha=0.6)
    ax3.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.92, edgecolor="#cccccc", fontsize=7.4)
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
        bbox=dict(boxstyle="round,pad=0.30", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
        fontsize=7.2
    )

    # =============================================================
    # BOTTOM-RIGHT [1, 1]: Reaction Force Validation
    # =============================================================
    ax4 = axes[1, 1]
    rf_title_prefix = f"[Conformal $Q={d['q_force']:.2f}$] " if is_conformal else ""
    steps_disp = np.arange(1, d["n_rf_steps"] + 1)
    ax4.set_xticks([1, 5, 10, 15, 20])
    ax4.set_xticks(np.arange(1, 21), minor=True)
    ax4.set_xlim(0.3, 20.7)
    ax4.set_xlabel("Load Step (1 to 20)", fontsize=10.0)
    ax4.tick_params(axis="both", which="major", labelsize=8.5)
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

        ax4.plot(steps_disp, mu_rx, color="#1f77b4", lw=1.3, linestyle="--", alpha=0.7)
        ax4.fill_between(steps_disp, low_rx, high_rx, color="#1f77b4", alpha=0.15)
        ax4.errorbar(
            steps_disp, mu_rx, yerr=err_x, fmt="o",
            color="#1f77b4", ecolor="#1f77b4", elinewidth=1.3, capsize=2.5, markersize=3.5, label=lbl_rx
        )
        ax4.scatter(
            steps_disp, r_obs_x, color="#084594", edgecolors="black",
            marker="s", s=22, zorder=5, label=r"Obs. $R_{\mathrm{obs}, x}$"
        )

        ax4.plot(steps_disp, mu_ry, color="#2ca02c", lw=1.3, linestyle="--", alpha=0.7)
        ax4.fill_between(steps_disp, low_ry, high_ry, color="#2ca02c", alpha=0.15)
        ax4.errorbar(
            steps_disp, mu_ry, yerr=err_y, fmt="o",
            color="#2ca02c", ecolor="#2ca02c", elinewidth=1.3, capsize=2.5, markersize=3.5, label=lbl_ry
        )
        ax4.scatter(
            steps_disp, r_obs_y, color="#006d2c", edgecolors="black",
            marker="s", s=22, zorder=5, label=r"Obs. $R_{\mathrm{obs}, y}$"
        )

        ax4.set_ylim(-0.08, 1.68)
        ax4.set_ylabel("Reaction Force", fontsize=10.0)
        ax4.set_title(rf"{rf_title_prefix}Reaction Force", fontsize=11.2, pad=6)

        handles, labels = ax4.get_legend_handles_labels()
        order = [1, 0, 3, 2]
        ax4.legend(
            [handles[i] for i in order], [labels[i] for i in order],
            loc="upper left", fontsize=7.0, framealpha=0.92, edgecolor="#cccccc",
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

        header_title = rf"$\mathbf{{Conformal\ Metrics}}\ (Q_{{0.95}}={q_force:.2f})$:" if is_conformal else r"$\mathbf{Metrics}$:"
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
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
            fontsize=6.8
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

        ax4.plot(steps_disp, mu_ry, color="#2ca02c", lw=1.3, linestyle="--", alpha=0.7)
        ax4.fill_between(steps_disp, low_ry, high_ry, color="#2ca02c", alpha=0.15, label=band_lbl)
        ax4.errorbar(
            steps_disp, mu_ry, yerr=err_y, fmt="o",
            color="#2ca02c", ecolor="#2ca02c", elinewidth=1.3, capsize=2.5, markersize=3.5, label=pts_lbl
        )
        ax4.scatter(
            steps_disp, r_obs_y, color="#006d2c", edgecolors="black",
            marker="s", s=22, zorder=5, label=r"Obs. $R_{\mathrm{obs}, y}$"
        )

        ax4.set_ylim(-0.08, 1.55)
        ax4.set_ylabel(r"Reaction Force $R_y$", fontsize=10.0)
        ax4.set_title(rf"{rf_title_prefix}Reaction Force", fontsize=11.2, pad=6)
        ax4.legend(loc="upper left", fontsize=7.2, framealpha=0.92, edgecolor="#cccccc")

        hit_y = (r_obs_y >= low_ry) & (r_obs_y <= high_ry)
        ec_ry = float(np.mean(hit_y) * 100.0)
        ss_tot_y = np.sum((r_obs_y - np.mean(r_obs_y))**2)
        ss_res_y = np.sum((r_obs_y - mu_ry)**2)
        r2_ry = float(1.0 - ss_res_y / (ss_tot_y + 1e-12))
        rmse_ry = float(np.sqrt(np.mean((r_obs_y - mu_ry)**2)))

        header_title = rf"$\mathbf{{Conformal\ Metrics}}\ (Q_{{0.95}}={q_force:.2f})$:" if is_conformal else r"$\mathbf{Metrics}$:"
        rf_box_text = (
            header_title + "\n"
            rf"$r^2_{{R_y}}: {r2_ry:.4f}$" + "\n"
            rf"$\mathrm{{RMSE}}_{{R_y}} = {format_sci(rmse_ry, 2)}$" + "\n"
            rf"$\mathrm{{EC}}_{{R_y}}: {int(np.sum(hit_y))}/{n_rf_steps} \ ({ec_ry:.1f}\%)$"
        )
        ax4.text(
            0.98, 0.04, rf_box_text, transform=ax4.transAxes,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="white", alpha=0.92, edgecolor="#cccccc", lw=0.6),
            fontsize=7.0
        )

    ax4.set_box_aspect(1)

    os.makedirs(save_path, exist_ok=True)
    out_pdf = os.path.join(save_path, f"{base_name}.pdf")
    out_png = os.path.join(save_path, f"{base_name}.png")

    plt.savefig(out_pdf)
    if make_png:
        plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"✅ Generated 2x2 figure {base_name}: {out_pdf} and {out_png}")

    artifact_dir = Path("/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03")
    if artifact_dir.exists():
        try:
            shutil.copy2(out_png, artifact_dir / f"{base_name}.png")
            shutil.copy2(out_pdf, artifact_dir / f"{base_name}.pdf")
        except Exception:
            pass

    return out_png

def main():
    parser = argparse.ArgumentParser(description="Generate 2x2 merged displacement and force figures.")
    parser.add_argument("--exp_dir", type=str, required=True, help="Path to experiment directory.")
    parser.add_argument("--seed", type=int, default=14, help="Random seed directory to evaluate.")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save figures.")
    parser.add_argument("--make_png", action="store_true", default=True, help="Save PNG in addition to PDF.")
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir)
    seed = args.seed
    out_dir = Path(args.out_dir) if args.out_dir else exp_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    block_val_dir = exp_dir / str(seed) / "fem_validation" / "block"
    holes_val_dir = exp_dir / str(seed) / "fem_validation" / "holes"

    for use_clean in [False, True]:
        for is_conf in [False, True]:
            conf_tag = "Conformal" if is_conf else "Raw"
            clean_tag = " (Clean FEM)" if use_clean else " (Observed)"
            print(f"\n==================================================")
            print(f"Extracting 2x2 Validation Data ({conf_tag}{clean_tag})")
            print(f"==================================================")
            d_block = extract_plot_data(
                geom="block",
                data_file=str(block_val_dir / "fem_distilled_samples.npz"),
                calib_disp_file=str(block_val_dir / "conformal_calibration_metrics.json"),
                calib_force_file=str(block_val_dir / "conformal_force_calibration.json"),
                reaction_cache_file=str(block_val_dir / "reaction_forces_cache_block.npz"),
                is_conformal=is_conf,
                use_clean=use_clean
            )
            d_holes = extract_plot_data(
                geom="holes",
                data_file=str(holes_val_dir / "fem_distilled_samples.npz"),
                calib_disp_file=str(block_val_dir / "conformal_calibration_metrics.json"),
                calib_force_file=str(block_val_dir / "conformal_force_calibration.json"),
                reaction_cache_file=str(holes_val_dir / "reaction_forces_cache_holes.npz"),
                is_conformal=is_conf,
                use_clean=use_clean
            )

            print(f"Generating 2x2 Figure for Block ({conf_tag}{clean_tag})...")
            create_merged_2x2_figure(d_block, str(out_dir), make_png=args.make_png)

            print(f"Generating 2x2 Figure for Holes ({conf_tag}{clean_tag})...")
            create_merged_2x2_figure(d_holes, str(out_dir), make_png=args.make_png)

if __name__ == "__main__":
    main()

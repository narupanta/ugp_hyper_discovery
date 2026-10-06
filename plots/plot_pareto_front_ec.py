#!/usr/bin/env python3
"""
plots/plot_pareto_front_ec.py

Plots the Pareto Front for:
1. Empirical Coverage (EC) vs R^2 (both maximized)
2. Empirical Coverage (EC) vs RMSE (EC maximized, RMSE minimized)

Analyzes both:
- Full-Field Reaction Force evaluation (true displacement & observed displacement)
- GP Stress / Invariant Validation set

Generates:
1. 1x2 focused figure for Reaction Force (EC vs R2, EC vs RMSE)
2. 2x2 comprehensive figure covering both Reaction Force and GP Stress Validation
3. JSON summary of non-dominated vs dominated points across all metrics
"""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from plots.theme import apply_style, save_figure


def compute_pareto_front_2d(x: np.ndarray, y: np.ndarray, maximize_x: bool = True, maximize_y: bool = True):
    """
    Computes indices of non-dominated (Pareto optimal) points for two objectives.
    """
    n = len(x)
    is_pareto = np.ones(n, dtype=bool)
    
    # Transform so both are maximization
    x_eff = x if maximize_x else -x
    y_eff = y if maximize_y else -y
    
    for i in range(n):
        for j in range(n):
            if i != j:
                # If j dominates i:
                # j is at least as good in both, and strictly better in at least one
                if (x_eff[j] >= x_eff[i] and y_eff[j] >= y_eff[i]) and (x_eff[j] > x_eff[i] or y_eff[j] > y_eff[i]):
                    is_pareto[i] = False
                    break
                    
    pareto_indices = np.where(is_pareto)[0]
    # Sort Pareto points along x
    sorted_order = np.argsort(x[pareto_indices])
    return pareto_indices[sorted_order]


def plot_pareto_front():
    apply_style("paper")
    
    base_dir = Path("/home/mmdiscovery/shared/results/20261002T123703_isihara_0.0005_0.05_1.0_0.5_5_0.01_isotropic_block/1/extracted_candidates")
    rx_json_path = base_dir / "reaction_force_beta_comparison.json"
    sel_json_path = base_dir / "beta_selection_summary.json"
    
    with open(rx_json_path, "r") as f:
        rx_data = json.load(f)
        
    with open(sel_json_path, "r") as f:
        sel_data = json.load(f)
        
    betas = np.array([item["beta"] for item in rx_data])
    names = [item["name"] for item in rx_data]
    
    # Reaction Force Metrics
    ec_rx_true = np.array([item["tot_ec_true"] for item in rx_data])
    ec_rx_obs  = np.array([item["tot_ec_obs"] for item in rx_data])
    r2_rx_tot  = np.array([item["r2_tot"] for item in rx_data])
    rmse_rx_tot= np.array([item["rmse_tot"] for item in rx_data])
    
    # Stress Validation Metrics
    # Match candidate by beta
    sel_cand_map = {c["beta"]: c for c in sel_data["candidates"]}
    ec_val = np.array([sel_cand_map[b]["ec_val"] for b in betas])
    r2_val = np.array([sel_cand_map[b]["r2_val"] for b in betas])
    rmse_val = np.array([sel_cand_map[b]["rmse_val"] for b in betas])
    
    # ---------------------------------------------------------
    # 1. Focused 1x2 Plot: Reaction Force EC vs R2 and EC vs RMSE
    # ---------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    
    # (a) EC vs R^2 (True Disp)
    # Objectives: Maximize EC (x-axis), Maximize R2 (y-axis)
    pareto_r2_true = compute_pareto_front_2d(ec_rx_true, r2_rx_tot, maximize_x=True, maximize_y=True)
    pareto_r2_obs  = compute_pareto_front_2d(ec_rx_obs,  r2_rx_tot, maximize_x=True, maximize_y=True)
    
    # Step/Line connecting Pareto points
    ax1.plot(ec_rx_true[pareto_r2_true], r2_rx_tot[pareto_r2_true], color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.8, label="Pareto Front (True Disp)")
    ax1.plot(ec_rx_obs[pareto_r2_obs],   r2_rx_tot[pareto_r2_obs],   color="#ff7f0e", linestyle=":",  linewidth=1.8, alpha=0.8, label="Pareto Front (Obs Disp)")
    
    # Scatter all points
    # True Disp
    ax1.scatter(ec_rx_true, r2_rx_tot, color="#1f77b4", s=65, edgecolors="navy", linewidth=1.2, zorder=4, label=r"Candidates ($\mathbf{u}_{\mathrm{true}}$)")
    # Obs Disp
    ax1.scatter(ec_rx_obs, r2_rx_tot, color="#ff7f0e", marker="^", s=65, edgecolors="#b25900", linewidth=1.2, zorder=4, label=r"Candidates ($\mathbf{u}_{\mathrm{exp}}$)")
    
    # Highlight Selected Beta = 20.0
    idx_20 = np.where(betas == 20.0)[0][0]
    ax1.scatter(ec_rx_true[idx_20], r2_rx_tot[idx_20], color="#d62728", marker="*", s=260, edgecolors="black", linewidth=1.5, zorder=6, label=r"$\beta=20.0$ (Selected Winner)")
    ax1.scatter(ec_rx_obs[idx_20],  r2_rx_tot[idx_20],  color="#d62728", marker="*", s=260, edgecolors="black", linewidth=1.5, zorder=6)
    
    # Annotate beta labels
    for i, b in enumerate(betas):
        # Annotate True disp
        offset_y = 0.003 if b != 0.1 else -0.007
        ax1.annotate(f"$\\beta={b}$", (ec_rx_true[i], r2_rx_tot[i]), textcoords="offset points", xytext=(6, offset_y * 1000), fontsize=8.5, color="#0b3c5d", fontweight="medium")
        
    # Mark target / ideal point
    ax1.scatter([95.0], [1.0], marker="X", s=140, color="green", zorder=5, label="Nominal Target (95%, 1.0)")
    ax1.axvline(95.0, color="green", linestyle="--", alpha=0.3, linewidth=1.0)
    ax1.axhline(1.0, color="black", linestyle=":", alpha=0.3, linewidth=1.0)
    
    ax1.set_xlabel("Empirical Coverage EC (%) [Higher is better $\\rightarrow$]", fontweight="bold")
    ax1.set_ylabel(r"Total Reaction Force $R^2$ [Higher is better $\rightarrow$]", fontweight="bold")
    ax1.set_title(r"(a) Reaction Force: EC vs. $R^2$", fontsize=12, pad=10)
    ax1.set_xlim(-2, 102)
    ax1.set_ylim(0.90, 1.01)
    ax1.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    ax1.grid(True, alpha=0.25, linestyle="--")
    
    # (b) EC vs RMSE
    # Objectives: Maximize EC (x-axis), Minimize RMSE (y-axis)
    pareto_rmse_true = compute_pareto_front_2d(ec_rx_true, rmse_rx_tot, maximize_x=True, maximize_y=False)
    pareto_rmse_obs  = compute_pareto_front_2d(ec_rx_obs,  rmse_rx_tot, maximize_x=True, maximize_y=False)
    
    # Pareto boundary lines
    ax2.plot(ec_rx_true[pareto_rmse_true], rmse_rx_tot[pareto_rmse_true], color="#1f77b4", linestyle="--", linewidth=1.8, alpha=0.8, label="Pareto Front (True Disp)")
    ax2.plot(ec_rx_obs[pareto_rmse_obs],   rmse_rx_tot[pareto_rmse_obs],   color="#ff7f0e", linestyle=":",  linewidth=1.8, alpha=0.8, label="Pareto Front (Obs Disp)")
    
    # Scatter
    ax2.scatter(ec_rx_true, rmse_rx_tot, color="#1f77b4", s=65, edgecolors="navy", linewidth=1.2, zorder=4, label=r"Candidates ($\mathbf{u}_{\mathrm{true}}$)")
    ax2.scatter(ec_rx_obs,  rmse_rx_tot, color="#ff7f0e", marker="^", s=65, edgecolors="#b25900", linewidth=1.2, zorder=4, label=r"Candidates ($\mathbf{u}_{\mathrm{exp}}$)")
    
    # Highlight Selected Beta = 20.0
    ax2.scatter(ec_rx_true[idx_20], rmse_rx_tot[idx_20], color="#d62728", marker="*", s=260, edgecolors="black", linewidth=1.5, zorder=6, label=r"$\beta=20.0$ (Selected Winner)")
    ax2.scatter(ec_rx_obs[idx_20],  rmse_rx_tot[idx_20],  color="#d62728", marker="*", s=260, edgecolors="black", linewidth=1.5, zorder=6)
    
    # Annotate beta labels
    for i, b in enumerate(betas):
        offset_y = 5 if b != 0.1 else -15
        ax2.annotate(f"$\\beta={b}$", (ec_rx_true[i], rmse_rx_tot[i]), textcoords="offset points", xytext=(6, offset_y), fontsize=8.5, color="#0b3c5d", fontweight="medium")
        
    # Mark target / ideal point
    ax2.scatter([95.0], [0.0], marker="X", s=140, color="green", zorder=5, label="Nominal Target (95%, 0.0 N)")
    ax2.axvline(95.0, color="green", linestyle="--", alpha=0.3, linewidth=1.0)
    ax2.axhline(0.0, color="black", linestyle=":", alpha=0.3, linewidth=1.0)
    
    ax2.set_xlabel("Empirical Coverage EC (%) [Higher is better $\\rightarrow$]", fontweight="bold")
    ax2.set_ylabel(r"Total Reaction Force RMSE (N) [Lower is better $\leftarrow$]", fontweight="bold")
    ax2.set_title("(b) Reaction Force: EC vs. RMSE", fontsize=12, pad=10)
    ax2.set_xlim(-2, 102)
    ax2.set_ylim(-0.03, 0.80)
    ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9)
    ax2.grid(True, alpha=0.25, linestyle="--")
    
    # Save 1x2 figure
    out_1x2_png = base_dir / "pareto_front_reaction_force.png"
    out_1x2_pdf = base_dir / "pareto_front_reaction_force.pdf"
    save_figure(fig, str(out_1x2_pdf), dpi=300, close=False, make_png=True)
    fig.savefig(str(out_1x2_png), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_1x2_png}")

    # ---------------------------------------------------------
    # 2. Comprehensive 2x2 Plot: Reaction Force & GP Validation
    # ---------------------------------------------------------
    fig2, axs = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    
    # Top Row: Reaction Force (True Disp)
    # (0, 0): EC vs R2 (Reaction Force)
    axs[0, 0].plot(ec_rx_true[pareto_r2_true], r2_rx_tot[pareto_r2_true], color="#1f77b4", linestyle="--", linewidth=2.0, alpha=0.85, label="Pareto Frontier")
    axs[0, 0].scatter(ec_rx_true, r2_rx_tot, color="#1f77b4", s=70, edgecolors="navy", linewidth=1.2, zorder=4, label="Candidate Models")
    axs[0, 0].scatter(ec_rx_true[idx_20], r2_rx_tot[idx_20], color="#d62728", marker="*", s=280, edgecolors="black", linewidth=1.5, zorder=6, label=r"Winner ($\beta=20.0$)")
    for i, b in enumerate(betas):
        axs[0, 0].annotate(f"$\\beta={b}$", (ec_rx_true[i], r2_rx_tot[i]), textcoords="offset points", xytext=(6, 2), fontsize=8.5, color="#1f77b4")
    axs[0, 0].axvline(95.0, color="green", linestyle="--", alpha=0.4)
    axs[0, 0].set_xlabel("Empirical Coverage EC (%) [$\\rightarrow$ higher]", fontweight="bold")
    axs[0, 0].set_ylabel(r"Reaction Force $R^2_{\mathrm{tot}}$ [$\rightarrow$ higher]", fontweight="bold")
    axs[0, 0].set_title(r"Full-Field FEM Reaction Force: EC vs. $R^2$", fontsize=11.5)
    axs[0, 0].set_xlim(-2, 102)
    axs[0, 0].set_ylim(0.90, 1.01)
    axs[0, 0].legend(loc="lower left", fontsize=8.5)
    axs[0, 0].grid(True, alpha=0.25, linestyle="--")

    # (0, 1): EC vs RMSE (Reaction Force)
    axs[0, 1].plot(ec_rx_true[pareto_rmse_true], rmse_rx_tot[pareto_rmse_true], color="#1f77b4", linestyle="--", linewidth=2.0, alpha=0.85, label="Pareto Frontier")
    axs[0, 1].scatter(ec_rx_true, rmse_rx_tot, color="#1f77b4", s=70, edgecolors="navy", linewidth=1.2, zorder=4, label="Candidate Models")
    axs[0, 1].scatter(ec_rx_true[idx_20], rmse_rx_tot[idx_20], color="#d62728", marker="*", s=280, edgecolors="black", linewidth=1.5, zorder=6, label=r"Winner ($\beta=20.0$)")
    for i, b in enumerate(betas):
        axs[0, 1].annotate(f"$\\beta={b}$", (ec_rx_true[i], rmse_rx_tot[i]), textcoords="offset points", xytext=(6, 2), fontsize=8.5, color="#1f77b4")
    axs[0, 1].axvline(95.0, color="green", linestyle="--", alpha=0.4)
    axs[0, 1].set_xlabel("Empirical Coverage EC (%) [$\\rightarrow$ higher]", fontweight="bold")
    axs[0, 1].set_ylabel(r"Reaction Force RMSE (N) [$\leftarrow$ lower]", fontweight="bold")
    axs[0, 1].set_title("Full-Field FEM Reaction Force: EC vs. RMSE", fontsize=11.5)
    axs[0, 1].set_xlim(-2, 102)
    axs[0, 1].set_ylim(-0.02, 0.80)
    axs[0, 1].legend(loc="upper left", fontsize=8.5)
    axs[0, 1].grid(True, alpha=0.25, linestyle="--")

    # Bottom Row: GP Stress Validation Set
    pareto_val_r2 = compute_pareto_front_2d(ec_val, r2_val, maximize_x=True, maximize_y=True)
    pareto_val_rmse = compute_pareto_front_2d(ec_val, rmse_val, maximize_x=True, maximize_y=False)
    
    # (1, 0): EC vs R2 (Validation Stress)
    axs[1, 0].plot(ec_val[pareto_val_r2], r2_val[pareto_val_r2], color="#2ca02c", linestyle="--", linewidth=2.0, alpha=0.85, label="Pareto Frontier")
    axs[1, 0].scatter(ec_val, r2_val, color="#2ca02c", s=70, edgecolors="darkgreen", linewidth=1.2, zorder=4, label="Candidate Models")
    axs[1, 0].scatter(ec_val[idx_20], r2_val[idx_20], color="#d62728", marker="*", s=280, edgecolors="black", linewidth=1.5, zorder=6, label=r"Winner ($\beta=20.0$)")
    for i, b in enumerate(betas):
        # Indicate dominated points for beta >= 50
        offset_y = 3 if b in [0.01, 0.1, 1.0, 10.0, 20.0] else -12
        axs[1, 0].annotate(f"$\\beta={b}$", (ec_val[i], r2_val[i]), textcoords="offset points", xytext=(6, offset_y), fontsize=8.5, color="#1b651b")
    axs[1, 0].axvline(95.0, color="green", linestyle="--", alpha=0.4)
    axs[1, 0].set_xlabel("Empirical Coverage EC (%) [$\\rightarrow$ higher]", fontweight="bold")
    axs[1, 0].set_ylabel(r"Validation Stress $R^2$ [$\rightarrow$ higher]", fontweight="bold")
    axs[1, 0].set_title(r"GP Constitutive Validation: EC vs. $R^2$", fontsize=11.5)
    axs[1, 0].set_xlim(-2, 102)
    axs[1, 0].set_ylim(0.84, 1.01)
    axs[1, 0].legend(loc="lower left", fontsize=8.5)
    axs[1, 0].grid(True, alpha=0.25, linestyle="--")

    # (1, 1): EC vs RMSE (Validation Stress)
    axs[1, 1].plot(ec_val[pareto_val_rmse], rmse_val[pareto_val_rmse], color="#2ca02c", linestyle="--", linewidth=2.0, alpha=0.85, label="Pareto Frontier")
    axs[1, 1].scatter(ec_val, rmse_val, color="#2ca02c", s=70, edgecolors="darkgreen", linewidth=1.2, zorder=4, label="Candidate Models")
    axs[1, 1].scatter(ec_val[idx_20], rmse_val[idx_20], color="#d62728", marker="*", s=280, edgecolors="black", linewidth=1.5, zorder=6, label=r"Winner ($\beta=20.0$)")
    for i, b in enumerate(betas):
        offset_y = 3 if b in [0.01, 0.1, 1.0, 10.0, 20.0] else -12
        axs[1, 1].annotate(f"$\\beta={b}$", (ec_val[i], rmse_val[i]), textcoords="offset points", xytext=(6, offset_y), fontsize=8.5, color="#1b651b")
    axs[1, 1].axvline(95.0, color="green", linestyle="--", alpha=0.4)
    axs[1, 1].set_xlabel("Empirical Coverage EC (%) [$\\rightarrow$ higher]", fontweight="bold")
    axs[1, 1].set_ylabel(r"Validation Stress RMSE (MPa) [$\leftarrow$ lower]", fontweight="bold")
    axs[1, 1].set_title("GP Constitutive Validation: EC vs. RMSE", fontsize=11.5)
    axs[1, 1].set_xlim(-2, 102)
    axs[1, 1].set_ylim(0.10, 0.65)
    axs[1, 1].legend(loc="upper left", fontsize=8.5)
    axs[1, 1].grid(True, alpha=0.25, linestyle="--")

    # Save 2x2 figure
    out_2x2_png = base_dir / "pareto_front_ec_vs_metrics_2x2.png"
    out_2x2_pdf = base_dir / "pareto_front_ec_vs_metrics_2x2.pdf"
    save_figure(fig2, str(out_2x2_pdf), dpi=300, close=False, make_png=True)
    fig2.savefig(str(out_2x2_png), dpi=300, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved: {out_2x2_png}")

    # Copy to artifact dir if exists
    art_dir = Path("/root/.gemini/antigravity/brain/737a06df-42ad-49d1-9b57-e67225b038cf")
    if art_dir.exists():
        import shutil
        for fname in ["pareto_front_reaction_force.png", "pareto_front_reaction_force.pdf",
                      "pareto_front_ec_vs_metrics_2x2.png", "pareto_front_ec_vs_metrics_2x2.pdf"]:
            src = base_dir / fname
            if src.exists():
                shutil.copy2(src, art_dir / fname)
        print(f"Copied plots to artifact dir: {art_dir}")

    # Save Pareto analysis JSON
    pareto_summary = {
        "reaction_force_true_disp": {
            "pareto_optimal_betas_r2": [float(betas[i]) for i in pareto_r2_true],
            "pareto_optimal_betas_rmse": [float(betas[i]) for i in pareto_rmse_true],
            "points": [
                {
                    "beta": float(betas[i]),
                    "ec_true": float(ec_rx_true[i]),
                    "r2": float(r2_rx_tot[i]),
                    "rmse": float(rmse_rx_tot[i]),
                    "is_pareto_r2": bool(i in pareto_r2_true),
                    "is_pareto_rmse": bool(i in pareto_rmse_true),
                }
                for i in range(len(betas))
            ]
        },
        "reaction_force_obs_disp": {
            "pareto_optimal_betas_r2": [float(betas[i]) for i in pareto_r2_obs],
            "pareto_optimal_betas_rmse": [float(betas[i]) for i in pareto_rmse_obs],
        },
        "validation_stress": {
            "pareto_optimal_betas_r2": [float(betas[i]) for i in pareto_val_r2],
            "pareto_optimal_betas_rmse": [float(betas[i]) for i in pareto_val_rmse],
            "points": [
                {
                    "beta": float(betas[i]),
                    "ec_val": float(ec_val[i]),
                    "r2_val": float(r2_val[i]),
                    "rmse_val": float(rmse_val[i]),
                    "is_pareto_r2": bool(i in pareto_val_r2),
                    "is_pareto_rmse": bool(i in pareto_val_rmse),
                }
                for i in range(len(betas))
            ]
        }
    }
    
    with open(base_dir / "pareto_analysis.json", "w") as f:
        json.dump(pareto_summary, f, indent=4)
    print("Saved pareto_analysis.json")


if __name__ == "__main__":
    plot_pareto_front()

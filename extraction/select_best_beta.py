#!/usr/bin/env python3
"""
extraction/select_best_beta.py

Automated Model Selection for Physics-Informed Variational Hyperelasticity GPs.
Evaluates candidate models trained across different beta (KL weight) values,
assesses calibration against target Empirical Coverage (default 95.0%),
and automatically promotes the best-calibrated, physically sound model to
the target extracted directory for downstream distillation and FEM validation.
"""

import os
import sys
import glob
import json
import shutil
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(
        description="Select the best-calibrated model across multiple beta candidates."
    )
    parser.add_argument(
        "--candidates_dir", type=str, required=True,
        help="Directory containing candidate extraction directories (e.g., beta_1.0, beta_10.0, etc.)"
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Target output directory (e.g., $SEED_DIR/extracted) to copy/populate with winning model"
    )
    parser.add_argument(
        "--target_ec", type=float, default=95.0,
        help="Target Empirical Coverage percentage (default: 95.0%%)"
    )
    parser.add_argument(
        "--min_r2", type=float, default=0.90,
        help="Minimum validation R2 required for physics fidelity (default: 0.90)"
    )
    parser.add_argument(
        "--selection_metric", type=str, default="ec_calibration",
        choices=["ec_calibration", "composite", "highest_r2"],
        help="Selection rule: 'ec_calibration' (closest to target EC among R2 >= min_r2), "
             "'composite' (R2 - lambda * |EC - target|), or 'highest_r2'"
    )
    return parser.parse_args()


def load_candidate_metrics(candidate_dir: str):
    """Loads metrics and config for a candidate extraction run."""
    metrics_path = os.path.join(candidate_dir, "extraction_metrics.json")
    if not os.path.exists(metrics_path):
        return None

    try:
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    except Exception as e:
        print(f"Warning: Could not read {metrics_path}: {e}")
        return None

    # Determine beta
    beta = metrics.get("beta", None)
    if beta is None:
        # Try reading config.yaml or config.json
        cfg_yaml = os.path.join(candidate_dir, "config.yaml")
        cfg_json = os.path.join(candidate_dir, "config.json")
        if os.path.exists(cfg_yaml):
            try:
                import yaml
                with open(cfg_yaml, "r") as f:
                    cfg = yaml.safe_load(f)
                beta = cfg.get("beta", None)
            except Exception:
                pass
        if beta is None and os.path.exists(cfg_json):
            try:
                with open(cfg_json, "r") as f:
                    cfg = json.load(f)
                beta = cfg.get("beta", None)
            except Exception:
                pass
        if beta is None:
            # Try extracting from folder name (e.g. beta_50.0)
            base = os.path.basename(candidate_dir)
            if "beta_" in base:
                try:
                    beta = float(base.split("beta_")[-1])
                except Exception:
                    beta = 1.0
            else:
                beta = 1.0

    r2_val = metrics.get("r2_val", metrics.get("r2", float("nan")))
    rmse_val = metrics.get("rmse_val", metrics.get("rmse", float("nan")))
    ec_val = metrics.get("ec_val", metrics.get("ec", float("nan")))

    return {
        "dir": candidate_dir,
        "name": os.path.basename(candidate_dir),
        "beta": float(beta),
        "r2_val": float(r2_val),
        "rmse_val": float(rmse_val),
        "ec_val": float(ec_val),
        "r2_train": float(metrics.get("r2_train", float("nan"))),
        "ec_train": float(metrics.get("ec_train", float("nan"))),
        "r2_test": float(metrics.get("r2_test", float("nan"))),
        "ec_test": float(metrics.get("ec_test", float("nan"))),
        "elbo": float(metrics.get("elbo", float("nan"))),
        "kl": float(metrics.get("kl", float("nan"))),
        "phy": float(metrics.get("phy", float("nan"))),
    }


def select_winner(candidates, target_ec=95.0, min_r2=0.90, metric="ec_calibration"):
    """
    Selects the winning candidate based on the specified criterion.
    Default: Filters candidates with R2 >= min_r2, then chooses the one with
    EC closest to target_ec.
    """
    if not candidates:
        raise ValueError("No valid candidates provided for selection.")

    # Calculate calibration discrepancy
    for c in candidates:
        c["ec_diff"] = abs(c["ec_val"] - target_ec) if not np.isnan(c["ec_val"]) else float("inf")
        # Composite score: R2 penalized by calibration discrepancy
        c["composite_score"] = c["r2_val"] - 0.005 * c["ec_diff"] if not np.isnan(c["r2_val"]) else -float("inf")

    if metric == "highest_r2":
        candidates.sort(key=lambda x: (not np.isnan(x["r2_val"]), x["r2_val"]), reverse=True)
        return candidates[0]

    elif metric == "composite":
        candidates.sort(key=lambda x: x["composite_score"], reverse=True)
        return candidates[0]

    else:  # 'ec_calibration' (recommended)
        # Stage 1: Filter candidates meeting physical fidelity
        physically_sound = [c for c in candidates if not np.isnan(c["r2_val"]) and c["r2_val"] >= min_r2]

        if physically_sound:
            # Sort physically sound models by calibration discrepancy
            physically_sound.sort(key=lambda x: (x["ec_diff"], -x["r2_val"]))
            return physically_sound[0]
        else:
            # Fallback if no candidate meets min_r2: pick highest R2
            print(f"Warning: No candidate met R2_val >= {min_r2}. Falling back to highest R2_val.")
            candidates.sort(key=lambda x: (not np.isnan(x["r2_val"]), x["r2_val"]), reverse=True)
            return candidates[0]


def generate_calibration_plot(candidates, winner, target_ec, save_dir):
    """Generates a dual-axis diagnostic plot of R2 and EC vs Beta."""
    sorted_c = sorted([c for c in candidates if not np.isnan(c["beta"])], key=lambda x: x["beta"])
    if len(sorted_c) < 2:
        return

    betas = [c["beta"] for c in sorted_c]
    r2_vals = [c["r2_val"] for c in sorted_c]
    ec_vals = [c["ec_val"] for c in sorted_c]

    fig, ax1 = plt.subplots(figsize=(7.5, 5.0), dpi=300)

    # Color scheme
    c_r2 = "#1f77b4"
    c_ec = "#2ca02c"
    c_win = "#d62728"

    # Left Axis: R2
    ax1.set_xlabel(r"Variational Regularization Weight ($\beta$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel(r"Validation Fit ($R^2_{\mathrm{val}}$)", color=c_r2, fontsize=11, fontweight="bold")
    line1 = ax1.plot(betas, r2_vals, color=c_r2, marker="o", lw=2, label=r"$R^2_{\mathrm{val}}$")
    ax1.tick_params(axis="y", labelcolor=c_r2)
    ax1.set_xscale("log")
    ax1.grid(True, alpha=0.25, linestyle="--")

    # Right Axis: EC (%)
    ax2 = ax1.twinx()
    ax2.set_ylabel(r"Empirical Coverage ($\mathrm{EC}_{\mathrm{val}}$ %)", color=c_ec, fontsize=11, fontweight="bold")
    line2 = ax2.plot(betas, ec_vals, color=c_ec, marker="s", lw=2, label=r"$\mathrm{EC}_{\mathrm{val}}$ (%)")
    line_target = ax2.axhline(target_ec, color="gray", linestyle="--", lw=1.5, label=f"Target EC ({target_ec}%)")
    ax2.tick_params(axis="y", labelcolor=c_ec)
    ax2.set_ylim(-2, 102)

    # Highlight winning beta
    win_beta = winner["beta"]
    ax1.axvline(win_beta, color=c_win, linestyle=":", lw=2, alpha=0.8, label=f"Selected $\\beta^*={win_beta}$")
    ax1.scatter([win_beta], [winner["r2_val"]], color=c_win, s=120, zorder=5, marker="*")
    ax2.scatter([win_beta], [winner["ec_val"]], color=c_win, s=120, zorder=5, marker="*")

    # Combined legend
    lines = line1 + line2 + [line_target]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower left", framealpha=0.9, fontsize=9.5)

    plt.title(f"Model Selection & Uncertainty Calibration\nSelected $\\beta^* = {win_beta}$ (EC = {winner['ec_val']:.1f}%, $R^2$ = {winner['r2_val']:.4f})",
              fontsize=12, pad=10)
    plt.tight_layout()

    out_png = os.path.join(save_dir, "beta_calibration_curve.png")
    out_pdf = os.path.join(save_dir, "beta_calibration_curve.pdf")
    plt.savefig(out_png, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"📈 Saved calibration curve to {out_png} and {out_pdf}")


def main():
    args = parse_args()

    candidates_dir = os.path.abspath(args.candidates_dir)
    output_dir = os.path.abspath(args.output_dir)

    if not os.path.exists(candidates_dir):
        print(f"❌ Error: Candidates directory not found: {candidates_dir}")
        sys.exit(1)

    # Find all subdirectories
    subdirs = [os.path.join(candidates_dir, d) for d in os.listdir(candidates_dir)
               if os.path.isdir(os.path.join(candidates_dir, d))]

    if not subdirs:
        print(f"❌ Error: No candidate subdirectories found in {candidates_dir}")
        sys.exit(1)

    candidates = []
    for d in subdirs:
        c = load_candidate_metrics(d)
        if c is not None:
            candidates.append(c)

    if not candidates:
        print(f"❌ Error: No candidate in {candidates_dir} had a valid extraction_metrics.json")
        sys.exit(1)

    # Sort candidates by beta for display
    candidates.sort(key=lambda x: x["beta"])

    # Select winner
    winner = select_winner(
        candidates, target_ec=args.target_ec, min_r2=args.min_r2, metric=args.selection_metric
    )

    # Print summary table
    print("\n" + "=" * 90)
    print(f"🏆 Multi-Beta Model Calibration & Selection Summary (Target EC: {args.target_ec}%)")
    print("=" * 90)
    print(f"{'Candidate':<18} {'Beta':<8} {'R2_val':<10} {'RMSE_val':<11} {'EC_val (%)':<12} {'|EC-95%|':<10} {'Status'}")
    print("-" * 90)

    for c in candidates:
        is_winner = (c["dir"] == winner["dir"])
        status = "🏆 SELECTED" if is_winner else ("Eligible" if c["r2_val"] >= args.min_r2 else "❌ Low R2")
        if not is_winner and c["r2_val"] >= args.min_r2:
            if c["ec_val"] < args.target_ec - 30.0:
                status = "❌ Overconfident"
            elif c["ec_val"] > args.target_ec + 3.0:
                status = "❌ Underconfident"
            else:
                status = "Eligible"

        print(f"{c['name']:<18} {c['beta']:<8.2f} {c['r2_val']:<10.4f} {c['rmse_val']:<11.4f} "
              f"{c['ec_val']:<12.1f} {c['ec_diff']:<10.1f} {status}")
    print("=" * 90)
    print(f"🎯 Winning Model: {winner['name']} (Beta = {winner['beta']})")
    print(f"   Validation Metrics: R2 = {winner['r2_val']:.4f} | RMSE = {winner['rmse_val']:.4f} | EC = {winner['ec_val']:.1f}%\n")

    # Ensure output_dir exists
    os.makedirs(output_dir, exist_ok=True)

    # Copy all files from winner to output_dir
    for item in os.listdir(winner["dir"]):
        src_item = os.path.join(winner["dir"], item)
        dst_item = os.path.join(output_dir, item)
        if os.path.isdir(src_item):
            if os.path.exists(dst_item):
                shutil.rmtree(dst_item)
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

    print(f"✅ Successfully copied winning model files from {winner['name']} to {output_dir}")

    # Generate calibration plot
    try:
        generate_calibration_plot(candidates, winner, args.target_ec, output_dir)
        # Also copy plot to candidates_dir
        shutil.copy2(os.path.join(output_dir, "beta_calibration_curve.png"),
                     os.path.join(candidates_dir, "beta_calibration_curve.png"))
        shutil.copy2(os.path.join(output_dir, "beta_calibration_curve.pdf"),
                     os.path.join(candidates_dir, "beta_calibration_curve.pdf"))
    except Exception as e:
        print(f"Warning: Could not generate calibration plot: {e}")

    # Save structured summary JSON
    summary_data = {
        "selected_candidate": winner["name"],
        "selected_beta": winner["beta"],
        "target_ec": args.target_ec,
        "selection_metric": args.selection_metric,
        "min_r2": args.min_r2,
        "candidates": candidates
    }
    summary_path_out = os.path.join(output_dir, "beta_selection_summary.json")
    summary_path_cand = os.path.join(candidates_dir, "beta_selection_summary.json")
    with open(summary_path_out, "w") as f:
        json.dump(summary_data, f, indent=4)
    with open(summary_path_cand, "w") as f:
        json.dump(summary_data, f, indent=4)

    print(f"📋 Saved selection summary to {summary_path_out}")


if __name__ == "__main__":
    main()


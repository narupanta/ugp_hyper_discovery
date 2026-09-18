#!/usr/bin/env python3
"""
scripts/run_all_validation_plots.py

Executes full validation and UQ analysis across all seeds in an experiment:
1. Displacement UQ verification (uq_verification_disp.py)
2. Pointwise Reaction Force calibration (plot_reaction_force_distilled.py)
3. Free Node Residual analysis with GP overlay (plot_free_node_residuals.py)
4. Master cross-seed ranking & summary compilation (generate_experiment_summary.py)
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run complete validation pipeline across all seeds")
    parser.add_argument(
        "--experiment_dir", type=str,
        default="results/20260910T085807_gentthomas_0.0001_0.01_2.0_0.95_5_0.01_isotropic_block",
        help="Path to experiment directory"
    )
    parser.add_argument("--step", type=int, default=16, help="Step index for distribution plots (default: 16)")
    parser.add_argument("--seeds", type=str, nargs="+", default=None, help="Specific seeds to evaluate (default: all)")
    return parser.parse_args()


def run_command(cmd_list):
    print(f">> Running: {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Command failed with code {res.returncode}:\n{res.stderr}\n{res.stdout}")
    return res.returncode == 0


def main():
    args = parse_args()
    exp_dir = Path(args.experiment_dir).resolve()
    if not exp_dir.exists():
        print(f"Error: {exp_dir} not found.")
        sys.exit(1)

    all_seed_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name.isdigit()], key=lambda x: int(x.name))
    if args.seeds:
        seed_filter = set([str(s) for s in args.seeds])
        all_seed_dirs = [d for d in all_seed_dirs if d.name in seed_filter]

    print(f"=== Running Full Validation Pipeline for {len(all_seed_dirs)} seeds in {exp_dir.name} ===")

    for seed_dir in all_seed_dirs:
        seed = seed_dir.name
        fem_val_dir = seed_dir / "fem_validation"
        if not fem_val_dir.exists():
            continue

        print(f"\n------------------------------------------------------------")
        print(f"--- Processing Seed {seed} ({fem_val_dir}) ---")
        print(f"------------------------------------------------------------")

        # 1. Displacement UQ verification
        run_command([
            "python3", "plots/uq_verification_disp.py",
            "--model_path", str(fem_val_dir),
            "--step", str(args.step),
            "--dist_step", str(args.step)
        ])

        # 2. Reaction Force Calibration & Distribution
        run_command([
            "python3", "plots/plot_reaction_force_distilled.py",
            "--model_path", str(fem_val_dir),
            "--step", str(args.step)
        ])

        # 3. Free Node Residual Analysis
        run_command([
            "python3", "plots/plot_free_node_residuals.py",
            "--model_path", str(fem_val_dir),
            "--step", str(args.step)
        ])

    # 4. Master Cross-Seed Summary & Ranking
    print(f"\n============================================================")
    print(f"=== Compiling Master Summary Across Seeds ===")
    print(f"============================================================")
    run_command([
        "python3", "plots/generate_experiment_summary.py",
        "--experiment_dir", str(exp_dir)
    ])

    print("\n🎉 Full validation pipeline execution complete across all seeds!")


if __name__ == "__main__":
    main()


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from plots.theme import apply_style, save_figure

def main():
    apply_style()

    parser = argparse.ArgumentParser(description="Plot invariant comparisons between training (block) and validation (holes) datasets")
    parser.add_argument("--block_dataset", type=str, default="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block_12.npz",
                        help="Path to block dataset npz")
    parser.add_argument("--holes_dataset", type=str, default="dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_holes_12.npz",
                        help="Path to holes dataset npz")
    parser.add_argument("--train_steps", type=int, nargs="+", default=[2, 6, 10, 14],
                        help="Training load step indices for block")
    parser.add_argument("--output_dir", type=str, default="results/20260918T223851_isihara_1e-05_0.01_1.0_0.5_5_0.01_isotropic_block/12/plots",
                        help="Output directory")
    args = parser.parse_args()

    # Fallbacks if _12 does not exist
    if not os.path.exists(args.block_dataset):
        alt = "dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_block.npz"
        if os.path.exists(alt):
            args.block_dataset = alt
    if not os.path.exists(args.holes_dataset):
        alt = "dataset/preprocessed/syn_f/isihara_1e-05_0.01_1.0_0.5_holes.npz"
        if os.path.exists(alt):
            args.holes_dataset = alt

    print(f"Loading block dataset: {args.block_dataset}")
    print(f"Loading holes dataset: {args.holes_dataset}")

    block_data = np.load(args.block_dataset)
    holes_data = np.load(args.holes_dataset)

    # Invariants for training (block) at train_steps
    train_steps = [s for s in args.train_steps if s < len(block_data["obs_I1_bar"])]
    b_I1 = block_data["obs_I1_bar"][train_steps].flatten()
    b_I2 = block_data["obs_I2_bar"][train_steps].flatten()
    b_J = block_data["obs_J"][train_steps].flatten()

    # Invariants for validation (holes) across all steps
    h_I1 = holes_data["obs_I1_bar"].flatten()
    h_I2 = holes_data["obs_I2_bar"].flatten()
    h_J = holes_data["obs_J"].flatten()

    # 3 subplots:
    # 1. I1_bar - 3 vs I2_bar - 3
    # 2. I1_bar vs J
    # 3. I2_bar vs J
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), gridspec_kw={"wspace": 0.28})

    color_block = "#888888"    # Grey
    color_holes = "#D55E00"    # Orange

    # Subplot 1: (I1_bar - 3) vs (I2_bar - 3)
    ax1 = axes[0]
    ax1.set_box_aspect(1)
    ax1.scatter(b_I1 - 3.0, b_I2 - 3.0, color=color_block, alpha=0.35, s=12, edgecolors="none", label="Training (Block)")
    ax1.scatter(h_I1 - 3.0, h_I2 - 3.0, color=color_holes, alpha=0.35, s=12, edgecolors="none", label="Validation (Holes)")
    ax1.set_xlabel(r"$\bar{I}_1 - 3$", fontsize=10, labelpad=2)
    ax1.set_ylabel(r"$\bar{I}_2 - 3$", fontsize=10, labelpad=2)
    ax1.tick_params(axis="both", which="major", labelsize=8)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Subplot 2: I1_bar vs J
    ax2 = axes[1]
    ax2.set_box_aspect(1)
    ax2.scatter(b_I1, b_J, color=color_block, alpha=0.35, s=12, edgecolors="none", label="Training (Block)")
    ax2.scatter(h_I1, h_J, color=color_holes, alpha=0.35, s=12, edgecolors="none", label="Validation (Holes)")
    ax2.set_xlabel(r"$\bar{I}_1$", fontsize=10, labelpad=2)
    ax2.set_ylabel(r"$J$", fontsize=10, labelpad=2)
    ax2.tick_params(axis="both", which="major", labelsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # Subplot 3: I2_bar vs J
    ax3 = axes[2]
    ax3.set_box_aspect(1)
    ax3.scatter(b_I2, b_J, color=color_block, alpha=0.35, s=12, edgecolors="none", label="Training (Block)")
    ax3.scatter(h_I2, h_J, color=color_holes, alpha=0.35, s=12, edgecolors="none", label="Validation (Holes)")
    ax3.set_xlabel(r"$\bar{I}_2$", fontsize=10, labelpad=2)
    ax3.set_ylabel(r"$J$", fontsize=10, labelpad=2)
    ax3.tick_params(axis="both", which="major", labelsize=8)
    ax3.grid(True, linestyle=":", alpha=0.5)

    # Clean legend directly above the subplots
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=2, frameon=False, fontsize=9.0, markerscale=1.8, handletextpad=0.3, columnspacing=1.5)

    plt.subplots_adjust(top=0.86, bottom=0.14, left=0.06, right=0.98)

    os.makedirs(args.output_dir, exist_ok=True)
    pdf_path = os.path.join(args.output_dir, "invariant_comparison_block_holes.pdf")
    png_path = os.path.join(args.output_dir, "invariant_comparison_block_holes.png")

    plt.savefig(pdf_path, bbox_inches="tight", dpi=300)
    plt.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved figure to:\n  {pdf_path}\n  {png_path}")

if __name__ == "__main__":
    main()

"""
dataset/preprocess_filter.py: Standalone CLI Tool to Preprocess & Filter Displacement Noise

Filters full-field displacement measurements in an .npz archive, recomputes deformation gradients
and strain invariants while preserving boundary conditions, and saves a filtered .npz dataset.
"""

import os
import sys
import argparse
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.filters import filter_dataset_kinematics
from plots.training import plot_domain_invariants


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and filter displacement noise in FE datasets.")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input .npz dataset file")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to output filtered .npz file (default: <input>_filtered.npz)")
    parser.add_argument("--method", type=str, default="laplacian", choices=["laplacian", "gaussian"], help="Filtering method")
    parser.add_argument("--alpha", type=float, default=0.5, help="Laplacian smoothing relaxation factor in (0, 1]")
    parser.add_argument("--passes", type=int, default=3, help="Number of smoothing passes")
    parser.add_argument("--radius", type=float, default=None, help="Gaussian kernel radius")
    parser.add_argument("--sigma", type=float, default=None, help="Gaussian kernel sigma")
    parser.add_argument("--plot", action="store_true", default=True, help="Generate domain invariants before/after plot")
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    if args.output is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_filtered{ext}"
    else:
        output_path = os.path.abspath(args.output)

    print(f"Loading raw dataset: {input_path}")
    raw_dict = dict(np.load(input_path, allow_pickle=True))

    print(f"Applying {args.method} displacement filter (alpha={args.alpha}, passes={args.passes})...")
    filtered_dict = filter_dataset_kinematics(
        raw_dict,
        method=args.method,
        alpha=args.alpha,
        passes=args.passes,
        radius=args.radius,
        sigma=args.sigma,
        verbose=True
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **filtered_dict)
    print(f"✅ Filtered dataset saved to: {output_path}")

    if args.plot:
        plot_dir = os.path.dirname(output_path)
        print(f"Generating domain invariants visualization in: {plot_dir}...")
        try:
            plot_domain_invariants(
                prep_data=filtered_dict,
                save_path=plot_dir,
                step_idx=-1,
                make_png=True,
                save_comparison=True
            )
            print("Domain invariants plot generated successfully.")
        except Exception as e:
            print(f"Warning: Could not generate domain plot: {e}")


if __name__ == "__main__":
    main()


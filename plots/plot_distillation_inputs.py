import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os
import argparse

jax.config.update("jax_enable_x64", True)

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.utils import fto3x3 as core_fto3x3, farthest_point_sampling

def fto3x3(f):
    f3x3 = jnp.array([[f[0,0], f[0,1], 0.0],
                      [f[1,0], f[1,1], 0.0],
                      [0.0, 0.0, 1.0]])
    return f3x3

def invariants(f):
    C = f.T @ f
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    I3 = jnp.linalg.det(C)
    return jnp.array([I1, I2, jnp.sqrt(I3)])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset/preprocessed/syn_f/isihara_0.0001_0.01_8.0_0.95.npz")
    parser.add_argument("--export_file", type=str, default="extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1/pytorch_export_dataset_f/f3x3.npy", help="Path to exported f3x3.npy to visualize exact sampled points")
    parser.add_argument("--num_samples", type=int, default=192)
    parser.add_argument("--save_path", type=str, default="distillation/distilled_models/20260727T081506_isihara_gmr_dataset_f_uqmodeldisc/input_invariant_space.pdf")
    args = parser.parse_args()

    print(f"Loading dataset {args.dataset}...")
    data = np.load(args.dataset)
    F_all_steps_2x2 = data["F"]
    
    load_steps = None
    if args.export_file and os.path.exists(args.export_file):
        model_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.export_file)))
        log_file = os.path.join(model_dir, "optimization_log.txt")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as lf:
                first_line = lf.readline()
                if "[" in first_line and "]" in first_line:
                    steps_str = first_line.split("]")[0].split("[")[1].strip()
                    if steps_str:
                        load_steps = [int(x.strip()) for x in steps_str.split(",") if x.strip().isdigit()]
                        
    if load_steps and len(load_steps) > 0 and max(load_steps) < F_all_steps_2x2.shape[0]:
        print(f"Filtering dataset points to extraction load steps: {load_steps}")
        F_2x2 = F_all_steps_2x2[load_steps].reshape(-1, 2, 2)
    else:
        F_2x2 = F_all_steps_2x2.reshape(-1, 2, 2)
    
    print("Converting dataset points to 3x3 plane strain and calculating invariants...")
    F_3x3 = jax.vmap(fto3x3)(F_2x2)
    invs = jax.vmap(invariants)(F_3x3)
    
    I1_all = invs[:, 0]
    I2_all = invs[:, 1]
    J_all = invs[:, 2]

    if args.export_file and os.path.exists(args.export_file):
        print(f"Loading exact sampled distillation deformations from {args.export_file}...")
        F_sampled_3x3 = jnp.array(np.load(args.export_file))
        invs_sampled = jax.vmap(invariants)(F_sampled_3x3)
        I1_sampled = invs_sampled[:, 0]
        I2_sampled = invs_sampled[:, 1]
        J_sampled = invs_sampled[:, 2]
        is_all = ("dataset_all" in args.export_file or len(I1_sampled) == len(I1_all))
        sample_label = "All Distillation Inputs (No FPS)" if is_all else "FPS Distillation Inputs"
        title_str = f"Invariant Space: All Dataset Deformations ({len(I1_sampled)} points)" if is_all else f"Invariant Space via Farthest Point Sampling ({len(I1_sampled)} points)"
    else:
        print("Applying Farthest Point Sampling (FPS) on dataset deformations...")
        pts = jnp.array(F_2x2.reshape(-1, 4), dtype=jnp.float64)
        if len(F_2x2) <= args.num_samples:
            indices = np.arange(len(F_2x2))
            title_str = f"Invariant Space: All Dataset Deformations ({len(F_2x2)} points)"
        else:
            indices = np.array(farthest_point_sampling(pts, args.num_samples))
            title_str = f"Invariant Space via Farthest Point Sampling ({len(indices)} points)"
        I1_sampled = I1_all[indices]
        I2_sampled = I2_all[indices]
        J_sampled = J_all[indices]
        sample_label = "FPS Distillation Inputs"

    # Plotting
    plt.rcParams.update({"font.size": 14})
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    scatter_kwargs = {
        'facecolors': 'none', 
        'edgecolors': 'dodgerblue', 
        'marker': 'o', 
        'alpha': 0.4, 
        's': 15, 
        'label': 'All FE Dataset Points'
    }
    
    sampled_kwargs = {
        'color': 'crimson', 
        'marker': '*' if "FPS" in sample_label else 'x', 
        's': 90 if "FPS" in sample_label else 25, 
        'label': sample_label
    }

    # I1 vs I2
    axes[0].scatter(I1_all, I2_all, **scatter_kwargs)
    axes[0].scatter(I1_sampled, I2_sampled, **sampled_kwargs)
    axes[0].set_xlabel("$I_1$")
    axes[0].set_ylabel("$I_2$")
    axes[0].set_title("$I_1$ vs $I_2$")
    axes[0].grid(True, alpha=0.3)

    # I1 vs J
    axes[1].scatter(I1_all, J_all, **scatter_kwargs)
    axes[1].scatter(I1_sampled, J_sampled, **sampled_kwargs)
    axes[1].set_xlabel("$I_1$")
    axes[1].set_ylabel("$J$")
    axes[1].set_title("$I_1$ vs $J$")
    axes[1].grid(True, alpha=0.3)

    # I2 vs J
    axes[2].scatter(I2_all, J_all, **scatter_kwargs)
    axes[2].scatter(I2_sampled, J_sampled, **sampled_kwargs)
    axes[2].set_xlabel("$I_2$")
    axes[2].set_ylabel("$J$")
    axes[2].set_title("$I_2$ vs $J$")
    axes[2].grid(True, alpha=0.3)

    axes[0].legend(loc="upper left")

    plt.suptitle(title_str, fontsize=16)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    plt.savefig(args.save_path, bbox_inches='tight', dpi=150)
    print(f"Saved plot to {args.save_path}")

if __name__ == "__main__":
    main()

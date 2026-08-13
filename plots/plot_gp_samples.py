import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys

# Try to import material models for true curve
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from core.material_models import get_material
except ImportError:
    get_material = None

def get_gamma_from_F(F, mode_idx):
    if mode_idx == 0:   # UT
        return F[:, 0, 0] - 1.0
    elif mode_idx == 1: # EBT
        return F[:, 0, 0] - 1.0
    elif mode_idx == 2: # PS
        return F[:, 0, 0] - 1.0
    elif mode_idx == 3: # UC
        return 1.0 / F[:, 0, 0] - 1.0
    elif mode_idx == 4: # EBC
        return 1.0 / F[:, 0, 0] - 1.0
    elif mode_idx == 5: # SS
        return F[:, 0, 1]
    return np.zeros(len(F))

def main():
    parser = argparse.ArgumentParser(description="Plot GP samples from exported PyTorch matrices")
    parser.add_argument("--export_dir", type=str, required=True, help="Path to pytorch_export directory containing cov_psi.npy, etc.")
    parser.add_argument("--num_samples", type=int, default=32, help="Number of GP samples to draw")
    parser.add_argument("--model_name", type=str, default="nh4", help="True model name for reference")
    args = parser.parse_args()

    f3x3_path = os.path.join(args.export_dir, "f3x3.npy")
    mean_path = os.path.join(args.export_dir, "mean_psi.npy")
    cov_path = os.path.join(args.export_dir, "cov_psi.npy")

    if not all(os.path.exists(p) for p in [f3x3_path, mean_path, cov_path]):
        print(f"Error: Required .npy files not found in {args.export_dir}")
        return

    f3x3 = np.load(f3x3_path)
    mean_psi = np.load(mean_path)
    cov_psi = np.load(cov_path)

    # Ensure symmetric positive semi-definite
    cov_psi = 0.5 * (cov_psi + cov_psi.T)

    print(f"Loaded f3x3 shape: {f3x3.shape}")
    print(f"Loaded mean_psi shape: {mean_psi.shape}")
    print(f"Loaded cov_psi shape: {cov_psi.shape}")

    # Draw samples
    np.random.seed(42)
    
    # Rebuild a perfectly smooth, mathematically valid PSD matrix just in case of float truncations
    w, v = np.linalg.eigh(cov_psi)
    w = np.clip(w, a_min=1e-8, a_max=None)
    cov_psi_smooth = v @ np.diag(w) @ v.T
    
    try:
        samples = np.random.multivariate_normal(mean_psi, cov_psi_smooth, size=args.num_samples)
    except Exception as e:
        print(f"Sampling failed entirely: {e}")
        samples = np.zeros((args.num_samples, len(mean_psi)))

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # Calculate true model values if available
    true_psi = None
    if get_material is not None and args.model_name:
        true_model = get_material(args.model_name, jit_P=False)
        true_psi = np.zeros(len(f3x3))
        for i in range(len(f3x3)):
            # Fallback to pure numpy extraction just in case
            try:
                import jax.numpy as jnp
                true_psi[i] = true_model.psi(jnp.array(f3x3[i]))
            except:
                pass
                
    num_modes = 6
    pts_per_mode = len(f3x3) // num_modes

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for i, name in enumerate(mode_names):
        ax = axes[i // 3, i % 3]
        start_idx = i * pts_per_mode
        end_idx = (i + 1) * pts_per_mode

        F_mode = f3x3[start_idx:end_idx]
        gamma = get_gamma_from_F(F_mode, i)
        
        # Plot all samples
        for s in range(args.num_samples):
            ax.plot(gamma, samples[s, start_idx:end_idx], color="lightblue", lw=0.6, alpha=0.3)
        
        # Plot mean
        ax.plot(gamma, mean_psi[start_idx:end_idx], color="blue", lw=2.0, label="GP Mean")
        
        # Plot ±1.96 std (Using the true structural variance)
        raw_var = np.diag(cov_psi_smooth)[start_idx:end_idx]
        std_psi = np.sqrt(np.clip(raw_var, a_min=1e-8, a_max=None))
        ax.fill_between(gamma, 
                        mean_psi[start_idx:end_idx] - 1.96 * std_psi,
                        mean_psi[start_idx:end_idx] + 1.96 * std_psi,
                        color="blue", alpha=0.1, label="95% CI")
                        
        # Plot True
        if true_psi is not None:
            ax.plot(gamma, true_psi[start_idx:end_idx], 'k--', lw=1.5, label="True Model")

        ax.set_title(name)
        ax.set_xlabel(r"Stretch $\gamma$")
        ax.set_ylabel(r"Strain Energy $\Psi$")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend()

    plt.tight_layout()
    save_path = os.path.join(args.export_dir, "gp_energy_samples.png")
    plt.savefig(save_path, dpi=200)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    main()

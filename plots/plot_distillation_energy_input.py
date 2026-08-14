import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, default="extraction/extracted_models/20260723T152354_nh_0.0001_0.01_1.5_0.95_5_80.0_1")
    args = parser.parse_args()
    
    export_dir = os.path.join(args.saved_model_dir, "pytorch_export")
    mean_psi = torch.tensor(np.load(os.path.join(export_dir, "mean_psi.npy")), dtype=torch.float64)
    cov_psi = torch.tensor(np.load(os.path.join(export_dir, "cov_psi.npy")), dtype=torch.float64)
    cov_psi = (cov_psi + cov_psi.T) / 2.0
    cov_psi += 1e-3 * torch.eye(cov_psi.shape[0], dtype=torch.float64)
    
    dist = torch.distributions.MultivariateNormal(mean_psi, cov_psi)
    samples = dist.sample((32,)).numpy()
    
    mean_psi_np = mean_psi.numpy()
    std_psi_np = torch.sqrt(torch.diag(cov_psi)).numpy()
    
    num_points = 32
    max_gamma = 0.8
    gamma = np.linspace(0.0, max_gamma, num_points)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]
                  
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Energy Distribution Input to Distillation (GP Posterior over Standard Modes)", fontsize=18)
    
    for i, name in enumerate(mode_names):
        ax = axes[i // 2, i % 2]
        start_idx = i * num_points
        end_idx = (i + 1) * num_points
        
        mode_samples = samples[:, start_idx:end_idx]
        mode_mean = mean_psi_np[start_idx:end_idx]
        mode_std = std_psi_np[start_idx:end_idx]
        
        # Plot samples
        ax.plot(gamma, mode_samples.T, color="orange", lw=0.8, alpha=0.4, zorder=1)
        ax.plot([], [], color="orange", lw=2.0, label="GP Energy Samples")
        
        # Plot mean and std
        ax.plot(gamma, mode_mean, color="blue", lw=2, label="GP Mean", zorder=2)
        ax.fill_between(gamma, mode_mean - 1.96 * mode_std, mode_mean + 1.96 * mode_std, 
                        color="blue", alpha=0.1, zorder=0, label="95% CI")
        
        ax.set_title(name)
        ax.set_xlabel(r"Stretch Measure ($\gamma$)")
        ax.set_ylabel(r"Energy ($\Psi$)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left')
            
    plt.tight_layout()
    save_file = os.path.join(args.saved_model_dir, "distillation_energy_input.pdf")
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    print(f"Saved plot to {save_file}")

if __name__ == "__main__":
    main()

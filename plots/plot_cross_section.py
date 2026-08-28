import argparse
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from scipy.stats import norm

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.utils import infer_material_model_name

def generate_single_mode(gamma_val=0.5):
    gamma = jnp.array([gamma_val])
    F_all = jnp.zeros((6, 1, 3, 3))
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((1, 3, 3))
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(1 + gamma_val, 1.0, 1.0))            
    F_all = F_all.at[1].set(set_F(1 + gamma_val, 1 + gamma_val, 1.0))    
    F_all = F_all.at[2].set(set_F(1 + gamma_val, 1/(1 + gamma_val), 1.0)) 
    F_all = F_all.at[3].set(set_F(1/(1 + gamma_val), 1.0, 1.0))       
    F_all = F_all.at[4].set(set_F(1/(1 + gamma_val), 1/(1 + gamma_val), 1.0)) 
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma_val))      
    return F_all, gamma

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--material_model", type=str, required=True, choices=["ogden", "gmr", "isihara"])
    parser.add_argument("--gamma", type=float, default=0.5)
    args = parser.parse_args()
    
    # 1. Load True Model dynamically from saved_model_dir
    true_model_name = infer_material_model_name(args.saved_model_dir)
    true_model = get_material(true_model_name, jit_P=False)
    
    # 2. Load GP Model
    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(args.saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev, min_vol = jnp.min(dev_z, axis=0), jnp.min(vol_z, axis=0)
    max_dev, max_vol = jnp.max(dev_z, axis=0), jnp.max(vol_z, axis=0)
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    # 3. Load Distilled Samples
    samples = np.load(os.path.join(args.distilled_dir, "flow_samples.npy"))
    
    # 4. Generate Data for specific gamma
    F_all, gamma = generate_single_mode(args.gamma)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # 5. Evaluate True and GP
    psi_true = jax.vmap(true_model.psi)(F_all)
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean.flatten()[0] for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var.flatten()[0] for mode in range(len(mode_names))]
    
    def get_distilled_energy(theta, F_chunk):
        if args.material_model == "ogden":
            mu = theta[:3]
            alpha = theta[3:6]
            vol = theta[6:9]
            mat = get_material("ogden", mu_params=mu, alpha_params=alpha, vol_params=vol, jit_P=False)
        elif args.material_model == "gmr":
            dev = theta[:9]
            vol = theta[9:12]
            mat = get_material("gmr", dev_params=dev, vol_params=vol, jit_P=False)
        elif args.material_model == "isihara":
            mat = get_material("isihara", c10=theta[0], c01=theta[1], c20=theta[2], d1=theta[3], jit_P=False)
        return jax.vmap(mat.psi)(F_chunk)
        
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f"Energy Distribution Cross-Section at $\\gamma={args.gamma}$", fontsize=18)
    axes = axes.flatten()
    
    for i, name in enumerate(mode_names):
        ax = axes[i]
        
        # Ground Truth (Black Dash Line)
        true_val = psi_true[i, 0]
        ax.axvline(true_val, color='black', linestyle='--', lw=2, label="True Physics")
        
        # GP Distribution (Blue)
        mu_gp = psi_dist_mean[i]
        std_gp = jnp.sqrt(psi_dist_var[i])
        x = np.linspace(mu_gp - 4*std_gp, mu_gp + 4*std_gp, 100)
        p = norm.pdf(x, mu_gp, std_gp)
        ax.plot(x, p, color='blue', lw=2, label="GP Posterior")
        ax.fill_between(x, p, alpha=0.2, color='blue')
        
        # Distilled Distribution (Yellow Histogram)
        mode_F = F_all[i]
        s_psi = jax.vmap(lambda t: get_distilled_energy(t, mode_F))(samples)
        s_psi = np.array(s_psi).flatten()
        sns.histplot(s_psi, color='gold', ax=ax, label="Distilled Flow", stat='density', alpha=0.4, bins=30)
        
        ax.set_title(name)
        ax.set_xlabel(r"Strain Energy ($\Psi$)")
        ax.set_ylabel("Density")
        if i == 0:
            ax.legend(loc='upper right')
            
    plt.tight_layout()
    save_file = os.path.join(args.distilled_dir, f"energy_cross_section_g{args.gamma}.pdf")
    plt.savefig(save_file, dpi=150)
    plt.close()
    print(f"Plot saved to {save_file}")

if __name__ == "__main__":
    main()

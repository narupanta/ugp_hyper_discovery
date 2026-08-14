import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import os

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material

def generate_standard_modes(num_points=100, max_gamma=2.0):
    gamma = jnp.linspace(0.0, max_gamma, num_points)
    
    F_all = jnp.zeros((6, num_points, 3, 3))
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(1 + gamma, 1.0, 1.0))            
    F_all = F_all.at[1].set(set_F(1 + gamma, 1 + gamma, 1.0))    
    F_all = F_all.at[2].set(set_F(1 + gamma, 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[3].set(set_F(1/(1 + gamma), 1.0, 1.0))       
    F_all = F_all.at[4].set(set_F(1/(1 + gamma), 1/(1 + gamma), 1.0)) 
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))      
    return F_all, gamma

def main():
    saved_model_dir = "extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1"
    
    # 1. Load True Model
    true_model = get_material("isihara", jit_P=False)
    
    # 2. Load GP Model
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=1.0)
    
    # 3. Load Transitions
    transitions = np.load("extrapolation_transitions.npy")
    
    # 4. Generate Data
    F_all, gamma = generate_standard_modes(num_points=100, max_gamma=2.0)
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # 5. Evaluate True and GP
    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    P_dist_mean = [learned_gp.piola_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    P_dist_var = [learned_gp.piola_dist(F_all[mode]).var for mode in range(len(mode_names))]
    
    # 6. Plotting
    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    fig.suptitle(f"GP Extraction Validation: Interpolation vs Extrapolation Regions", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        if name == "Pure Shear":
            idx_comp = (1, 1); label_P = r"$P_{22}$"
        elif name == "Simple Shear":
            idx_comp = (0, 1); label_P = r"$P_{12}$"
        else:
            idx_comp = (0, 0); label_P = r"$P_{11}$"
            
        trans_g = transitions[i]

        # Column 0: Energy
        ax_psi = axes[i, 0]
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=2.0, label="True Physics", zorder=5)
        ax_psi.plot(gamma, psi_dist_mean[i], color="blue", lw=2, label="GP Mean", zorder=3)
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96*jnp.sqrt(psi_dist_var[i]), 
                           psi_dist_mean[i] + 1.96*jnp.sqrt(psi_dist_var[i]), color="blue", alpha=0.2, zorder=2)
                           
        # Add shaded regions
        ax_psi.axvspan(0, min(trans_g, 2.0), color='green', alpha=0.1, label="Interpolation" if i == 0 else "")
        if trans_g < 2.0:
            ax_psi.axvspan(trans_g, 2.0, color='red', alpha=0.1, label="Extrapolation" if i == 0 else "")
            ax_psi.axvline(trans_g, color='black', linestyle=':', alpha=0.6)
            ax_psi.text(trans_g + 0.02, ax_psi.get_ylim()[1]*0.9, f"$\\gamma_{{trans}}$={trans_g:.3f}", rotation=90, verticalalignment='top')
        
        y_min, y_max = jnp.min(psi_true[i]), jnp.max(psi_true[i])
        pad = (y_max - y_min) * 0.1 if y_max != y_min else 0.1
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())

        # Column 1: Stress
        ax_p = axes[i, 1]
        p_true_comp = P_true[i, :, idx_comp[0], idx_comp[1]]
        p_mean_comp = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(P_dist_var[i][:, idx_comp[0], idx_comp[1]])

        ax_p.plot(gamma, p_true_comp, 'k--', lw=2.0, label="True Physics", zorder=5)
        ax_p.plot(gamma, p_mean_comp, color="blue", lw=2, label="GP Mean", zorder=3)
        ax_p.fill_between(gamma, p_mean_comp - 1.96*p_std_comp, 
                         p_mean_comp + 1.96*p_std_comp, color="blue", alpha=0.2, zorder=2)
                         
        # Add shaded regions
        ax_p.axvspan(0, min(trans_g, 2.0), color='green', alpha=0.1, label="Interpolation" if i == 0 else "")
        if trans_g < 2.0:
            ax_p.axvspan(trans_g, 2.0, color='red', alpha=0.1, label="Extrapolation" if i == 0 else "")
            ax_p.axvline(trans_g, color='black', linestyle=':', alpha=0.6)

        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())

        # Formatting
        ax_psi.set_title(f"{name}: Energy ($\\Psi$)")
        ax_p.set_title(f"{name}: Stress ({label_P})")
        for ax in [ax_psi, ax_p]:
            ax.set_xlabel(r"Stretch Measure ($\gamma$)")
            ax.grid(True, alpha=0.3)
            if i == 0: 
                ax.legend(loc="upper left")

    plt.tight_layout()
    save_file = "extrapolation_analysis.pdf"
    plt.savefig(save_file, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Validation plot saved to: {save_file}")

if __name__ == "__main__":
    main()

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jax
import jax.numpy as jnp
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.utils import fto3x3

def generate_standard_modes(num_points=500, max_gamma=2.0):
    gammas = np.linspace(0, max_gamma, num_points)
    lam = gammas + 1.0

    F_uni = np.array([np.diag([l, 1/np.sqrt(l)]) for l in lam])
    F_bi = np.array([np.diag([l, l]) for l in lam])
    F_shear = np.array([np.diag([l, 1.0]) for l in lam])
    F_ss = np.array([np.array([[1.0, g], [0.0, 1.0]]) for g in gammas])
    F_ss_uni = np.array([np.array([[l, g], [0.0, 1/np.sqrt(l)]]) for l, g in zip(lam, gammas)])
    F_ss_bi = np.array([np.array([[l, g], [0.0, l]]) for l, g in zip(lam, gammas)])
    
    return np.concatenate([F_uni, F_bi, F_shear, F_ss, F_ss_uni, F_ss_bi], axis=0)

def invariants(f):
    F = fto3x3(f)
    C = F.T @ F
    I1 = jnp.trace(C)
    I2 = 0.5 * (I1**2 - jnp.trace(C @ C))
    J = jnp.linalg.det(F)
    return jnp.array([I1, I2, J])

def gmr_energy(C10, C01, C20, C11, C02, C30, C21, C12, C03, D1, D2, D3, f):
    inv = invariants(f)
    I1, I2, J = inv[0], inv[1], inv[2]
    I1_bar = J**(-2/3) * I1
    I2_bar = J**(-4/3) * I2
    I1_m3 = I1_bar - 3.0
    I2_m3 = I2_bar - 3.0
    J_m1 = J - 1.0
    
    W_dev = (C10 * I1_m3 + C01 * I2_m3 +
             C20 * I1_m3**2 + C11 * I1_m3 * I2_m3 + C02 * I2_m3**2 +
             C30 * I1_m3**3 + C21 * (I1_m3**2) * I2_m3 + C12 * I1_m3 * (I2_m3**2) + C03 * I2_m3**3)
    W_vol = D1 * J_m1**2 + D2 * J_m1**4 + D3 * J_m1**6
    return W_dev + W_vol

def get_gt_energy(f, material):
    if material == 'nh':
        inv = invariants(f)
        I1, J = inv[0], inv[2]
        I1_bar = J**(-2/3) * I1
        return 0.5 * (I1_bar - 3.0) + 0.5 * (J - 1.0)**2
    return 0.0 # simplified

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_dir", type=str, required=True)
    args = parser.parse_args()

    # Load Extracted GP
    best_params_dict = np.load(os.path.join(args.saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(args.saved_model_dir, "I_z.npy"))
    dev_z = I_z[:, :2]; vol_z = I_z[:, 2:]
    gp_model = SparseHyperelasticityGP(gp_params, I_z, jnp.min(dev_z, axis=0), jnp.min(vol_z, axis=0), jnp.max(dev_z, axis=0), jnp.max(vol_z, axis=0), beta=1.0)
    
    # Load Distilled Samples (From PyTorch Distillation)
    pytorch_samples_path = os.path.join(args.saved_model_dir, "uqmodeldisc_distilled", "pytorch_flow_samples.npy")
    distilled_params = np.load(pytorch_samples_path)
    
    f3x3_flat = generate_standard_modes(num_points=100, max_gamma=0.8)
    
    print("Evaluating GT, Extracted, and Distilled Energies...")
    # Extracted GP Energies
    extracted_energies = np.array(gp_model.psi_gp_mean(f3x3_flat))
    
    # GT Energies
    gt_energies = np.array(jax.vmap(lambda f: get_gt_energy(f, 'nh'))(f3x3_flat))
    
    # Distilled Energies
    vmap_energy = jax.vmap(gmr_energy, in_axes=(0,0,0,0,0,0,0,0,0,0,0,0, None))
    # average over the first 50 samples
    samples = distilled_params[:50]
    distilled_energies_all = []
    for f in f3x3_flat:
        e = vmap_energy(*[samples[:, i] for i in range(12)], f)
        distilled_energies_all.append(np.mean(e))
    distilled_energies = np.array(distilled_energies_all)

    # Plot Distribution (KDE)
    plt.figure(figsize=(10, 6))
    sns.kdeplot(gt_energies, label='Ground Truth', color='blue', lw=2)
    sns.kdeplot(extracted_energies, label='Extracted GP', color='green', lw=2, linestyle='--')
    sns.kdeplot(distilled_energies, label='Distilled GMR', color='red', lw=2, linestyle=':')
    
    plt.title('Energy Distribution Comparison over Standard Modes')
    plt.xlabel('Strain Energy Density')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig('energy_distribution.pdf', dpi=300)
    print("Saved to energy_distribution.pdf")

if __name__ == "__main__":
    main()

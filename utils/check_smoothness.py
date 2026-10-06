from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp
import os
import numpy as np

from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams

def check_smoothness():
    saved_model_dir = "extraction/extracted_models/20260714T093804_isihara_0.0001_0.01_8.0_0.95_5_80.0_1"
    
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    min_dev = jnp.min(dev_z, axis=0)
    min_vol = jnp.min(vol_z, axis=0)
    max_dev = jnp.max(dev_z, axis=0)
    max_vol = jnp.max(vol_z, axis=0)
    
    learned_gp = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=80.0)
    
    num_points = 50
    gamma = jnp.linspace(0.0, 2.0, num_points)
    F_all = jnp.zeros((6, num_points, 3, 3))
    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(1 + gamma, 1.0, 1.0))            
    psi_dist_mean = learned_gp.psi_dist(F_all[0]).mean
    
    diff = jnp.diff(psi_dist_mean)
    diff2 = jnp.diff(diff)
    print("Max absolute second derivative of GP mean (x64):", jnp.max(jnp.abs(diff2)))
    
    config.update("jax_enable_x64", False)
    learned_gp_32 = SparseHyperelasticityGP(gp_params, I_z, min_dev, min_vol, max_dev, max_vol, beta=80.0)
    psi_dist_mean_32 = learned_gp_32.psi_dist(F_all[0]).mean
    diff_32 = jnp.diff(psi_dist_mean_32)
    diff2_32 = jnp.diff(diff_32)
    print("Max absolute second derivative of GP mean (x32):", jnp.max(jnp.abs(diff2_32)))

if __name__ == "__main__":
    check_smoothness()

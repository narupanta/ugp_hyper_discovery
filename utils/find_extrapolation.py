import jax
import jax.numpy as jnp
import numpy as np
import os
from core.features import IsotropicFeatureExtractor

def generate_standard_modes(num_points=500, max_gamma=3.0):
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
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:]
    
    true_min_dev = jnp.min(dev_z, axis=0) - 1e-4
    true_max_dev = jnp.max(dev_z, axis=0) + 1e-4
    true_min_vol = jnp.min(vol_z, axis=0) - 1e-4
    true_max_vol = jnp.max(vol_z, axis=0) + 1e-4
    
    extractor = IsotropicFeatureExtractor()
    F_all, gamma = generate_standard_modes()
    
    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]
    
    transitions = []

    for i in range(6):
        dev_m, vol_m = jax.vmap(extractor.extract)(F_all[i])
        
        in_bounds_dev0 = (dev_m[:, 0] >= true_min_dev[0]) & (dev_m[:, 0] <= true_max_dev[0])
        in_bounds_dev1 = (dev_m[:, 1] >= true_min_dev[1]) & (dev_m[:, 1] <= true_max_dev[1])
        in_bounds_vol = (vol_m[:, 0] >= true_min_vol[0]) & (vol_m[:, 0] <= true_max_vol[0])
        
        in_bounds = in_bounds_dev0 & in_bounds_dev1 & in_bounds_vol
        
        if jnp.any(~in_bounds):
            idx = jnp.argmax(~in_bounds)
            transitions.append(gamma[idx].item())
            print(f"Mode {i} ({mode_names[i]}): Interpolation region ends at gamma = {gamma[idx]:.4f}")
        else:
            transitions.append(3.0)
            print(f"Mode {i} ({mode_names[i]}): Entirely within interpolation region up to gamma = 3.0")
            
    np.save("extrapolation_transitions.npy", np.array(transitions))

if __name__ == "__main__":
    main()

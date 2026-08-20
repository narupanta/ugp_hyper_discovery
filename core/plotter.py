import jax 
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from core.utils import *
import jax.random as jr
# helper: per-element edge-based neumann traction contribution
import os
def plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path) :
    fig1, axs = plt.subplots(1, 4, figsize=(22, 5))
    fig1.suptitle("Optimization Objectives and Physics Noise", fontsize=16)

    # Total Loss
    axs[0].plot(steps_history, loss_components_hist["total_loss"], 'k-')
    axs[0].set_title("Total ELBO")
    axs[0].set_yscale('symlog')

    # Log-Likelihood (Data Fit)
    axs[1].plot(steps_history, loss_components_hist["log_like"], color='blue')
    axs[1].set_title("Log-Likelihood")

    # KL Divergence (Regularization)
    axs[2].plot(steps_history, loss_components_hist["kl"], color='green')
    axs[2].set_title("KL Divergence")

    # Physics Residual & Physics Noise Scale
    axs[3].plot(steps_history, loss_components_hist["phy"], color='red', label="Residual")
    ax3_twin = axs[3].twinx()
    ax3_twin.plot(steps_history, params_hist["sigma_free_x"], linestyle='--', label=r"$\sigma_free_x$")
    ax3_twin.plot(steps_history, params_hist["sigma_free_y"], linestyle='--', label=r"$\sigma_free_y$")
    axs[3].set_title("Physics (Resid vs Noise)")
    axs[3].set_yscale('log')
    ax3_twin.set_yscale('log')
    axs[3].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    plt.tight_layout()
    fig1.savefig(os.path.join(save_path, "loss_and_physics.pdf"))

def plot_parameters_hist(params_hist, steps_history, save_path):
    # --- FIGURE 1: Inducing Variables & Positions ---
    has_aniso_inducing = "aniso_u_mean" in params_hist and len(params_hist["aniso_u_mean"]) > 0 and params_hist["aniso_u_mean"][0] is not None
    rows1 = 3 if has_aniso_inducing else 2
    fig1, axes1 = plt.subplots(rows1, 3, figsize=(18, 5 * rows1))
    fig1.suptitle(r"Evolution of Inducing Variables and Positions ($Z, \mathbf{u}$)", fontsize=16)

    # ROW 0: DEVIATORIC GP
    axes1[0, 0].plot(steps_history, np.array(params_hist["dev_u_mean"]))
    axes1[0, 0].set_title(r"Deviatoric Mean ($\mathbf{m}_{dev}$)")
    
    dev_u_var_arr = np.array(params_hist["dev_u_var"])
    if dev_u_var_arr.ndim == 3:
        dev_u_var_arr = np.diagonal(dev_u_var_arr, axis1=1, axis2=2)
    axes1[0, 1].plot(steps_history, dev_u_var_arr)
    axes1[0, 1].set_title(r"Deviatoric Variance ($\mathbf{S}_{dev}$)")
    
    dev_z_1 = np.array(params_hist["dev_z"])[:, :, 0]
    dev_z_2 = np.array(params_hist["dev_z"])[:, :, 1]
    axes1[0, 2].plot(steps_history, dev_z_1) 
    axes1[0, 2].plot(steps_history, dev_z_2)
    axes1[0, 2].set_title(r"Dev. Inducing Positions ($Z_{dev, I_1}$)")

    # ROW 1: VOLUMETRIC GP
    axes1[1, 0].plot(steps_history, np.array(params_hist["vol_u_mean"]))
    axes1[1, 0].set_title(r"Volumetric Mean ($\mathbf{m}_{vol}$)")
    
    vol_u_var_arr = np.array(params_hist["vol_u_var"])
    if vol_u_var_arr.ndim == 3:
        vol_u_var_arr = np.diagonal(vol_u_var_arr, axis1=1, axis2=2)
    axes1[1, 1].plot(steps_history, vol_u_var_arr)
    axes1[1, 1].set_title(r"Volumetric Variance ($\mathbf{S}_{vol}$)")
    
    actual_vol_z = np.array(params_hist["vol_z"])[:, :, 0]
    axes1[1, 2].plot(steps_history, actual_vol_z)
    axes1[1, 2].set_title(r"Vol. Inducing Positions ($Z_{vol, J}$)")

    if has_aniso_inducing:
        # ROW 2: ANISOTROPIC GP
        axes1[2, 0].plot(steps_history, np.array(params_hist["aniso_u_mean"]))
        axes1[2, 0].set_title(r"Anisotropic Mean ($\mathbf{m}_{aniso}$)")
        
        aniso_u_var_arr = np.array(params_hist["aniso_u_var"])
        if aniso_u_var_arr.ndim == 3:
            aniso_u_var_arr = np.diagonal(aniso_u_var_arr, axis1=1, axis2=2)
        axes1[2, 1].plot(steps_history, aniso_u_var_arr)
        axes1[2, 1].set_title(r"Anisotropic Variance ($\mathbf{S}_{aniso}$)")
        
        actual_aniso_z = np.array(params_hist["aniso_z"])[:, :, 0]
        axes1[2, 2].plot(steps_history, actual_aniso_z)
        axes1[2, 2].set_title(r"Aniso. Inducing Positions ($Z_{aniso, I_4}$)")

    for ax in axes1.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig1.savefig(os.path.join(save_path, "inducing_state_evolution.pdf"))
    # --- FIGURE 2: Kernel Hyperparameters ---
    has_aniso = "aniso_gp_lengthscales" in params_hist and len(params_hist["aniso_gp_lengthscales"]) > 0 and params_hist["aniso_gp_lengthscales"][0] is not None
    rows = 3 if has_aniso else 2
    fig2, axes2 = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    fig2.suptitle("Evolution of Kernel Hyperparameters", fontsize=16)

    # 0,0: Deviatoric Lengthscales
    axes2[0, 0].plot(steps_history, np.array(params_hist["dev_gp_lengthscales"]))
    axes2[0, 0].set_title(r"Deviatoric Lengthscales ($\ell_{dev}$)")
    
    # 0,1: Deviatoric Sigma Scaling
    axes2[0, 1].plot(steps_history, np.array(params_hist["dev_gp_sigma_scaling"]))
    axes2[0, 1].set_title(r"Deviatoric Signal Scale ($\sigma_{dev}$)")
    
    # 1,0: Volumetric Lengthscales
    axes2[1, 0].plot(steps_history, np.array(params_hist["vol_gp_lengthscales"]))
    axes2[1, 1].set_yscale('log') # Useful if lengthscales vary widely
    axes2[1, 0].set_title(r"Volumetric Lengthscales ($\ell_{vol}$)")
    
    # 1,1: Volumetric Sigma Scaling
    axes2[1, 1].plot(steps_history, np.array(params_hist["vol_gp_sigma_scaling"]))
    axes2[1, 1].set_title(r"Volumetric Signal Scale ($\sigma_{vol}$)")

    if has_aniso:
        # 2,0: Anisotropic Lengthscales
        axes2[2, 0].plot(steps_history, np.array(params_hist["aniso_gp_lengthscales"]))
        axes2[2, 0].set_title(r"Anisotropic Lengthscales ($\ell_{aniso}$)")
        
        # 2,1: Anisotropic Sigma Scaling
        axes2[2, 1].plot(steps_history, np.array(params_hist["aniso_gp_sigma_scaling"]))
        axes2[2, 1].set_title(r"Anisotropic Signal Scale ($\sigma_{aniso}$)")

    for ax in axes2.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.3)
        ax.set_ylabel("Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig2.savefig(os.path.join(save_path, "hyperparameters_evolution.pdf"))

    # Optional: If you want to track the physics noise parameter separately:

    plt.figure(figsize=(8, 4))
    plt.plot(steps_history, np.array(params_hist["sigma_free_x"]))
    plt.plot(steps_history, np.array(params_hist["sigma_free_y"]))
    plt.plot(steps_history, np.array(params_hist["sigma_fix_x"]))
    plt.plot(steps_history, np.array(params_hist["sigma_fix_y"]))
    plt.title(r"Physics Residual Noise ($\sigma_{physic}$)")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_path, "physics_noise_evolution.pdf"))

    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle("Evolution of Trend Function (Mean) Parameters", fontsize=16)

    # Subplot 1: Deviatoric Trend Parameters (c20, c02, c11, c10, c01)
    dev_params = ["c10", "c01", "c20", "c02", "c11"]
    for p in dev_params:
        if p in params_hist:
            axes3[0].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    
    axes3[0].set_title("Deviatoric Trend Parameters")
    axes3[0].set_xlabel("Iteration Step")
    axes3[0].set_ylabel("Value")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.3)

    # Subplot 2: Volumetric Trend Parameters (k, q, vol_kappa)
    vol_params = ["k", "q", "s", "vol_kappa"]
    for p in vol_params:
        if p in params_hist:
            axes3[1].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    
    axes3[1].set_title("Volumetric Trend Parameters")
    axes3[1].set_xlabel("Iteration Step")
    axes3[1].set_ylabel("Value")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(os.path.join(save_path, "trend_parameters_evolution.pdf"))
    
    if "aniso_theta_mean" in params_hist and len(params_hist["aniso_theta_mean"]) > 0:
        plt.figure(figsize=(8, 4))
        angles_mean = np.degrees(np.array(params_hist["aniso_theta_mean"]))
        angles_std = np.degrees(np.array(params_hist["aniso_theta_var"])) # It's actually stddev
        plt.plot(steps_history, angles_mean, label="Predicted Angle (Mean)")
        plt.fill_between(steps_history, angles_mean - angles_std, angles_mean + angles_std, alpha=0.3, label="±1 Std Dev")
        plt.axhline(30.0, color='r', linestyle='--', label="True Angle (30 deg)")
        plt.title(r"Variational Fiber Orientation Evolution ($\theta$)")
        plt.xlabel("Iteration Step")
        plt.ylabel("Angle (degrees)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_path, "fiber_angle_evolution.pdf"))
        plt.close()
def plot_r2_strain_energy_function(psi_pred, psi_true, psi_dev_pred, psi_dev_true, psi_vol_pred, psi_vol_true, save_path) :
    plt.figure(figsize=(8, 6))
    
    # Flatten the arrays for plotting and R2 calculation
    psi_pred_flat = psi_pred.flatten()
    psi_true_flat = psi_true.flatten()

    
# --- Combined Energy Parity Plots (1x3 Figure) ---
    fig_energy, axs = plt.subplots(1, 3, figsize=(22, 6))
    fig_energy.suptitle(f"Strain Energy Prediction Parity (R² Analysis)", fontsize=16)

    # Data configurations for the loop
    plot_configs = [
        (psi_true, psi_pred, "Total Energy", r"$\psi$"),
        (psi_dev_true, psi_dev_pred, "Deviatoric Energy", r"$\psi_{dev}$"),
        (psi_vol_true, psi_vol_pred, "Volumetric Energy", r"$\psi_{vol}$")
    ]

    for i, (true_data, pred_data, title, label) in enumerate(plot_configs):
        # Flatten and calculate R2
        t_flat = true_data.flatten()
        p_flat = pred_data.flatten()
        r2_val = jnp.corrcoef(t_flat, p_flat)[0, 1]**2
        
        # Scatter plot
        axs[i].scatter(t_flat, p_flat, alpha=0.6, s=10, color='tab:blue')
        
        # Identity line (y=x)
        min_val = min(t_flat.min(), p_flat.min())
        max_val = max(t_flat.max(), p_flat.max())
        axs[i].plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', label='y=x')
        
        # Formatting
        axs[i].set_title(f"{title}\n$R^2 = {r2_val:.4f}$")
        axs[i].set_xlabel(f"True {label}")
        axs[i].set_ylabel(f"Predicted {label}")
        axs[i].grid(True, alpha=0.3)
        axs[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save to your timestamped folder
    energy_plot_path = os.path.join(save_path, "energy_parity_combined.pdf")
    plt.savefig(energy_plot_path)
    print(f"Energy parity plots saved to: {energy_plot_path}")

# def plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path, step):
#     num_points = 50
#     num_samples = 10 # Number of GP posterior samples to draw
#     gamma = jnp.linspace(0.0, 1.0, num_points)
    
#     # Define deformation modes in a dictionary for easy iteration
#     modes = {
#         "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
#         "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
#         "Pure Shear": jnp.zeros((num_points, 3, 3)),
#         "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
#         "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
#         "Simple Shear": jnp.zeros((num_points, 3, 3))
#     }

#     # Populate Deformation Gradients (F)
#     modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
#     modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

#     modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
#     modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

#     modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
#     modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

#     modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
#     modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

#     modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
#     modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

#     modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
#     modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma

#     # JIT compile the vmapped prediction function
#     # psi_pred_func = jax.vmap(learned_gp.psi, in_axes=(0, None))
#     psi_dist = jax.vmap(learned_gp.psi_dist)
#     psi = jax.vmap(learned_gp.psi, in_axes = (0, None))

#     # Create 2x3 grid for all modes
#     fig, axes = plt.subplots(2, 3, figsize=(18, 10))
#     fig.suptitle("Material Model Validation: Strain Energy Density ($\psi$) vs Standard Deformation Modes", fontsize=16)
#     axes = axes.flatten()

#     for idx, (name, F_mode) in enumerate(modes.items()):
#         ax = axes[idx]
        
#         # 1. Calculate and plot True Energy
#         psi_true = true_model.phi(F_mode)
#         ax.plot(gamma, psi_true, label="True", color="grey", linewidth=2.5, zorder=5)

#         # 2. Plot GP Samples
#         for i in range(num_samples):
#             # Pass a unique key for each sample
#             psi_sample = psi(F_mode, jr.PRNGKey(i))
#             label = "GP Samples" if i == 0 else None
#             ax.plot(gamma, psi_sample, color="royalblue", alpha=0.1, linewidth=0.8, label=label, zorder=1)
#         psi_mean = psi_dist(F_mode).mean
#         psi_var = psi_dist(F_mode).var
#         psi_std = jnp.sqrt(psi_var)
#         lower_bound = psi_mean - 1.96 * psi_std
#         upper_bound = psi_mean + 1.96 * psi_std

#         # Plot the Mean

#         # psi_mean = psi_pred_func(F_mode, None)
#         ax.plot(gamma, psi_mean, color="navy", alpha=0.9, linewidth=2, label="GP Mean", zorder=3)

#         # Plot the 95% Confidence Interval
#         ax.fill_between(gamma, lower_bound, upper_bound, 
#                         color="navy", alpha=0.2, label="95% CI", zorder=2)
        

#         # Formatting
#         ax.set_title(name)
#         ax.set_xlabel("Deformation Measure ($\gamma$)")
#         ax.set_ylabel("Energy ($\psi$)")
#         ax.grid(True, linestyle='--', alpha=0.5)
#         y_min = jnp.min(jnp.array([psi_true.min(), psi_mean.min()]))
#         y_max = jnp.max(jnp.array([psi_true.max(), psi_mean.max()]))

#         # Add a 10% buffer so the lines aren't touching the edge
#         padding = (y_max - y_min) * 0.1
#         ax.set_ylim(y_min - padding, y_max + padding)
#         if idx == 0: # Only show legend on first plot to avoid clutter
#             ax.legend()

#     plt.tight_layout(rect=[0, 0.03, 1, 0.95])
#     save_file = os.path.join(save_path, f"material_modes_validation_{step}.pdf")
#     plt.savefig(save_file)
#     print(f"Loading mode validation plots saved to: {save_file}")

import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import os

def _compute_regime_transitions(learned_gp, F_all, gamma):
    """
    Computes interpolation/extrapolation transition gamma values for each deformation mode.
    Returns: trans_tot, trans_dev, trans_vol as lists of float gamma values.
    """
    true_min_dev = jnp.array(learned_gp.min_dev) - 1e-4
    true_max_dev = jnp.array(learned_gp.max_dev) + 1e-4
    true_min_vol = jnp.array(learned_gp.min_vol) - 1e-4
    true_max_vol = jnp.array(learned_gp.max_vol) + 1e-4
    
    trans_tot, trans_dev, trans_vol = [], [], []
    
    for mode in range(F_all.shape[0]):
        feats = jax.vmap(learned_gp.feature_extractor.extract)(F_all[mode])
        dev_m, vol_m = feats[0], feats[1]
        in_dev = ((dev_m[:, 0] >= true_min_dev[0]) & (dev_m[:, 0] <= true_max_dev[0]) &
                  (dev_m[:, 1] >= true_min_dev[1]) & (dev_m[:, 1] <= true_max_dev[1]))
        in_vol = (vol_m[:, 0] >= true_min_vol[0]) & (vol_m[:, 0] <= true_max_vol[0])
        in_tot = in_dev & in_vol
        
        def _get_trans(mask):
            if not jnp.all(mask):
                idx = int(jnp.argmax(~mask))
                val = float(gamma[idx])
                return float(gamma[1]) if idx == 0 else val
            return float(gamma.max())
            
        trans_dev.append(_get_trans(in_dev))
        trans_vol.append(_get_trans(in_vol))
        trans_tot.append(_get_trans(in_tot))
        
    return trans_tot, trans_dev, trans_vol

def plot_combined_validation(learned_gp, true_model, save_path, step):
    num_points = 50
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
    # --- 1. Pre-calculate Deformation Gradients (Bulk) ---
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

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # --- 2. Vectorized Computations ---
    # psi_dist_vmap = learned_gp.psi_dist
    # psi_det_vmap = jax.vmap(jax.vmap(learned_gp.psi_det, in_axes=(0, None)), in_axes=(0, None))
    psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)),  in_axes=(0, None))
    # piola_dist_vmap = jax.vmap(jax.vmap(learned_gp.piola_dist))
    # piola_det_vmap = jax.vmap(jax.vmap(learned_gp.piola_det, in_axes=(0, None)), in_axes=(0, None))
    piola_vmap = jax.vmap(jax.vmap(learned_gp.piola, in_axes=(0, None)), in_axes=(0, None))

    psi_true = jax.vmap(true_model.psi)(F_all)
    P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    P_dist_mean = [learned_gp.piola_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    P_dist_var = [learned_gp.piola_dist(F_all[mode]).var for mode in range(len(mode_names))]


    keys = jax.random.split(jax.random.PRNGKey(step), num_samples)
    psi_samples = jax.vmap(psi_vmap, in_axes=(None, 0))(F_all, keys)
    psi_dets = [jax.vmap(learned_gp.psi_det)(F_all[mode]) for mode in range(len(mode_names))]
    
    P_samples = jax.vmap(piola_vmap, in_axes=(None, 0))(F_all, keys)
    P_dets = [jax.vmap(learned_gp.piola_det)(F_all[mode]) for mode in range(len(mode_names))]
    trans_tot, _, _ = _compute_regime_transitions(learned_gp, F_all, gamma)

    # --- 3. Plotting ---
    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    fig.suptitle(f"Material Discovery Validation - Step {step}", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        if name == "Pure Shear":
            idx_comp = (1, 1); label_P = r"$P_{22}$"
        elif name == "Simple Shear":
            idx_comp = (0, 1); label_P = r"$P_{12}$"
        else:
            idx_comp = (0, 0); label_P = r"$P_{11}$"

        # Column 0: Energy
        ax_psi = axes[i, 0]
        ax_psi.plot(gamma, psi_true[i], 'k--', lw=1.2, label="True", zorder=5)
        # plot energy prediction (deterministic)
        ax_psi.plot(gamma, psi_dets[i], color = "red", lw = 0.8, alpha = 0.3, zorder = 1)
        ax_psi.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_dist_mean[i], color="blue", lw=2, label="GP Mean", zorder=3)
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96*jnp.sqrt(psi_dist_var[i]), 
                           psi_dist_mean[i] + 1.96*jnp.sqrt(psi_dist_var[i]), color="blue", alpha=0.1, zorder=2)
        
        # Limit Energy plot to True model range
        y_min, y_max = jnp.min(psi_true[i]), jnp.max(psi_true[i])
        pad = (y_max - y_min) * 0.1
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())

        # Column 1: Stress
        ax_p = axes[i, 1]
        p_true_comp = P_true[i, :, idx_comp[0], idx_comp[1]]
        p_mean_comp = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(P_dist_var[i][:, idx_comp[0], idx_comp[1]])
        p_samples_comp = P_samples[:, i, :, idx_comp[0], idx_comp[1]]
        p_det_comp = P_dets[i][:, idx_comp[0], idx_comp[1]]

        ax_p.plot(gamma, p_true_comp, 'k--', lw=1.2, label="True", zorder=5)
        ax_p.plot(gamma, p_det_comp, color = "red", lw = 0.8, alpha = 0.3, zorder = 1)

        ax_p.plot(gamma, p_samples_comp.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_mean_comp, color="blue", lw=2, label="GP Mean", zorder=3)
        ax_p.fill_between(gamma, p_mean_comp - 1.96*p_std_comp, 
                         p_mean_comp + 1.96*p_std_comp, color="blue", alpha=0.1, zorder=2)

        # Limit Stress plot to True model range
        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())

        # Formatting
        ax_psi.set_title(f"{name}: Energy")
        ax_p.set_title(f"{name}: Stress")
        trans_g = trans_tot[i]
        max_g = float(gamma.max())
        for ax in [ax_psi, ax_p]:
            ax.axvspan(0, min(trans_g, max_g), color='green', alpha=0.12, zorder=1, label="Interpolation" if (i == 0 and ax == ax_psi) else "")
            if trans_g < max_g:
                ax.axvspan(trans_g, max_g, color='red', alpha=0.12, zorder=1, label="Extrapolation" if (i == 0 and ax == ax_psi) else "")
                ax.axvline(x=trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8, zorder=4)
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.2)
            if i == 0: ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_file = os.path.join(save_path, f"clamped_validation_{step}.pdf")
    plt.savefig(save_file, bbox_inches='tight')
    plt.close()
    print(f"Clamped limits plot saved to: {save_file}")

def plot_ut_ebt_ps_uc_ebc_ss(learned_gp, true_model, save_path, step):
    num_points = 50
    num_samples = 32 
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
    # 1. Pre-calculate all deformation gradients in a single array for bulk processing
    # Shape: (6 modes, num_points, 3, 3)
    F_all = jnp.zeros((6, num_points, 3, 3))
    
    # Helper to fill modes efficiently
    def set_F(idx, f11, f22, f33, f12=0.0):
        # We broadcast the scalar values or arrays to the num_points dimension
        arr = jnp.zeros((num_points, 3, 3))
        arr = arr.at[:, 0, 0].set(f11)
        arr = arr.at[:, 1, 1].set(f22)
        arr = arr.at[:, 2, 2].set(f33)
        arr = arr.at[:, 0, 1].set(f12)
        return arr

    F_all = F_all.at[0].set(set_F(0, 1 + gamma, 1.0, 1.0))            # Uniaxial Tension
    F_all = F_all.at[1].set(set_F(1, 1 + gamma, 1 + gamma, 1.0))    # Equibiaxial Tension
    F_all = F_all.at[2].set(set_F(2, 1 + gamma, 1/(1 + gamma), 1.0)) # Pure Shear
    F_all = F_all.at[3].set(set_F(3, 1/(1 + gamma), 1.0, 1.0))       # Uniaxial Compression
    F_all = F_all.at[4].set(set_F(4, 1/(1 + gamma), 1/(1 + gamma), 1.0)) # Equibiaxial Compression
    F_all = F_all.at[5].set(set_F(5, 1.0, 1.0, 1.0, f12=gamma))      # Simple Shear

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    # 2. Vectorize the GP calls
    # Vectorize across the points (axis 0) AND the modes (new axis)
    psi_dist_vmap = jax.vmap(jax.vmap(learned_gp.psi_dist))
    
    # Vectorize sample generation: (samples, modes, points)
    # We vmap over keys and then over the double-vmapped psi function
    sample_keys = jax.random.split(jax.random.PRNGKey(0), num_samples)
    psi_sample_vmap = jax.vmap(jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None)), in_axes=(None, 0))

    # 3. Compute everything in bulk (The heavy lifting)
    # psi_true: (6, num_points)
    psi_true_all = jax.vmap(true_model.phi)(F_all)
    
    # stats: (6, num_points)
    dist = psi_dist_vmap(F_all)
    psi_mean_all = dist.mean
    psi_std_all = jnp.sqrt(dist.var)
    
    # samples: (num_samples, 6, num_points)
    psi_samples_all = psi_sample_vmap(F_all, sample_keys)

    # 4. Plotting
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Material Model Validation: Strain Energy Density ($\psi$) vs Standard Deformation Modes", fontsize=16)
    axes = axes.flatten()

    for idx, name in enumerate(mode_names):
        ax = axes[idx]
        
        # 1. Plot True Energy (Dashed and thinner as requested)
        ax.plot(gamma, psi_true_all[idx], label="True", color="black", 
                linestyle='--', linewidth=1.5, zorder=5)

        # 2. Plot GP Samples (From pre-computed bulk array)
        ax.plot(gamma, psi_samples_all[:, idx, :].T, color="lightblue", 
                alpha=0.1, linewidth=0.8, zorder=1)
        
        # 3. Plot Mean and CI
        lower_bound = psi_mean_all[idx] - 1.96 * psi_std_all[idx]
        upper_bound = psi_mean_all[idx] + 1.96 * psi_std_all[idx]

        ax.plot(gamma, psi_mean_all[idx], color="blue", alpha=0.9, linewidth=2, label="GP Mean", zorder=3)
        ax.fill_between(gamma, lower_bound, upper_bound, color="blue", alpha=0.2, label="95% CI", zorder=2)

        # Formatting
        ax.set_title(name)
        ax.set_xlabel("Deformation Measure ($\gamma$)")
        ax.set_ylabel("Energy ($\psi$)")
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Buffering
        y_min, y_max = psi_true_all[idx].min(), psi_true_all[idx].max()
        padding = (y_max - y_min) * 0.15
        ax.set_ylim(y_min - padding, y_max + padding)
        
        if idx == 0:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_file = os.path.join(save_path, f"material_modes_validation_{step}.pdf")
    plt.savefig(save_file)
    print(f"Loading mode validation plots saved to: {save_file}")

def plot_stress_validation(gp_model, true_model, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Piola Stress Validation: Model Discovery vs. True Physics", fontsize=16)
    axes = axes.flatten()
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma
    
    for i, (mode_name, F_stack) in enumerate(modes.items()):
        # num_samples = 10
        # keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
        # 1. Compute Piola Stress for the whole stack
        # Piola_stress function uses jax.grad(psi)
        # P_predicted = jax.vmap(lambda f: gp_model.piola(f, None))(F_stack)
        # P_predicted = jnp.array(P_predicted)
        P_mean = jax.vmap(lambda f: gp_model.piola_dist(f).mean)(F_stack)
        P_var = jax.vmap(lambda f: gp_model.piola_dist(f).var)(F_stack)
        P_std = jnp.sqrt(P_var)
        P_lower_bound = P_mean - 1.96 * P_std
        P_upper_bound = P_mean + 1.96 * P_std
        
        P_true = jax.vmap(true_model.P)(F_stack)
        # 2. Select the relevant component based on the mode
        if mode_name == "Pure Shear":
            y_pred = P_mean[:, 1, 1] # P22
            y_true = P_true[:, 1, 1]
            lower = P_lower_bound[:, 1, 1]
            upper = P_upper_bound[:, 1, 1]
            label = r"$P_{22}$"
        elif mode_name == "Simple Shear":
            y_pred = P_mean[:, 0, 1] # P12
            y_true = P_true[:, 0, 1]
            lower = P_lower_bound[:, 0, 1]
            upper = P_upper_bound[:, 0, 1]
            label = r"$P_{12}$"
        else:
            y_pred = P_mean[:, 0, 0] # P11
            y_true = P_true[:, 0, 0]
            lower = P_lower_bound[:, 0, 0]
            upper = P_upper_bound[:, 0, 0]
            label = r"$P_{11}$"

        # 3. Plotting
        gamma = jnp.linspace(0, 1, len(F_stack)) # Match your gamma range
        axes[i].plot(gamma, y_pred, color='blue', label='GP Predicted')
        axes[i].fill_between(gamma, lower, upper, color='blue', alpha=0.2, label='95% CI')
        # Assuming you have ground truth stress 'y_true'
        axes[i].plot(gamma, y_true, 'k--', alpha=0.6, label='True')
        
        axes[i].set_title(mode_name)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel(r"$\gamma$")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "piola_stress_validation.pdf"))

def plot_inducing_points(dev_z, vol_z, dev_I, vol_I, save_path):
    # Setup Figure 1: Inducing points in Feature Space
    fig1, axes1 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: I1_dev vs I2_dev (Inducing points for the Deviatoric GP)
    
    axes1[0].scatter(dev_I[:, 0], dev_I[:, 1], marker='o', label='Invariants (Dev)')
    axes1[0].scatter(dev_z[:, 0], dev_z[:, 1], c='red', marker='x', label='Inducing Points (Dev)')
    axes1[0].set_xlabel(r"$\bar{I}_1$")
    axes1[0].set_ylabel(r"$\bar{I}_2$")
    axes1[0].set_title("Deviatoric Inducing Points")
    axes1[0].legend()

    # Plot 2: J vs -2*J (Inducing points for the Volumetric GP)
    axes1[1].scatter(vol_I[:, 0], vol_I[:, 1], marker='o', label='J and -2 * J (Vol)')
    axes1[1].scatter(vol_z[:, 0], vol_z[:, 1], c='red', marker='x', label='Inducing Points (Vol)')
    # Reference constraint line
    axes1[1].set_xlabel(r"$J$")
    axes1[1].set_ylabel(r"$-2J$")
    axes1[1].set_title("Volumetric Inducing Points")
    axes1[1].legend()
    
    fig1.savefig(os.path.join(save_path, "inducing_points_features.pdf"))

    # --- Setup Figure 2: Standard Load Paths ---
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma
    # Reuse your 'modes' dictionary logic here (assuming 'modes' is accessible)
    # For brevity, we compute and plot the lines for each mode:
    axes2[0].scatter(dev_I[:, 0] - 3, dev_I[:, 1] - 3, marker='o')
    axes2[1].scatter(dev_I[:, 0] - 3, (vol_I[:, 0] - 1)**2, marker='o')
    axes2[2].scatter(dev_I[:, 1] - 3, (vol_I[:, 0] - 1)**2, marker='o')

    axes2[0].scatter(dev_z[:, 0] - 3, dev_z[:, 1] - 3, color = "red", marker='x')
    axes2[1].scatter(dev_z[:, 0] - 3, (vol_z[:, 0] - 1)**2, color = "red", marker='x')
    axes2[2].scatter(dev_z[:, 1] - 3, (vol_z[:, 0] - 1)**2, color = "red", marker='x')
    linestyles = {
        "Uniaxial Tension": "-",
        "Uniaxial Compression": "--",
        "Equibiaxial Tension": "-.",
        "Equibiaxial Compression": ":",
        "Pure Shear": (0, (3, 1, 1, 1)),
        "Simple Shear": (0, (5, 2))
    }
    for mode_name, F_stack in modes.items():
        i, _  = jax.vmap(invariants_and_derivatives)(F_stack)
        js = jnp.sqrt(i[:, 2])
        i1_bar = js**(-2/3) * i[:, 0]
        i2_bar = js**(-4/3) * i[:, 1]
        # Plot 1: I1_bar - 3 vs I2_bar - 3
        axes2[0].plot(i1_bar - 3, i2_bar - 3, label=mode_name,linestyle=linestyles[mode_name])
        

        # Plot 2: I1_bar - 3 vs (J - 1)**2
        axes2[1].plot(i1_bar - 3, (js - 1)**2, label=mode_name,linestyle=linestyles[mode_name])
        
        # Plot 3: I2_bar - 3 vs (J - 1)**2
        axes2[2].plot(i2_bar - 3, (js - 1)**2, label=mode_name,linestyle=linestyles[mode_name])
    # Labeling Figure 2
    axes2[0].set_title(r"$\bar{I}_1-3$ vs $\bar{I}_2-3$")
    axes2[1].set_title(r"$\bar{I}_1-3$ vs $(J-1)^2$")
    axes2[2].set_title(r"$\bar{I}_2-3$ vs $(J-1)^2$")
    
    for ax in axes2:
        ax.legend(fontsize='small')
        ax.grid(True, alpha=0.2)
        
    plt.tight_layout()
    fig2.savefig(os.path.join(save_path, "standard_loading_paths.pdf"))


def plot_stress_validation(gp_model, true_model, save_path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Piola Stress Validation: Model Discovery vs. True Physics", fontsize=16)
    axes = axes.flatten()
    num_points = 100
    gamma = jnp.linspace(0.0, 1.0, num_points)
    modes = {
        "Uniaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Tension": jnp.zeros((num_points, 3, 3)),
        "Pure Shear": jnp.zeros((num_points, 3, 3)),
        "Uniaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Equibiaxial Compression": jnp.zeros((num_points, 3, 3)),
        "Simple Shear": jnp.zeros((num_points, 3, 3))
    }

    # Populate Deformation Gradients (F)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)

    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 1, 1].set(1).at[:, 2, 2].set(1)

    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1/(1 + gamma))
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 1, 1].set(1/(1 + gamma)).at[:, 2, 2].set(1)

    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 1].set(gamma) # Standard simple shear gamma
    
    for i, (mode_name, F_stack) in enumerate(modes.items()):
        # num_samples = 10
        # keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
        # 1. Compute Piola Stress for the whole stack
        # Piola_stress function uses jax.grad(psi)
        # P_predicted = jax.vmap(lambda f: gp_model.piola(f, None))(F_stack)
        # P_predicted = jnp.array(P_predicted)
        P_mean = jax.vmap(lambda f: gp_model.piola_dist(f).mean)(F_stack)
        P_var = jax.vmap(lambda f: gp_model.piola_dist(f).var)(F_stack)
        P_std = jnp.sqrt(P_var)
        P_lower_bound = P_mean - 1.96 * P_std
        P_upper_bound = P_mean + 1.96 * P_std
        
        P_true = jax.vmap(true_model.P)(F_stack)
        # 2. Select the relevant component based on the mode
        if mode_name == "Pure Shear":
            y_pred = P_mean[:, 1, 1] # P22
            y_true = P_true[:, 1, 1]
            lower = P_lower_bound[:, 1, 1]
            upper = P_upper_bound[:, 1, 1]
            label = r"$P_{22}$"
        elif mode_name == "Simple Shear":
            y_pred = P_mean[:, 0, 1] # P12
            y_true = P_true[:, 0, 1]
            lower = P_lower_bound[:, 0, 1]
            upper = P_upper_bound[:, 0, 1]
            label = r"$P_{12}$"
        else:
            y_pred = P_mean[:, 0, 0] # P11
            y_true = P_true[:, 0, 0]
            lower = P_lower_bound[:, 0, 0]
            upper = P_upper_bound[:, 0, 0]
            label = r"$P_{11}$"

        # 3. Plotting
        gamma = jnp.linspace(0, 1, len(F_stack)) # Match your gamma range
        axes[i].plot(gamma, y_pred, color='blue', label='GP Predicted')
        axes[i].fill_between(gamma, lower, upper, color='blue', alpha=0.2, label='95% CI')
        # Assuming you have ground truth stress 'y_true'
        axes[i].plot(gamma, y_true, 'k--', alpha=0.6, label='True')
        
        axes[i].set_title(mode_name)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel(r"$\gamma$")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(save_path, "piola_stress_validation.pdf"))

def plot_deterministic_against_true() :
    raise NotImplementedError

def plot_noise_model_verification(gp_model, data, save_path) :
    pass

from sklearn.metrics import r2_score

# --- FIGURE 1: Energy Decomposition Validation ---
def plot_energy_decomposition_validation(learned_gp, true_model, save_path):
    print("Generating Energy Decomposition Validation Plot...")
    num_points = 50
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points)
    
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

    mode_names = ["Uniaxial Tension", "Equibiaxial Tension", "Pure Shear", 
                  "Uniaxial Compression", "Equibiaxial Compression", "Simple Shear"]

    psi_true_tot = jax.vmap(jax.vmap(true_model.psi))(F_all)
    psi_true_dev = jax.vmap(jax.vmap(true_model.psi_dev))(F_all)
    psi_true_vol = jax.vmap(jax.vmap(true_model.psi_vol))(F_all)
    if hasattr(true_model, 'psi_aniso'):
        psi_true_aniso = jax.vmap(jax.vmap(true_model.psi_aniso))(F_all)
    else:
        psi_true_aniso = jnp.zeros_like(psi_true_tot)

    dist_tot = jax.vmap(learned_gp.psi_dist)(F_all)
    psi_mean_tot, psi_std_tot = dist_tot.mean, jnp.sqrt(dist_tot.var)
    
    dist_dev = jax.vmap(learned_gp.dev_psi_dist)(F_all)
    psi_mean_dev, psi_std_dev = dist_dev.mean, jnp.sqrt(dist_dev.var)

    if hasattr(learned_gp, 'is_anisotropic') and learned_gp.is_anisotropic:
        dist_aniso = jax.vmap(learned_gp.aniso_psi_dist)(F_all)
        psi_mean_aniso, psi_std_aniso = dist_aniso.mean, jnp.sqrt(dist_aniso.var)
    else:
        psi_mean_aniso = jnp.zeros_like(psi_true_tot)
        psi_std_aniso = jnp.zeros_like(psi_true_tot)


    dist_vol = jax.vmap(learned_gp.vol_psi_dist)(F_all)
    psi_mean_vol, psi_std_vol = dist_vol.mean, jnp.sqrt(dist_vol.var)

    keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
    
    psi_samples_dev = []
    psi_samples_vol = []
    psi_samples_aniso_list = []
    psi_samples_tot = []
    for k in keys:
        fn = learned_gp.get_path_components_psi_fn(k)
        dev_s, vol_s, aniso_s = jax.vmap(jax.vmap(fn))(F_all)
        psi_samples_dev.append(dev_s)
        psi_samples_vol.append(vol_s)
        psi_samples_aniso_list.append(aniso_s)
        psi_samples_tot.append(dev_s + vol_s + aniso_s)
    psi_samples_dev = jnp.stack(psi_samples_dev, axis=0) 
    psi_samples_vol = jnp.stack(psi_samples_vol, axis=0)
    psi_samples_aniso = jnp.stack(psi_samples_aniso_list, axis=0)
    psi_samples_tot = jnp.stack(psi_samples_tot, axis=0)
    trans_tot, trans_dev, trans_vol = _compute_regime_transitions(learned_gp, F_all, gamma)

    def calc_metrics(true, mean, std):
        rmse = jnp.sqrt(jnp.mean((true - mean)**2))
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        coverage = jnp.mean((true >= lower) & (true <= upper)) * 100
        return rmse, coverage
    
    fig, axes = plt.subplots(6, 4, figsize=(24, 24))
    fig.suptitle(f"Energy Decomposition Validation", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        configs = [
            (0, "Deviatoric", psi_true_dev[i], psi_mean_dev[i], psi_std_dev[i], psi_samples_dev[:, i, :], trans_dev[i]),
            (1, "Volumetric", psi_true_vol[i], psi_mean_vol[i], psi_std_vol[i], psi_samples_vol[:, i, :], trans_vol[i]),
            (2, "Anisotropic", psi_true_aniso[i], psi_mean_aniso[i], psi_std_aniso[i], psi_samples_aniso[:, i, :], trans_tot[i]),
            (3, "Total Energy", psi_true_tot[i], psi_mean_tot[i], psi_std_tot[i], psi_samples_tot[:, i, :], trans_tot[i])
        ]
        for col, col_name, true_val, mean_val, std_val, samples, trans_g in configs:
            ax = axes[i, col]
            ax.plot(gamma, true_val, 'k--', lw=1.5, label="True", zorder=5)
            ax.plot(gamma, samples.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
            ax.plot(gamma, mean_val, color="blue", lw=2, label="GP Mean", zorder=3)
            ax.fill_between(gamma, mean_val - 1.96*std_val, mean_val + 1.96*std_val, color="blue", alpha=0.2, zorder=2)
            
            max_g = float(gamma.max())
            ax.axvspan(0, min(trans_g, max_g), color='green', alpha=0.12, zorder=1, label="Interpolation" if (i == 0 and col == 3) else "")
            if trans_g < max_g:
                ax.axvspan(trans_g, max_g, color='red', alpha=0.12, zorder=1, label="Extrapolation" if (i == 0 and col == 3) else "")
                ax.axvline(x=trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8, zorder=4)

            rmse, coverage = calc_metrics(true_val, mean_val, std_val)
            
            ax.set_title(f"{name}: {col_name}\nRMSE: {rmse:.4f} | Cov: {coverage:.1f}%")
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.2)
            
            y_min, y_max = jnp.min(true_val), jnp.max(true_val)
            pad = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlim(0, max_g)
            if i == 0 and col == 3:
                ax.legend(loc="upper left", framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "energy_decomposition.pdf"), bbox_inches='tight')
    plt.close()

# --- FIGURE 2: R2 Plot on Training Data ---
def plot_training_r2(learned_gp, true_model, F_train_full, save_path):
    print("Generating Training Data R2 Plot...")
    num_steps = F_train_full.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_title("Training Data Energy R2 (GP Mean vs Truth)", fontsize=16)
    
    colors = plt.cm.jet(np.linspace(0, 1, num_steps))
    
    all_true = []
    all_mean = []
    all_std = []
    
    for step in range(num_steps):
        F_step = F_train_full[step]
        true_psi = jax.vmap(true_model.psi)(F_step)
        dist = learned_gp.psi_dist(F_step)
        mean_psi = dist.mean
        std_psi = jnp.sqrt(dist.var)
        
        all_true.append(true_psi)
        all_mean.append(mean_psi)
        all_std.append(std_psi)
        
        ax.errorbar(true_psi, mean_psi, yerr=1.96*std_psi, fmt='o', color=colors[step], 
                    alpha=0.2, markersize=3, label=f"Step {step}" if step % 5 == 0 else "")
                    
    all_true = jnp.concatenate(all_true)
    all_mean = jnp.concatenate(all_mean)
    all_std = jnp.concatenate(all_std)
    
    r2 = r2_score(all_true, all_mean)
    rmse = jnp.sqrt(jnp.mean((all_true - all_mean)**2))
    lower = all_mean - 1.96 * all_std
    upper = all_mean + 1.96 * all_std
    coverage = jnp.mean((all_true >= lower) & (all_true <= upper)) * 100
    
    min_val = min(all_true.min(), all_mean.min())
    max_val = max(all_true.max(), all_mean.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label="Parity")
    
    ax.text(0.05, 0.95, f"R2: {r2:.4f}\nRMSE: {rmse:.4f}\nCoverage: {coverage:.1f}%", 
            transform=ax.transAxes, verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=14)
            
    ax.set_xlabel("True Total Energy")
    ax.set_ylabel("Predicted Total Energy")
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_r2_energy.pdf"), bbox_inches='tight')
    plt.close()
    
    return float(r2), float(rmse), float(coverage)
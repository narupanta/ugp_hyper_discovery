"""
plots/training.py: Centralized Training and Extraction Visualizations

Contains all plotting routines used during unsupervised Gaussian Process training
and extraction, unified with the repository's styling theme.
"""

import os
import json
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from core.utils import C_func, I3_func
from plots.theme import (
    apply_style,
    save_figure,
    MODE_NAMES,
    MODE_COLORS,
    MODE_LINESTYLES,
    COMPONENT_COLORS,
    CURVE_STYLES,
)


def plot_loss_analysis(loss_components_hist, params_hist, steps_history, save_path):
    """Plots ELBO, log-likelihood, KL divergence, and physics residual losses."""
    apply_style()
    fig, axs = plt.subplots(1, 4, figsize=(22, 5))
    fig.suptitle("Optimization Objectives and Physics Noise", fontsize=15)

    # Total Loss
    axs[0].plot(steps_history, loss_components_hist["total_loss"], 'k-')
    axs[0].set_title("Total ELBO")
    axs[0].set_yscale('symlog')

    # Log-Likelihood (Data Fit)
    axs[1].plot(steps_history, loss_components_hist["log_like"], color='#1f77b4')
    axs[1].set_title("Log-Likelihood")

    # KL Divergence (Regularization)
    axs[2].plot(steps_history, loss_components_hist["kl"], color='#2ca02c')
    axs[2].set_title("KL Divergence")

    # Physics Residual & Physics Noise Scale
    axs[3].plot(steps_history, loss_components_hist["phy"], color='#d62728', label="Residual")
    ax3_twin = axs[3].twinx()
    ax3_twin.plot(steps_history, params_hist["sigma_free_x"], linestyle='--', label=r"$\sigma_{\mathrm{free}, x}$")
    ax3_twin.plot(steps_history, params_hist["sigma_free_y"], linestyle='--', label=r"$\sigma_{\mathrm{free}, y}$")
    axs[3].set_title("Physics (Resid vs Noise)")
    axs[3].set_yscale('log')
    ax3_twin.set_yscale('log')
    axs[3].legend(loc='upper left')
    ax3_twin.legend(loc='upper right')

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "loss_and_physics.pdf"))
    plt.close(fig)


def plot_parameters_hist(params_hist, steps_history, save_path):
    """Plots evolution of inducing points, kernel hyperparameters, and physics noise."""
    apply_style()

    # --- FIGURE 1: Inducing Variables & Positions ---
    has_aniso_inducing = "aniso_u_mean" in params_hist and len(params_hist["aniso_u_mean"]) > 0 and params_hist["aniso_u_mean"][0] is not None
    rows1 = 3 if has_aniso_inducing else 2
    fig1, axes1 = plt.subplots(rows1, 3, figsize=(18, 5 * rows1))
    fig1.suptitle(r"Evolution of Inducing Variables and Positions ($Z, \mathbf{u}$)", fontsize=15)

    # ROW 0: DEVIATORIC GP
    axes1[0, 0].plot(steps_history, np.array(params_hist["dev_u_mean"]))
    axes1[0, 0].set_title(r"Deviatoric Mean ($\mathbf{m}_{\mathrm{dev}}$)")

    dev_u_var_arr = np.array(params_hist["dev_u_var"])
    if dev_u_var_arr.ndim == 3:
        dev_u_var_arr = np.diagonal(dev_u_var_arr, axis1=1, axis2=2)
    axes1[0, 1].plot(steps_history, dev_u_var_arr)
    axes1[0, 1].set_title(r"Deviatoric Variance ($\mathbf{S}_{\mathrm{dev}}$)")

    dev_z_1 = np.array(params_hist["dev_z"])[:, :, 0]
    dev_z_2 = np.array(params_hist["dev_z"])[:, :, 1]
    axes1[0, 2].plot(steps_history, dev_z_1)
    axes1[0, 2].plot(steps_history, dev_z_2)
    axes1[0, 2].set_title(r"Dev. Inducing Positions ($Z_{\mathrm{dev}, I_1}$)")

    # ROW 1: VOLUMETRIC GP
    axes1[1, 0].plot(steps_history, np.array(params_hist["vol_u_mean"]))
    axes1[1, 0].set_title(r"Volumetric Mean ($\mathbf{m}_{\mathrm{vol}}$)")

    vol_u_var_arr = np.array(params_hist["vol_u_var"])
    if vol_u_var_arr.ndim == 3:
        vol_u_var_arr = np.diagonal(vol_u_var_arr, axis1=1, axis2=2)
    axes1[1, 1].plot(steps_history, vol_u_var_arr)
    axes1[1, 1].set_title(r"Volumetric Variance ($\mathbf{S}_{\mathrm{vol}}$)")

    actual_vol_z = np.array(params_hist["vol_z"])[:, :, 0]
    axes1[1, 2].plot(steps_history, actual_vol_z)
    axes1[1, 2].set_title(r"Vol. Inducing Positions ($Z_{\mathrm{vol}, J}$)")

    if has_aniso_inducing:
        # ROW 2: ANISOTROPIC GP
        axes1[2, 0].plot(steps_history, np.array(params_hist["aniso_u_mean"]))
        axes1[2, 0].set_title(r"Anisotropic Mean ($\mathbf{m}_{\mathrm{aniso}}$)")

        aniso_u_var_arr = np.array(params_hist["aniso_u_var"])
        if aniso_u_var_arr.ndim == 3:
            aniso_u_var_arr = np.diagonal(aniso_u_var_arr, axis1=1, axis2=2)
        axes1[2, 1].plot(steps_history, aniso_u_var_arr)
        axes1[2, 1].set_title(r"Anisotropic Variance ($\mathbf{S}_{\mathrm{aniso}}$)")

        actual_aniso_z = np.array(params_hist["aniso_z"])[:, :, 0]
        axes1[2, 2].plot(steps_history, actual_aniso_z)
        axes1[2, 2].set_title(r"Aniso. Inducing Positions ($Z_{\mathrm{aniso}, I_4}$)")

    for ax in axes1.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig1, os.path.join(save_path, "inducing_state_evolution.pdf"))
    plt.close(fig1)

    # --- FIGURE 2: Kernel Hyperparameters ---
    has_aniso = "aniso_gp_lengthscales" in params_hist and len(params_hist["aniso_gp_lengthscales"]) > 0 and params_hist["aniso_gp_lengthscales"][0] is not None
    rows = 3 if has_aniso else 2
    fig2, axes2 = plt.subplots(rows, 2, figsize=(14, 5 * rows))
    fig2.suptitle("Evolution of Kernel Hyperparameters", fontsize=15)

    # 0,0: Deviatoric Lengthscales
    axes2[0, 0].plot(steps_history, np.array(params_hist["dev_gp_lengthscales"]))
    axes2[0, 0].set_title(r"Deviatoric Lengthscales ($\ell_{\mathrm{dev}}$)")

    # 0,1: Deviatoric Sigma Scaling
    axes2[0, 1].plot(steps_history, np.array(params_hist["dev_gp_sigma_scaling"]))
    axes2[0, 1].set_title(r"Deviatoric Signal Scale ($\sigma_{\mathrm{dev}}$)")

    # 1,0: Volumetric Lengthscales
    axes2[1, 0].plot(steps_history, np.array(params_hist["vol_gp_lengthscales"]))
    axes2[1, 0].set_title(r"Volumetric Lengthscales ($\ell_{\mathrm{vol}}$)")

    # 1,1: Volumetric Sigma Scaling
    axes2[1, 1].plot(steps_history, np.array(params_hist["vol_gp_sigma_scaling"]))
    axes2[1, 1].set_yscale('log')
    axes2[1, 1].set_title(r"Volumetric Signal Scale ($\sigma_{\mathrm{vol}}$)")

    if has_aniso:
        # 2,0: Anisotropic Lengthscales
        axes2[2, 0].plot(steps_history, np.array(params_hist["aniso_gp_lengthscales"]))
        axes2[2, 0].set_title(r"Anisotropic Lengthscales ($\ell_{\mathrm{aniso}}$)")

        # 2,1: Anisotropic Sigma Scaling
        axes2[2, 1].plot(steps_history, np.array(params_hist["aniso_gp_sigma_scaling"]))
        axes2[2, 1].set_title(r"Anisotropic Signal Scale ($\sigma_{\mathrm{aniso}}$)")

    for ax in axes2.flatten():
        ax.set_xlabel("Iteration Step")
        ax.grid(True, alpha=0.25)
        ax.set_ylabel("Value")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig2, os.path.join(save_path, "hyperparameters_evolution.pdf"))
    plt.close(fig2)

    # Track physics noise parameter separately:
    fig_pn, ax_pn = plt.subplots(figsize=(8, 4))
    ax_pn.plot(steps_history, np.array(params_hist["sigma_free_x"]), label=r"$\sigma_{\mathrm{free}, x}$")
    ax_pn.plot(steps_history, np.array(params_hist["sigma_free_y"]), label=r"$\sigma_{\mathrm{free}, y}$")
    ax_pn.plot(steps_history, np.array(params_hist["sigma_fix_x"]), label=r"$\sigma_{\mathrm{fix}, x}$")
    ax_pn.plot(steps_history, np.array(params_hist["sigma_fix_y"]), label=r"$\sigma_{\mathrm{fix}, y}$")
    ax_pn.set_title(r"Physics Residual Noise ($\sigma_{\mathrm{physic}}$)")
    ax_pn.set_yscale('log')
    ax_pn.set_xlabel("Iteration Step")
    ax_pn.legend()
    ax_pn.grid(True, alpha=0.25)
    plt.tight_layout()
    save_figure(fig_pn, os.path.join(save_path, "physics_noise_evolution.pdf"))
    plt.close(fig_pn)

    # Evolution of Trend Function (Mean) Parameters
    fig3, axes3 = plt.subplots(1, 2, figsize=(16, 6))
    fig3.suptitle("Evolution of Trend Function (Mean) Parameters", fontsize=15)

    dev_params = ["c10", "c01", "c20", "c02", "c11"]
    for p in dev_params:
        if p in params_hist:
            axes3[0].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    axes3[0].set_title("Deviatoric Trend Parameters")
    axes3[0].set_xlabel("Iteration Step")
    axes3[0].set_ylabel("Value")
    axes3[0].legend()
    axes3[0].grid(True, alpha=0.25)

    vol_params = ["k", "q", "s"]
    for p in vol_params:
        if p in params_hist:
            axes3[1].plot(steps_history, np.array(params_hist[p]), label=fr"${p}$")
    axes3[1].set_title("Volumetric Trend Parameters")
    axes3[1].set_xlabel("Iteration Step")
    axes3[1].set_ylabel("Value")
    axes3[1].legend()
    axes3[1].grid(True, alpha=0.25)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig3, os.path.join(save_path, "trend_parameters_evolution.pdf"))
    plt.close(fig3)

    if "aniso_theta_mean" in params_hist and len(params_hist["aniso_theta_mean"]) > 0:
        fig_th, ax_th = plt.subplots(figsize=(8, 4))
        angles_mean = np.degrees(np.array(params_hist["aniso_theta_mean"]))
        ax_th.plot(steps_history, angles_mean, label="Predicted Angle (Mean)")
        if "aniso_theta_var" in params_hist and len(params_hist["aniso_theta_var"]) > 0:
            angles_std = np.degrees(np.array(params_hist["aniso_theta_var"]))
            ax_th.fill_between(steps_history, angles_mean - angles_std, angles_mean + angles_std, alpha=0.3, label=r"$\pm 1$ Std Dev")
        ax_th.axhline(30.0, color='r', linestyle='--', label="True Angle (30 deg)")
        ax_th.set_title(r"Fiber Orientation Evolution ($\theta$)")
        ax_th.set_xlabel("Iteration Step")
        ax_th.set_ylabel("Angle (degrees)")
        ax_th.legend()
        ax_th.grid(True, alpha=0.25)
        plt.tight_layout()
        save_figure(fig_th, os.path.join(save_path, "fiber_angle_evolution.pdf"))
        plt.close(fig_th)


def plot_inducing_points(dev_z, vol_z, dev_I, vol_I, save_path, aniso_z=None, aniso_I=None, feature_extractor=None):
    """Plots inducing points relative to training invariants in feature and deformation space."""
    apply_style()
    has_aniso1 = aniso_z is not None and aniso_I is not None
    has_aniso2 = has_aniso1 and aniso_z.shape[-1] >= 2

    n_fig1 = 2
    if has_aniso2:
        n_fig1 = 4
    elif has_aniso1:
        n_fig1 = 3

    fig1, axes1 = plt.subplots(1, n_fig1, figsize=(6 * n_fig1, 5), squeeze=False)
    axes1 = axes1[0]

    # Plot 1: I1_dev vs I2_dev (Inducing points for Deviatoric GP)
    axes1[0].scatter(dev_I[:, 0], dev_I[:, 1], marker='o', alpha=0.3, color=COMPONENT_COLORS["dev"], label='Invariants (Dev)')
    axes1[0].scatter(dev_z[:, 0], dev_z[:, 1], c='red', marker='x', s=50, label='Inducing Points (Dev)')
    axes1[0].set_xlabel(r"$\bar{I}_1$")
    axes1[0].set_ylabel(r"$\bar{I}_2$")
    axes1[0].set_title("Deviatoric Inducing Points")
    axes1[0].legend()
    axes1[0].grid(True, alpha=0.25)

    # Plot 2: J (Inducing points for Volumetric GP)
    axes1[1].scatter(vol_I[:, 0], -2.0 * vol_I[:, 0], marker='o', alpha=0.3, color=COMPONENT_COLORS["vol"], label=r'$J$ and $-2J$ (Vol)')
    axes1[1].scatter(vol_z[:, 0], -2.0 * vol_z[:, 0], c='red', marker='x', s=50, label='Inducing Points (Vol)')
    axes1[1].set_xlabel(r"$J$")
    axes1[1].set_ylabel(r"$-2J$")
    axes1[1].set_title("Volumetric Inducing Points")
    axes1[1].legend()
    axes1[1].grid(True, alpha=0.25)

    if has_aniso1 and not has_aniso2:
        axes1[2].scatter(aniso_I[:, 0], jnp.zeros_like(aniso_I[:, 0]), marker='o', alpha=0.3, color=COMPONENT_COLORS["aniso"], label=r'Invariants ($\bar{I}_4$)')
        axes1[2].scatter(aniso_z[:, 0], jnp.zeros_like(aniso_z[:, 0]), c='red', marker='x', s=50, label='Inducing Points')
        axes1[2].set_xlabel(r"$\bar{I}_4$")
        axes1[2].set_title("Anisotropic Inducing Points")
        axes1[2].legend()
        axes1[2].grid(True, alpha=0.25)
    elif has_aniso2:
        axes1[2].scatter(aniso_I[:, 0], aniso_I[:, 1], marker='o', alpha=0.3, color=COMPONENT_COLORS["aniso"], label=r'Invariants ($\bar{I}_4$ vs $\bar{I}_6$)')
        axes1[2].scatter(aniso_z[:, 0], aniso_z[:, 1], c='red', marker='x', s=50, label='Inducing Points')
        axes1[2].set_xlabel(r"$\bar{I}_4$")
        axes1[2].set_ylabel(r"$\bar{I}_6$")
        axes1[2].set_title(r"Anisotropic Inducing Points ($\bar{I}_4$ vs $\bar{I}_6$)")
        axes1[2].legend()
        axes1[2].grid(True, alpha=0.25)

        axes1[3].scatter(aniso_I[:, 0], jnp.zeros_like(aniso_I[:, 0]), marker='o', alpha=0.3, color=COMPONENT_COLORS["aniso"], label=r'Invariants ($\bar{I}_4$)')
        axes1[3].scatter(aniso_z[:, 0], jnp.zeros_like(aniso_z[:, 0]), c='red', marker='x', s=50, label='Inducing Points')
        axes1[3].set_xlabel(r"$\bar{I}_4$")
        axes1[3].set_title("Anisotropic Inducing Points")
        axes1[3].legend()
        axes1[3].grid(True, alpha=0.25)

    plt.tight_layout()
    save_figure(fig1, os.path.join(save_path, "inducing_points_features.pdf"))
    plt.close(fig1)

    # --- Setup Figure 2: Standard Load Paths ---
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

    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1 / (1 + gamma)).at[:, 2, 2].set(1)
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1 / (1 + gamma)).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1 / (1 + gamma)).at[:, 1, 1].set(1 / (1 + gamma)).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1).at[:, 0, 1].set(gamma)

    modes_data = {}
    for mode_name, F_stack in modes.items():
        C = jax.vmap(C_func)(F_stack)
        I3 = jnp.clip(jax.vmap(I3_func)(C), 1.0e-8, 1.0e8)
        C_bar = (I3**(-1/3))[..., None, None] * C
        js = jnp.sqrt(I3)
        i1_bar = jnp.trace(C_bar, axis1=-2, axis2=-1)
        i2_bar = 0.5 * (i1_bar**2 - jnp.trace(C_bar @ C_bar, axis1=-2, axis2=-1))

        m_dict = {
            "dev1": i1_bar - 3.0,
            "dev2": i2_bar - 3.0,
            "vol": js - 1.0
        }
        if feature_extractor is not None and hasattr(feature_extractor, "a0"):
            a0 = feature_extractor.a0
            m_dict["aniso1"] = jnp.einsum('i,...ij,j->...', a0, C_bar, a0) - 1.0
            if hasattr(feature_extractor, "a1") and feature_extractor.a1 is not None:
                a1 = feature_extractor.a1
                m_dict["aniso2"] = jnp.einsum('i,...ij,j->...', a1, C_bar, a1) - 1.0
        modes_data[mode_name] = m_dict

    data_dict = {
        "dev1": dev_I[:, 0] - 3.0,
        "dev2": dev_I[:, 1] - 3.0,
        "vol": vol_I[:, 0] - 1.0
    }
    z_dict = {
        "dev1": dev_z[:, 0] - 3.0,
        "dev2": dev_z[:, 1] - 3.0,
        "vol": vol_z[:, 0] - 1.0
    }

    if aniso_I is not None and aniso_z is not None:
        shift_data_a1 = 1.0 if float(jnp.mean(aniso_I[:, 0])) > 0.5 else 0.0
        shift_z_a1 = 1.0 if float(jnp.mean(aniso_z[:, 0])) > 0.5 else 0.0
        data_dict["aniso1"] = aniso_I[:, 0] - shift_data_a1
        z_dict["aniso1"] = aniso_z[:, 0] - shift_z_a1
        if aniso_I.shape[-1] >= 2 and aniso_z.shape[-1] >= 2:
            shift_data_a2 = 1.0 if float(jnp.mean(aniso_I[:, 1])) > 0.5 else 0.0
            shift_z_a2 = 1.0 if float(jnp.mean(aniso_z[:, 1])) > 0.5 else 0.0
            data_dict["aniso2"] = aniso_I[:, 1] - shift_data_a2
            z_dict["aniso2"] = aniso_z[:, 1] - shift_z_a2

    pairs = [
        (r"$\bar{I}_1-3$ vs $\bar{I}_2-3$", "dev1", "dev2", r"$\bar{I}_1-3$", r"$\bar{I}_2-3$"),
        (r"$\bar{I}_1-3$ vs $J-1$", "dev1", "vol", r"$\bar{I}_1-3$", r"$J-1$"),
        (r"$\bar{I}_2-3$ vs $J-1$", "dev2", "vol", r"$\bar{I}_2-3$", r"$J-1$"),
    ]

    if "aniso1" in data_dict and "aniso1" in modes_data[list(modes.keys())[0]]:
        pairs.extend([
            (r"$\bar{I}_4-1$ vs $\bar{I}_1-3$", "aniso1", "dev1", r"$\bar{I}_4-1$", r"$\bar{I}_1-3$"),
            (r"$\bar{I}_4-1$ vs $\bar{I}_2-3$", "aniso1", "dev2", r"$\bar{I}_4-1$", r"$\bar{I}_2-3$"),
            (r"$\bar{I}_4-1$ vs $J-1$", "aniso1", "vol", r"$\bar{I}_4-1$", r"$J-1$"),
        ])
    if "aniso2" in data_dict and "aniso2" in modes_data[list(modes.keys())[0]]:
        pairs.extend([
            (r"$\bar{I}_6-1$ vs $\bar{I}_1-3$", "aniso2", "dev1", r"$\bar{I}_6-1$", r"$\bar{I}_1-3$"),
            (r"$\bar{I}_6-1$ vs $\bar{I}_2-3$", "aniso2", "dev2", r"$\bar{I}_6-1$", r"$\bar{I}_2-3$"),
            (r"$\bar{I}_6-1$ vs $J-1$", "aniso2", "vol", r"$\bar{I}_6-1$", r"$J-1$"),
            (r"$\bar{I}_4-1$ vs $\bar{I}_6-1$", "aniso1", "aniso2", r"$\bar{I}_4-1$", r"$\bar{I}_6-1$"),
        ])

    n_pairs = len(pairs)
    if n_pairs == 10:
        nrows, ncols = 2, 5
        figsize = (25, 9)
    elif n_pairs == 6:
        nrows, ncols = 2, 3
        figsize = (18, 9)
    else:
        nrows, ncols = 1, 3
        figsize = (18, 5)

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes2_flat = axes2.flatten()

    for idx, (title, x_k, y_k, x_lbl, y_lbl) in enumerate(pairs):
        ax = axes2_flat[idx]
        ax.scatter(data_dict[x_k], data_dict[y_k], marker='o', alpha=0.3, color='#888888', label='Training Data' if idx == 0 else "")
        ax.scatter(z_dict[x_k], z_dict[y_k], color='red', marker='x', s=45, label='Inducing Points' if idx == 0 else "")

        for mode_name, m_vals in modes_data.items():
            ax.plot(
                m_vals[x_k], m_vals[y_k],
                label=mode_name if idx == 0 else "",
                color=MODE_COLORS.get(mode_name, '#333333'),
                linestyle=MODE_LINESTYLES.get(mode_name, '-')
            )

        ax.set_title(title, fontsize=12)
        ax.set_xlabel(x_lbl, fontsize=11)
        ax.set_ylabel(y_lbl, fontsize=11)
        ax.grid(True, alpha=0.25)
        if idx == 0:
            ax.legend(fontsize='small', loc='best')

    for idx in range(n_pairs, len(axes2_flat)):
        axes2_flat[idx].set_visible(False)

    plt.tight_layout()
    save_figure(fig2, os.path.join(save_path, "standard_loading_paths.pdf"))
    plt.close(fig2)


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
    """Plots clamped strain energy density and stress predictions against ground truth across 6 deformation modes."""
    apply_style()
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
    F_all = F_all.at[2].set(set_F(1 + gamma, 1 / (1 + gamma), 1.0))
    F_all = F_all.at[3].set(set_F(1 / (1 + gamma), 1.0, 1.0))
    F_all = F_all.at[4].set(set_F(1 / (1 + gamma), 1 / (1 + gamma), 1.0))
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))

    mode_names = MODE_NAMES

    psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None))
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

    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    fig.suptitle(f"Material Discovery Validation - Step {step}", fontsize=18, y=1.01)

    for i, name in enumerate(mode_names):
        if name == "Pure Shear":
            idx_comp = (1, 1)
        elif name == "Simple Shear":
            idx_comp = (0, 1)
        else:
            idx_comp = (0, 0)

        # Column 0: Energy
        ax_psi = axes[i, 0]
        ax_psi.plot(gamma, psi_true[i], **CURVE_STYLES["gt"])
        ax_psi.plot(gamma, psi_dets[i], color="red", lw=0.8, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_dist_mean[i], **CURVE_STYLES["gp"])
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96 * jnp.sqrt(psi_dist_var[i]),
                            psi_dist_mean[i] + 1.96 * jnp.sqrt(psi_dist_var[i]), **CURVE_STYLES["gp_ci"])

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

        ax_p.plot(gamma, p_true_comp, **CURVE_STYLES["gt"])
        ax_p.plot(gamma, p_det_comp, color="red", lw=0.8, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_samples_comp.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_mean_comp, **CURVE_STYLES["gp"])
        ax_p.fill_between(gamma, p_mean_comp - 1.96 * p_std_comp,
                          p_mean_comp + 1.96 * p_std_comp, **CURVE_STYLES["gp_ci"])

        y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())

        ax_psi.set_title(f"{name}: Energy")
        ax_p.set_title(f"{name}: Stress")
        trans_g = trans_tot[i]
        max_g = float(gamma.max())
        for ax in [ax_psi, ax_p]:
            ax.axvspan(0, min(trans_g, max_g), color='green', alpha=0.10, zorder=1, label="Interpolation" if (i == 0 and ax == ax_psi) else "")
            if trans_g < max_g:
                ax.axvspan(trans_g, max_g, color='red', alpha=0.10, zorder=1, label="Extrapolation" if (i == 0 and ax == ax_psi) else "")
                ax.axvline(x=trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8, zorder=4)
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_file = os.path.join(save_path, f"clamped_validation_{step}.pdf")
    save_figure(fig, save_file)
    plt.close(fig)
    print(f"Clamped limits plot saved to: {save_file}")


def plot_stress_validation(gp_model, true_model, save_path):
    """Plots Piola stress components across all 6 deformation modes."""
    apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Piola Stress Validation: Model Discovery vs. True Physics", fontsize=15)
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

    modes["Uniaxial Tension"] = modes["Uniaxial Tension"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Equibiaxial Tension"] = modes["Equibiaxial Tension"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1 + gamma).at[:, 2, 2].set(1)
    modes["Pure Shear"] = modes["Pure Shear"].at[:, 0, 0].set(1 + gamma).at[:, 1, 1].set(1 / (1 + gamma)).at[:, 2, 2].set(1)
    modes["Uniaxial Compression"] = modes["Uniaxial Compression"].at[:, 0, 0].set(1 / (1 + gamma)).at[:, 1, 1].set(1).at[:, 2, 2].set(1)
    modes["Equibiaxial Compression"] = modes["Equibiaxial Compression"].at[:, 0, 0].set(1 / (1 + gamma)).at[:, 1, 1].set(1 / (1 + gamma)).at[:, 2, 2].set(1)
    modes["Simple Shear"] = modes["Simple Shear"].at[:, 0, 0].set(1).at[:, 1, 1].set(1).at[:, 2, 2].set(1).at[:, 0, 1].set(gamma)

    for i, (mode_name, F_stack) in enumerate(modes.items()):
        P_mean = jax.vmap(lambda f: gp_model.piola_dist(f).mean)(F_stack)
        P_var = jax.vmap(lambda f: gp_model.piola_dist(f).var)(F_stack)
        P_std = jnp.sqrt(P_var)
        P_lower_bound = P_mean - 1.96 * P_std
        P_upper_bound = P_mean + 1.96 * P_std

        P_true = jax.vmap(true_model.P)(F_stack)
        if mode_name == "Pure Shear":
            y_pred = P_mean[:, 1, 1]
            y_true = P_true[:, 1, 1]
            lower = P_lower_bound[:, 1, 1]
            upper = P_upper_bound[:, 1, 1]
            label = r"$P_{22}$"
        elif mode_name == "Simple Shear":
            y_pred = P_mean[:, 0, 1]
            y_true = P_true[:, 0, 1]
            lower = P_lower_bound[:, 0, 1]
            upper = P_upper_bound[:, 0, 1]
            label = r"$P_{12}$"
        else:
            y_pred = P_mean[:, 0, 0]
            y_true = P_true[:, 0, 0]
            lower = P_lower_bound[:, 0, 0]
            upper = P_upper_bound[:, 0, 0]
            label = r"$P_{11}$"

        gamma_pts = jnp.linspace(0, 1, len(F_stack))
        axes[i].plot(gamma_pts, y_pred, color='#1f77b4', lw=2.0, label='GP Mean')
        axes[i].fill_between(gamma_pts, lower, upper, color='#1f77b4', alpha=0.2, label='95% CI')
        axes[i].plot(gamma_pts, y_true, 'k--', lw=1.8, label='True')

        axes[i].set_title(mode_name)
        axes[i].set_ylabel(label)
        axes[i].set_xlabel(r"$\gamma$")
        axes[i].grid(True, alpha=0.25)
        axes[i].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_figure(fig, os.path.join(save_path, "piola_stress_validation.pdf"))
    plt.close(fig)


def plot_energy_decomposition_validation(learned_gp, true_model, save_path):
    """Plots energy decomposition (Deviatoric, Volumetric, Anisotropic, Total) across 6 deformation modes."""
    apply_style()
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
    F_all = F_all.at[2].set(set_F(1 + gamma, 1 / (1 + gamma), 1.0))
    F_all = F_all.at[3].set(set_F(1 / (1 + gamma), 1.0, 1.0))
    F_all = F_all.at[4].set(set_F(1 / (1 + gamma), 1 / (1 + gamma), 1.0))
    F_all = F_all.at[5].set(set_F(1.0, 1.0, 1.0, f12=gamma))

    mode_names = MODE_NAMES

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

    psi_samples_dev, psi_samples_vol, psi_samples_aniso_list, psi_samples_tot = [], [], [], []
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
    fig.suptitle("Energy Decomposition Validation", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        configs = [
            (0, "Deviatoric", psi_true_dev[i], psi_mean_dev[i], psi_std_dev[i], psi_samples_dev[:, i, :], trans_dev[i]),
            (1, "Volumetric", psi_true_vol[i], psi_mean_vol[i], psi_std_vol[i], psi_samples_vol[:, i, :], trans_vol[i]),
            (2, "Anisotropic", psi_true_aniso[i], psi_mean_aniso[i], psi_std_aniso[i], psi_samples_aniso[:, i, :], trans_tot[i]),
            (3, "Total Energy", psi_true_tot[i], psi_mean_tot[i], psi_std_tot[i], psi_samples_tot[:, i, :], trans_tot[i])
        ]
        for col, col_name, true_val, mean_val, std_val, samples, trans_g in configs:
            ax = axes[i, col]
            ax.plot(gamma, true_val, 'k--', lw=1.8, label="True", zorder=5)
            ax.plot(gamma, samples.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
            ax.plot(gamma, mean_val, color="#1f77b4", lw=2, label="GP Mean", zorder=3)
            ax.fill_between(gamma, mean_val - 1.96 * std_val, mean_val + 1.96 * std_val, color="#1f77b4", alpha=0.2, zorder=2)

            max_g = float(gamma.max())
            ax.axvspan(0, min(trans_g, max_g), color='green', alpha=0.10, zorder=1, label="Interpolation" if (i == 0 and col == 3) else "")
            if trans_g < max_g:
                ax.axvspan(trans_g, max_g, color='red', alpha=0.10, zorder=1, label="Extrapolation" if (i == 0 and col == 3) else "")
                ax.axvline(x=trans_g, color='darkred', linestyle=':', lw=1.5, alpha=0.8, zorder=4)

            rmse, coverage = calc_metrics(true_val, mean_val, std_val)

            ax.set_title(f"{name}: {col_name}\nRMSE: {rmse:.4f} | Cov: {coverage:.1f}%")
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.25)

            y_min, y_max = jnp.min(true_val), jnp.max(true_val)
            pad = (y_max - y_min) * 0.1
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlim(0, max_g)
            if i == 0 and col == 3:
                ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "energy_decomposition.pdf"))
    plt.close(fig)


class ExtractionR2Metrics(tuple):
    """
    3-tuple (r2, rmse, coverage) ensuring backwards compatibility for callers
    expecting a 3-tuple return value, while exposing .train_metrics, .val_metrics,
    and .by_component for detailed evaluation.
    """
    def __new__(cls, r2, rmse, coverage, train_metrics=None, val_metrics=None, by_component=None):
        return super().__new__(cls, (r2, rmse, coverage))

    def __init__(self, r2, rmse, coverage, train_metrics=None, val_metrics=None, by_component=None):
        self.r2 = r2
        self.rmse = rmse
        self.coverage = coverage
        self.train_metrics = train_metrics or {}
        self.val_metrics = val_metrics or {}
        self.by_component = by_component or {}


def _format_step_indices(steps):
    """Utility to format a collection of load step indices into compact strings."""
    if not steps:
        return "None"
    steps = sorted(list(steps))
    if len(steps) > 2 and steps == list(range(steps[0], steps[-1] + 1)):
        return f"{steps[0]}-{steps[-1]}"
    elif len(steps) <= 5:
        return ", ".join(map(str, steps))
    else:
        return f"{steps[0]}...{steps[-1]}"


def plot_training_r2(learned_gp, true_model, F_train_full, save_path, train_steps=None, val_steps=None):
    """Plots parity and computes R2/RMSE/Coverage metrics across training and validation steps."""
    apply_style()
    print("Generating Training & Validation Data R2 Plot...")
    num_steps = F_train_full.shape[0]

    # Resolve train_steps and val_steps from config files if omitted
    if (val_steps is None or train_steps is None) and save_path:
        for cfg_file in ["recipe_config.yaml", "config.yaml", "config.json"]:
            cfg_path = os.path.join(save_path, cfg_file)
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        if cfg_file.endswith(".json"):
                            cfg = json.load(f)
                        else:
                            cfg = yaml.safe_load(f)
                        if cfg:
                            if val_steps is None and "val_load_steps_indices" in cfg:
                                val_steps = cfg["val_load_steps_indices"]
                            if train_steps is None and "train_load_steps_indices" in cfg:
                                train_steps = cfg["train_load_steps_indices"]
                        if val_steps is not None and train_steps is not None:
                            break
                except Exception:
                    pass

    if val_steps is not None:
        if isinstance(val_steps, (int, float)):
            val_steps = [int(val_steps)]
        else:
            val_steps = [int(s) for s in val_steps if int(s) < num_steps]
    else:
        val_steps = []

    if train_steps is not None and len(train_steps) > 0:
        train_steps_list = [int(s) for s in train_steps if int(s) < num_steps]
        start_step = min(train_steps_list)
        if len(val_steps) == 0 and max(train_steps_list) < num_steps - 1:
            val_steps = list(range(max(train_steps_list) + 1, num_steps))
    else:
        start_step = 0

    val_set = set(val_steps)
    val_steps_clean = [s for s in range(num_steps) if s in val_set]
    train_steps_clean = [s for s in range(start_step, num_steps) if s not in val_set]
    if len(train_steps_clean) == 0:
        train_steps_clean = [s for s in range(num_steps) if s not in val_set]

    train_steps_str = _format_step_indices(train_steps_clean)
    val_steps_str = _format_step_indices(val_steps_clean)

    has_aniso = (hasattr(learned_gp, 'is_anisotropic') and learned_gp.is_anisotropic) and hasattr(true_model, 'psi_aniso')

    if hasattr(true_model, 'psi_dev') and hasattr(true_model, 'psi_vol'):
        psi_dev_fn = lambda f: jax.vmap(true_model.psi_dev)(f)
        psi_vol_fn = lambda f: jax.vmap(true_model.psi_vol)(f)
    elif hasattr(true_model, 'dev_params') and hasattr(true_model, 'vol_params'):
        from core.material_models import get_material
        model_name = getattr(true_model, 'name', 'gmr')
        t_dev = get_material(model_name, dev_params=list(true_model.dev_params), vol_params=[0] * len(true_model.vol_params), jit_P=False)
        t_vol = get_material(model_name, dev_params=[0] * len(true_model.dev_params), vol_params=list(true_model.vol_params), jit_P=False)
        psi_dev_fn = lambda f: jax.vmap(t_dev.psi)(f)
        psi_vol_fn = lambda f: jax.vmap(t_vol.psi)(f)
    else:
        psi_dev_fn = None
        psi_vol_fn = None

    components = []
    if psi_dev_fn is not None and psi_vol_fn is not None:
        components.append(("Deviatoric", r"Deviatoric Energy ($\Psi_{\mathrm{dev}}$)", psi_dev_fn, lambda f: learned_gp.dev_psi_dist(f)))
        components.append(("Volumetric", r"Volumetric Energy ($\Psi_{\mathrm{vol}}$)", psi_vol_fn, lambda f: learned_gp.vol_psi_dist(f)))
    if has_aniso:
        components.append(("Anisotropic", r"Anisotropic Energy ($\Psi_{\mathrm{aniso}}$)", lambda f: jax.vmap(true_model.psi_aniso)(f), lambda f: learned_gp.aniso_psi_dist(f)))
    components.append(("Total Energy", r"Total Energy ($\Psi_{\mathrm{total}}$)", lambda f: jax.vmap(true_model.psi)(f), lambda f: learned_gp.psi_dist(f)))

    n_panels = len(components)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 6), squeeze=False)
    axes = axes[0]

    comp_metrics = {}
    ret_train = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
    ret_val = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
    ret_overall = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}

    def compute_metrics(y_t_list, y_m_list, y_s_list):
        if len(y_t_list) == 0:
            return np.nan, np.nan, np.nan
        y_t = jnp.concatenate(y_t_list)
        y_m = jnp.concatenate(y_m_list)
        y_s = jnp.concatenate(y_s_list)

        ss_tot = jnp.sum((y_t - jnp.mean(y_t)) ** 2)
        ss_res = jnp.sum((y_t - y_m) ** 2)
        r2_val_c = 1.0 - ss_res / (ss_tot + 1e-12)
        rmse_val_c = jnp.sqrt(jnp.mean((y_t - y_m) ** 2))
        lower = y_m - 1.96 * y_s
        upper = y_m + 1.96 * y_s
        cov_val_c = jnp.mean((y_t >= lower) & (y_t <= upper)) * 100.0
        return float(r2_val_c), float(rmse_val_c), float(cov_val_c)

    for ax_idx, (comp_name, comp_label, true_fn, gp_dist_fn) in enumerate(components):
        ax = axes[ax_idx]
        ax.set_title(f"Energy Parity: {comp_name}", fontsize=14)

        train_true, train_mean, train_std = [], [], []
        val_true, val_mean, val_std = [], [], []

        has_train_lbl = False
        has_val_lbl = False

        for step in range(num_steps):
            if step not in train_steps_clean and step not in val_steps_clean:
                continue

            F_step = F_train_full[step]
            true_psi = true_fn(F_step)
            dist = gp_dist_fn(F_step)
            mean_psi = dist.mean
            std_psi = jnp.sqrt(dist.var)

            if step in val_set:
                val_true.append(true_psi)
                val_mean.append(mean_psi)
                val_std.append(std_psi)
                lbl = f"Val Steps ({val_steps_str})" if not has_val_lbl else ""
                has_val_lbl = True
                ax.errorbar(true_psi, mean_psi, yerr=1.96 * std_psi, fmt='o',
                            color='#d62728', ecolor='#e45756', alpha=0.35,
                            markersize=3, elinewidth=0.8, label=lbl)
            else:
                train_true.append(true_psi)
                train_mean.append(mean_psi)
                train_std.append(std_psi)
                lbl = f"Train Steps ({train_steps_str})" if not has_train_lbl else ""
                has_train_lbl = True
                ax.errorbar(true_psi, mean_psi, yerr=1.96 * std_psi, fmt='o',
                            color='gray', ecolor='#b0b0b0', alpha=0.25,
                            markersize=3, elinewidth=0.8, label=lbl)

        r2_tr, rmse_tr, cov_tr = compute_metrics(train_true, train_mean, train_std)
        r2_v, rmse_v, cov_v = compute_metrics(val_true, val_mean, val_std)
        r2_tot, rmse_tot, cov_tot = compute_metrics(train_true + val_true, train_mean + val_mean, train_std + val_std)

        all_pts_true = train_true + val_true
        all_pts_mean = train_mean + val_mean
        if len(all_pts_true) > 0:
            cat_true = jnp.concatenate(all_pts_true)
            cat_mean = jnp.concatenate(all_pts_mean)
            min_val = min(float(cat_true.min()), float(cat_mean.min()))
            max_val = max(float(cat_true.max()), float(cat_mean.max()))
            margin = max((max_val - min_val) * 0.05, 1e-4)
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', lw=1.5, label="Parity")

        box_lines = []
        if len(train_true) > 0:
            box_lines.append(f"Train (Steps {train_steps_str}):")
            box_lines.append(f"  $R^2$: {r2_tr:.4f}")
            box_lines.append(f"  RMSE: {rmse_tr:.4f}")
            box_lines.append(f"  EC: {cov_tr:.1f}%")

        if len(val_true) > 0:
            if len(box_lines) > 0:
                box_lines.append("")
            box_lines.append(f"Val (Steps {val_steps_str}):")
            box_lines.append(f"  $R^2$: {r2_v:.4f}")
            box_lines.append(f"  RMSE: {rmse_v:.4f}")
            box_lines.append(f"  EC: {cov_v:.1f}%")

        box_text = "\n".join(box_lines)
        ax.text(0.05, 0.95, box_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'), fontsize=10.5)

        ax.set_xlabel(f"True {comp_label}", fontsize=11)
        ax.set_ylabel(f"Predicted GP Mean {comp_label}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.85)

        comp_metrics[comp_name] = {
            "train": {"r2": r2_tr, "rmse": rmse_tr, "ec": cov_tr},
            "val": {"r2": r2_v, "rmse": rmse_v, "ec": cov_v},
            "overall": {"r2": r2_tot, "rmse": rmse_tot, "ec": cov_tot}
        }
        if comp_name == "Total Energy":
            ret_train = {"r2": r2_tr, "rmse": rmse_tr, "ec": cov_tr, "steps": train_steps_clean}
            ret_val = {"r2": r2_v, "rmse": rmse_v, "ec": cov_v, "steps": val_steps_clean}
            ret_overall = {"r2": r2_tot, "rmse": rmse_tot, "ec": cov_tot, "steps": list(range(num_steps))}

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "training_r2_energy.pdf"))
    save_figure(fig, os.path.join(save_path, "training_r2_energy.png"))
    plt.close(fig)

    primary_r2 = ret_train["r2"] if not np.isnan(ret_train["r2"]) else ret_overall["r2"]
    primary_rmse = ret_train["rmse"] if not np.isnan(ret_train["rmse"]) else ret_overall["rmse"]
    primary_cov = ret_train["ec"] if not np.isnan(ret_train["ec"]) else ret_overall["ec"]

    return ExtractionR2Metrics(
        primary_r2, primary_rmse, primary_cov,
        train_metrics=ret_train, val_metrics=ret_val, by_component=comp_metrics
    )


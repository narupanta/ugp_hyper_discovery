import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import stats

def plot_parameter_posteriors(param_metrics, save_path):
    """
    Level 1: Plot parameter posteriors & true ground truth values.
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    n_params = param_metrics["n_params"]
    indices = np.arange(1, n_params + 1)
    means = np.array(param_metrics["mean_params"])
    stds = np.array(param_metrics["std_params"])
    gts = np.array(param_metrics["gt_params"])

    ax.errorbar(indices, means, yerr=1.96 * stds, fmt='o', color='#1f77b4',
                linewidth=2, capsize=5, label='Predicted Mean (95% CI)')
    ax.scatter(indices, gts, color='red', marker='x', s=80, zorder=5, label='Ground Truth (\\theta_{true})')

    for i in range(n_params):
        ax.text(indices[i], means[i] + 1.96 * stds[i] + 0.05, f"RelErr: {param_metrics['rel_error']:.1%}",
                ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('Material Parameter Index', fontweight='bold', fontsize=12)
    ax.set_ylabel('Parameter Value', fontweight='bold', fontsize=12)
    ax.set_title(f'Level 1: Material Parameter Validation (D_M^2 = {param_metrics["dm2_theta"]:.2f}, p = {param_metrics["p_value"]:.3f})',
                 fontweight='bold', fontsize=13)
    ax.set_xticks(indices)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "level1_parameter_validation.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_energy_calibration(energy_metrics, save_path):
    """
    Level 2: Plot strain energy calibration curve & ECE.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    nominal = np.array(energy_metrics["nominal_levels"])
    picp = np.array(energy_metrics["picp_values"])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label=r'Ideal Calibration ($y=x$)')
    ax.plot(nominal, picp, 'o-', color='#ff7f0e', linewidth=2.5, markersize=6,
            label=f'Energy W(F) (ECE = {energy_metrics["ece_W"]:.4f})')
    ax.fill_between(nominal, nominal, picp, color='#ff7f0e', alpha=0.15, label='Calibration Gap')

    info_text = (f"Energy RMSE: {energy_metrics['rmse_W']:.4e}\n"
                 f"Rel-RMSE: {energy_metrics['rel_rmse_W']:.2%}\n"
                 f"PICP (95% CI): {energy_metrics['picp_W_95']*100:.1f}%\n"
                 f"MPIW (95% CI): {energy_metrics['mpiw_W_95']:.4e}")
    ax.text(0.05, 0.60, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.set_xlabel(r'Nominal Confidence Level ($1 - \alpha$)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Empirical Coverage (PICP)', fontweight='bold', fontsize=12)
    ax.set_title(r'Level 2: Strain Energy $W(\mathbf{F})$ Calibration', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "level2_energy_calibration.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_subspace_chi2_distribution(subspace_metrics, save_path):
    """
    Level 3: Plot Subspace Chi-Square distribution with observed D_M_sub^2 marker.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.8))

    k = subspace_metrics["k_components"]
    dm2_sub = subspace_metrics["dm2_sub"]
    chi2_low = subspace_metrics["chi2_lower_95"]
    chi2_up = subspace_metrics["chi2_upper_95"]
    p_val = subspace_metrics["p_value"]

    x_max = max(chi2_up * 1.3, dm2_sub * 1.3, k * 2.5)
    x = np.linspace(0.001, x_max, 300)
    pdf = stats.chi2.pdf(x, df=k)

    ax.plot(x, pdf, 'k-', linewidth=2.5, label=f'Theoretical $\\chi^2(K={k})$ PDF')
    ax.axvspan(chi2_low, chi2_up, color='green', alpha=0.15, label=r'95% Acceptance Interval $[\chi^2_{0.025}, \chi^2_{0.975}]$')
    ax.axvline(chi2_low, color='green', linestyle='--', linewidth=1.5)
    ax.axvline(chi2_up, color='green', linestyle='--', linewidth=1.5)

    ax.axvline(dm2_sub, color='red', linestyle='-', linewidth=2.5, label=f'Observed $D_{{M, \\text{{sub}}}}^2 = {dm2_sub:.3f}$')

    pass_status = "PASSED (Well-Calibrated)" if (chi2_low <= dm2_sub <= chi2_up) else ("UNDERCONFIDENT" if dm2_sub < chi2_low else "OVERCONFIDENT")
    info_text = (f"Subspace Modes (K): {k} (99% var)\n"
                 f"Observed $D_{{M, \\text{{sub}}}}^2$: {dm2_sub:.3f}\n"
                 f"95% Bounds: [{chi2_low:.2f}, {chi2_up:.2f}]\n"
                 f"Multivariate $p$-value: {p_val:.3f}\n"
                 f"Status: {pass_status}")
    ax.text(0.42, 0.88, info_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel(r'Subspace Squared Mahalanobis Distance $D_{M, \text{sub}}^2$', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title(r'Level 3: Thin-SVD Subspace $\chi^2(K)$ Calibration Test', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "level3_subspace_chi2_calibration.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_reliability_diagrams(calib_metrics, save_path):
    """
    Level 3: Plot Full-Field Kinematics Reliability Diagram (Coverage vs Nominal).
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    nominal = np.array(calib_metrics["nominal_levels"])
    picp = np.array(calib_metrics["picp_values"])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label=r'Ideal Calibration ($y=x$)')
    ax.plot(nominal, picp, 'o-', color='#1f77b4', linewidth=2.5, markersize=6,
            label=f'Full-Field Kinematics (ECE = {calib_metrics["ece"]:.4f})')
    ax.fill_between(nominal, nominal, picp, color='#1f77b4', alpha=0.15, label='Calibration Gap')

    info_text = (f"ECE: {calib_metrics['ece']:.4f}\n"
                 f"PICP (95% CI): {calib_metrics['picp_95']*100:.1f}%\n"
                 f"MPIW (95% CI): {calib_metrics['mpiw_95']:.4f}")
    ax.text(0.05, 0.60, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.set_xlabel(r'Nominal Confidence Level ($1 - \alpha$)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Empirical Coverage Probability (PICP)', fontweight='bold', fontsize=12)
    ax.set_title('Level 3: Full-Field Kinematic Reliability Diagram', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "level3_reliability_diagram.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_pit_histogram(calib_metrics, save_path):
    """
    Level 3: Plot Probability Integral Transform (PIT) Uniformity Histogram.
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    pit_values = np.array(calib_metrics["pit_values"])

    ax.hist(pit_values, bins=20, density=True, color='#2ca02c', alpha=0.6, edgecolor='darkgreen', label='Empirical PIT')
    ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Ideal Uniform $U[0, 1]$')

    ax.set_xlabel('Probability Integral Transform (PIT) Value', fontweight='bold', fontsize=12)
    ax.set_ylabel('Density', fontweight='bold', fontsize=12)
    ax.set_title('Level 3: PIT Histogram Uniformity Test', fontweight='bold', fontsize=14)
    ax.set_xlim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "level3_pit_histogram.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_calibration_maps(node_coords, cells, u_obs, u_pred_samples, save_path, step_idx=-1):
    """
    Level 3: Plot 2x2 Spatial Heatmaps of Observed, Mean, Absolute Error, and 95% MPIW.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    u_obs_step = u_obs[step_idx] # (n_nodes, 2)
    u_pred_step = u_pred_samples[:, step_idx, :, :] # (n_samples, n_nodes, 2)

    u_mean_step = np.mean(u_pred_step, axis=0)
    q_low = np.quantile(u_pred_step, 0.025, axis=0)
    q_high = np.quantile(u_pred_step, 0.975, axis=0)

    u_obs_norm = np.linalg.norm(u_obs_step, axis=1)
    u_mean_norm = np.linalg.norm(u_mean_step, axis=1)
    err_norm = np.linalg.norm(u_obs_step - u_mean_step, axis=1)
    mpiw_norm = np.linalg.norm(q_high - q_low, axis=1)

    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)

    def add_map(ax, val, title, cmap='viridis'):
        im = ax.tricontourf(triang, val, levels=30, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    add_map(axes[0, 0], u_obs_norm, r"Observed Kinematics $\|\mathbf{u}_{obs}\|$", 'viridis')
    add_map(axes[0, 1], u_mean_norm, r"Predicted Mean $\|\boldsymbol{\mu}_{u}\|$", 'viridis')
    add_map(axes[1, 0], err_norm, r"Absolute Error $\|\mathbf{u}_{obs} - \boldsymbol{\mu}_{u}\|$", 'magma')
    add_map(axes[1, 1], mpiw_norm, r"95% Credible Interval Width $\text{MPIW}_{95}$", 'plasma')

    for ax in axes.flat:
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')

    plt.suptitle(f'Level 3: Spatial Kinematic Calibration (Step {step_idx if step_idx >= 0 else len(u_obs) + step_idx})',
                 fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_path, "level3_spatial_calibration_map.pdf"), dpi=300, bbox_inches='tight')
    plt.close()

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.tri as tri
from scipy import stats

def compute_calibration_metrics(u_exp, u_pred_samples, alpha_levels=None):
    """
    Compute quantitative UQ calibration metrics for vector displacement fields.

    Parameters
    ----------
    u_exp : np.ndarray
        Observed experimental displacement field, shape (n_steps, n_nodes, dim)
    u_pred_samples : np.ndarray
        Predicted forward FEM displacement realizations, shape (n_samples, n_steps, n_nodes, dim)
    alpha_levels : list or np.ndarray, optional
        Significance levels for prediction interval quantiles.

    Returns
    -------
    dict
        Dictionary containing ECE, PICP values, MPIW, RMSE, MAE, R2, and z-score statistics.
    """
    if alpha_levels is None:
        alpha_levels = np.linspace(0.05, 0.95, 19)

    n_samples, n_steps, n_nodes, dim = u_pred_samples.shape
    u_exp_flat = u_exp.flatten()
    u_pred_flat = u_pred_samples.reshape(n_samples, -1)

    # 1. Empirical Coverage (PICP) & ECE
    nominal_confs = 1.0 - alpha_levels
    picp_values = []
    mpiw_values = []

    for alpha in alpha_levels:
        lower_q = np.quantile(u_pred_flat, alpha / 2.0, axis=0)
        upper_q = np.quantile(u_pred_flat, 1.0 - alpha / 2.0, axis=0)
        
        # Check if u_exp falls within [lower_q, upper_q]
        inside = (u_exp_flat >= lower_q) & (u_exp_flat <= upper_q)
        picp = np.mean(inside)
        mpiw = np.mean(upper_q - lower_q)
        
        picp_values.append(float(picp))
        mpiw_values.append(float(mpiw))

    picp_values = np.array(picp_values)
    mpiw_values = np.array(mpiw_values)
    nominal_confs = np.array(nominal_confs)

    # ECE: Expected Calibration Error
    ece = float(np.mean(np.abs(picp_values - nominal_confs)))

    # 2. Mean and Std predictions
    u_mean = np.mean(u_pred_samples, axis=0)
    u_std = np.std(u_pred_samples, axis=0)

    u_mean_flat = u_mean.flatten()
    u_std_flat = u_std.flatten()

    # Accuracy Metrics
    err_flat = u_exp_flat - u_mean_flat
    mae = float(np.mean(np.abs(err_flat)))
    rmse = float(np.sqrt(np.mean(err_flat**2)))
    ss_res = np.sum(err_flat**2)
    ss_tot = np.sum((u_exp_flat - np.mean(u_exp_flat))**2)
    r2 = float(1.0 - (ss_res / (ss_tot + 1e-12)))

    # Standardized Residuals (Z-Score) on free deforming nodes (u_std > 1e-6)
    free_mask = u_std_flat > 1e-6
    if np.sum(free_mask) > 0:
        z_scores = err_flat[free_mask] / u_std_flat[free_mask]
    else:
        z_scores = err_flat / (u_std_flat + 1e-8)
    z_mean = float(np.mean(z_scores))
    z_std = float(np.std(z_scores))

    # 90% Confidence Interval Metrics
    idx_90 = np.argmin(np.abs(nominal_confs - 0.90))
    picp_90 = float(picp_values[idx_90])
    mpiw_90 = float(mpiw_values[idx_90])

    # 95% Confidence Interval Metrics
    idx_95 = np.argmin(np.abs(nominal_confs - 0.95))
    picp_95 = float(picp_values[idx_95])
    mpiw_95 = float(mpiw_values[idx_95])

    return {
        "n_samples": int(n_samples),
        "ece": ece,
        "picp_90": picp_90,
        "mpiw_90": mpiw_90,
        "picp_95": picp_95,
        "mpiw_95": mpiw_95,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "z_score_mean": z_mean,
        "z_score_std": z_std,
        "nominal_confs": nominal_confs.tolist(),
        "picp_values": picp_values.tolist(),
        "mpiw_values": mpiw_values.tolist()
    }


def plot_reliability_diagram(metrics, save_path):
    """Plot Nominal Confidence Level vs Empirical Coverage (PICP) curve."""
    fig, ax = plt.subplots(figsize=(6, 5.5))

    nominal_confs = np.array(metrics["nominal_confs"])
    picp_values = np.array(metrics["picp_values"])

    ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label=r'Ideal Calibration ($y=x$)')
    ax.plot(nominal_confs, picp_values, 'o-', color='#1f77b4', linewidth=2.5, 
            markersize=6, label=f'Model Predictions (ECE = {metrics["ece"]:.4f})')
    
    ax.fill_between(nominal_confs, nominal_confs, picp_values, color='#1f77b4', alpha=0.15, label='Calibration Gap')

    ax.set_xlabel(r'Nominal Confidence Level ($1 - \alpha$)', fontweight='bold', fontsize=12)
    ax.set_ylabel('Empirical Coverage Probability (PICP)', fontweight='bold', fontsize=12)
    ax.set_title('UQ Model Displacement Reliability Diagram', fontweight='bold', fontsize=14)

    info_text = (f"Samples: {metrics['n_samples']}\n"
                 f"ECE: {metrics['ece']:.4f}\n"
                 f"PICP (95% CI): {metrics['picp_95']*100:.2f}%\n"
                 f"MPIW (95% CI): {metrics['mpiw_95']:.4f}\n"
                 f"PICP (90% CI): {metrics['picp_90']*100:.2f}%\n"
                 f"MPIW (90% CI): {metrics['mpiw_90']:.4f}\n"
                 f"$R^2$: {metrics['r2']:.4f}")
    ax.text(0.05, 0.60, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray'))

    ax.set_xlim([0, 1.0])
    ax.set_ylim([0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "reliability_diagram.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_error_vs_uncertainty(u_exp, u_pred_samples, save_path, n_bins=15):
    """Plot binned predicted standard deviation vs empirical absolute error."""
    fig, ax = plt.subplots(figsize=(6, 5))

    u_mean = np.mean(u_pred_samples, axis=0)
    u_std = np.std(u_pred_samples, axis=0)

    abs_err = np.abs(u_exp - u_mean).flatten()
    std_vals = u_std.flatten()

    # Binning
    bin_edges = np.linspace(np.min(std_vals), np.max(std_vals), n_bins + 1)
    bin_centers = []
    bin_mean_err = []
    bin_std_err = []

    for j in range(n_bins):
        mask = (std_vals >= bin_edges[j]) & (std_vals < bin_edges[j+1])
        if np.sum(mask) > 5:
            bin_centers.append(np.mean(std_vals[mask]))
            bin_mean_err.append(np.mean(abs_err[mask]))
            bin_std_err.append(np.std(abs_err[mask]) / np.sqrt(np.sum(mask)))

    bin_centers = np.array(bin_centers)
    bin_mean_err = np.array(bin_mean_err)
    bin_std_err = np.array(bin_std_err)

    # Ideal Gaussian relationship: E[|err|] = std * sqrt(2/pi) approx 0.7979 * std
    x_ref = np.linspace(0, np.max(std_vals), 100)
    y_ref = x_ref * np.sqrt(2.0 / np.pi)

    ax.plot(x_ref, y_ref, 'k--', linewidth=2, label=r'Ideal Gaussian ($E[|e|] = \sqrt{2/\pi} \sigma$)')
    ax.errorbar(bin_centers, bin_mean_err, yerr=bin_std_err, fmt='o-', color='#d62728', 
                linewidth=2, capsize=4, label='Binned Nodal Predictions')

    corr, _ = stats.spearmanr(std_vals, abs_err)
    ax.text(0.05, 0.88, f"Spearman Correlation: {corr:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85))

    ax.set_xlabel(r'Predicted Standard Deviation ($\sigma_u$)', fontweight='bold', fontsize=12)
    ax.set_ylabel(r'Empirical Absolute Error ($|u_{exp} - \mu_u|$)', fontweight='bold', fontsize=12)
    ax.set_title('Error vs. Uncertainty Calibration', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "error_vs_uncertainty.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_z_score_distribution(u_exp, u_pred_samples, metrics, save_path):
    """Plot standardized residual (Z-Score) distribution vs Standard Normal PDF."""
    fig, ax = plt.subplots(figsize=(6, 5))

    u_mean = np.mean(u_pred_samples, axis=0)
    u_std = np.std(u_pred_samples, axis=0)

    err = (u_exp - u_mean).flatten()
    std = u_std.flatten()
    
    # Filter free nodes with u_std > 1e-6
    free_mask = std > 1e-6
    if np.sum(free_mask) > 0:
        z_scores = err[free_mask] / std[free_mask]
    else:
        z_scores = err / (std + 1e-8)

    # Filter extreme outliers for clean histogram display
    z_clipped = z_scores[np.abs(z_scores) < 5.0]

    ax.hist(z_clipped, bins=40, density=True, alpha=0.5, color='#2ca02c', edgecolor='darkgreen', label=r'Empirical $Z$-Scores')

    x = np.linspace(-4, 4, 200)
    ax.plot(x, stats.norm.pdf(x, 0, 1), 'r--', linewidth=2.5, label=r'Standard Normal $\mathcal{N}(0, 1)$')

    z_m = metrics["z_score_mean"]
    z_s = metrics["z_score_std"]

    ax.text(0.05, 0.85, f"Mean ($z$): {z_m:.3f}\nStd ($z$): {z_s:.3f}", transform=ax.transAxes,
            fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85))

    ax.set_xlabel(r'Standardized Residual $z = (u_{exp} - \mu_u) / \sigma_u$', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title('Standardized Residual Calibration ($Z$-Score)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "z_score_distribution.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def compute_mahalanobis_distance_metrics(u_exp, u_pred_samples, var_threshold=0.999):
    """
    Compute Multivariate Mahalanobis Distance for high-dimensional displacement predictions via SVD/PCA projection.
    """
    N, S, V, C = u_pred_samples.shape
    X = u_pred_samples.reshape(N, -1)
    y = u_exp.reshape(1, -1)

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    y_centered = y - mu

    # SVD projection for stable covariance inversion in high-D
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = (s**2) / np.sum(s**2)
    cum_var = np.cumsum(var_explained)
    k_components = int(np.searchsorted(cum_var, var_threshold) + 1)
    k_components = max(1, min(k_components, N - 1))

    V_k = Vt[:k_components].T
    s_k = s[:k_components] / np.sqrt(N - 1) # Singular values to std devs

    # Project centered data onto top K PCs
    X_proj = X_centered @ V_k # (N, K)
    y_proj = y_centered @ V_k # (1, K)

    # Squared Mahalanobis distances
    DM2_samples = np.sum((X_proj / s_k)**2, axis=1) # (N,)
    DM2_exp = float(np.sum((y_proj / s_k)**2, axis=1)[0])

    # Chi-Square reference distribution properties
    chi2_95_threshold = float(stats.chi2.ppf(0.95, df=k_components))
    p_value = float(1.0 - stats.chi2.cdf(DM2_exp, df=k_components))

    return {
        "k_components": k_components,
        "variance_explained": float(cum_var[k_components-1]),
        "dm2_exp": DM2_exp,
        "dm2_samples_mean": float(np.mean(DM2_samples)),
        "dm2_samples_std": float(np.std(DM2_samples)),
        "chi2_95_threshold": chi2_95_threshold,
        "p_value": p_value,
        "dm2_samples": DM2_samples.tolist()
    }


def plot_mahalanobis_distance(mahal_metrics, save_path):
    """Plot distribution of Multivariate Mahalanobis Distances vs Chi-Square reference distribution."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    dm2_samples = np.array(mahal_metrics["dm2_samples"])
    dm2_exp = mahal_metrics["dm2_exp"]
    k = mahal_metrics["k_components"]
    chi2_95 = mahal_metrics["chi2_95_threshold"]
    p_val = mahal_metrics["p_value"]

    # Histogram of ensemble realization Mahalanobis distances
    ax.hist(dm2_samples, bins=35, density=True, alpha=0.5, color='#9467bd', edgecolor='purple', 
            label=f'Predicted Samples ($N={len(dm2_samples)}$)')

    # Overlaid theoretical Chi-Square PDF
    x_max = max(np.max(dm2_samples), chi2_95, dm2_exp) * 1.15
    x = np.linspace(0.001, x_max, 200)
    ax.plot(x, stats.chi2.pdf(x, df=k), 'k--', linewidth=2.5, label=f'Theoretical $\\chi^2(K={k})$ PDF')

    # Shaded 95% Confidence Region
    ax.axvspan(0, chi2_95, color='green', alpha=0.10, label=r'95% $\chi^2$ Confidence Region')
    ax.axvline(chi2_95, color='green', linestyle=':', linewidth=2, label=f'95% Threshold ({chi2_95:.2f})')

    # Observed Experimental Mahalanobis Distance Line
    ax.axvline(dm2_exp, color='red', linestyle='-', linewidth=2.5, label=f'Observed $D_M^2 = {dm2_exp:.3f}$')

    info_text = (f"PCA Modes ($K$): {k} (99.9% var)\n"
                 f"Observed $D_M^2$: {dm2_exp:.3f}\n"
                 f"Sample Mean $D_M^2$: {mahal_metrics['dm2_samples_mean']:.3f}\n"
                 f"$\\chi^2_{{0.95}}$ Limit: {chi2_95:.3f}\n"
                 f"Multivariate $p$-value: {p_val:.3f}")
    ax.text(0.48, 0.88, info_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

    ax.set_xlabel(r'Squared Mahalanobis Distance $D_M^2(\mathbf{u})$', fontweight='bold', fontsize=12)
    ax.set_ylabel('Probability Density', fontweight='bold', fontsize=12)
    ax.set_title(r'Multivariate Mahalanobis Distance Calibration', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "mahalanobis_distance.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_calibration_maps(node_coords, cells, u_exp, u_pred_samples, save_path, step_idx=-1):
    """Plot 2x2 spatial heatmaps of observed, predicted mean, error norm, and nodal Mahalanobis distance."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    u_exp_step = u_exp[step_idx] # (n_nodes, 2)
    u_pred_step = u_pred_samples[:, step_idx, :, :] # (n_samples, n_nodes, 2)

    u_mean_step = np.mean(u_pred_step, axis=0)
    u_std_step = np.std(u_pred_step, axis=0)

    u_exp_norm = np.linalg.norm(u_exp_step, axis=1)
    u_mean_norm = np.linalg.norm(u_mean_step, axis=1)
    err_norm = np.linalg.norm(u_exp_step - u_mean_step, axis=1)
    std_norm = np.linalg.norm(u_std_step, axis=1)

    # 2D Nodal Mahalanobis distance calculation
    n_nodes = node_coords.shape[0]
    nodal_dm2 = np.zeros(n_nodes)
    for a in range(n_nodes):
        e_a = u_exp_step[a] - u_mean_step[a] # (2,)
        cov_a = np.cov(u_pred_step[:, a, :], rowvar=False) + 1e-10 * np.eye(2) # (2, 2)
        nodal_dm2[a] = float(e_a @ np.linalg.solve(cov_a, e_a))
    nodal_dm = np.sqrt(nodal_dm2)

    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)

    def add_map(ax, val, title, cmap='viridis'):
        im = ax.tricontourf(triang, val, levels=30, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=12)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    add_map(axes[0, 0], u_exp_norm, r"Observed Displacement $\|\mathbf{u}_{exp}\|$", 'viridis')
    add_map(axes[0, 1], u_mean_norm, r"Predicted Mean $\|\boldsymbol{\mu}_{u}\|$", 'viridis')
    add_map(axes[1, 0], err_norm, r"Absolute Error $\|\mathbf{u}_{exp} - \boldsymbol{\mu}_{u}\|$", 'magma')
    add_map(axes[1, 1], nodal_dm, r"Nodal Mahalanobis Distance $D_{M,\text{nodal}}$", 'plasma')

    for ax in axes.flat:
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')

    plt.suptitle(f'Spatial Model Calibration (Load Step {step_idx if step_idx >= 0 else len(u_exp) + step_idx})', 
                 fontweight='bold', fontsize=15, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(save_path, "spatial_calibration_map.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def compute_subspace_hypothesis_metrics(u_exp, u_pred_samples, var_threshold=0.999, p_lower=0.05, p_upper=0.95):
    """
    Perform Subspace Hypothesis Testing (p in [0.05, 0.95]) and Orthogonal Residual Decomposition.
    """
    N, S, V, C = u_pred_samples.shape
    X = u_pred_samples.reshape(N, -1)
    y = u_exp.reshape(1, -1)

    mu = np.mean(X, axis=0)
    X_centered = X - mu
    y_centered = y - mu

    # SVD for principal subspace decomposition
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = (s**2) / np.sum(s**2)
    cum_var = np.cumsum(var_explained)
    k_components = int(np.searchsorted(cum_var, var_threshold) + 1)
    k_components = max(1, min(k_components, N - 1))

    V_k = Vt[:k_components].T # (D, K)
    s_k = s[:k_components] / np.sqrt(N - 1) # Standard deviations along mode k

    # Projection coordinates
    y_proj = (y_centered @ V_k)[0] # (K,)
    z_proj = y_proj / s_k # Mode z-scores

    # Two-sided p-value per principal mode under standard normal hypothesis N(0, 1)
    mode_p_values = [float(2.0 * (1.0 - stats.norm.cdf(np.abs(z_proj[k])))) for k in range(k_components)]
    mode_passed = [bool(p_lower <= p <= p_upper) for p in mode_p_values]
    all_passed = bool(all(mode_passed))

    # Orthogonal Residual Decomposition: e = e_parallel + e_perp
    e_total = y_centered[0] # (D,)
    e_parallel = (y_centered @ V_k @ V_k.T)[0] # (D,)
    e_perp = e_total - e_parallel # (D,)

    norm_total = float(np.linalg.norm(e_total))
    norm_parallel = float(np.linalg.norm(e_parallel))
    norm_perp = float(np.linalg.norm(e_perp))

    ratio_parallel = norm_parallel / (norm_total + 1e-12)
    ratio_perp = norm_perp / (norm_total + 1e-12)

    # Spatial step decomposition for final step
    step_shape = (V, C)
    e_total_spatial = e_total.reshape(S, V, C)
    e_parallel_spatial = e_parallel.reshape(S, V, C)
    e_perp_spatial = e_perp.reshape(S, V, C)

    return {
        "k_components": k_components,
        "variance_explained": float(cum_var[k_components-1]),
        "mode_z_scores": z_proj.tolist(),
        "mode_p_values": mode_p_values,
        "p_acceptance_range": [p_lower, p_upper],
        "mode_passed": mode_passed,
        "all_modes_passed": all_passed,
        "norm_total": norm_total,
        "norm_parallel": norm_parallel,
        "norm_perp": norm_perp,
        "ratio_parallel": ratio_parallel,
        "ratio_perp": ratio_perp,
        "e_total_spatial": e_total_spatial,
        "e_parallel_spatial": e_parallel_spatial,
        "e_perp_spatial": e_perp_spatial
    }


def plot_subspace_hypothesis_test(subspace_metrics, save_path):
    """Plot Subspace Hypothesis Test p-values and z-scores for principal modes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    k = subspace_metrics["k_components"]
    modes = np.arange(1, k + 1)
    p_vals = subspace_metrics["mode_p_values"]
    z_scores = subspace_metrics["mode_z_scores"]

    p_lower, p_upper = subspace_metrics["p_acceptance_range"]

    # 1. p-values plot
    colors = ['#2ca02c' if passed else '#d62728' for passed in subspace_metrics["mode_passed"]]
    bars = ax1.bar(modes, p_vals, color=colors, alpha=0.8, edgecolor='black', width=0.4)
    ax1.axhspan(p_lower, p_upper, color='green', alpha=0.15, label=f'Acceptance Zone [{p_lower}, {p_upper}]')
    ax1.axhline(p_lower, color='red', linestyle='--', linewidth=1.5, label=f'Limit p = {p_lower}')
    ax1.axhline(p_upper, color='red', linestyle='--', linewidth=1.5, label=f'Limit p = {p_upper}')

    for bar, p in zip(bars, p_vals):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02, f"{p:.3f}",
                 ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax1.set_xlabel('Principal Mode ($k$)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Subspace Mode $p$-value', fontweight='bold', fontsize=12)
    ax1.set_title(f'Subspace Hypothesis Test ($p \\in [{p_lower}, {p_upper}]$)', fontweight='bold', fontsize=13)
    ax1.set_ylim([0, 1.05])
    ax1.set_xticks(modes)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right', fontsize=9)

    # 2. z-scores plot
    ax2.bar(modes, z_scores, color='#1f77b4', alpha=0.8, edgecolor='black', width=0.4)
    ax2.axhspan(-1.96, 1.96, color='blue', alpha=0.10, label=r'95% $z$-Interval $[-1.96, 1.96]$')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.axhline(1.96, color='gray', linestyle=':', linewidth=1.5)
    ax2.axhline(-1.96, color='gray', linestyle=':', linewidth=1.5)

    for i, z in enumerate(z_scores):
        offset = 0.08 if z >= 0 else -0.15
        ax2.text(modes[i], z + offset, f"{z:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax2.set_xlabel('Principal Mode ($k$)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Subspace Standardized Projection $z_k$', fontweight='bold', fontsize=12)
    ax2.set_title('Subspace Mode Standardized Projections ($z_k$)', fontweight='bold', fontsize=13)
    ax2.set_xticks(modes)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "subspace_hypothesis_test.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_orthogonal_residual_analysis(subspace_metrics, save_path):
    """Plot Orthogonal Decomposition of Total Displacement Error."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    labels = [r'Total Residual $\|e\|$', r'In-Subspace $\|e_\|\|$', r'Orthogonal Residual $\|e_\perp\|$']
    norms = [subspace_metrics["norm_total"], subspace_metrics["norm_parallel"], subspace_metrics["norm_perp"]]
    ratios = [1.0, subspace_metrics["ratio_parallel"], subspace_metrics["ratio_perp"]]
    colors = ['#333333', '#1f77b4', '#d62728']

    bars = ax.bar(labels, norms, color=colors, alpha=0.85, edgecolor='black', width=0.45)

    for bar, norm, ratio in zip(bars, norms, ratios):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.003, f"{norm:.4f}\n({ratio*100:.1f}%)",
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylabel('Displacement Error Norm', fontweight='bold', fontsize=12)
    ax.set_title('Orthogonal Error Subspace Decomposition', fontweight='bold', fontsize=14)
    ax.set_ylim([0, max(norms) * 1.25])
    ax.grid(True, alpha=0.3, axis='y')

    info_text = (f"PCA Subspace Modes: K = {subspace_metrics['k_components']}\n"
                 f"Variance Explained: {subspace_metrics['variance_explained']*100:.2f}%\n"
                 f"In-Subspace Error: {subspace_metrics['ratio_parallel']*100:.2f}%\n"
                 f"Orthogonal Error: {subspace_metrics['ratio_perp']*100:.2f}%")
    ax.text(0.58, 0.65, info_text, transform=ax.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "orthogonal_residual_analysis.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_spatial_residual_maps(node_coords, cells, subspace_metrics, save_path, step_idx=-1):
    """Plot spatial maps of Total Residual ||e||, In-Subspace ||e_||||, and Orthogonal ||e_perp||."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    e_total_step = subspace_metrics["e_total_spatial"][step_idx] # (n_nodes, 2)
    e_parallel_step = subspace_metrics["e_parallel_spatial"][step_idx] # (n_nodes, 2)
    e_perp_step = subspace_metrics["e_perp_spatial"][step_idx] # (n_nodes, 2)

    norm_total = np.linalg.norm(e_total_step, axis=1)
    norm_parallel = np.linalg.norm(e_parallel_step, axis=1)
    norm_perp = np.linalg.norm(e_perp_step, axis=1)

    triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)

    def add_map(ax, val, title, cmap='viridis', vmax=None):
        if vmax is not None:
            im = ax.tricontourf(triang, val, levels=30, cmap=cmap, vmin=0, vmax=vmax)
        else:
            im = ax.tricontourf(triang, val, levels=30, cmap=cmap)
        ax.set_aspect('equal')
        ax.set_title(title, fontweight='bold', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    vmax_common = max(np.max(norm_total), np.max(norm_parallel))
    add_map(axes[0], norm_total, r"Total Error $\|e(x, y)\|$", 'magma', vmax=vmax_common)
    add_map(axes[1], norm_parallel, r"In-Subspace Error $\|e_\|(x, y)\|$", 'magma', vmax=vmax_common)
    add_map(axes[2], norm_perp, r"Orthogonal Residual $\|e_\perp(x, y)\|$", 'Reds')

    for ax in axes.flat:
        ax.set_xlabel('X', fontweight='bold')
        ax.set_ylabel('Y', fontweight='bold')

    plt.suptitle(f'Spatial Orthogonal Residual Decomposition (Step {step_idx if step_idx >= 0 else 10 + step_idx})', 
                 fontweight='bold', fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(save_path, "spatial_residual_map.pdf"), dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_calibration(data_file_path, output_dir=None):
    """
    Main programmatic entry point to evaluate UQ model calibration.
    """
    data_path = Path(data_file_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Consolidated dataset file not found: {data_file_path}")

    if output_dir is None:
        output_dir = data_path.parent
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading consolidated dataset from: {data_path}")
    data = np.load(data_path, allow_pickle=True)

    u_pred_samples = data["u_pred"] # Shape: (n_samples, n_steps, n_nodes, 2)
    node_coords = data["node_coords"]
    cells = data["cells"]

    if "u_exp" in data:
        u_exp = data["u_exp"]
        print("Using observed experimental displacement field (u_exp).")
    elif "u_true" in data:
        u_exp = data["u_true"]
        print("Using ground truth displacement field (u_true).")
    else:
        raise KeyError("Neither 'u_exp' nor 'u_true' found in consolidated dataset!")

    print(f"Dataset summary: {u_pred_samples.shape[0]} samples, {u_pred_samples.shape[1]} load steps, {u_pred_samples.shape[2]} nodes.")

    # Compute metrics
    metrics = compute_calibration_metrics(u_exp, u_pred_samples)
    mahal_metrics = compute_mahalanobis_distance_metrics(u_exp, u_pred_samples)
    subspace_metrics = compute_subspace_hypothesis_metrics(u_exp, u_pred_samples)

    metrics["mahalanobis"] = {
        "k_components": mahal_metrics["k_components"],
        "variance_explained": mahal_metrics["variance_explained"],
        "dm2_exp": mahal_metrics["dm2_exp"],
        "dm2_samples_mean": mahal_metrics["dm2_samples_mean"],
        "chi2_95_threshold": mahal_metrics["chi2_95_threshold"],
        "p_value": mahal_metrics["p_value"]
    }
    metrics["subspace_hypothesis"] = {
        "k_components": subspace_metrics["k_components"],
        "mode_z_scores": subspace_metrics["mode_z_scores"],
        "mode_p_values": subspace_metrics["mode_p_values"],
        "p_acceptance_range": subspace_metrics["p_acceptance_range"],
        "all_modes_passed": subspace_metrics["all_modes_passed"]
    }
    metrics["orthogonal_residuals"] = {
        "norm_total": subspace_metrics["norm_total"],
        "norm_parallel": subspace_metrics["norm_parallel"],
        "norm_perp": subspace_metrics["norm_perp"],
        "ratio_parallel": subspace_metrics["ratio_parallel"],
        "ratio_perp": subspace_metrics["ratio_perp"]
    }

    # Save metrics JSON
    json_path = output_dir / "calibration_metrics.json"
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved calibration metrics to: {json_path}")

    # Generate diagnostic plots
    plot_reliability_diagram(metrics, output_dir)
    plot_error_vs_uncertainty(u_exp, u_pred_samples, output_dir)
    plot_z_score_distribution(u_exp, u_pred_samples, metrics, output_dir)
    plot_mahalanobis_distance(mahal_metrics, output_dir)
    plot_subspace_hypothesis_test(subspace_metrics, output_dir)
    plot_orthogonal_residual_analysis(subspace_metrics, output_dir)
    plot_spatial_residual_maps(node_coords, cells, subspace_metrics, output_dir)
    plot_spatial_calibration_maps(node_coords, cells, u_exp, u_pred_samples, output_dir)

    print(f"🎉 Model calibration evaluation completed successfully! Plots saved in: {output_dir}")
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate UQ Model Displacement Calibration")
    parser.add_argument('--distilled_dir', type=str, required=True, help="Path to distilled model folder or fem_validation folder")
    parser.add_argument('--subfolder', type=str, default="fem_validation", help="Subfolder name containing fem_distilled_samples.npz")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    base_path = Path(args.distilled_dir)

    if (base_path / args.subfolder / "fem_distilled_samples.npz").exists():
        data_file = base_path / args.subfolder / "fem_distilled_samples.npz"
        out_dir = base_path / args.subfolder
    elif (base_path / "fem_distilled_samples.npz").exists():
        data_file = base_path / "fem_distilled_samples.npz"
        out_dir = base_path
    else:
        raise FileNotFoundError(f"Could not locate fem_distilled_samples.npz in {base_path} or {base_path / args.subfolder}")

    evaluate_calibration(data_file, out_dir)

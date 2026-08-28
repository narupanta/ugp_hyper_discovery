import numpy as np
from scipy import stats

def compute_parameter_metrics(theta_samples, theta_gt):
    """
    Level 1: Material Parameter Validation Metrics.

    Parameters
    ----------
    theta_samples : np.ndarray
        Array of sampled material parameter vectors, shape (n_samples, n_params)
    theta_gt : np.ndarray
        Ground truth parameter vector, shape (n_params,)

    Returns
    -------
    dict
        Relative error, parameter z-scores, Mahalanobis distance D_M_theta^2, p-value, and entropy/volume.
    """
    theta_samples = np.asarray(theta_samples)
    theta_gt = np.asarray(theta_gt)
    n_samples, n_params = theta_samples.shape

    mu_theta = np.mean(theta_samples, axis=0)
    std_theta = np.std(theta_samples, axis=0) + 1e-12

    # Relative Parameter Error
    rel_error = float(np.linalg.norm(mu_theta - theta_gt) / (np.linalg.norm(theta_gt) + 1e-12))

    # Normalized Z-Scores
    z_scores = (mu_theta - theta_gt) / std_theta

    # SVD rank determination across the ENTIRE parameter space
    X_centered = theta_samples - mu_theta
    diff = mu_theta - theta_gt

    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    s_deg = s / np.sqrt(n_samples - 1)
    valid_modes = s_deg > 1e-5
    k_rank = int(np.sum(valid_modes)) if np.sum(valid_modes) > 0 else n_params

    V_k = Vt[:k_rank].T
    s_k = s_deg[:k_rank]
    z_k = (diff @ V_k) / s_k

    dm2_theta = float(np.sum(z_k**2))
    p_value = float(1.0 - stats.chi2.cdf(dm2_theta, df=k_rank))
    chi2_95 = float(stats.chi2.ppf(0.95, df=k_rank))

    # Log-Determinant Uncertainty Volume
    cov_theta = np.cov(theta_samples, rowvar=False) + 1e-10 * np.eye(n_params)
    sign, logdet = np.linalg.slogdet(cov_theta)
    entropy_volume = float(0.5 * logdet)

    return {
        "n_samples": int(n_samples),
        "n_params": int(n_params),
        "k_rank": k_rank,
        "rel_error": rel_error,
        "mean_params": mu_theta.tolist(),
        "std_params": std_theta.tolist(),
        "gt_params": theta_gt.tolist(),
        "z_scores": z_scores.tolist(),
        "dm2_theta": dm2_theta,
        "chi2_95_threshold": chi2_95,
        "p_value": p_value,
        "entropy_volume": entropy_volume
    }


def compute_energy_metrics(W_pred_samples, W_gt, nominal_levels=None):
    """
    Level 2: Strain Energy Functional W(F) & Stress Calibration Metrics.

    Parameters
    ----------
    W_pred_samples : np.ndarray
        Predicted strain energy samples across deformation domain, shape (n_samples, n_points)
    W_gt : np.ndarray
        Ground truth strain energy values, shape (n_points,)
    nominal_levels : list or np.ndarray, optional
        Confidence levels for calibration evaluation.

    Returns
    -------
    dict
        Energy RMSE, Rel-RMSE, PICP_95, ECE_W, and MPIW_W.
    """
    W_pred_samples = np.asarray(W_pred_samples)
    W_gt = np.asarray(W_gt)
    n_samples, n_points = W_pred_samples.shape

    if nominal_levels is None:
        nominal_levels = np.linspace(0.05, 0.95, 19)

    mu_W = np.mean(W_pred_samples, axis=0)

    # 1. Energy Accuracy Metrics
    err = mu_W - W_gt
    rmse_W = float(np.sqrt(np.mean(err**2)))
    rel_rmse_W = float(rmse_W / (np.mean(np.abs(W_gt)) + 1e-12))

    # 2. Calibration & ECE
    picp_values = []
    mpiw_values = []

    for conf in nominal_levels:
        alpha = 1.0 - conf
        q_low = np.quantile(W_pred_samples, alpha / 2.0, axis=0)
        q_high = np.quantile(W_pred_samples, 1.0 - alpha / 2.0, axis=0)

        inside = (W_gt >= q_low) & (W_gt <= q_high)
        picp_values.append(float(np.mean(inside)))
        mpiw_values.append(float(np.mean(q_high - q_low)))

    picp_values = np.array(picp_values)
    mpiw_values = np.array(mpiw_values)
    nominal_levels = np.array(nominal_levels)
    ece_W = float(np.mean(np.abs(picp_values - nominal_levels)))

    # 95% Confidence Interval Metrics
    idx_95 = np.argmin(np.abs(nominal_levels - 0.95))
    picp_95 = float(picp_values[idx_95])
    mpiw_95 = float(mpiw_values[idx_95])

    return {
        "n_samples": int(n_samples),
        "n_points": int(n_points),
        "rmse_W": rmse_W,
        "rel_rmse_W": rel_rmse_W,
        "ece_W": ece_W,
        "picp_W_95": picp_95,
        "mpiw_W_95": mpiw_95,
        "nominal_levels": nominal_levels.tolist(),
        "picp_values": picp_values.tolist(),
        "mpiw_values": mpiw_values.tolist()
    }


def compute_subspace_mahalanobis(Y_pred, y_obs, energy_thresh=0.99):
    """
    Level 3: Thin-SVD Subspace Mahalanobis Distance and Orthogonal Residual Decomposition.

    Parameters
    ----------
    Y_pred : np.ndarray
        Array of ensemble realization predictions, shape (n_samples, d)
    y_obs : np.ndarray
        Observed field vector, shape (d,)
    energy_thresh : float
        Variance threshold for retaining principal components (default: 0.99)

    Returns
    -------
    dict
        Subspace Mahalanobis D_M^2, K_components, p-value, in-subspace norm, orthogonal residual norm.
    """
    Y_pred = np.asarray(Y_pred)
    y_obs = np.asarray(y_obs)
    n_samples, d = Y_pred.shape

    mu_y = np.mean(Y_pred, axis=0)
    X_centered = Y_pred - mu_y # (N, d)
    y_centered = y_obs - mu_y # (d,)

    # Thin SVD on centered realizations
    U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
    var_explained = (s**2) / np.sum(s**2)
    cum_var = np.cumsum(var_explained)

    k_components = int(np.searchsorted(cum_var, energy_thresh) + 1)
    k_components = max(1, min(k_components, n_samples - 1))

    V_k = Vt[:k_components].T # (d, K)
    s_k = s[:k_components] / np.sqrt(n_samples - 1) # Mode standard deviations

    # Project centered data onto top K PCs
    z_k = (y_centered @ V_k) # (K,)
    z_k_std = z_k / s_k # Standardized mode projections

    # Subspace Mahalanobis Distance D_M^2 ~ Chi2(K)
    dm2_sub = float(np.sum(z_k_std**2))

    # Chi-Square hypothesis test
    p_value = float(1.0 - stats.chi2.cdf(dm2_sub, df=k_components))
    chi2_lower = float(stats.chi2.ppf(0.025, df=k_components))
    chi2_upper = float(stats.chi2.ppf(0.975, df=k_components))

    # Orthogonal Residual Decomposition
    e_parallel = (y_centered @ V_k @ V_k.T) # In-subspace component
    e_perp = y_centered - e_parallel # Orthogonal residual

    norm_total = float(np.linalg.norm(y_centered))
    norm_parallel = float(np.linalg.norm(e_parallel))
    norm_perp = float(np.linalg.norm(e_perp))

    return {
        "k_components": k_components,
        "variance_explained": float(cum_var[k_components - 1]),
        "dm2_sub": dm2_sub,
        "chi2_lower_95": chi2_lower,
        "chi2_upper_95": chi2_upper,
        "p_value": p_value,
        "mode_z_scores": z_k_std.tolist(),
        "norm_total": norm_total,
        "norm_parallel": norm_parallel,
        "norm_perp": norm_perp,
        "ratio_parallel": norm_parallel / (norm_total + 1e-12),
        "ratio_perp": norm_perp / (norm_total + 1e-12),
        "e_parallel": e_parallel,
        "e_perp": e_perp
    }


def compute_multivariate_energy_score(Y_pred, y_obs):
    """
    Compute Multivariate Energy Score (proper scoring rule for multivariate UQ).

    ES(y_obs) = 1/N sum_k ||y_k - y_obs||_2 - 1/(2 N^2) sum_k sum_j ||y_k - y_j||_2

    Parameters
    ----------
    Y_pred : np.ndarray
        Array of ensemble realizations, shape (n_samples, d)
    y_obs : np.ndarray
        Observed field vector, shape (d,)

    Returns
    -------
    float
        Multivariate Energy Score.
    """
    Y_pred = np.asarray(Y_pred)
    y_obs = np.asarray(y_obs)
    n_samples = Y_pred.shape[0]

    # Term 1: Mean distance from realizations to observation
    term1 = float(np.mean(np.linalg.norm(Y_pred - y_obs, axis=1)))

    # Term 2: Mean pairwise distance between realizations (memory-efficient)
    total_pairwise_dist = 0.0
    for i in range(n_samples):
        dists = np.linalg.norm(Y_pred - Y_pred[i], axis=1)
        total_pairwise_dist += float(np.sum(dists))

    term2 = float(0.5 * (total_pairwise_dist / (n_samples**2)))

    energy_score = term1 - term2
    return float(energy_score)


def compute_marginal_calibration(Y_pred, y_obs, nominal_levels=None):
    """
    Compute marginal field calibration metrics (PICP, ECE, PIT).

    Parameters
    ----------
    Y_pred : np.ndarray
        Array of ensemble realizations, shape (n_samples, d)
    y_obs : np.ndarray
        Observed field vector, shape (d,)
    nominal_levels : list or np.ndarray, optional
        Confidence levels to evaluate.

    Returns
    -------
    dict
        ECE, PICP levels, MPIW, and PIT values.
    """
    Y_pred = np.asarray(Y_pred)
    y_obs = np.asarray(y_obs)
    n_samples, d = Y_pred.shape

    if nominal_levels is None:
        nominal_levels = np.linspace(0.05, 0.95, 19)

    # Probability Integral Transform (PIT) values
    pit_values = np.mean(Y_pred <= y_obs[None, :], axis=0) # Shape: (d,)

    picp_values = []
    mpiw_values = []

    for conf in nominal_levels:
        alpha = 1.0 - conf
        q_low = np.quantile(Y_pred, alpha / 2.0, axis=0)
        q_high = np.quantile(Y_pred, 1.0 - alpha / 2.0, axis=0)

        inside = (y_obs >= q_low) & (y_obs <= q_high)
        picp_values.append(float(np.mean(inside)))
        mpiw_values.append(float(np.mean(q_high - q_low)))

    picp_values = np.array(picp_values)
    mpiw_values = np.array(mpiw_values)
    nominal_levels = np.array(nominal_levels)

    ece = float(np.mean(np.abs(picp_values - nominal_levels)))

    idx_95 = np.argmin(np.abs(nominal_levels - 0.95))
    picp_95 = float(picp_values[idx_95])
    mpiw_95 = float(mpiw_values[idx_95])

    return {
        "ece": ece,
        "picp_95": picp_95,
        "mpiw_95": mpiw_95,
        "nominal_levels": nominal_levels.tolist(),
        "picp_values": picp_values.tolist(),
        "mpiw_values": mpiw_values.tolist(),
        "pit_values": pit_values.tolist()
    }


def compute_mpiw(Y_pred, alpha=0.05):
    """
    Compute Mean Prediction Interval Width (MPIW) for a specified significance level alpha.
    """
    Y_pred = np.asarray(Y_pred)
    q_low = np.quantile(Y_pred, alpha / 2.0, axis=0)
    q_high = np.quantile(Y_pred, 1.0 - alpha / 2.0, axis=0)
    return float(np.mean(q_high - q_low))

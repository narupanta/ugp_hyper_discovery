import numpy as np
from typing import Tuple, Dict, Any, Optional

def compute_conformal_scale(
    y_true: np.ndarray,
    y_pred_samples: np.ndarray,
    alpha: float = 0.05,
    min_std: float = 1e-12
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the nonconformity score and the empirical (1 - alpha) quantile scale factor Q.
    
    Parameters:
    -----------
    y_true : np.ndarray
        Ground truth or observed values, shape (..., )
    y_pred_samples : np.ndarray
        Predictive samples from distilled posterior, shape (N_samples, ...)
    alpha : float
        Significance level (default 0.05 for 95% coverage)
    min_std : float
        Regularization floor for standard deviation to avoid division by zero.
        
    Returns:
    --------
    q_val : float
        Conformal scale factor Q_{1-alpha}
    scores : np.ndarray
        Pointwise nonconformity scores s_i = |y_true - y_mean| / y_std
    y_std : np.ndarray
        Standard deviation across parameter posterior samples
    """
    y_mean = np.mean(y_pred_samples, axis=0)
    y_std = np.std(y_pred_samples, axis=0)
    denom = np.maximum(y_std, min_std)
    
    scores = np.abs(y_true - y_mean) / denom
    scores_flat = scores.flatten()
    
    # Compute the (1 - alpha) empirical quantile
    # Conformal prediction with finite sample correction: ceil((n + 1) * (1 - alpha)) / n
    n = len(scores_flat)
    q_level = min(100.0, np.ceil((n + 1) * (1.0 - alpha)) / n * 100.0)
    q_val = float(np.percentile(scores_flat, q_level))
    
    return q_val, scores, y_std


def apply_conformal_band(
    y_pred_samples: np.ndarray,
    q_val: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the calibrated prediction interval:
    [y_mean - Q * y_std, y_mean + Q * y_std]
    
    Returns:
    --------
    y_mean : np.ndarray
        Predictive mean
    lower_bound : np.ndarray
        Calibrated lower bound
    upper_bound : np.ndarray
        Calibrated upper bound
    """
    y_mean = np.mean(y_pred_samples, axis=0)
    y_std = np.std(y_pred_samples, axis=0)
    lower = y_mean - q_val * y_std
    upper = y_mean + q_val * y_std
    return y_mean, lower, upper


def compute_empirical_coverage(
    y_true: np.ndarray,
    lower_bound: np.ndarray,
    upper_bound: np.ndarray
) -> float:
    """Computes empirical percentage of ground truth observations within [lower_bound, upper_bound]."""
    inside = (y_true >= lower_bound) & (y_true <= upper_bound)
    return float(np.mean(inside) * 100.0)


def compute_variance_budget(
    sigma2_param: float,
    sigma2_noise: float,
    q_val: float,
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Decomposes the total predictive variance into parameter, measurement noise, and model discrepancy:
    
    sigma_total^2 = (Q / z_alpha)^2 * sigma_param^2
    sigma_total^2 = sigma_param^2 + sigma_noise^2 + sigma_discrepancy^2
    sigma_discrepancy^2 = max(0, sigma_total^2 - sigma_param^2 - sigma_noise^2)
    
    If sigma_discrepancy^2 <= sigma_noise^2, model discrepancy is within the experimental noise floor.
    """
    from scipy.stats import norm
    z_alpha = norm.ppf(1.0 - alpha / 2.0)  # ~1.95996 for alpha=0.05
    
    # Calibrated total variance
    scaling_ratio = q_val / z_alpha
    sigma2_total = (scaling_ratio ** 2) * sigma2_param
    
    # Discrepancy variance
    raw_discrepancy = sigma2_total - sigma2_param - sigma2_noise
    sigma2_discrepancy = max(0.0, float(raw_discrepancy))
    
    # Relative percentages
    total_denom = max(sigma2_total, 1e-15)
    pct_param = (sigma2_param / total_denom) * 100.0
    pct_noise = (sigma2_noise / total_denom) * 100.0
    pct_discrepancy = (sigma2_discrepancy / total_denom) * 100.0
    
    is_zero_discrepancy = (sigma2_discrepancy <= sigma2_noise) or (raw_discrepancy <= 0.0)
    
    return {
        "q_val": float(q_val),
        "z_alpha": float(z_alpha),
        "scaling_ratio": float(scaling_ratio),
        "sigma2_total": float(sigma2_total),
        "sigma2_param": float(sigma2_param),
        "sigma2_noise": float(sigma2_noise),
        "sigma2_discrepancy": float(sigma2_discrepancy),
        "pct_param": float(pct_param),
        "pct_noise": float(pct_noise),
        "pct_discrepancy": float(pct_discrepancy),
        "is_zero_discrepancy": bool(is_zero_discrepancy),
        "status": "Zero / Noise-Floor Discrepancy" if is_zero_discrepancy else "Significant Model Discrepancy"
    }


def test_conformal():
    np.random.seed(42)
    n_samples, n_points = 500, 100
    y_true = np.sin(np.linspace(0, 3, n_points))
    # Simulated posterior with slight undercoverage
    noise = np.random.randn(n_samples, n_points) * 0.05
    y_pred_samples = y_true[None, :] + noise + 0.02
    
    q_val, scores, y_std = compute_conformal_scale(y_true, y_pred_samples, alpha=0.05)
    y_mean, lower, upper = apply_conformal_band(y_pred_samples, q_val)
    cov = compute_empirical_coverage(y_true, lower, upper)
    budget = compute_variance_budget(np.mean(y_std**2), 1e-4, q_val)
    
    print(f"Test Q_0.95: {q_val:.4f}, Coverage: {cov:.2f}%")
    print(f"Variance Budget: {budget}")
    assert cov >= 94.0, f"Expected coverage >= 94%, got {cov}"
    print("✅ Conformal test passed successfully!")


if __name__ == "__main__":
    test_conformal()


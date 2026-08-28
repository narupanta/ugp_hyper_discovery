import os
import json
from pathlib import Path
import numpy as np

from validation.metrics import (
    compute_parameter_metrics,
    compute_energy_metrics,
    compute_subspace_mahalanobis,
    compute_multivariate_energy_score,
    compute_marginal_calibration,
    compute_mpiw
)
from validation.plots import (
    plot_parameter_posteriors,
    plot_energy_calibration,
    plot_subspace_chi2_distribution,
    plot_reliability_diagrams,
    plot_pit_histogram,
    plot_spatial_calibration_maps
)

def validate_material_parameters(theta_samples, theta_gt, output_dir):
    """
    Level 1 Validation Pipeline: Material Parameter Validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_parameter_metrics(theta_samples, theta_gt)
    plot_parameter_posteriors(metrics, output_dir)
    return metrics


def validate_energy_functional(W_pred_samples, W_gt, output_dir):
    """
    Level 2 Validation Pipeline: Strain Energy Functional W(F) & Stress Calibration.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_energy_metrics(W_pred_samples, W_gt)
    plot_energy_calibration(metrics, output_dir)
    return metrics


def validate_full_field_kinematics(U_pred_samples, U_obs, R_pred_samples=None, R_obs=None, 
                                   node_coords=None, cells=None, sigma_dic=None, output_dir=None):
    """
    Level 3 Validation Pipeline: Full-Field Kinematics & Reaction Force Validation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    N, S, V, C = U_pred_samples.shape
    d = S * V * C

    U_pred_flat = U_pred_samples.reshape(N, -1)
    U_obs_flat = U_obs.flatten()

    # 1. Kinematic Accuracy (NRMSE)
    u_mean_flat = np.mean(U_pred_flat, axis=0)
    disp_range = np.max(U_obs_flat) - np.min(U_obs_flat) + 1e-12
    nrmse_u = float(np.linalg.norm(u_mean_flat - U_obs_flat) / (np.sqrt(d) * disp_range))
    rmse_u = float(np.sqrt(np.mean((u_mean_flat - U_obs_flat)**2)))

    # 2. Reaction Force Misfit
    if R_pred_samples is not None and R_obs is not None:
        r_mean = np.mean(R_pred_samples, axis=0)
        reaction_misfit = float(np.linalg.norm(r_mean - R_obs) / (np.linalg.norm(R_obs) + 1e-12))
    else:
        reaction_misfit = None

    # 3. Subspace Mahalanobis Distance & Chi2 Test
    subspace_results = compute_subspace_mahalanobis(U_pred_flat, U_obs_flat, energy_thresh=0.99)

    # 4. Proper Scoring Rule: Multivariate Energy Score (ES)
    energy_score = compute_multivariate_energy_score(U_pred_flat, U_obs_flat)

    # 5. Marginal Calibration (PICP, ECE, PIT)
    calib_results = compute_marginal_calibration(U_pred_flat, U_obs_flat)

    # 6. Sharpness (MPIW) & Noise Floor comparison
    mpiw_95 = compute_mpiw(U_pred_flat, alpha=0.05)
    mpiw_ratio_dic = float(mpiw_95 / sigma_dic) if sigma_dic is not None else None

    # Generate Level 3 Diagnostic Plots
    plot_subspace_chi2_distribution(subspace_results, output_dir)
    plot_reliability_diagrams(calib_results, output_dir)
    plot_pit_histogram(calib_results, output_dir)

    if node_coords is not None and cells is not None:
        plot_spatial_calibration_maps(node_coords, cells, U_obs, U_pred_samples, output_dir)

    results = {
        "nrmse_u": nrmse_u,
        "rmse_u": rmse_u,
        "reaction_misfit": reaction_misfit,
        "subspace_mahalanobis": {
            "k_components": subspace_results["k_components"],
            "variance_explained": subspace_results["variance_explained"],
            "dm2_sub": subspace_results["dm2_sub"],
            "chi2_lower_95": subspace_results["chi2_lower_95"],
            "chi2_upper_95": subspace_results["chi2_upper_95"],
            "p_value": subspace_results["p_value"],
            "ratio_parallel": subspace_results["ratio_parallel"],
            "ratio_perp": subspace_results["ratio_perp"]
        },
        "multivariate_energy_score": energy_score,
        "marginal_calibration": {
            "ece": calib_results["ece"],
            "picp_95": calib_results["picp_95"],
            "mpiw_95": calib_results["mpiw_95"],
            "mpiw_ratio_dic": mpiw_ratio_dic
        }
    }
    return results


def evaluate_acceptance_checklist(level1_results=None, level2_results=None, level3_results=None):
    """
    Evaluate Section 4 Acceptance Criteria Checklist across Level 1, 2, and 3 results.
    """
    checklist = []

    # Check 1: Mean Field Accuracy (NRMSE_u <= 5%)
    if level3_results is not None:
        nrmse = level3_results["nrmse_u"]
        passed = bool(nrmse <= 0.05)
        checklist.append({
            "check_item": "Mean Field Accuracy (NRMSE_u <= 5%)",
            "target": "NRMSE_u <= 5%",
            "observed_value": f"{nrmse:.2%}",
            "status": "PASSED" if passed else "FAILED",
            "diagnostic_action": "Check for missing constitutive terms, kinematic boundary mismatch, or optimization local minima." if not passed else "N/A"
        })

        # Check 2: Subspace Calibration (D_M^2 in [chi2_0.025, chi2_0.975])
        sub = level3_results["subspace_mahalanobis"]
        dm2 = sub["dm2_sub"]
        p_val = sub["p_value"]
        passed_sub = bool(0.05 <= p_val <= 0.95)
        status_sub = "PASSED" if passed_sub else ("FAILED (Underconfident)" if p_val > 0.95 else "FAILED (Overconfident)")
        diag_sub = "Model is overconfident (underestimating parameter variance)." if p_val < 0.05 else ("Model is underconfident (posterior too loose)." if p_val > 0.95 else "N/A")
        checklist.append({
            "check_item": "Subspace Calibration (D_M^2 ~ Chi2)",
            "target": "p in [0.05, 0.95]",
            "observed_value": f"D_M^2 = {dm2:.2f}, p = {p_val:.3f}",
            "status": status_sub,
            "diagnostic_action": diag_sub
        })

        # Check 3: Marginal Field Coverage (PICP_95 in [93%, 97%])
        picp = level3_results["marginal_calibration"]["picp_95"]
        passed_picp = bool(0.93 <= picp <= 0.97)
        checklist.append({
            "check_item": "Marginal Field Coverage (PICP_95%)",
            "target": "PICP_95 in [93.0%, 97.0%]",
            "observed_value": f"{picp*100:.1f}%",
            "status": "PASSED" if passed_picp else "FAILED",
            "diagnostic_action": "Inspect localized spatial residual maps to identify boundary stress concentration errors." if not passed_picp else "N/A"
        })

        # Check 4: Regression ECE (ECE <= 0.05)
        ece = level3_results["marginal_calibration"]["ece"]
        passed_ece = bool(ece <= 0.05)
        checklist.append({
            "check_item": "Regression ECE",
            "target": "ECE <= 0.05",
            "observed_value": f"{ece:.4f}",
            "status": "PASSED" if passed_ece else "FAILED",
            "diagnostic_action": "Recalibrate surrogate variance or temperature scale." if not passed_ece else "N/A"
        })

    # Check 5: Level 1 Parameter Relative Error (RelErr <= 10%)
    if level1_results is not None:
        rel_err = level1_results["rel_error"]
        passed_rel = bool(rel_err <= 0.10)
        checklist.append({
            "check_item": "Material Parameter Accuracy (RelError <= 10%)",
            "target": "RelError <= 10%",
            "observed_value": f"{rel_err:.2%}",
            "status": "PASSED" if passed_rel else "FAILED",
            "diagnostic_action": "Review parameter identifiability and GP likelihood noise scale." if not passed_rel else "N/A"
        })

    # Check 6: Level 2 Energy Functional Relative RMSE (RelRMSE_W <= 5%)
    if level2_results is not None:
        rel_rmse = level2_results["rel_rmse_W"]
        passed_w = bool(rel_rmse <= 0.05)
        checklist.append({
            "check_item": "Strain Energy Accuracy (RelRMSE_W <= 5%)",
            "target": "RelRMSE_W <= 5%",
            "observed_value": f"{rel_rmse:.2%}",
            "status": "PASSED" if passed_w else "FAILED",
            "diagnostic_action": "Check strain domain coverage and functional basis complexity." if not passed_w else "N/A"
        })

    return checklist

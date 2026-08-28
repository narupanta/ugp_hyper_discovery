import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np

# Ensure validation package is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validation.pipelines import (
    validate_material_parameters,
    validate_energy_functional,
    validate_full_field_kinematics,
    evaluate_acceptance_checklist
)

def compute_synthetic_strain_energy_samples(selected_samples, F_grid, model_type="gmr"):
    """
    Compute strain energy W(F) samples for Level 2 functional validation across a grid of deformation gradients F.
    """
    n_samples = selected_samples.shape[0]
    n_points = F_grid.shape[0]
    W_pred_samples = np.zeros((n_samples, n_points))

    # Evaluate strain energy W(F) for each material parameter vector realization
    for i in range(n_samples):
        params = selected_samples[i]
        # For Neo-Hookean nh2 (c10 = params[0], d1 = params[10]):
        # W(F) = c10 * (I1_dev - 3) + 1/d1 * (J - 1)^2
        c10 = params[0]
        d1 = params[10] if len(params) > 10 else 1.5

        for p in range(n_points):
            F = F_grid[p]
            J = np.linalg.det(F)
            C = F.T @ F
            I1 = np.trace(C)
            I1_dev = J**(-2.0/3.0) * I1
            W = c10 * (I1_dev - 3.0) + (1.0 / (d1 + 1e-12)) * (J - 1.0)**2
            W_pred_samples[i, p] = W

    return W_pred_samples


def run_multi_level_validation(distilled_dir, subfolder="fem_validation", sigma_dic=0.0001):
    """
    Run full Multi-Level Uncertainty Quantification Validation Framework (Levels 1, 2, 3).
    """
    base_path = Path(distilled_dir)
    if (base_path / subfolder / "fem_distilled_samples.npz").exists():
        data_file = base_path / subfolder / "fem_distilled_samples.npz"
        output_dir = base_path / subfolder / "multi_level_validation"
    elif (base_path / "fem_distilled_samples.npz").exists():
        data_file = base_path / "fem_distilled_samples.npz"
        output_dir = base_path / "multi_level_validation"
    else:
        raise FileNotFoundError(f"Could not locate fem_distilled_samples.npz in {base_path} or {base_path / subfolder}")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"================================================================")
    print(f"🚀 Running Multi-Level UQ Validation Framework")
    print(f"Dataset File: {data_file}")
    print(f"Output Directory: {output_dir}")
    print(f"================================================================")

    data = np.load(data_file, allow_pickle=True)
    u_pred_samples = data["u_pred"] # (n_samples, n_steps, n_nodes, 2)
    selected_samples = data["selected_samples"] # (n_samples, n_params)
    node_coords = data["node_coords"]
    cells = data["cells"]

    if "u_exp" in data:
        u_obs = data["u_exp"]
        print("Using observed experimental kinematics (u_exp).")
    elif "u_true" in data:
        u_obs = data["u_true"]
        print("Using ground truth kinematics (u_true).")
    else:
        raise KeyError("Neither 'u_exp' nor 'u_true' found in dataset!")

    # -------------------------------------------------------------------------
    # LEVEL 1: Material Parameter Validation
    # -------------------------------------------------------------------------
    print("\n--- [Level 1] Validating Material Parameters (theta) ---")
    # Ground truth for Neo-Hookean (nh2): c10 = 0.5, d1 = 1.5
    theta_gt = np.zeros(selected_samples.shape[1])
    theta_gt[0] = 0.5 # c10
    if len(theta_gt) > 10:
        theta_gt[10] = 1.5 # d1

    level1_results = validate_material_parameters(selected_samples, theta_gt, output_dir)
    print(f"Level 1 RelError(theta): {level1_results['rel_error']:.2%}")
    print(f"Level 1 D_M_theta^2: {level1_results['dm2_theta']:.3f} (p = {level1_results['p_value']:.3f})")

    # -------------------------------------------------------------------------
    # LEVEL 2: Strain Energy Functional W(F) Calibration
    # -------------------------------------------------------------------------
    print("\n--- [Level 2] Validating Strain Energy Functional W(F) ---")
    # Synthetic grid of deformation gradients F (uniaxial tension lambda from 1.0 to 1.5)
    lambdas = np.linspace(1.0, 1.5, 50)
    F_grid = np.zeros((50, 3, 3))
    for i, lam in enumerate(lambdas):
        F_grid[i] = np.diag([lam, 1.0 / np.sqrt(lam), 1.0 / np.sqrt(lam)])

    # Compute predicted energy W(F) samples and GT energy W_gt(F)
    W_pred_samples = compute_synthetic_strain_energy_samples(selected_samples, F_grid)
    W_gt = compute_synthetic_strain_energy_samples(theta_gt[None, :], F_grid)[0]

    level2_results = validate_energy_functional(W_pred_samples, W_gt, output_dir)
    print(f"Level 2 Energy RMSE_W: {level2_results['rmse_W']:.4e} (Rel-RMSE: {level2_results['rel_rmse_W']:.2%})")
    print(f"Level 2 ECE_W: {level2_results['ece_W']:.4f}, PICP_95: {level2_results['picp_W_95']*100:.1f}%")

    # -------------------------------------------------------------------------
    # LEVEL 3: Full-Field Kinematics u & Reaction Forces R
    # -------------------------------------------------------------------------
    print("\n--- [Level 3] Validating Full-Field Kinematics & Reaction Forces ---")
    level3_results = validate_full_field_kinematics(
        U_pred_samples=u_pred_samples,
        U_obs=u_obs,
        node_coords=node_coords,
        cells=cells,
        sigma_dic=sigma_dic,
        output_dir=output_dir
    )
    print(f"Level 3 NRMSE_u: {level3_results['nrmse_u']:.2%}")
    print(f"Level 3 Subspace D_M_sub^2: {level3_results['subspace_mahalanobis']['dm2_sub']:.3f} (p = {level3_results['subspace_mahalanobis']['p_value']:.3f})")
    print(f"Level 3 Multivariate Energy Score: {level3_results['multivariate_energy_score']:.4e}")
    print(f"Level 3 PICP_95: {level3_results['marginal_calibration']['picp_95']*100:.1f}%, ECE: {level3_results['marginal_calibration']['ece']:.4f}")

    # -------------------------------------------------------------------------
    # ACCEPTANCE CRITERIA CHECKLIST
    # -------------------------------------------------------------------------
    checklist = evaluate_acceptance_checklist(level1_results, level2_results, level3_results)

    report = {
        "level1_material_parameters": level1_results,
        "level2_strain_energy": level2_results,
        "level3_full_field_kinematics": level3_results,
        "acceptance_checklist": checklist
    }

    report_path = output_dir / "multi_level_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n================================================================")
    print(f"📋 ACCEPTANCE CRITERIA CHECKLIST SUMMARY")
    print(f"================================================================")
    for item in checklist:
        status_symbol = "✅" if "PASSED" in item["status"] else "❌"
        print(f"{status_symbol} {item['check_item']:<48} | Target: {item['target']:<24} | Observed: {item['observed_value']:<20} | Status: {item['status']}")
        if "FAILED" in item["status"]:
            print(f"   ↳ Action: {item['diagnostic_action']}")
    print(f"================================================================")
    print(f"🎉 Full Multi-Level Validation Report saved to: {report_path}")
    print(f"   Plots and figures exported to: {output_dir}\n")

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Run Multi-Level Uncertainty Quantification Validation Framework")
    parser.add_argument('--distilled_dir', type=str, required=True, help="Path to distilled model folder")
    parser.add_argument('--subfolder', type=str, default="fem_validation", help="Subfolder containing fem_distilled_samples.npz")
    parser.add_argument('--sigma_dic', type=float, default=0.0001, help="DIC displacement noise standard deviation")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_multi_level_validation(args.distilled_dir, subfolder=args.subfolder, sigma_dic=args.sigma_dic)

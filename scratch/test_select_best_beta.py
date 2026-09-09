#!/usr/bin/env python3
"""
scratch/test_select_best_beta.py

Unit test for automated multi-beta model selection logic.
Creates mock candidate runs and verifies that select_best_beta.py correctly
identifies the best-calibrated model, copies files, and creates diagnostic plots.
"""

import os
import json
import shutil
import subprocess
import numpy as np

def run_test():
    test_root = "/home/mmdiscovery/shared/scratch/test_beta_selection"
    if os.path.exists(test_root):
        shutil.rmtree(test_root)
    os.makedirs(test_root)

    candidates_dir = os.path.join(test_root, "candidates")
    output_dir = os.path.join(test_root, "extracted")
    os.makedirs(candidates_dir)

    # Define mock candidates
    mock_data = [
        {"name": "beta_1.0", "beta": 1.0, "r2_val": 0.9967, "rmse_val": 0.1021, "ec_val": 5.4, "elbo": -9261.1},
        {"name": "beta_10.0", "beta": 10.0, "r2_val": 0.9920, "rmse_val": 0.1150, "ec_val": 42.1, "elbo": -8100.0},
        {"name": "beta_50.0", "beta": 50.0, "r2_val": 0.9850, "rmse_val": 0.1280, "ec_val": 93.8, "elbo": -7500.0},
        {"name": "beta_100.0", "beta": 100.0, "r2_val": 0.9710, "rmse_val": 0.1450, "ec_val": 98.1, "elbo": -7200.0},
        {"name": "beta_200.0", "beta": 200.0, "r2_val": 0.8200, "rmse_val": 0.3500, "ec_val": 99.5, "elbo": -6800.0}, # Low R2
    ]

    for c in mock_data:
        c_dir = os.path.join(candidates_dir, c["name"])
        os.makedirs(c_dir)
        metrics = {
            "beta": c["beta"],
            "r2_val": c["r2_val"],
            "rmse_val": c["rmse_val"],
            "ec_val": c["ec_val"],
            "r2_train": c["r2_val"] + 0.002,
            "ec_train": c["ec_val"] + 10.0,
            "elbo": c["elbo"],
            "kl": 88.0 * c["beta"],
            "phy": 0.03
        }
        with open(os.path.join(c_dir, "extraction_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
        with open(os.path.join(c_dir, "best_params.npy"), "wb") as f:
            np.save(f, {"dummy_key": c["beta"]})
        with open(os.path.join(c_dir, "config.yaml"), "w") as f:
            f.write(f"beta: {c['beta']}\n")

    # Run select_best_beta.py
    cmd = [
        "python3", "/home/mmdiscovery/shared/extraction/select_best_beta.py",
        "--candidates_dir", candidates_dir,
        "--output_dir", output_dir,
        "--target_ec", "95.0",
        "--min_r2", "0.90"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:")
    print(res.stdout)
    print("STDERR:")
    print(res.stderr)
    assert res.returncode == 0, f"select_best_beta failed with code {res.returncode}"

    # Verify summary JSON
    summary_path = os.path.join(output_dir, "beta_selection_summary.json")
    assert os.path.exists(summary_path), "Summary JSON missing in output_dir"
    with open(summary_path) as f:
        summary = json.load(f)

    print(f"Selected winner: {summary['selected_candidate']} (beta={summary['selected_beta']})")
    assert summary["selected_candidate"] == "beta_50.0", f"Expected beta_50.0 but got {summary['selected_candidate']}"
    assert summary["selected_beta"] == 50.0, f"Expected beta=50.0 but got {summary['selected_beta']}"

    # Verify files copied to output_dir
    assert os.path.exists(os.path.join(output_dir, "best_params.npy")), "best_params.npy missing in output_dir"
    assert os.path.exists(os.path.join(output_dir, "beta_calibration_curve.png")), "calibration png missing"
    assert os.path.exists(os.path.join(output_dir, "beta_calibration_curve.pdf")), "calibration pdf missing"

    # Verify copied dummy parameter corresponds to 50.0
    p = np.load(os.path.join(output_dir, "best_params.npy"), allow_pickle=True).item()
    assert p["dummy_key"] == 50.0, f"Expected copied file to have beta=50.0, got {p['dummy_key']}"

    print("🎉 ALL UNIT TESTS PASSED FOR select_best_beta.py!")

if __name__ == "__main__":
    run_test()


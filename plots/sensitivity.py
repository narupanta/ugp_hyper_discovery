"""
plots/sensitivity.py: Shared Utilities for Sobol Sensitivity Visualizations

Centralizes discovery of sensitivity CSV outputs, deformation gradient loading,
invariant extraction, and statistical aggregation across sensitivity plotting scripts.
"""

import os
from typing import Optional, Tuple, List, Dict
import pandas as pd
import numpy as np

from core.utils import compute_invariants_np, load_f3x3_from_distilled


def find_sensitivity_dir(distilled_dir: str, subdir: str) -> Optional[str]:
    """Locates the sensitivity results directory (supports both flat and output/ nested layouts)."""
    search_dirs = [
        os.path.join(distilled_dir, "output", subdir),
        os.path.join(distilled_dir, subdir)
    ]
    for d in search_dirs:
        if os.path.exists(d):
            return d
    return None


def load_sobol_csv(
    distilled_dir: str,
    subdir: str,
    filename: str = "total_sobol_indices_output_0.csv"
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Loads a Sobol index CSV file from the appropriate sensitivity directory."""
    s_dir = find_sensitivity_dir(distilled_dir, subdir)
    if not s_dir:
        return None, None
    csv_path = os.path.join(s_dir, filename)
    if not os.path.exists(csv_path):
        return None, None
    df = pd.read_csv(csv_path)
    return df, csv_path


def get_sensitivity_statistics(
    out_dir: str,
    prefix: str,
    subdir: str
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Aggregates first-order and total-order Sobol indices across output dimensions.
    Returns (tot_dict, first_dict).
    """
    valid_dir = find_sensitivity_dir(out_dir, subdir)
    if not valid_dir:
        return {}, {}

    dfs_tot = []
    dfs_first = []

    for i in range(4):
        for pfx in [prefix, ""]:
            p_tot = os.path.join(valid_dir, f"{pfx}total_sobol_indices_statistics_output_{i}.csv")
            p_first = os.path.join(valid_dir, f"{pfx}first_sobol_indices_statistics_output_{i}.csv")
            if os.path.exists(p_tot) and os.path.exists(p_first):
                dfs_tot.append(pd.read_csv(p_tot).fillna(0.0))
                dfs_first.append(pd.read_csv(p_first).fillna(0.0))
                break

    if not dfs_tot:
        return {}, {}

    param_cols = [c for c in dfs_tot[0].columns if c not in ["Unnamed: 0", ""]]

    tot_means_list = np.array([df.iloc[0][param_cols].values.astype(float) for df in dfs_tot])
    first_means_list = np.array([df.iloc[0][param_cols].values.astype(float) for df in dfs_first])

    tot_means = np.nanmean(tot_means_list, axis=0)
    first_means = np.nanmean(first_means_list, axis=0)

    tot_dict = {col: val for col, val in zip(param_cols, tot_means)}
    first_dict = {col: val for col, val in zip(param_cols, first_means)}

    return tot_dict, first_dict


def load_sensitivity_data(
    distilled_dir: str,
    component: str = "dev",
    distill_target: str = "sef",
    active_params: Optional[str] = None
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], List[str]]:
    """
    Unified loader that pairs Sobol CSV rows with corresponding deformation gradients and invariants.

    Returns:
        df: Filtered Sobol index DataFrame
        f3x3_kept: Matching deformation gradient array (N, 3, 3)
        invariants: (I1_bar, I2_bar, J)
        params: List of active parameter names
    """
    if distill_target == "sef_split":
        subdir = f"{component}_sensitivities"
    else:
        subdir = "sensitivities"

    df, csv_path = load_sobol_csv(distilled_dir, subdir, "total_sobol_indices_output_0.csv")
    if df is None:
        return None, None, None, []

    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]

    if active_params:
        active_list = [p.strip() for p in active_params.split(',') if p.strip() in params]
        if active_list:
            params = active_list

    f3x3 = load_f3x3_from_distilled(distilled_dir)
    if f3x3 is None:
        return df, None, None, params

    num_orig_points = f3x3.shape[0]
    if num_orig_points == len(df):
        f3x3_kept = f3x3
    else:
        # Filter out zero strain
        keep_mask = np.ones(num_orig_points, dtype=bool)
        identity_matrix = np.eye(3)
        is_zero_strain = np.max(np.max(np.abs(f3x3 - identity_matrix), axis=2), axis=1) < 1e-6
        keep_mask[is_zero_strain] = False
        f3x3_kept = f3x3[keep_mask]

        if len(f3x3_kept) != len(df):
            min_len = min(len(f3x3_kept), len(df))
            f3x3_kept = f3x3_kept[:min_len]
            df = df.iloc[:min_len]

    I1_bar, I2_bar, J = compute_invariants_np(f3x3_kept)
    return df, f3x3_kept, (I1_bar, I2_bar, J), params


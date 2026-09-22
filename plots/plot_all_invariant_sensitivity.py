"""
plots/plot_all_invariant_sensitivity.py: Unified scatter plots of Sobol indices vs strain invariants
combining all material components (deviatoric, volumetric, anisotropic) across all strain invariants.

Features:
- Subplots for isochoric invariants (I1_bar, I2_bar, J) and anisotropic invariants (I4_bar, I6_bar, I8_bar).
  * Automatically detects active anisotropic invariants (omits I8_bar when identically 0 for single-fiber or orthogonal fiber states).
- Color tone distinction by component:
  * Deviatoric parameters (C_ij, E): Blue tones
  * Volumetric parameters (D_k): Orange tones
  * Anisotropic parameters (C_4i, C_6i): Pink / Magenta tones
- Unified legend with LaTeX math symbols placed cleanly at the top.
- Publication-quality formatting adhering to repository styling.
"""

import os
import sys
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from plots.theme import apply_style, save_figure
from plots.sensitivity import load_sobol_csv
from core.utils import compute_invariants_np, load_f3x3_from_distilled
from core.fem_engine import compute_all_invariants


def to_latex(name: str) -> str:
    clean = name.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    if clean.startswith("C") and len(clean) == 3 and clean[1:].isdigit():
        return rf"$C_{{{clean[1:]}}}$"
    if clean.startswith("D") and len(clean) == 2 and clean[1:].isdigit():
        return rf"$D_{{{clean[1:]}}}$"
    if clean == "E":
        return r"$E$"
    return rf"${clean}$"


# Discrete Palette Maps for high contrast and elegance:
DEV_PALETTE = {
    "C10": "#08519c",  # Classic strong blue
    "C01": "#3182bd",  # True blue
    "C20": "#08306b",  # Dark navy blue
    "C11": "#009999",  # Ocean teal
    "C02": "#6baed6",  # Cornflower blue
    "C30": "#2171b5",  # Mid-dark royal blue
    "C21": "#4292c6",  # Sky blue
    "C12": "#006d2c",  # Accent dark sea green/blue-green
    "C03": "#9ecae1",  # Light soft ice blue
    "E":   "#02818a",  # Deep peacock blue
}

VOL_PALETTE = {
    "D1": "#d95f02",   # Vivid rust orange
    "D2": "#fdbb84",   # Soft warm peach/amber
    "D3": "#7f2704",   # Deep dark burnt orange
}

ANISO_PALETTE = {
    "C42": "#f768a1",  # Bright rose pink
    "C43": "#dd3497",  # Vivid magenta pink
    "C44": "#ae017e",  # Deep magenta
    "C62": "#e7298a",  # Vibrant magenta
    "C63": "#980043",  # Deep crimson wine
    "C64": "#67001f",  # Dark ruby
}


def get_param_color(name: str, comp_type: str = "dev", idx: int = 0, total_in_comp: int = 1) -> str:
    clean = name.replace("$", "").replace("{", "").replace("}", "").replace("_", "")
    if comp_type == "dev":
        if clean in DEV_PALETTE:
            return DEV_PALETTE[clean]
        return plt.cm.Blues(np.linspace(0.4, 0.95, max(total_in_comp, 1)))[idx]
    elif comp_type == "vol":
        if clean in VOL_PALETTE:
            return VOL_PALETTE[clean]
        return plt.cm.Oranges(np.linspace(0.45, 0.95, max(total_in_comp, 1)))[idx]
    elif comp_type == "aniso":
        if clean in ANISO_PALETTE:
            return ANISO_PALETTE[clean]
        return plt.cm.RdPu(np.linspace(0.4, 0.95, max(total_in_comp, 1)))[idx]
    return "#333333"


def load_fiber_vectors(distilled_dir: str):
    """Attempt to load fiber directions a0, a1 from configs or metadata."""
    angles = None
    candidate_paths = [
        os.path.join(distilled_dir, "config.yaml"),
        os.path.join(distilled_dir, "aniso_config.yaml"),
        os.path.join(distilled_dir, "..", "extracted", "config.yaml"),
        os.path.join(distilled_dir, "..", "config.yaml"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r") as f:
                    cfg = yaml.safe_load(f)
                    if isinstance(cfg, dict):
                        if "angles" in cfg and cfg["angles"]:
                            angles = cfg["angles"]
                            break
                        if "fiber_angle" in cfg:
                            angles = [cfg["fiber_angle"]]
                            break
                        if "fiber_angle1" in cfg:
                            angles = [cfg["fiber_angle1"]]
                            if "fiber_angle2" in cfg:
                                angles.append(cfg["fiber_angle2"])
                            break
            except Exception:
                continue

    if angles is None:
        angles = [30.0]

    th0 = np.radians(angles[0])
    a0 = np.array([np.cos(th0), np.sin(th0), 0.0])
    if len(angles) > 1:
        th1 = np.radians(angles[1])
        a1 = np.array([np.cos(th1), np.sin(th1), 0.0])
    else:
        # Default orthogonal complement for 1-fiber setting
        a1 = np.array([-np.sin(th0), np.cos(th0), 0.0])

    return a0, a1


def plot_all_invariant_sensitivity(
    distilled_dir: str,
    threshold: float = 1e-4,
    out_name: str = "all_invariant_sensitivity_scatter"
):
    apply_style()
    distilled_dir = os.path.abspath(distilled_dir)
    if not os.path.exists(os.path.join(distilled_dir, "f3x3.npy")) and os.path.isdir(os.path.join(distilled_dir, "distilled")):
        distilled_dir = os.path.join(distilled_dir, "distilled")

    # 1. Load f3x3
    f3x3 = load_f3x3_from_distilled(distilled_dir)
    if f3x3 is None:
        raise FileNotFoundError(f"Could not load f3x3.npy from {distilled_dir}")
    n_points = len(f3x3)

    # 2. Check available components and sensitivities
    df_dev, _ = load_sobol_csv(distilled_dir, "dev_sensitivities")
    if df_dev is None:
        df_dev, _ = load_sobol_csv(distilled_dir, "sensitivities")

    df_vol, _ = load_sobol_csv(distilled_dir, "vol_sensitivities")
    df_aniso, _ = load_sobol_csv(distilled_dir, "aniso_sensitivities")

    has_dev = df_dev is not None
    has_vol = df_vol is not None
    has_aniso = df_aniso is not None

    if not has_dev and not has_vol and not has_aniso:
        print(f"No sensitivity data found in {distilled_dir}")
        return

    exclude_cols = ['test cases', 'Unnamed: 0']
    components_data = []

    if has_dev:
        df_dev = df_dev.iloc[:n_points]
        dev_cols = [c for c in df_dev.columns if c not in exclude_cols and not c.startswith('Unnamed')]
        active_dev = [c for c in dev_cols if df_dev[c].max() > threshold]
        if not active_dev and dev_cols:
            active_dev = [dev_cols[0]]
        components_data.append(('Deviatoric', 'dev', df_dev, active_dev))

    if has_vol:
        df_vol = df_vol.iloc[:n_points]
        vol_cols = [c for c in df_vol.columns if c not in exclude_cols and not c.startswith('Unnamed')]
        active_vol = [c for c in vol_cols if df_vol[c].max() > threshold]
        if not active_vol and vol_cols:
            active_vol = [vol_cols[0]]
        components_data.append(('Volumetric', 'vol', df_vol, active_vol))

    if has_aniso:
        df_aniso = df_aniso.iloc[:n_points]
        aniso_cols = [c for c in df_aniso.columns if c not in exclude_cols and not c.startswith('Unnamed')]
        active_aniso = [c for c in aniso_cols if df_aniso[c].max() > threshold]
        if not active_aniso and aniso_cols:
            active_aniso = [aniso_cols[0]]
        components_data.append(('Anisotropic', 'aniso', df_aniso, active_aniso))

    # 3. Compute Invariants
    if has_aniso:
        a0, a1 = load_fiber_vectors(distilled_dir)
        inv_dict = compute_all_invariants(f3x3, a0=a0, a1=a1)
        invariants = [
            (r"$\bar{I}_1$", inv_dict["I1_bar"]),
            (r"$\bar{I}_2$", inv_dict["I2_bar"]),
            (r"$J$", inv_dict["J"]),
            (r"$\bar{I}_4$", inv_dict["I4_bar"]),
            (r"$\bar{I}_6$", inv_dict["I6_bar"]),
        ]
        # Include I8_bar only if it has non-zero variation
        I8_vals = inv_dict["I8_bar"]
        if np.ptp(I8_vals) > 1e-6 or np.max(np.abs(I8_vals)) > 1e-4:
            invariants.append((r"$\bar{I}_8$", I8_vals))

        n_inv = len(invariants)
        if n_inv == 6:
            n_rows, n_cols = 2, 3
            figsize = (18, 9)
        elif n_inv == 5:
            # 2 rows: row 1 has 3, row 2 has 2
            n_rows, n_cols = 2, 3
            figsize = (18, 9)
        else:
            n_rows, n_cols = 1, n_inv
            figsize = (6 * n_inv, 5)
    else:
        I1_bar, I2_bar, J = compute_invariants_np(f3x3)
        invariants = [
            (r"$\bar{I}_1$", I1_bar),
            (r"$\bar{I}_2$", I2_bar),
            (r"$J$", J),
        ]
        n_rows, n_cols = 1, 3
        figsize = (18, 5)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    # If 5 subplots in a 2x3 grid, hide the 6th subplot
    if n_rows * n_cols > len(invariants):
        for unused_idx in range(len(invariants), n_rows * n_cols):
            fig.delaxes(axes_flat[unused_idx])

    # Collect legend handles
    legend_handles = []
    legend_labels = []

    for comp_name, comp_type, df, active_params in components_data:
        for idx, p in enumerate(active_params):
            color = get_param_color(p, comp_type=comp_type, idx=idx, total_in_comp=len(active_params))
            handle = mlines.Line2D([], [], color=color, marker='o', linestyle='none', markersize=6)
            legend_handles.append(handle)
            legend_labels.append(to_latex(p))

    # Plot each invariant panel
    for i, (inv_name, inv_vals) in enumerate(invariants):
        ax = axes_flat[i]
        
        # Plot each component parameter
        for comp_name, comp_type, df, active_params in components_data:
            for idx, p in enumerate(active_params):
                color = get_param_color(p, comp_type=comp_type, idx=idx, total_in_comp=len(active_params))
                y_vals = df[p].values
                ax.scatter(inv_vals, y_vals, color=color, alpha=0.65, s=24, edgecolors='none')

        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_title(f"Sensitivity vs {inv_name}", fontsize=13)
        ax.set_xlabel(inv_name, fontsize=12)
        ax.set_ylabel(r"Total-Order Sobol Index ($S_T$)", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)

    # Place unified legend at the top
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.05 if n_rows == 2 else 1.08),
            ncol=min(len(legend_handles), 10),
            fontsize=11,
            framealpha=0.9,
            handletextpad=0.2,
            columnspacing=1.0
        )

    plt.tight_layout()

    out_prefix = out_name if out_name else "all_invariant_sensitivity_scatter"
    out_pdf = os.path.join(distilled_dir, f"{out_prefix}.pdf")
    out_png = os.path.join(distilled_dir, f"{out_prefix}.png")
    save_figure(fig, out_pdf)
    save_figure(fig, out_png)
    plt.close(fig)
    print(f"Unified invariant sensitivity plot saved to: {out_png}")


def main():
    parser = argparse.ArgumentParser(description="Plot unified invariant-dependent Sobol indices across all components")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to distilled model directory")
    parser.add_argument("--threshold", type=float, default=1e-4, help="Sobol index threshold for active parameters")
    parser.add_argument("--out_name", type=str, default="all_invariant_sensitivity_scatter", help="Base name of output plot")
    args = parser.parse_args()

    plot_all_invariant_sensitivity(args.distilled_dir, threshold=args.threshold, out_name=args.out_name)


if __name__ == "__main__":
    main()


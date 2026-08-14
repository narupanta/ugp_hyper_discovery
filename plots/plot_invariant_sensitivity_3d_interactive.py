import os
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.colors as pcolors

def compute_invariants(F):
    J = np.linalg.det(F)
    F_T = np.transpose(F, (0, 2, 1))
    C = np.einsum('nij,njk->nik', F_T, F)
    J_safe = np.clip(J, 1e-8, 1e8)
    C_bar = C / (J_safe**(2/3))[:, None, None]
    
    I1_bar = np.trace(C_bar, axis1=1, axis2=2)
    C_bar_sq = np.einsum('nij,njk->nik', C_bar, C_bar)
    I2_bar = 0.5 * (I1_bar**2 - np.trace(C_bar_sq, axis1=1, axis2=2))
    
    return I1_bar, I2_bar, J

def main():
    parser = argparse.ArgumentParser(description="Interactive Plot invariant-dependent Sobol indices in 3D pairs")
    parser.add_argument("--distilled_dir", type=str, required=True, help="Path to the distilled model directory")
    parser.add_argument("--active_params", type=str, default=None, help="Comma-separated list of active parameters")
    parser.add_argument("--component", type=str, default="dev", choices=["dev", "vol"], help="Component for split model")
    parser.add_argument("--distill_target", type=str, default="sef", choices=["sef", "sef_stress", "sef_cauchy", "sef_split"])
    args = parser.parse_args()
    
    if args.distill_target == "sef_split":
        sens_dir_name = f"{args.component}_sensitivities"
        source_txt_name = f"{args.component}_source_extraction_dir.txt"
        img_prefix = f"{args.component}_"
    else:
        sens_dir_name = "sensitivities"
        source_txt_name = "source_extraction_dir.txt"
        img_prefix = ""

    csv_path = os.path.join(args.distilled_dir, "output", sens_dir_name, "total_sobol_indices_output_0.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(args.distilled_dir, sens_dir_name, "total_sobol_indices_output_0.csv")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    exclude_cols = ['test cases', 'Unnamed: 0']
    params = [c for c in df.columns if c not in exclude_cols and not c.startswith('Unnamed')]
    
    source_txt = os.path.join(args.distilled_dir, source_txt_name)
    if not os.path.exists(source_txt):
        print(f"Error: {source_txt} not found.")
        return
        
    with open(source_txt, 'r') as f:
        source_dir = f.read().strip()
        
    f3x3_path = None
    best_f3x3 = None
    
    for root, dirs, files in os.walk(source_dir):
        if "f3x3.npy" in files:
            candidate_path = os.path.join(root, "f3x3.npy")
            f3x3_candidate = np.load(candidate_path)
            num_orig_points = f3x3_candidate.shape[0]
            
            if num_orig_points == len(df):
                f3x3_path = candidate_path
                best_f3x3 = f3x3_candidate
                break
                
            dropped_indices = [0, 25, 39]
            keep_mask = np.ones(num_orig_points, dtype=bool)
            if num_orig_points > max(dropped_indices):
                keep_mask[dropped_indices] = False
            
            identity_matrix = np.eye(3)
            is_zero_strain = np.max(np.max(np.abs(f3x3_candidate - identity_matrix), axis=2), axis=1) < 1e-6
            keep_mask[is_zero_strain] = False
            
            f3x3_kept = f3x3_candidate[keep_mask]
            
            if len(f3x3_kept) == len(df):
                f3x3_path = candidate_path
                best_f3x3 = f3x3_kept
                break
                
            if best_f3x3 is None:
                best_f3x3 = f3x3_kept

    if best_f3x3 is None:
        print(f"Error: f3x3.npy not found in {source_dir}")
        return
        
    f3x3_kept = best_f3x3
    
    if len(f3x3_kept) != len(df):
        print(f"Warning: Number of remaining f3x3 points ({len(f3x3_kept)}) does not match CSV rows ({len(df)})!")
        min_len = min(len(f3x3_kept), len(df))
        f3x3_kept = f3x3_kept[:min_len]
        df = df.iloc[:min_len]
        
    I1_bar, I2_bar, J = compute_invariants(f3x3_kept)
    
    I1_bar_m3 = I1_bar - 3.0
    I2_bar_m3 = I2_bar - 3.0
    J_m1_sq = (J - 1.0)**2
    
    if args.active_params:
        active_params = [p.strip() for p in args.active_params.split(',') if p.strip() in df.columns]
    else:
        active_params = []
        for param in params:
            if np.max(np.abs(df[param].values)) > 1e-4:
                active_params.append(param)
            
    num_params = len(active_params)
    if num_params == 0:
        print("No active parameters found with sensitivity > 1e-4.")
        return
        
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(
            "Sensitivity vs I1_bar - 3 and I2_bar - 3",
            "Sensitivity vs I1_bar - 3 and (J - 1)^2",
            "Sensitivity vs I2_bar - 3 and (J - 1)^2"
        )
    )
    
    pairs = [
        (I1_bar_m3, "I1_bar - 3", I2_bar_m3, "I2_bar - 3"),
        (I1_bar_m3, "I1_bar - 3", J_m1_sq, "(J - 1)^2"),
        (I2_bar_m3, "I2_bar - 3", J_m1_sq, "(J - 1)^2")
    ]
    
    colors = pcolors.qualitative.Plotly
    
    for c_idx, (x_data, x_label, y_data, y_label) in enumerate(pairs):
        for r_idx, param in enumerate(active_params):
            y_vals = df[param].values
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_data,
                    y=y_data,
                    z=y_vals,
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors[r_idx % len(colors)],
                        opacity=0.7
                    ),
                    name=param,
                    showlegend=(c_idx == 0) # Only show legend once per parameter
                ),
                row=1, col=c_idx + 1
            )
            
        fig.update_layout(**{
            f"scene{c_idx + 1 if c_idx > 0 else ''}": dict(
                xaxis_title=x_label,
                yaxis_title=y_label,
                zaxis_title="Sensitivity (S_T)"
            )
        })

    fig.update_layout(
        title="Interactive 3D Invariant Sensitivity",
        height=600,
        width=1800,
        margin=dict(l=0, r=0, b=0, t=50)
    )

    out_path = os.path.join(args.distilled_dir, f"{img_prefix}invariant_sensitivity_3d_interactive.html")
    fig.write_html(out_path)
    print(f"Interactive 3D plot saved to: {out_path}")

if __name__ == "__main__":
    main()

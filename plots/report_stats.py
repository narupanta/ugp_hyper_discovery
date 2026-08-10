import numpy as np
import pandas as pd
import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.material_models import get_material

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--saved_model_dir", type=str, required=True)
    args = parser.parse_args()
    
    out_dir = args.distilled_dir
    
    samples_np = np.load(os.path.join(out_dir, "flow_samples.npy"))
    full_param_names = ("C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "D1", "D2", "D3")
    df = pd.DataFrame(samples_np, columns=full_param_names)
    
    csv_dir = os.path.join(out_dir, "output", "sensitivities")
    if not os.path.exists(os.path.join(csv_dir, "total_sobol_indices_statistics_output_0.csv")):
        csv_dir = os.path.join(out_dir, "sensitivities")
    
    dfs = []
    for i in range(4):
        p = os.path.join(csv_dir, f"total_sobol_indices_statistics_output_{i}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            
    if dfs:
        param_cols = [c for c in dfs[0].columns if c not in ["Unnamed: 0", ""]]
        means_list = np.array([df_sub.iloc[0][param_cols].values.astype(float) for df_sub in dfs])
        mean_total_indices_outputs = np.nan_to_num(np.nanmean(means_list, axis=0), nan=0.0)
        relevant_parameter_indices = np.where(mean_total_indices_outputs >= 1e-4)[0].tolist()
        active_parameter_names = [full_param_names[i] for i in relevant_parameter_indices]
    else:
        active_parameter_names = full_param_names
        
    df_active = df[active_parameter_names]
    
    model_folder_name = os.path.basename(os.path.normpath(args.saved_model_dir))
    parts = model_folder_name.split('_')
    true_model_name = parts[1] if len(parts) > 1 else "isihara"
    true_model = get_material(true_model_name, jit_P=False)
    
    true_params = {}
    if true_model_name == "isihara":
        true_params = {"C10": true_model.c10, "C01": true_model.c01, "C20": true_model.c20, "D1": true_model.d1}
    elif true_model_name in ["nh", "neohookean2", "nh2"]:
        true_params = {"C10": true_model.dev_params[0], "D1": true_model.vol_params[0]}
    elif true_model_name in ["gentthomas"]:
        true_params = {"C10": true_model.dev_params[0], "CL2": true_model.dev_params[10], "D1": true_model.vol_params[0]}
    
    print("="*75)
    print(f"PARAMETER STATISTICS (Mean & 95% CI vs Ground Truth)")
    print("="*75)
    print(f"{'Parameter':<10} | {'True Value':<10} | {'Mean':<10} | {'95% CI':<30}")
    print("-" * 75)
    for col in active_parameter_names:
        true_val = true_params.get(col, 0.0)
        mean_val = df_active[col].mean()
        ci_lower = np.percentile(df_active[col], 2.5)
        ci_upper = np.percentile(df_active[col], 97.5)
        ci_str = f"[{ci_lower:.4f}, {ci_upper:.4f}]"
        print(f"{col:<10} | {true_val:<10.4f} | {mean_val:<10.4f} | {ci_str:<30}")
    
    print("\n" + "="*75)
    print("PARAMETER CORRELATION MATRIX (Pearson)")
    print("="*75)
    corr_matrix = df_active.corr()
    
    print(f"{'':<8}", end="")
    for col in active_parameter_names:
        print(f"{col:>8}", end="")
    print("\n" + "-" * (8 + 8 * len(active_parameter_names)))
    
    for row_col in active_parameter_names:
        print(f"{row_col:<8}", end="")
        for col in active_parameter_names:
            val = corr_matrix.loc[row_col, col]
            print(f"{val:>8.3f}", end="")
        print()

if __name__ == "__main__":
    main()

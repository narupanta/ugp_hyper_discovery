import os
import glob
import argparse
import numpy as np

def merge_worker_files(folder_path, pattern="fem_distilled_samples_worker*.npz"):
    worker_files = sorted(glob.glob(os.path.join(folder_path, pattern)))
    if not worker_files:
        print(f"No worker files found matching {pattern} in {folder_path}")
        return

    print(f"Found {len(worker_files)} worker files to merge in {folder_path}...")
    
    all_u_preds = []
    all_selected_samples = []
    base_dict = {}

    for wf in worker_files:
        d = np.load(wf, allow_pickle=True)
        if "u_pred" in d and len(d["u_pred"]) > 0:
            all_u_preds.append(d["u_pred"])
            all_selected_samples.append(d["selected_samples"])
            if not base_dict:
                for k in ["node_coords", "cells", "node_type", "loads", "u_true", "u_exp"]:
                    if k in d:
                        base_dict[k] = d[k]

    if not all_u_preds:
        print("No samples found across worker files!")
        return

    combined_u = np.concatenate(all_u_preds, axis=0)
    combined_params = np.concatenate(all_selected_samples, axis=0)

    # Filter any potential duplicate parameter realizations if any
    unique_indices = []
    for i, p in enumerate(combined_params):
        if not any(np.allclose(p, combined_params[prev], atol=1e-7) for prev in unique_indices):
            unique_indices.append(i)
            
    final_u = combined_u[unique_indices]
    final_params = combined_params[unique_indices]

    target_file = os.path.join(folder_path, "fem_distilled_samples.npz")
    save_dict = {
        "u_pred": final_u,
        "selected_samples": final_params,
        **base_dict
    }

    np.savez_compressed(target_file, **save_dict)
    print(f"🎉 Successfully merged {len(worker_files)} worker files into {target_file}!")
    print(f"Total unique non-repeated samples: {final_u.shape[0]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    args = parser.parse_args()
    merge_worker_files(args.folder)

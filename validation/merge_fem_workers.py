import os
import glob
import argparse
import numpy as np

def merge_worker_files(folder_path, pattern="fem_distilled_samples_worker*.npz", keep_shards=False):
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
                for k in ["node_coords", "cells", "node_type", "loads", "schedule_solve", "control_mode", "stress_mode", "u_true", "u_exp", "lam3", "lam3_true"]:
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
        "u_pred": final_u.astype(np.float32),
        "selected_samples": final_params,
        "merged_complete": True,
        **base_dict
    }

    np.savez_compressed(target_file, **save_dict)
    print(f"🎉 Successfully merged {len(worker_files)} worker files into {target_file}!")
    print(f"Total unique non-repeated samples: {final_u.shape[0]}")

    # Shards duplicate the merged file; remove them once the merged file reads back intact.
    if not keep_shards:
        check = np.load(target_file, allow_pickle=True)
        if check["u_pred"].shape == final_u.shape and bool(check["merged_complete"]):
            for wf in worker_files:
                os.remove(wf)
            print(f"🧹 Removed {len(worker_files)} worker shard files (pass --keep_shards to keep them).")
        else:
            print("⚠️ Merged file failed read-back verification; keeping worker shards.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True)
    parser.add_argument("--keep_shards", action="store_true", help="Keep fem_distilled_samples_worker*.npz after merging")
    args = parser.parse_args()
    merge_worker_files(args.folder, keep_shards=args.keep_shards)

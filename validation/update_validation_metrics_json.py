import os
import glob
import json
import argparse
import numpy as np
from sklearn.metrics import r2_score

DEV_PARAM_NAMES = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]
VOL_PARAM_NAMES = ["D1", "D2", "D3"]
ANISO_PARAM_NAMES = ["C42", "C43", "C44", "C62", "C63", "C64"]

def compute_displacement_metrics(distilled_dir, subfolder, step_indices=[9]):
    file_path = os.path.join(distilled_dir, subfolder, "fem_distilled_samples.npz")
    if not os.path.exists(file_path):
        return None

    data = np.load(file_path, allow_pickle=True)
    if "u_pred" not in data or ("u_true" not in data and "u_exp" not in data):
        return None

    u_pred = data["u_pred"]  # (N, n_steps, n_nodes, 2)
    u_true_all = data["u_exp"] if "u_exp" in data else data["u_true"]

    # Filter and validate indices
    if isinstance(step_indices, int):
        step_indices = [step_indices]
    valid_steps = [s for s in step_indices if s < u_pred.shape[1]]
    if not valid_steps:
        valid_steps = [u_pred.shape[1] - 1]

    # Concatenate across all evaluation steps
    # Shape: (N, len(valid_steps) * n_nodes, 2)
    u_pred_steps = u_pred[:, valid_steps]
    N, S, num_nodes, D = u_pred_steps.shape
    u_pred_flat = u_pred_steps.reshape(N, S * num_nodes, D)

    # Shape: (len(valid_steps) * n_nodes, 2)
    u_true_steps = u_true_all[valid_steps]
    u_true_flat = u_true_steps.reshape(S * num_nodes, D)

    u_pred_mag = np.linalg.norm(u_pred_flat, axis=-1)  # (N, S * num_nodes)
    u_true_mag = np.linalg.norm(u_true_flat, axis=-1)  # (S * num_nodes,)

    ux_true = u_true_flat[:, 0]
    uy_true = u_true_flat[:, 1]
    umag_true = u_true_mag

    ux_pred_samples = u_pred_flat[:, :, 0]
    uy_pred_samples = u_pred_flat[:, :, 1]
    umag_pred_samples = u_pred_mag

    def get_comp_stats(y_true, y_samples):
        y_mean = np.mean(y_samples, axis=0)
        y_low, y_high = np.quantile(y_samples, [0.025, 0.975], axis=0)

        rmse = float(np.sqrt(np.mean((y_true - y_mean) ** 2)))
        r2 = float(r2_score(y_true, y_mean))
        inside = (y_true >= y_low) & (y_true <= y_high)
        cov = float(np.mean(inside) * 100.0)

        return {
            "rmse": rmse,
            "r2": r2,
            "coverage": cov
        }

    return {
        "ux": get_comp_stats(ux_true, ux_pred_samples),
        "uy": get_comp_stats(uy_true, uy_pred_samples),
        "norm": get_comp_stats(umag_true, umag_pred_samples)
    }

def extract_model_structure(distilled_dir):
    dev_file = os.path.join(distilled_dir, "dev_flow_samples.npy")
    vol_file = os.path.join(distilled_dir, "vol_flow_samples.npy")
    aniso_file = os.path.join(distilled_dir, "aniso_flow_samples.npy")

    model_structure = {}

    def process_file(file_path, param_names):
        if not os.path.exists(file_path):
            return
        samples = np.load(file_path)
        for i, name in enumerate(param_names[:samples.shape[1]]):
            col = samples[:, i]
            mean = float(np.mean(col))
            low = float(np.percentile(col, 2.5))
            high = float(np.percentile(col, 97.5))
            # Check if parameter survived (not identically zero / pruned)
            is_survived = not (np.isclose(mean, 0.0, atol=1e-8) and np.isclose(low, 0.0, atol=1e-8) and np.isclose(high, 0.0, atol=1e-8))
            if is_survived:
                model_structure[name] = {
                    "mean": mean,
                    "95ci_lower": low,
                    "95ci_upper": high
                }

    process_file(dev_file, DEV_PARAM_NAMES)
    process_file(vol_file, VOL_PARAM_NAMES)
    process_file(aniso_file, ANISO_PARAM_NAMES)

    return model_structure

def update_metrics(distilled_dir, step_idx=9):
    # Locate validation_metrics_*.json or create validation_metrics.json
    metric_files = glob.glob(os.path.join(distilled_dir, "validation_metrics_*.json"))
    if metric_files:
        val_json_path = metric_files[0]
    else:
        val_json_path = os.path.join(distilled_dir, "validation_metrics.json")

    existing_data = {}
    if os.path.exists(val_json_path):
        with open(val_json_path, "r") as f:
            try:
                existing_data = json.load(f)
            except Exception:
                existing_data = {}

    # 1. Strain Energy Function (SEF) metrics formatted as sef -> {gp, dist} -> {dev, vol, total}
    sef_data = {"gp": {}, "dist": {}}
    raw_sef = existing_data.get("sef", existing_data)

    # Check if raw_sef is already split into gp/dist or has dev/vol keys directly
    if "gp" in raw_sef and "dist" in raw_sef:
        sef_data = raw_sef
    else:
        for comp in ["dev", "vol", "aniso", "total"]:
            if comp in raw_sef:
                c_data = raw_sef[comp]
                # Distilled metrics
                sef_data["dist"][comp] = {
                    "rmse": c_data.get("rmse_dist", c_data.get("rmse")),
                    "r2": c_data.get("r2_dist", c_data.get("r2")),
                    "coverage": c_data.get("coverage_dist", c_data.get("coverage"))
                }
                # GP metrics
                sef_data["gp"][comp] = {
                    "rmse": c_data.get("rmse_gp"),
                    "r2": c_data.get("r2_gp"),
                    "coverage": c_data.get("coverage_gp")
                }

    # 2. Displacement Field metrics (Block and Holes)
    disp_block = compute_displacement_metrics(distilled_dir, "fem_validation", step_indices=val_steps)
    disp_holes = compute_displacement_metrics(distilled_dir, "fem_validation_holes", step_indices=val_steps)

    # 3. Discovered Model Structure (only survived parameters with 95ci)
    model_struct = extract_model_structure(distilled_dir)

    # 4. Execution Time (GP, Distillation Stages, and FEM)
    # 4.1 GP Extraction Time
    gp_time = None
    src_file = os.path.join(distilled_dir, "dev_source_extraction_dir.txt")
    if os.path.exists(src_file):
        with open(src_file, "r") as f:
            src_path = f.read().strip()
        metrics_file = os.path.join(src_path, "extraction_metrics.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    gp_time = json.load(f).get("extraction_time")
            except Exception:
                pass

    # 4.2 Distillation Times
    import re
    def parse_dist_log(log_name):
        log_path = os.path.join(distilled_dir, log_name)
        times = {"stage1_sec": None, "sensitivity_sec": None, "stage2_sec": None, "total_sec": None}
        if not os.path.exists(log_path):
            return times
        with open(log_path, "r") as f:
            content = f.read()
        m1 = re.search(r"Stage 1.*?:\s*[\d.]+\s*min\s*\(\s*([\d.]+)\s*sec\)", content)
        if m1: times["stage1_sec"] = float(m1.group(1))
        m2 = re.search(r"Stage 2.*?:\s*[\d.]+\s*min\s*\(\s*([\d.]+)\s*sec\)", content)
        if m2: times["sensitivity_sec"] = float(m2.group(1))
        m3 = re.search(r"Stage 3.*?:\s*[\d.]+\s*min\s*\(\s*([\d.]+)\s*sec\)", content)
        if m3: times["stage2_sec"] = float(m3.group(1))
        mt = re.search(r"Total Pipeline Duration.*?:\s*[\d.]+\s*min\s*\(\s*([\d.]+)\s*sec\)", content)
        if mt: times["total_sec"] = float(mt.group(1))
        return times

    dev_times = parse_dist_log("dev_distillation_log.txt")
    vol_times = parse_dist_log("vol_distillation_log.txt")
    aniso_times = parse_dist_log("aniso_distillation_log.txt")

    dist_time = {
        "stage1_sec": max([t["stage1_sec"] for t in [dev_times, vol_times, aniso_times] if t["stage1_sec"] is not None] or [0.0]),
        "sensitivity_sec": max([t["sensitivity_sec"] for t in [dev_times, vol_times, aniso_times] if t["sensitivity_sec"] is not None] or [0.0]),
        "stage2_sec": max([t["stage2_sec"] for t in [dev_times, vol_times, aniso_times] if t["stage2_sec"] is not None] or [0.0]),
        "total_sec": max([t["total_sec"] for t in [dev_times, vol_times, aniso_times] if t["total_sec"] is not None] or [0.0]),
        "components": {
            "dev": dev_times,
            "vol": vol_times
        }
    }
    if os.path.exists(os.path.join(distilled_dir, "aniso_distillation_log.txt")):
        dist_time["components"]["aniso"] = aniso_times

    # 4.3 FEM Validation Time
    def parse_fem_time(folder_name):
        fem_dir = os.path.join(distilled_dir, folder_name)
        if not os.path.exists(fem_dir): return None
        npz_file = os.path.join(fem_dir, "fem_distilled_samples.npz")
        if os.path.exists(npz_file):
            d = np.load(npz_file, allow_pickle=True)
            if "fem_time_sec" in d and float(d["fem_time_sec"]) > 0:
                return float(d["fem_time_sec"])
        wlogs = glob.glob(os.path.join(fem_dir, "worker_*.log"))
        max_duration = 0.0
        for wl in wlogs:
            try:
                with open(wl, "r") as f:
                    lines = [l for l in f if "[INFO] jax_fem:" in l]
                    if len(lines) >= 2:
                        import datetime
                        def parse_t(line):
                            m = re.search(r"\[09-(\d+) (\d+:\d+:\d+)\]", line)
                            if m: return datetime.datetime.strptime(f"2026-09-{m.group(1)} {m.group(2)}", "%Y-%m-%d %H:%M:%S")
                            return None
                        t0 = parse_t(lines[0])
                        t1 = parse_t(lines[-1])
                        if t0 and t1:
                            dur = (t1 - t0).total_seconds()
                            max_duration = max(max_duration, dur)
            except Exception:
                pass
        return max_duration if max_duration > 0 else None

    block_time = parse_fem_time("fem_validation")
    holes_time = parse_fem_time("fem_validation_holes")
    fem_time = {
        "block_sec": block_time,
        "holes_sec": holes_time,
        "total_sec": max([t for t in [block_time, holes_time] if t is not None] or [0.0])
    }

    time_taken = {
        "gp_sec": gp_time,
        "dist": dist_time,
        "fem": fem_time
    }

    # Build master unified dictionary
    unified_dict = {
        "time_taken": time_taken,
        "sef": sef_data,
        "disp": {
            "block": disp_block,
            "holes": disp_holes
        },
        "model_structure": model_struct
    }

    # Save to validation_metrics_{material}.json
    with open(val_json_path, "w") as f:
        json.dump(unified_dict, f, indent=4)
    print(f"✅ Updated master metrics JSON: {val_json_path}")

    # Also save to generic validation_metrics.json for universal access
    generic_path = os.path.join(distilled_dir, "validation_metrics.json")
    if generic_path != val_json_path:
        with open(generic_path, "w") as f:
            json.dump(unified_dict, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--distilled_dir", type=str, required=True)
    parser.add_argument("--step_idx", type=int, nargs="*", default=[9])
    parser.add_argument("--val_load_steps", type=int, nargs="*", default=None)
    args = parser.parse_args()
    val_steps = args.val_load_steps if args.val_load_steps is not None else args.step_idx
    update_metrics(args.distilled_dir, val_steps)

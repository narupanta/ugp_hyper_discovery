import os
import json
import argparse
import numpy as np
import re

def parse_distillation_log(log_path):
    times = {'stage1': None, 'stage2': None, 'stage3': None, 'total': None}
    if not os.path.exists(log_path): return times
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'([\d.]+)\s*min', line)
            if not match: continue
            val = float(match.group(1))
            
            if 'Stage 1' in line:
                times['stage1'] = val
            elif 'Stage 2' in line:
                times['stage2'] = val
            elif 'Stage 3' in line:
                times['stage3'] = val
            elif 'Total Pipeline Duration' in line:
                times['total'] = val
    return times

def calculate_param_stats(npy_path):
    if not os.path.exists(npy_path): return []
    samples = np.load(npy_path)
    stats = []
    num_params = samples.shape[1]
    for i in range(num_params):
        stats.append({
            "mean": float(np.mean(samples[:, i])),
            "ci_lower": float(np.percentile(samples[:, i], 2.5)),
            "ci_upper": float(np.percentile(samples[:, i], 97.5))
        })
    return stats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist_dir", type=str, required=True)
    parser.add_argument("--val_json", type=str, default=None)
    args = parser.parse_args()

    dist_dir = args.dist_dir
    times = parse_distillation_log(os.path.join(dist_dir, "dev_distillation_log.txt"))
    dev_stats = calculate_param_stats(os.path.join(dist_dir, "dev_flow_samples.npy"))
    vol_stats = calculate_param_stats(os.path.join(dist_dir, "vol_flow_samples.npy"))

    val_metrics = None
    if args.val_json and os.path.exists(args.val_json):
        with open(args.val_json, 'r') as f:
            val_metrics = json.load(f)

    summary = {
        "distillation_time_min": times,
        "validation_metrics": val_metrics,
        "discovered_parameters": {"dev": dev_stats, "vol": vol_stats}
    }

    out_path = os.path.join(dist_dir, "distillation_metrics.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"✅ Saved master metrics to {out_path}")

if __name__ == "__main__":
    main()

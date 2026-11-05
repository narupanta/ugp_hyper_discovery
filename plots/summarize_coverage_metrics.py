#!/usr/bin/env python3
"""
Energy accuracy of a coverage study: GP posterior (one per noise seed) and distilled model (one per seed x init),
from each run's validation_metrics.json (written by plots/plot_distilled_r2_energy.py: strain energy on the elements
of the validation load steps, against the true material).

Writes <study>/metrics_summary.md with
  - overview: mean / std of RMSE, R^2 and 95% coverage (EC) for the GP (n = seeds) and the distilled model (n = runs);
  - per noise seed: the GP values and the distilled mean / std over that seed's initialisations;
  - all distilled runs.
The GP of a seed is the same for all its initialisations (the values of the first one found are used; a mismatch
between initialisations is reported).

Usage: python3 plots/summarize_coverage_metrics.py <experiment>/coverage_study_<...> [--component total|dev|vol|aniso]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

KEYS = (("rmse", "RMSE"), ("r2", "R²"), ("coverage", "95% EC (%)"))


def read(run_dir, component):
    p = os.path.join(run_dir, "validation_metrics.json")
    if not os.path.exists(p):
        return None
    sef = json.load(open(p)).get("sef", {})
    if "gp" not in sef or "dist" not in sef:
        return None
    return {src: {k: float(sef[src][component][k]) for k, _ in KEYS} for src in ("gp", "dist")}


def fmt(x, k):
    return f"{x:.4f}" if k != "coverage" else f"{x:.1f}"


def ms(vals, k):
    v = np.asarray(vals, dtype=float)
    return f"{fmt(v.mean(), k)} ± {fmt(v.std(ddof=1) if len(v) > 1 else 0.0, k)}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("study_dir")
    p.add_argument("--component", default="total", help="energy part: total (default), dev, vol, aniso")
    args = p.parse_args()
    study = os.path.abspath(args.study_dir)
    runs = {}
    for d in glob.glob(os.path.join(study, "seed*", "init*")):
        m = re.search(r"seed(\d+)[/\\]init(\d+)$", d)
        r = read(d, args.component) if m else None
        if r is not None:
            runs[(int(m.group(1)), int(m.group(2)))] = r
    if not runs:
        raise SystemExit(f"no validation_metrics.json with gp/dist '{args.component}' entries under {study}")
    seeds = sorted({s for s, _ in runs})
    gp, notes = {}, []
    for s in seeds:
        inits = sorted(m for t, m in runs if t == s)
        gp[s] = runs[(s, inits[0])]["gp"]
        for m in inits[1:]:
            if any(abs(runs[(s, m)]["gp"][k] - gp[s][k]) > 1e-6 * (1 + abs(gp[s][k])) for k, _ in KEYS):
                notes.append(f"seed {s}: GP metrics differ between init {inits[0]} and init {m}")
    dist_all = list(runs.values())

    L = [f"# Energy accuracy: GP posterior and distilled model ({os.path.basename(os.path.dirname(study))})", "",
         f"Study: `{os.path.basename(study)}`. Strain energy ({args.component}) on the elements of the validation load "
         "steps against the true material (from each run's validation_metrics.json). RMSE in energy units; 95% EC = "
         "share of states whose 95% interval contains the true energy. Mean ± std (sample std).", "",
         "## Overview", "",
         "| Model | Entries | RMSE | R² | 95% EC (%) |", "|---|---|---|---|---|",
         f"| GP posterior | {len(seeds)} (noise seeds) | " + " | ".join(ms([gp[s][k] for s in seeds], k) for k, _ in KEYS) + " |",
         f"| Distilled | {len(dist_all)} (seeds x inits) | " + " | ".join(ms([r['dist'][k] for r in dist_all], k) for k, _ in KEYS) + " |",
         "", "## Per noise seed", "",
         "GP: one posterior per seed. Distilled: mean ± std over that seed's initialisations.", "",
         "| Seed | GP RMSE | GP R² | GP 95% EC (%) | Inits | Distilled RMSE | Distilled R² | Distilled 95% EC (%) |",
         "|---|---|---|---|---|---|---|---|"]
    for s in seeds:
        ds = [runs[(t, m)]["dist"] for t, m in sorted(runs) if t == s]
        L.append(f"| {s} | " + " | ".join(fmt(gp[s][k], k) for k, _ in KEYS) + f" | {len(ds)} | "
                 + " | ".join(ms([d[k] for d in ds], k) for k, _ in KEYS) + " |")
    L += ["", "## All distilled runs", "", "| Seed | Init | RMSE | R² | 95% EC (%) |", "|---|---|---|---|---|"]
    for (s, m) in sorted(runs):
        L.append(f"| {s} | {m} | " + " | ".join(fmt(runs[(s, m)]["dist"][k], k) for k, _ in KEYS) + " |")
    if notes:
        L += ["", "Notes:", ""] + [f"- {n}" for n in notes]
    out = os.path.join(study, "metrics_summary.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L[:12]))
    print(out)


if __name__ == "__main__":
    main()

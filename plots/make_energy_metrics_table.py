#!/usr/bin/env python3
"""
LaTeX table of the strain-energy accuracy of several coverage studies (one row block per test case) from the
per-state energies of every run (energy_parity_total.npz, written by plots/plot_distilled_r2_energy.py):
  mean relative error |Psi_pred - Psi_true| / (|Psi_true| + 1e-8) over all validation states, R^2 and EC95 of the GP posterior (one per
  noise seed: init 0) and of the distilled model (one per seed x init), mean +- std; plus the median distilled
  model (middle run of the ranking by RMSE, R^2 and |EC95 - 95|, ranks averaged, as plots/make_coverage_*).

Usage: python3 plots/make_energy_metrics_table.py <study_dir>:<label> [...] --out table.tex
"""
import argparse
import glob
import os
import re

import numpy as np


def metrics(t, mu, lo, hi):
    return dict(rel=100 * np.mean(np.abs(mu - t) / (np.abs(t) + 1e-8)),      # all states (none near zero here)
                r2=1 - np.sum((t - mu) ** 2) / np.sum((t - t.mean()) ** 2),
                rmse=np.sqrt(np.mean((t - mu) ** 2)),
                ec=100 * np.mean((t >= lo) & (t <= hi)))


def load(study):
    dist, gp = {}, {}
    for f in glob.glob(os.path.join(study, "seed*", "init*", "energy_parity_total.npz")):
        s, m = map(int, re.search(r"seed(\d+)[/\\]init(\d+)", f).groups())
        z = np.load(f)
        t = z["true"]
        dist[(s, m)] = metrics(t, z["dist_mean"], z["dist_q025"], z["dist_q975"])
        if m == 0:
            gp[s] = metrics(t, z["gp_mean"], z["gp_mean"] - 1.96 * z["gp_std"], z["gp_mean"] + 1.96 * z["gp_std"])
    keys = sorted(dist)
    rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable")
    # median model from the stored validation metrics (validation_metrics.json, as the coverage tables and the
    # discovery figures), so all of them show the same run; the re-evaluated metrics can reorder close ranks
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from summarize_coverage_metrics import read
    stored = {k: read(os.path.join(study, f"seed{k[0]}", f"init{k[1]}"), "total") for k in keys}
    if all(v is not None for v in stored.values()):
        rm, r2, ec = (np.array([stored[k]["dist"][q] for k in keys]) for q in ("rmse", "r2", "coverage"))
    else:
        rm, r2, ec = (np.array([dist[k][q] for k in keys]) for q in ("rmse", "r2", "ec"))
    score = (rank(rm) + rank(-r2) + rank(np.abs(ec - 95.0))) / 3.0
    med = keys[int(np.argsort(score, kind="stable")[(len(keys) - 1) // 2])]
    return dist, gp, med


def pm(v, d):
    v = np.asarray(v, float)
    return f"${v.mean():.{d}f} \\pm {v.std(ddof=1):.{d}f}$"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="+", help="<study_dir>:<label>")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = []
    for c in args.cases:
        study, label = c.rsplit(":", 1)
        dist, gp, med = load(study)
        D, G = list(dist.values()), list(gp.values())
        print(f"{label}: {len(D)} distilled runs, {len(G)} GP seeds, median seed {med[0]} init {med[1]}")
        v = dist[med]
        rows += [rf"\multirow{{3}}{{*}}{{{label}}} & GP posterior & {pm([g['rel'] for g in G], 2)} & "
                 rf"{pm([g['r2'] for g in G], 4)} & {pm([g['ec'] for g in G], 1)} \\",
                 rf" & Distilled & {pm([d['rel'] for d in D], 2)} & {pm([d['r2'] for d in D], 4)} & "
                 rf"{pm([d['ec'] for d in D], 1)} \\",
                 rf" & Median model (seed ${med[0]}$, init ${med[1]}$) & ${v['rel']:.2f}$ & ${v['r2']:.4f}$ & "
                 rf"${v['ec']:.1f}$ \\", r"\midrule"]
    rows = rows[:-1]
    tex = "\n".join([
        r"\begin{table}[H]", r"\centering", r"\small", r"\begin{tabular}{llccc}", r"\toprule",
        r"Case & Model & $\bar{\epsilon}_{\mathrm{rel}}$ (\%) & $R^2$ & EC$_{95}$ (\%) \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}",
        r"\caption{Strain-energy accuracy on the validation load steps: GP posterior (one per noise seed) and distilled "
        r"model (one per noise seed and initialisation), mean $\pm$ std; median model: the middle run of the ranking by "
        r"RMSE, $R^2$ and $|\mathrm{EC}_{95} - 95|$ (ranks averaged).}",
        r"\label{tab:coverage_metrics}", r"\end{table}"])
    with open(args.out, "w") as f:
        f.write(tex + "\n")
    print(tex)


if __name__ == "__main__":
    main()

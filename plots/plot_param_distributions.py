#!/usr/bin/env python3
"""
Histograms of the distilled parameter samples (<component>_distributions_<model>.pdf), only for the library terms:
terms disabled for the run (--<component>_params_disabled, and the GMR log term E unless it was fitted) are left out;
terms pruned by the sensitivity analysis are shown as pruned (fixed at 0). Mean (red dashed) and true value (green
dotted) when known.

Used by pipeline/distill.py; for an existing run:
  python3 plots/plot_param_distributions.py --distilled_dir <run>/distilled [--component dev]
(the true values are read from the nearest config.yaml above the run with material_params)
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

GMR_DEV = ["C10", "C01", "C20", "C11", "C02", "C30", "C21", "C12", "C03", "E"]
VOL = ["D1", "D2", "D3"]
ANISO = ["C42", "C43", "C44", "C62", "C63", "C64"]


def tex(n):
    """'C10' -> '$C_{10}$', 'D1' -> '$D_{1}$', 'E' -> '$E$'; names already in TeX are kept."""
    if n.startswith("$"):
        return n
    if n == "E":
        return "$E$"
    if n.startswith("mu") or n.startswith("alpha"):
        k = n.lstrip("mualph")
        return f"$\\{'mu' if n.startswith('mu') else 'alpha'}_{{{k}}}$"
    return f"${n[0]}_{{{n[1:]}}}$"


def plot_param_distributions(samples, names, shown, active, true_params, path, ncols=None):
    """samples (S, len(names)); shown: names to draw (in order); active: names that were fitted (others: pruned);
    true_params: {name: value} (TeX or plain names)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    true_tex = {tex(k): v for k, v in true_params.items()}
    n = len(shown)
    cols = ncols or (3 if n in (3, 5, 6, 9) else 4)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.6), squeeze=False)
    axes = axes.ravel()
    for ax, name in zip(axes, shown):
        x = samples[:, names.index(name)]
        t = tex(name)
        if name in active and np.ptp(x) > 0:
            sns.histplot(x, ax=ax, color="#2980b9", bins=30, kde=True, edgecolor="black", alpha=0.7)
            m = float(x.mean())
            ax.axvline(m, color="#c0392b", ls="--", lw=2, label=f"Mean: {m:.4g}")
            title = f"{t}\nMean: {m:.4g}"
            if t in true_tex:
                ax.axvline(true_tex[t], color="#27ae60", ls=":", lw=2, label=f"True: {true_tex[t]:.4g}")
                title += f" (True: {true_tex[t]:.4g})"
            ax.set_title(title, color="#2c3e50", fontsize=11, fontweight="bold")
            ax.legend(fontsize=9)
        else:
            ax.bar([0.0], [1.0], width=0.1, color="gray")
            ax.set_xlim(-1, 1)
            ax.set_yticks([])
            sub = f" (True: {true_tex[t]:.4g})" if t in true_tex else ""
            ax.set_title(f"{t}{sub}\n[pruned by sensitivity, fixed at 0]", color="#7f8c8d", fontsize=11, style="italic")
            ax.set_xlabel("value")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def find_truth(start):
    """material_params of the nearest config.yaml above start (as a plain-name dict of non-zero true values)."""
    import yaml
    from core.material_library import FULL_NAMES, true_full_vector
    d = os.path.abspath(start)
    while d != os.path.dirname(d):
        f = os.path.join(d, "config.yaml")
        if os.path.exists(f):
            mp = (yaml.safe_load(open(f)) or {}).get("material_params")
            if mp:
                v = true_full_vector(mp)
                if v is not None:
                    return {n: float(x) for n, x in zip(FULL_NAMES, v) if abs(x) > 1e-12}
        d = os.path.dirname(d)
    return {}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", required=True)
    p.add_argument("--component", default="dev", choices=["dev", "vol", "aniso"])
    args = p.parse_args()
    d = args.distilled_dir
    cfg = {}
    for f in (f"{args.component}_config.json", "config.json"):
        if os.path.exists(os.path.join(d, f)):
            cfg = json.load(open(os.path.join(d, f)))
            break
    model = str(cfg.get("material_model", "gmr"))
    samples = np.load(os.path.join(d, f"{args.component}_flow_samples.npy"))
    if args.component == "dev":
        from core.material_library import distilled_library, dev_names
        names = dev_names(distilled_library(d))[:samples.shape[1]]
        disabled = {names[i] for i in cfg.get("dev_params_disabled", []) if i < len(names)}
        if "E" in names and not (int(cfg.get("include_log_term", 0)) or "_log" in model):
            disabled.add("E")
    elif args.component == "vol":
        names = VOL[:samples.shape[1]]
        disabled = {names[i] for i in cfg.get("vol_params_disabled", []) if i < len(names)}
    else:
        names = ANISO[:samples.shape[1]]
        disabled = {names[i] for i in cfg.get("aniso_params_disabled", []) if i < len(names)}
    shown = [n for n in names if n not in disabled]
    active = [n for n in shown if np.any(samples[:, names.index(n)] != 0)]
    out = os.path.join(d, f"{args.component}_distributions_{model}.pdf")
    plot_param_distributions(samples, names, shown, active, find_truth(d), out)
    print(f"{out}: {len(shown)} terms ({', '.join(shown)}); left out: {', '.join(sorted(disabled)) or '-'}")


if __name__ == "__main__":
    main()

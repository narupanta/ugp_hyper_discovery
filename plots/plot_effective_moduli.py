"""
Distribution of the effective small-strain moduli of a distillation: shear modulus mu = 2 (C10 + C01 + E/3) + sum mu_k
and bulk modulus kappa = 2 D1 of every distilled parameter sample, their 95% interval, the true values (synthetic
data) and the HSGP posterior (mean +- 1.96 std). kappa is skipped for deviatoric-only distillations (incompressible).

Usage: python3 plots/plot_effective_moduli.py --distilled_dir <dir> [--saved_model_dir <extracted>] [--material_model gmr]
Writes <distilled_dir>/effective_moduli.pdf and effective_moduli.json.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.material_library import (effective_moduli, full_param_vectors, is_dev_only,  # noqa: E402
                                   library_from_material_model, true_effective_moduli)
from plots.theme import apply_style, save_figure  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", required=True)
    p.add_argument("--saved_model_dir", default=None, help="extraction directory (default: dev_source_extraction_dir.txt)")
    p.add_argument("--material_model", default=None, help="distilled material model (default: from the config)")
    p.add_argument("--level", type=float, default=0.95)
    args = p.parse_args()
    d = args.distilled_dir
    ext = args.saved_model_dir
    if ext is None:
        for f in ("source_extraction_dir.txt", "dev_source_extraction_dir.txt"):
            if os.path.exists(os.path.join(d, f)):
                ext = open(os.path.join(d, f)).read().strip()
                break
    model = args.material_model
    if model is None:
        for f in ("dev_config.yaml", "config.yaml"):
            if os.path.exists(os.path.join(d, f)):
                import yaml
                model = (yaml.safe_load(open(os.path.join(d, f))) or {}).get("distilled_material_model") or model
                if model:
                    break
    model = model or "gmr"
    library = library_from_material_model(model)

    dev = np.load(os.path.join(d, "dev_flow_samples.npy"))
    vol_path = os.path.join(d, "vol_flow_samples.npy")
    vol = np.load(vol_path) if os.path.exists(vol_path) else np.zeros((len(dev), 3))
    n = min(len(dev), len(vol))
    mu, kappa = effective_moduli(full_param_vectors(dev[:n], vol[:n], library))

    gp, incompressible = None, is_dev_only(model)
    if ext and os.path.exists(os.path.join(ext, "extraction_metrics.json")):
        gp = json.load(open(os.path.join(ext, "extraction_metrics.json"))).get("small_strain_moduli")
    if ext and os.path.exists(os.path.join(ext, "hsgp_posterior.npz")):
        from core.hsgp_model import HSGPHyperelasticity
        incompressible = incompressible or bool(HSGPHyperelasticity.load(os.path.join(ext, "hsgp_posterior.npz")).incompressible_plane_stress)
    truth = None
    try:
        from core.material_models import get_material_from_dir
        truth = true_effective_moduli(get_material_from_dir(ext or d, jit_P=False))
    except Exception as e:   # experimental data: no truth
        print(f"no true material ({type(e).__name__}): intervals only")

    a = 0.5 * (1.0 - args.level)
    rows = [("mu", r"shear modulus $\mu$", mu, 0, "mu_eff")]
    if not incompressible:
        rows.append(("kappa", r"bulk modulus $\kappa$", kappa, 1, "kappa_eff"))
    apply_style()
    fig, axes = plt.subplots(1, len(rows), figsize=(5.4 * len(rows), 4.0), squeeze=False)
    out = dict(level=args.level, library=library, n_samples=int(n))
    for ax, (key, label, x, ti, gk) in zip(axes[0], rows):
        lo, hi = np.quantile(x, [a, 1.0 - a])
        rec = dict(mean=float(x.mean()), std=float(x.std()), lo=float(lo), hi=float(hi))
        ax.hist(x, bins=60, density=True, color="#1565c0", alpha=0.45, label="distilled samples")
        ax.axvspan(lo, hi, color="#1565c0", alpha=0.10, label=f"distilled {100 * args.level:.0f}% interval")
        if gp is not None and gp.get(gk) is not None and (gk != "kappa_eff" or gp.get("kappa_eff", 0) > 0):
            gm, gs = float(gp[gk]), float(gp.get(gk + "_std", 0.0))
            ax.errorbar([gm], [0.0], xerr=[[1.96 * gs], [1.96 * gs]], fmt="o", color="#2e7d32", ms=5, capsize=4,
                        clip_on=False, zorder=5, label="GP posterior (95%)")
            rec["gp_mean"], rec["gp_std"] = gm, gs
        if truth is not None:
            t = truth[ti]
            inside = bool(lo <= t <= hi)
            ax.axvline(t, color="#c62828", lw=2, ls="--", label=f"true ({'inside' if inside else 'OUTSIDE'})")
            rec.update(true=float(t), covered=inside, z=float((x.mean() - t) / max(x.std(), 1e-15)))
        ax.set_xlabel(label)
        ax.set_ylabel("density")
        ax.set_title(f"{label}: {rec['mean']:.4g} [{lo:.4g}, {hi:.4g}]")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=0.3)
        out[key] = rec
    fig.tight_layout()
    save_figure(fig, os.path.join(d, "effective_moduli.pdf"))
    plt.close(fig)
    json.dump(out, open(os.path.join(d, "effective_moduli.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

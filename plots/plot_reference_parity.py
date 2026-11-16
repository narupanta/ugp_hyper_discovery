#!/usr/bin/env python3
"""
Energy parity of the distilled model against published reference models on the experimental deformation states
(the extraction's train_states_f3x3.npy): x = reference W, y = distilled W (posterior mean, 95% band as error bars on
a subset), one panel per reference, with R^2, relative RMSE and the share of states whose 95% band contains the
reference.

Usage: python3 plots/plot_reference_parity.py --distilled_dir <seed>/distilled --extract_dir <seed>/extracted \
           [--references euclid_ttf_ut euclid_noogden_ttf_ut] [--n_states 4000]
Writes <distilled_dir>/reference_parity.pdf and .png
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "validation"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", required=True)
    p.add_argument("--extract_dir", required=True)
    p.add_argument("--references", nargs="+", default=["euclid_ttf_ut", "euclid_noogden_ttf_ut"])
    p.add_argument("--n_states", type=int, default=4000)
    p.add_argument("--n_samples", type=int, default=200, help="parameter samples for the 95% band")
    p.add_argument("--n_bars", type=int, default=150, help="states drawn with error bars")
    args = p.parse_args()
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from core.material_library import distilled_library, full_param_vectors
    from forward_fem_full_specimen import FullVectorMaterial, load_reference
    from plots.theme import apply_style, save_figure

    dev = np.load(os.path.join(args.distilled_dir, "dev_flow_samples.npy"))
    full = full_param_vectors(dev, np.zeros((len(dev), 3)), distilled_library(args.distilled_dir))
    F = np.load(os.path.join(args.extract_dir, "train_states_f3x3.npy")).reshape(-1, 3, 3)
    rng = np.random.default_rng(0)
    F = jnp.asarray(F[rng.choice(len(F), min(args.n_states, len(F)), replace=False)])
    idx = rng.choice(len(full), min(args.n_samples, len(full)), replace=False)
    Ws = np.array([np.asarray(jax.vmap(FullVectorMaterial(full[i]).psi)(F)) for i in idx])
    lo, hi = np.quantile(Ws, [0.025, 0.975], axis=0)
    Wm = np.asarray(jax.vmap(FullVectorMaterial(full.mean(0)).psi)(F))

    apply_style()
    n = len(args.references)
    fig, axes = plt.subplots(1, n, figsize=(4.4 * n, 4.4), squeeze=False)
    bars = rng.choice(len(Wm), min(args.n_bars, len(Wm)), replace=False)
    for ax, key in zip(axes[0], args.references):
        ref = load_reference(key)
        We = np.asarray(ref.psi(F))
        r2 = 1 - np.sum((We - Wm) ** 2) / np.sum((We - We.mean()) ** 2)
        rel = np.sqrt(np.mean((We - Wm) ** 2)) / np.sqrt(np.mean(We ** 2))
        ec = np.mean((We >= lo) & (We <= hi))
        ax.errorbar(We[bars], Wm[bars], yerr=[Wm[bars] - lo[bars], hi[bars] - Wm[bars]], fmt="none",
                    ecolor="#009E73", alpha=0.45, lw=0.8, zorder=2, label="distilled 95% interval")
        ax.scatter(We, Wm, s=4, color="#00695c", alpha=0.5, lw=0, zorder=3, rasterized=True,
                   label="distilled mean")
        m = max(We.max(), hi.max()) * 1.03
        ax.plot([0, m], [0, m], "k--", lw=1.0, zorder=4, label="parity")
        ax.set_xlim(0, m)
        ax.set_ylim(0, m)
        ax.set_aspect("equal")
        lab = ref.label.replace(" (UT+TTf)", "")
        ax.set_xlabel(rf"$\Psi$ {lab} [MPa]")
        ax.set_ylabel(r"$\Psi$ discovered [MPa]")
        ax.set_title(lab, fontsize=11)
        ax.text(0.04, 0.96, f"$R^2$ = {r2:.4f}\nrel. RMSE = {100 * rel:.1f}%\nin 95% band: {100 * ec:.1f}%",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", alpha=0.9))
        ax.grid(alpha=0.25)
        print(f"{key}: R2 {r2:.4f}, rel RMSE {100 * rel:.2f}%, in band {100 * ec:.1f}%")
    axes[0, -1].legend(loc="lower right", fontsize=8.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    out = os.path.join(args.distilled_dir, "reference_parity.pdf")
    save_figure(fig, out)
    fig.set_dpi(200)
    fig.canvas.print_png(out[:-4] + ".png")
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

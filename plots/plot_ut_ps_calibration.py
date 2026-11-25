#!/usr/bin/env python3
"""
Calibration of the UT / PS global-stress predictions: for every nominal level alpha in [0, 1], the share of the measured
points (lambda, P11) that lie inside the central alpha credible interval of each model, at the measured stretches.
  distilled: (50 -/+ 50 alpha) percentiles of P11 = dPsi/dlambda over the parameter samples;
  GP:        mean -/+ z_{(1 + alpha)/2} sd of the exact dPsi/dlambda of the posterior.
A calibrated model lies on the diagonal; above: under-confident, below: over-confident. The legend gives the
miscalibration area (mean |empirical - nominal| over alpha, in %).

Same data (ut_ps_global_data.npz, written by plots/plot_ut_ps_global.py), models, ideal load paths and last-step cut
as plots/plot_ut_ps_global.py.

Usage: python3 plots/plot_ut_ps_calibration.py --distilled_dir results/<exp>/1/distilled
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from plots.theme import apply_style  # noqa: E402
from plots.plot_ut_ps_global import LAST_STEP, TESTS, distilled_energy_fn, path_F  # noqa: E402

COL = {"UT": "#1565c0", "PS": "#c62828"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled_dir", required=True)
    ap.add_argument("--extract_dir", default=None, help="default: <distilled_dir>/../extracted")
    ap.add_argument("--n_samples", type=int, default=1000, help="distilled parameter samples")
    ap.add_argument("--out_name", default="ut_ps_calibration")
    args = ap.parse_args()
    apply_style()
    from core.teacher import load_hsgp_teacher
    d = os.path.abspath(args.distilled_dir)
    ex = args.extract_dir or os.path.join(os.path.dirname(d), "extracted")
    cache = os.path.join(d, "ut_ps_global_data.npz")
    if not os.path.exists(cache):
        sys.exit(f"{cache} missing: run plots/plot_ut_ps_global.py first")
    z = np.load(cache)
    data = {t: {k[len(t) + 1:]: z[k] for k in z.files if k.startswith(t + "_")} for t in TESTS}
    for t, last in LAST_STEP.items():
        keep = data[t]["steps"] <= last
        data[t] = {k: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for k, v in data[t].items()}

    psi_d, theta = distilled_energy_fn(d)
    theta = jnp.asarray(theta[np.random.default_rng(0).choice(len(theta), min(args.n_samples, len(theta)),
                                                                replace=False)])
    gp = load_hsgp_teacher(ex)
    alpha = np.linspace(0.0, 1.0, 101)
    res = {}
    for t, (_, lat, _) in TESTS.items():
        lam, P = jnp.asarray(data[t]["lam"]), np.asarray(data[t]["P"])
        # distilled: P11 per parameter sample at the measured stretches
        dpsi = jax.grad(lambda l, th: psi_d(th, path_F(l, lat)))
        Pd = np.asarray(jax.vmap(lambda th: jax.vmap(lambda l: dpsi(l, th))(lam))(theta))      # (S, N)
        lo = np.percentile(Pd, 50 * (1 - alpha), axis=0)                                        # (A, N)
        hi = np.percentile(Pd, 50 * (1 + alpha), axis=0)
        res[(t, "Distilled")] = 100 * np.mean((P >= lo) & (P <= hi), axis=1)
        # GP: exact derivative of the posterior energy along the path
        Fp = jax.vmap(lambda l: path_F(l, lat))(lam)
        dF = jax.vmap(jax.jacfwd(lambda l: path_F(l, lat)))(lam)
        dphi = jax.vmap(lambda F, dF_: jnp.einsum("pij,ij->p", jax.jacfwd(gp.basis)(F), dF_))(Fp, dF)
        m = np.asarray(dphi @ gp.theta_mean)
        sd = np.asarray(jnp.sqrt(jnp.sum((dphi @ gp.theta_factor) ** 2, axis=1)))
        zq = norm.ppf(0.5 * (1 + np.clip(alpha, 0, 1 - 1e-12)))[:, None]
        res[(t, "GP")] = 100 * np.mean(np.abs(P - m) <= zq * sd, axis=1)

    fig, ax = plt.subplots(figsize=(110 / 25.4, 62 / 25.4))
    ax.plot([0, 100], [0, 100], "k--", lw=0.6, zorder=1)
    ax.fill_between([0, 100], [0, 100], [100, 100], color="0.93", lw=0, zorder=0)
    ax.text(28, 90, "under-confident", fontsize=6, ha="center", color="0.4")
    ax.text(72, 10, "over-confident", fontsize=6, ha="center", color="0.4")
    print(f"{'test':4s} {'model':10s} {'EC50':>6s} {'EC95':>6s} {'miscal. area':>13s}")
    for (t, name), cov in res.items():
        area = float(np.mean(np.abs(cov - 100 * alpha)))
        ls = "-" if name == "Distilled" else "-."
        c = COL[t] if name == "Distilled" else COL[t]
        ax.plot(100 * alpha, cov, ls, color=c, lw=1.3 if name == "Distilled" else 1.0, zorder=3,
                label=f"{t} {'UGP + Distillation' if name == 'Distilled' else 'GP'} ({area:.0f}%)")
        print(f"{t:4s} {name:10s} {cov[50]:6.1f} {cov[95]:6.1f} {area:13.1f}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal")
    ax.set_xlabel(r"nominal level $\alpha$ (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.grid(False)
    ax.legend(fontsize=6, frameon=False, loc="center left", handlelength=2.0, borderaxespad=0.0,
              title="(miscalibration area)", title_fontsize=6, bbox_to_anchor=(1.04, 0.5))
    fig.tight_layout(pad=0.3)
    out = os.path.join(d, args.out_name)
    fig.savefig(out + ".pdf", dpi=300)
    fig.set_dpi(250)
    fig.canvas.print_png(out + ".png")
    np.savez(out + ".npz", alpha=alpha, **{f"{t}_{n}": v for (t, n), v in res.items()})
    print(f"saved {out}.pdf/.png/.npz")


if __name__ == "__main__":
    main()

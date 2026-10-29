"""
GP posterior energy of an experimental extraction against published reference models (incompressible, isotropic):
the models identified on the same data set by Flaschel et al. (EUCLID and the GMR / Gent-Thomas / Ogden fits).

Panels (top) W along uniaxial tension, pure shear and equibiaxial tension (incompressible, plane stress) with the GP
mean and 95% band; (bottom) the nominal stress P11 = dW/dlambda along UT and PS (1/2 dW/dlambda for EBT) and the
energy parity at all observed states. The stretch range the data cover (the convex hull of the observed (I1, I2))
is unshaded; beyond it the GP extrapolates.

Usage:
  python3 plots/plot_reference_energy.py --extract_dir results/<exp>/1/extracted [--references euclid gmr3 ogden2]
Writes <extract_dir>/reference_energy_comparison.pdf and reference_energy_comparison.json.
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.hsgp_model import HSGPHyperelasticity  # noqa: E402
from plots.theme import apply_style, save_figure  # noqa: E402


def _sum_pow(c, a):
    """sum_i lambda_i^a from the squared principal stretches c = (c1, c2, c3)."""
    return sum(ci ** (0.5 * a) for ci in c)


# W(c1, c2, c3) with c_i = lambda_i^2, I1 = sum c_i, I2 = c1 c2 + c2 c3 + c3 c1 (incompressible: c1 c2 c3 = 1)
def _inv(c):
    return c[0] + c[1] + c[2], c[0] * c[1] + c[1] * c[2] + c[2] * c[0]


REFERENCES = {
    "euclid": ("EUCLID", lambda c: (lambda I1, I2: 0.0814 * (I1 - 3) + 0.0006 * (I1 - 3) ** 3
                                    + 0.0917 * (_sum_pow(c, 2.0) - 3) + 0.4682 * np.log(I2 / 3))(*_inv(c))),
    "gmr1": ("1st-order GMR", lambda c: 0.2938 * (_inv(c)[0] - 3)),
    "gmr2": ("2nd-order GMR", lambda c: (lambda I1, I2: 0.2842 * (I1 - 3) + 0.0016 * (I1 - 3) ** 2)(*_inv(c))),
    "gmr3": ("3rd-order GMR", lambda c: (lambda I1, I2: 0.1354 * (I1 - 3) + 0.1528 * (I2 - 3)
                                         + 3.7369e-6 * (I1 - 3) ** 2 * (I2 - 3) + 8.6685e-5 * (I1 - 3) ** 2
                                         + 6.5870e-4 * (I1 - 3) ** 3)(*_inv(c))),
    "gt": ("Gent-Thomas", lambda c: (lambda I1, I2: 0.2842 * (I1 - 3) + 0.0445 * np.log(I2 / 3))(*_inv(c))),
    "ogden1": ("1-term Ogden", lambda c: 0.2881 * (_sum_pow(c, 2.0160) - 3)),
    "ogden2": ("2-term Ogden", lambda c: 0.0006 * (_sum_pow(c, 5.9364) - 3) + 10.9138 * (_sum_pow(c, 0.3435) - 3)),
}
REF_STYLES = {"euclid": dict(color="#c62828", lw=2.0, ls="--")}
OTHER_COLORS = ["#6d4c41", "#00897b", "#8e24aa", "#f9a825", "#546e7a", "#3949ab"]

# in-plane stretches (lambda1, lambda2) of each mode; lambda3 = 1 / (lambda1 lambda2); P11 = dW/dlambda / factor
MODES = {
    "UT": ("Uniaxial tension", lambda l: (l, l ** -0.5), 1.0),
    "PS": ("Pure shear", lambda l: (l, 0.0 * l + 1.0), 1.0),
    "EBT": ("Equibiaxial tension", lambda l: (l, l), 2.0),
}


def stretches_from_invariants(I1, I2):
    """Squared principal stretches (N, 3) from (I1, I2) with I3 = 1: roots of x^3 - I1 x^2 + I2 x - 1."""
    comp = np.zeros((len(I1), 3, 3))
    comp[:, 0, :] = np.stack([I1, -I2, np.ones_like(I1)], axis=1)
    comp[:, 1, 0] = comp[:, 2, 1] = 1.0
    return np.sort(np.clip(np.real(np.linalg.eigvals(comp)), 1e-12, None), axis=1).T


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract_dir", required=True)
    p.add_argument("--references", nargs="+", default=["euclid"], choices=list(REFERENCES),
                   help="reference models to draw (default: euclid)")
    p.add_argument("--lambda_max", type=float, default=None, help="largest stretch (default: 1.2 x the data range)")
    p.add_argument("--n_points", type=int, default=300)
    p.add_argument("--ylim_from_mean", type=int, default=1,
                   help="1 (default): y limits from the GP mean and reference curves + margin (band clipped); 0: auto")
    p.add_argument("--y_margin", type=float, default=0.15, help="margin as a fraction of the mean curves' range")
    p.add_argument("--ylim_range", choices=["data", "all"], default="data",
                   help="stretches the y limits are taken over: inside the data range (default) or the whole plot")
    args = p.parse_args()
    apply_style()

    gp = HSGPHyperelasticity.load(os.path.join(args.extract_dir, "hsgp_posterior.npz"))
    if not getattr(gp, "incompressible_plane_stress", False):
        print("warning: the GP is not incompressible plane stress; the reference models are incompressible")
    I_obs = np.load(os.path.join(args.extract_dir, "I_obs_all.npy")).reshape(-1, 3)
    dev_obs = I_obs[:, :2]

    from scipy.spatial import ConvexHull
    hull = ConvexHull(np.vstack([dev_obs, [[3.0, 3.0]]]))
    inside = lambda d: np.all(d @ hull.equations[:, :-1].T + hull.equations[:, -1] <= 1e-6 * (1 + np.abs(d).max()), axis=1)

    phi_feat = jax.jit(lambda dev: gp.basis_from_features(dev, jnp.ones(dev.shape[:-1] + (1,))))
    m, Fc = np.asarray(gp.theta_mean), np.asarray(gp.theta_factor)

    def gp_W_along(mode, lam):
        """GP energy and dW/dlambda (mean, std) along a mode path."""
        f_l = MODES[mode][1]

        def feats(l):
            l1, l2 = f_l(l)
            c = jnp.stack([l1 ** 2, l2 ** 2, 1.0 / (l1 * l2) ** 2])
            return jnp.stack([c.sum(), c[0] * c[1] + c[1] * c[2] + c[2] * c[0]])
        lam_j = jnp.asarray(lam)
        phi = np.asarray(phi_feat(jax.vmap(feats)(lam_j)))
        dphi = np.asarray(jax.vmap(jax.jacfwd(lambda l: gp.basis_from_features(feats(l), jnp.ones(1))))(lam_j))
        W, dW = phi @ m, dphi @ m
        return W, np.sqrt(np.sum((phi @ Fc) ** 2, 1)), dW, np.sqrt(np.sum((dphi @ Fc) ** 2, 1))

    def ref_W_along(key, mode, lam):
        l1, l2 = MODES[mode][1](lam)
        c = (l1 ** 2, l2 ** 2, 1.0 / (l1 * l2) ** 2)
        W = REFERENCES[key][1](c)
        h = 1e-6
        l1p, l2p = MODES[mode][1](lam + h)
        l1m, l2m = MODES[mode][1](lam - h)
        dW = (REFERENCES[key][1]((l1p ** 2, l2p ** 2, 1 / (l1p * l2p) ** 2))
              - REFERENCES[key][1]((l1m ** 2, l2m ** 2, 1 / (l1m * l2m) ** 2))) / (2 * h)
        return W, dW

    def data_lambda_max(mode):
        lam = np.linspace(1.0, 10.0, 4000)
        l1, l2 = MODES[mode][1](lam)
        c = np.stack([l1 ** 2, l2 ** 2, 1 / (l1 * l2) ** 2])
        d = np.stack([c.sum(0), c[0] * c[1] + c[1] * c[2] + c[2] * c[0]], 1)
        ins = inside(d)
        out = np.where(~ins[1:])[0]
        return float(lam[out[0]]) if len(out) else float(lam[-1])

    lam_data = {k: data_lambda_max(k) for k in MODES}
    print("stretch range inside the data hull:", {k: round(v, 3) for k, v in lam_data.items()})
    refs = list(dict.fromkeys(args.references))
    style = {}
    for i, k in enumerate(r for r in refs if r != "euclid"):
        style[k] = dict(color=OTHER_COLORS[i % len(OTHER_COLORS)], lw=1.2, ls=":")
    style.update({k: v for k, v in REF_STYLES.items() if k in refs})

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    summary = dict(lambda_data_max=lam_data, modes={})
    gp_col = "#1565c0"
    for j, mode in enumerate(MODES):
        lmax = args.lambda_max or 1.2 * lam_data[mode]
        lam = np.linspace(1.0, lmax, args.n_points)
        W, sW, dW, sdW = gp_W_along(mode, lam)
        fac = MODES[mode][2]
        P, sP = dW / fac, sdW / fac
        rows = [(axes[0, j], W, sW, "W")]
        if mode != "EBT":
            rows.append((axes[1, j], P, sP, "P11"))
        summary["modes"][mode] = {}
        for ax, y, s, what in rows:
            ax.fill_between(lam, y - 1.96 * s, y + 1.96 * s, color=gp_col, alpha=0.22, lw=0, label="GP 95%")
            ax.plot(lam, y, color=gp_col, lw=1.8, label="GP mean")
            curves = [y]
            for k in refs:
                Wr, dWr = ref_W_along(k, mode, lam)
                yr = Wr if what == "W" else dWr / fac
                ax.plot(lam, yr, label=REFERENCES[k][0], **style[k])
                curves.append(yr)
                msk = lam <= lam_data[mode]
                rel = np.abs(y[msk] - yr[msk]).max() / max(np.abs(yr[msk]).max(), 1e-12)
                cover = float(np.mean(np.abs(y[msk] - yr[msk]) <= 1.96 * s[msk]))
                summary["modes"][mode][f"{what}_{k}"] = dict(max_rel_diff_in_data=float(rel), band_contains=cover)
            ax.axvspan(lam_data[mode], lam[-1], color="#9e9e9e", alpha=0.15, lw=0,
                       label="outside data" if j == 0 else None)
            ax.set_xlim(1.0, lam[-1])
            if args.ylim_from_mean:   # y range from the mean and reference curves; the 95% band is clipped
                rng = lam <= lam_data[mode] if args.ylim_range == "data" else np.ones_like(lam, bool)
                lo, hi = min(float(np.min(c[rng])) for c in curves), max(float(np.max(c[rng])) for c in curves)
                pad = args.y_margin * max(hi - lo, 1e-12)
                ax.set_ylim(min(lo, 0.0) - pad, hi + pad)
            ax.set_xlabel(r"$\lambda$")
            ax.grid(alpha=0.3)
        axes[0, j].set_title(MODES[mode][0])
        axes[0, j].set_ylabel(r"$W$ [MPa]")
        if mode != "EBT":
            axes[1, j].set_ylabel(r"$P_{11}$ [MPa]")
            axes[1, j].set_title(f"{MODES[mode][0]}: nominal stress")

    # energy parity at all observed states
    ax = axes[1, 2]
    rng = np.random.default_rng(0)
    sel = rng.choice(len(dev_obs), min(len(dev_obs), 4000), replace=False)
    d = dev_obs[sel]
    phi = np.asarray(phi_feat(jnp.asarray(d)))
    Wg, sWg = phi @ m, np.sqrt(np.sum((phi @ Fc) ** 2, 1))
    c = stretches_from_invariants(d[:, 0], d[:, 1])
    We = REFERENCES["euclid"][1](tuple(c))
    ax.errorbar(We, Wg, yerr=1.96 * sWg, fmt="none", ecolor=gp_col, alpha=0.15, lw=0.6)
    ax.scatter(We, Wg, s=3, color=gp_col, lw=0, alpha=0.6, label="observed states")
    lim = [0, 1.05 * max(We.max(), Wg.max())]
    ax.plot(lim, lim, color="#424242", lw=1)
    ss = np.sum((Wg - We) ** 2)
    r2 = 1 - ss / np.sum((We - We.mean()) ** 2)
    cover = float(np.mean(np.abs(Wg - We) <= 1.96 * sWg))
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(r"$W_{\mathrm{EUCLID}}$ [MPa]"); ax.set_ylabel(r"$W_{\mathrm{GP}}$ [MPa]")
    ax.set_title(f"Observed states ($R^2$ = {r2:.3f}, in 95% band {100 * cover:.0f}%)")
    ax.grid(alpha=0.3)
    summary["data_states"] = dict(r2_vs_euclid=float(r2), band_contains_euclid=cover, n=int(len(sel)))

    axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("HSGP posterior energy against reference models (Flaschel et al.)", fontsize=12)
    fig.tight_layout()
    out = os.path.join(args.extract_dir, "reference_energy_comparison.pdf")
    save_figure(fig, out)
    plt.close(fig)
    with open(os.path.join(args.extract_dir, "reference_energy_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()

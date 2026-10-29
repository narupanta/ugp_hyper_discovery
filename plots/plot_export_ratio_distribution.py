"""
Where the distillation inputs sit in the mode-separating coordinates, one row per selection of states:
left, panel (a) of invariant_space_modes (I1_bar - 3 on a log axis against the ratio (I2_bar - 3) / (I1_bar - 3),
with the training invariants, their hull and the six standard mode paths) with the picked states on top;
right, the density of the ratio (shared y axis) for the picked states and for all training states.

Rows: every --export_dir (its f3x3.npy), then one random subset per --random_n (the export's dataset_f_random mode,
same seed), then one FPS selection per --fps_n (the export's dataset_f mode: farthest
point sampling over the training-step F_2D; taken from pytorch_export_dataset_f_n<N> when it exists, else computed).

Usage: python3 plots/plot_export_ratio_distribution.py --saved_model_dir <extracted> \
           --export_dir <extracted>/pytorch_export_dataset_f_random_n128 --fps_n 512
Writes <saved_model_dir>/export_ratio_distribution.pdf (or --out).
"""
import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core.hsgp_model import HSGPHyperelasticity  # noqa: E402
from plots.theme import MODE_COLORS, MODE_NAMES, MODE_SHORT_NAMES, apply_style, save_figure  # noqa: E402
from plots.training import standard_mode_F  # noqa: E402

EPS = 1e-8
Y_MAX = 2.5


def to_a(x1, x2):
    a = np.maximum(np.asarray(x1) - 3, EPS)
    return a, (np.asarray(x2) - 3) / a


def features(gp, F):
    dev, vol = jax.vmap(gp.feature_extractor.extract)(jnp.asarray(F))[:2]
    return np.concatenate([np.asarray(dev), np.asarray(vol).reshape(len(dev), -1)[:, :1]], axis=1)   # I1b, I2b, J


def training_F(saved_model_dir, incompressible):
    """F (N, 3, 3) of the training load steps, as the export loads them."""
    pooled = os.path.join(saved_model_dir, "train_states_f3x3.npy")
    if os.path.exists(pooled):
        return np.load(pooled)
    from core.dataset_store import load_dataset
    cfg = json.load(open(os.path.join(saved_model_dir, "config.json")))
    data = load_dataset(cfg["dataset_path"])
    steps = cfg["train_load_steps_indices"]
    F2 = np.asarray(data["F"])[steps].reshape(-1, 2, 2)
    F3 = data.get("F_3d")
    if F3 is not None and not incompressible:
        return np.asarray(F3)[steps].reshape(-1, 3, 3)
    out = np.zeros((len(F2), 3, 3))
    out[:, :2, :2] = F2
    out[:, 2, 2] = 1.0 / np.linalg.det(F2) if incompressible else 1.0
    return out


def random_selection(saved_model_dir, F_train, n, seed=0):
    existing = os.path.join(saved_model_dir, f"pytorch_export_dataset_f_random_n{n}", "f3x3.npy")
    if os.path.exists(existing):
        return np.load(existing)
    idx = np.sort(np.random.default_rng(seed).choice(len(F_train), size=min(n, len(F_train)), replace=False))
    return F_train[idx]


def fps_selection(saved_model_dir, F_train, n):
    existing = os.path.join(saved_model_dir, f"pytorch_export_dataset_f_n{n}", "f3x3.npy")
    if os.path.exists(existing):
        return np.load(existing)
    from core.utils import farthest_point_sampling
    pts = jnp.asarray(F_train[:, :2, :2].reshape(-1, 4), dtype=jnp.float64)
    idx = np.arange(len(F_train)) if len(F_train) <= n else np.asarray(farthest_point_sampling(pts, n))
    return F_train[idx]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saved_model_dir", required=True)
    p.add_argument("--export_dir", action="append", default=[])
    p.add_argument("--random_n", type=int, action="append", default=[])
    p.add_argument("--random_seed", type=int, default=0)
    p.add_argument("--fps_n", type=int, action="append", default=[])
    p.add_argument("--out", default=None)
    args = p.parse_args()
    jax.config.update("jax_enable_x64", True)
    plot_ratio_distribution(args.saved_model_dir, export_dirs=args.export_dir, random_n=args.random_n,
                            fps_n=args.fps_n, random_seed=args.random_seed, out=args.out)


def plot_ratio_distribution(ext, export_dirs=(), random_n=(), fps_n=(), random_seed=0, out=None, gp=None,
                            verbose=True):
    """One row per export folder / random subset / FPS selection; returns the output path. Also called by
    distillation/export_gp_to_pytorch.py after every export (one row: that export, saved in its folder)."""
    from types import SimpleNamespace
    args = SimpleNamespace(export_dir=list(export_dirs), random_n=list(random_n), fps_n=list(fps_n),
                           random_seed=random_seed, out=out)
    gp = gp or HSGPHyperelasticity.load(os.path.join(ext, "hsgp_posterior.npz"))
    incompressible = bool(getattr(gp, "incompressible_plane_stress", False))
    I_all = np.load(os.path.join(ext, "I_obs_all.npy"))
    dev_obs = I_all[:, :2]
    J_obs = I_all[:, 2] if I_all.shape[1] > 2 else np.ones(len(I_all))

    rows = []
    for d in args.export_dir:
        rows.append((os.path.basename(os.path.normpath(d)).replace("pytorch_export_", "").replace("dataset_f_", "")
                     .replace("_n", " n = "), features(gp, np.load(os.path.join(d, "f3x3.npy")))))
    F_train = training_F(ext, incompressible) if (args.fps_n or args.random_n) else None
    for n in args.random_n:
        rows.append((f"random n = {n}", features(gp, random_selection(ext, F_train, n, args.random_seed))))
    if args.fps_n:
        for n in args.fps_n:
            rows.append((f"FPS n = {n}", features(gp, fps_selection(ext, F_train, n))))
    if not rows:
        raise SystemExit("nothing to plot: pass --export_dir, --random_n and/or --fps_n")

    # data hull and standard mode paths (gamma in [0, 1]), as in plot_invariant_space_modes
    from scipy.spatial import ConvexHull
    hull = ConvexHull(dev_obs)
    poly = dev_obs[hull.vertices]
    closed = np.vstack([poly, poly[:1]])
    dense = np.concatenate([np.linspace(closed[k], closed[k + 1], 60) for k in range(len(closed) - 1)])
    gamma = jnp.linspace(0.0, 1.0, 400)
    F_modes = standard_mode_F(gp, gamma)
    mode_dev = [np.asarray(jax.vmap(gp.feature_extractor.extract)(F_modes[m])[0]) for m in range(6)]

    isub = np.random.default_rng(0).choice(len(dev_obs), min(len(dev_obs), 6000), replace=False)
    sub, J_sub = dev_obs[isub], J_obs[isub]
    from matplotlib.colors import Normalize
    J_lo = min(J_obs.min(), *(d[:, 2].min() for _, d in rows))
    J_hi = max(J_obs.max(), *(d[:, 2].max() for _, d in rows))
    if J_hi - J_lo < 1e-6:          # incompressible (J = 1): a constant colour scale
        J_lo, J_hi = J_lo - 0.5, J_hi + 0.5
    norm, cmap = Normalize(J_lo, J_hi), "viridis"
    _, r_all = to_a(dev_obs[:, 0], dev_obs[:, 1])
    bins = np.linspace(0.0, Y_MAX, 61)
    pos = (dev_obs[:, 0] - 3)[(dev_obs[:, 0] - 3) > 0]
    x_left = max(1e-6, 0.3 * float(np.quantile(pos, 0.001))) if len(pos) else 1e-6

    apply_style()
    fig, axes = plt.subplots(len(rows), 2, figsize=(10.5, 4.2 * len(rows)), squeeze=False,
                             gridspec_kw=dict(width_ratios=[3.0, 1.0]), sharey=True,
                             layout="constrained")
    summary = {}
    for (label, dev), (ax, axh) in zip(rows, axes):
        ax.scatter(*to_a(sub[:, 0], sub[:, 1]), s=3, c=J_sub, cmap=cmap, norm=norm, alpha=0.25, lw=0,
                   label="training invariants")
        ax.plot(*to_a(dense[:, 0], dense[:, 1]), color="#2e7d32", lw=1.2, label="interpolation region (hull)")
        for m in range(6):
            ax.plot(*to_a(mode_dev[m][1:, 0], mode_dev[m][1:, 1]), color=MODE_COLORS[m], lw=1.6,
                    label=MODE_SHORT_NAMES[m])
        x, r = to_a(dev[:, 0], dev[:, 1])
        sc = ax.scatter(x, r, s=30, marker="o", c=dev[:, 2], cmap=cmap, norm=norm, edgecolors="#111111", lw=0.6,
                        zorder=6, label=f"picked: {label}")
        fig.colorbar(sc, ax=axh, label="J", pad=0.02)
        ax.set_xscale("log")
        ax.set_xlim(left=x_left)
        ax.set_ylim(0.0, Y_MAX)
        ax.set_xlabel(r"$\bar I_1 - 3$")
        ax.set_ylabel(r"$(\bar I_2 - 3)\,/\,(\bar I_1 - 3)$")
        ax.set_title(f"Picked states: {label} (N = {len(dev)})")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="upper left", frameon=False, ncol=2)

        axh.hist(r_all, bins=bins, density=True, orientation="horizontal", color="#9e9e9e", alpha=0.45,
                 label=f"all training (N = {len(r_all)})")
        axh.hist(r, bins=bins, density=True, orientation="horizontal", histtype="step", color="#111111", lw=1.5,
                 label="picked")
        axh.axhline(1.0, color="#555555", lw=0.8, ls=":")
        axh.set_xlabel("density")
        axh.set_title("ratio distribution")
        axh.grid(alpha=0.3)
        axh.legend(fontsize=7.5, loc="upper right", frameon=False)
        summary[label] = dict(n=int(len(dev)), ratio_quantiles=dict(zip(["5%", "25%", "50%", "75%", "95%"],
                              np.quantile(r, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist())),
                              frac_ratio_gt_1=float(np.mean(r > 1.0)),
                              I1m3_quantiles=np.quantile(x, [0.05, 0.5, 0.95]).round(4).tolist())
    summary["all training"] = dict(n=int(len(r_all)), ratio_quantiles=dict(zip(["5%", "25%", "50%", "75%", "95%"],
                                   np.quantile(r_all, [0.05, 0.25, 0.5, 0.75, 0.95]).round(3).tolist())),
                                   frac_ratio_gt_1=float(np.mean(r_all > 1.0)))
    out = args.out or os.path.join(ext, "export_ratio_distribution.pdf")
    save_figure(fig, out)
    plt.close(fig)
    json.dump(summary, open(os.path.splitext(out)[0] + ".json", "w"), indent=2)
    if verbose:
        print(json.dumps(summary, indent=2))
        print(out)
    return out


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Displacement comparison drawn on the full specimen (style of plot_ttc_full_mesh_roi.py), at one load step:
  top row:    (a) measured DIC field on the ROI, inside the deformed specimen outline; then one panel per model with its
              FEM field on the whole deformed full mesh and the measured ROI outline on top;
  bottom row: parity of the ROI displacements (all models, validation steps), then the per-node error
              |u_FEM - u_DIC| on the ROI for each model.
Everything in the specimen frame (DIC data mapped with the inverse of the affine map X -> X_dic of the mesh).

Inputs: u_full_<model>.npz from validation/forward_fem_full_specimen.py.
Usage: python3 plots/plot_full_mesh_displacement.py --fem_dir <seed>/fem_validation/ttf_full \
           --mesh mesh/TTf_full_mesh.npz --dataset dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz [--step 19]
Writes <fem_dir>/displacement_full_mesh_roi.pdf
"""
import argparse
import glob
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def boundary_edges(cells):
    E = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return u[c == 1]


def r2(a, b):
    return 1.0 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fem_dir", required=True)
    p.add_argument("--mesh", default=os.path.join(ROOT, "mesh", "TTf_full_mesh.npz"))
    p.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "preprocessed", "experiment", "ttf_20_steps_ps_incomp.npz"))
    p.add_argument("--models", nargs="*", default=None,
                   help="model keys (u_full_<key>.npz); default: distilled first, then the others found")
    p.add_argument("--step", type=int, default=-1, help="load step index of the field panels (default: last)")
    p.add_argument("--val_steps", type=int, nargs="*", default=None, help="steps of the parity panel (default: all)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.lines import Line2D
    from plots.theme import apply_style, save_figure

    m = np.load(args.mesh, allow_pickle=True)
    X, cells = np.asarray(m["X"], float), np.asarray(m["tris"])
    ds = np.load(args.dataset, allow_pickle=True)
    xd, dcells, u_obs = np.asarray(ds["mesh_pos"])[:, :2].astype(float), np.asarray(ds["cells"]), np.asarray(ds["u_exp"])
    steps = np.asarray(ds["steps"])
    k = args.step if args.step >= 0 else len(steps) + args.step
    vs = np.arange(len(steps)) if args.val_steps is None else np.asarray(args.val_steps)
    if "X_dic" in m.files:      # specimen frame: inverse of the affine map X -> X_dic
        M = np.linalg.lstsq(np.c_[X, np.ones(len(X))], np.asarray(m["X_dic"], float), rcond=None)[0]
        Ai = np.linalg.inv(M[:2])
        xs, to_s = (xd - M[2]) @ Ai, (lambda v: v @ Ai)
    else:                       # mesh already in the DIC frame
        xs, to_s = xd, (lambda v: v)
    u_meas = to_s(u_obs)                                                   # (steps, m, 2)

    keys = args.models
    if keys is None:
        found = [os.path.basename(f)[7:-4] for f in sorted(glob.glob(os.path.join(args.fem_dir, "u_full_*.npz")))]
        keys = [f for f in found if f.startswith("distilled")] + [f for f in found if not f.startswith("distilled")]
    models = []
    for key in keys:
        d = np.load(os.path.join(args.fem_dir, f"u_full_{key}.npz"), allow_pickle=True)
        lab = str(d["label"])
        short = ("Distilled (mean)" if key == "distilled_mean" else "Distilled (samples)" if key.startswith("distilled")
                 else lab.replace(" (reference)", ""))
        models.append(dict(key=key, label=short, u_full=np.asarray(d["u_full"]),
                           u_roi=to_s(np.asarray(d["u_dic"])), grip=float(np.asarray(d["grip"])[k])))
    if not models:
        raise SystemExit(f"no u_full_*.npz in {args.fem_dir}")

    BE, BD = boundary_edges(cells), boundary_edges(dcells)
    xs_def = xs + u_meas[k]
    mag_meas = np.linalg.norm(u_meas[k], axis=1)
    vmax = max([mag_meas.max()] + [np.linalg.norm(md["u_full"][k], axis=1).max() for md in models])
    errs = [np.linalg.norm(md["u_roi"][k] - u_meas[k], axis=1) for md in models]
    emax = max(e.max() for e in errs)

    apply_style()
    n = len(models) + 1
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 10.0), layout="constrained")
    roi_col = "#e65100"
    cmap_u, cmap_e = "Blues", "Oranges"

    def outline(ax, Xf, color="#111111", lw=0.9, z=4):
        for a, b in BE:
            ax.plot(Xf[[a, b], 0], Xf[[a, b], 1], color=color, lw=lw, zorder=z)

    def roi_outline(ax, P, z=5):
        for a, b in BD:
            ax.plot(P[[a, b], 0], P[[a, b], 1], color=roi_col, lw=1.3, zorder=z)

    # (a) measured field on the ROI, inside the specimen outline of the first model's FEM
    ax = axes[0, 0]
    outline(ax, X + models[0]["u_full"][k], color="#9e9e9e", lw=0.8)
    tpc = ax.tripcolor(mtri.Triangulation(xs_def[:, 0], xs_def[:, 1], dcells), mag_meas, cmap=cmap_u, vmin=0,
                       vmax=vmax, shading="gouraud", zorder=2)
    roi_outline(ax, xs_def)
    ax.set_title(f"(a) Measured (DIC ROI)\nstep {int(steps[k])}, grip {models[0]['grip']:.1f} mm", fontsize=10.5)

    letters = "bcdefghij"
    for j, md in enumerate(models):
        ax = axes[0, j + 1]
        Xf = X + md["u_full"][k]
        ax.tripcolor(mtri.Triangulation(Xf[:, 0], Xf[:, 1], cells), np.linalg.norm(md["u_full"][k], axis=1),
                     cmap=cmap_u, vmin=0, vmax=vmax, shading="gouraud", zorder=1)
        outline(ax, Xf)
        roi_outline(ax, xs_def)
        ax.set_title(f"({letters[j]}) FEM: {md['label']}", fontsize=10.5)
    for ax in axes[0]:
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("y (mm)")
    cb = fig.colorbar(tpc, ax=list(axes[0]), fraction=0.03, pad=0.01)
    cb.set_label(r"$\|\mathbf{u}\|$ (mm)")

    # bottom row: parity, then the error maps on the ROI
    ax = axes[1, 0]
    lim = [min(u_meas[vs].min(), min(md["u_roi"][vs].min() for md in models)),
           max(u_meas[vs].max(), max(md["u_roi"][vs].max() for md in models))]
    cols = ["#1565c0", "#6a1b9a", "#2e7d32", "#c62828"]
    lines = []
    for j, md in enumerate(models):
        o, q = u_meas[vs].reshape(-1, 2), md["u_roi"][vs].reshape(-1, 2)
        for c, mk in ((0, "o"), (1, "^")):
            ax.scatter(o[:, c], q[:, c], s=2, marker=mk, color=cols[j % 4], alpha=0.15 if c == 0 else 0.3,
                       rasterized=True, lw=0, zorder=3 + len(models) - j)     # first model on top
        rmse = np.sqrt(np.mean((o - q) ** 2, axis=0))
        lines.append(Line2D([], [], color=cols[j % 4], marker="o", ls="", ms=5,
                            label=f"{md['label']}: $R^2_x$ {r2(o[:, 0], q[:, 0]):.3f}, $R^2_y$ {r2(o[:, 1], q[:, 1]):.3f}; "
                                  f"\nRMSE$_{{x,y}}$ {rmse[0]:.2f} / {rmse[1]:.2f} mm"))
    ax.plot(lim, lim, "k--", lw=0.9)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$u_{\mathrm{DIC}}$ (mm)")
    ax.set_ylabel(r"$u_{\mathrm{FEM}}$ (mm)")
    ax.set_title(r"(%s) Parity on the ROI" "\n" r"($u_x$: o, $u_y$: $\triangle$; specimen frame)" % letters[len(models)],
                 fontsize=10.5)
    ax.legend(handles=lines, fontsize=7, loc="upper left", frameon=True, framealpha=0.9)
    ax.grid(alpha=0.25)

    for j, (md, e) in enumerate(zip(models, errs)):
        ax = axes[1, j + 1]
        Xf = X + md["u_full"][k]
        outline(ax, Xf, color="#9e9e9e", lw=0.8)
        P = xs + md["u_roi"][k]          # ROI at the FEM-deformed positions
        tpe = ax.tripcolor(mtri.Triangulation(P[:, 0], P[:, 1], dcells), e, cmap=cmap_e, vmin=0, vmax=emax,
                           shading="gouraud", zorder=2)
        roi_outline(ax, xs_def, z=5)
        ax.set_title(f"({letters[len(models) + 1 + j]}) Error: {md['label']}\n"
                     f"mean {e.mean():.2f} mm, max {e.max():.2f} mm", fontsize=10.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.grid(alpha=0.2)
    cb = fig.colorbar(tpe, ax=list(axes[1, 1:]), fraction=0.035, pad=0.01)
    cb.set_label(r"$\|\mathbf{u}_{\mathrm{FEM}} - \mathbf{u}_{\mathrm{DIC}}\|$ (mm)")
    fig.legend(handles=[Line2D([], [], color=roi_col, lw=1.5, label="measured ROI outline (DIC)"),
                        Line2D([], [], color="#111111", lw=1, label="FEM specimen outline")],
               loc="outside lower center", ncol=2, fontsize=8.5, frameon=False)
    out = args.out or os.path.join(args.fem_dir, "displacement_full_mesh_roi.pdf")
    save_figure(fig, out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

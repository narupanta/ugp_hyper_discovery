#!/usr/bin/env python3
"""
FEM with the measured DIC displacements prescribed on the two end cuts (validation/forward_fem_dic_boundary.py):
discovered model vs reference model on the DIC mesh.

Rows: models (default: distilled mean = ours, then the reference). Columns at the plotted load step:
  deformed domain (FEM mesh, measured deformed outline on top), |u| FEM, |u| DIC (measured), nodal error
  |u_FEM - u_DIC|, and the parity plot u_FEM vs u_DIC over all load steps with R^2 of u_x and u_y.
Each field column shares one colour scale across the rows.

Usage:
  python3 plots/plot_dic_bc_comparison.py --fem_dir results/<exp>/1/fem_validation/ttf_dic_bc \
      [--models distilled_mean euclid_ttf_ut] [--dataset dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz]
Writes <fem_dir>/dic_bc_comparison.pdf/.png and dic_bc_comparison.json (R^2, RMSE, mean / max error).
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.theme import apply_style  # noqa: E402

ROW_NAMES = {"distilled_mean": "Ours (distilled)", "distilled_samples": "Ours (distilled)"}


def boundary_edges(cells):
    e = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


def r2(o, p):
    return 1.0 - np.sum((o - p) ** 2) / np.sum((o - o.mean()) ** 2)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fem_dir", required=True)
    p.add_argument("--models", nargs="+", default=["distilled_mean", "euclid_ttf_ut"])
    p.add_argument("--dataset", default="dataset/preprocessed/experiment/ttf_20_steps_ps_incomp.npz")
    p.add_argument("--plot_step", type=int, default=-1)
    p.add_argument("--figsize", type=float, nargs=2, default=None, help="inches (default: A4 width)")
    args = p.parse_args()
    apply_style()
    d = np.load(args.dataset, allow_pickle=True)
    X, cells = np.asarray(d["mesh_pos"], float)[:, :2], np.asarray(d["cells"])
    u_obs, steps, bcx = np.asarray(d["u_exp"], float), np.asarray(d["steps"]), np.asarray(d["bcx"]).ravel()
    k = args.plot_step % len(steps)
    res = {}
    for m in args.models:
        z = np.load(os.path.join(args.fem_dir, f"u_dicbc_{m}.npz"), allow_pickle=True)
        lab = str(z["label"]).replace(" (reference)", "")
        res[m] = dict(u=np.asarray(z["u"], float), label=ROW_NAMES.get(m, lab))
    X = X - X.min(0)
    bed = boundary_edges(cells)
    ends = [np.where(bcx == e)[0] for e in (1, 2)]
    obs_def = X + u_obs[k]
    mag_obs = np.linalg.norm(u_obs[k], axis=1)
    vmax = max([mag_obs.max()] + [np.linalg.norm(r["u"][k], axis=1).max() for r in res.values()])
    emax = max(np.linalg.norm(r["u"][k] - u_obs[k], axis=1).max() for r in res.values())
    o_all = u_obs.reshape(-1, 2)
    lim = [min(o_all.min(), min(r["u"].min() for r in res.values())), max(o_all.max(), max(r["u"].max() for r in res.values()))]

    nr, nc = len(res), 5
    W = args.figsize[0] if args.figsize else 210 / 25.4
    H = args.figsize[1] if args.figsize else 1.95 * nr + 0.55
    fig = plt.figure(figsize=(W, H))
    gs = fig.add_gridspec(nr + 1, nc, height_ratios=[1] * nr + [0.05], hspace=0.10, wspace=0.18,
                          left=0.06, right=0.995, top=0.94, bottom=0.08)
    titles = ["Deformed domain", r"$|\mathbf{u}|$ FEM", r"$|\mathbf{u}|$ DIC (measured)",
              r"$|\mathbf{u}_{\mathrm{FEM}} - \mathbf{u}_{\mathrm{DIC}}|$", "Parity (all load steps)"]
    summary = {}
    tp = te = None
    for i, (m, r) in enumerate(res.items()):
        u = r["u"]
        P = X + u[k]
        tri = mtri.Triangulation(P[:, 0], P[:, 1], cells)
        tri_o = mtri.Triangulation(obs_def[:, 0], obs_def[:, 1], cells)
        err = np.linalg.norm(u[k] - u_obs[k], axis=1)
        axs = [fig.add_subplot(gs[i, j]) for j in range(nc)]
        # 1 deformed domain: FEM mesh, measured deformed outline, the prescribed end cuts
        axs[0].triplot(tri, color="#bdbdbd", lw=0.08)
        for a, b in bed:
            axs[0].plot(obs_def[[a, b], 0], obs_def[[a, b], 1], color="#c62828", lw=0.6)
        for n in ends:
            n = n[np.argsort(X[n, 1])]
            axs[0].plot(P[n, 0], P[n, 1], color="k", lw=1.6)
        # 2-4 fields on the deformed configuration
        tp = axs[1].tripcolor(tri, np.linalg.norm(u[k], axis=1), cmap="Blues", vmin=0, vmax=vmax, shading="gouraud")
        axs[2].tripcolor(tri_o, mag_obs, cmap="Blues", vmin=0, vmax=vmax, shading="gouraud")
        te = axs[3].tripcolor(tri, err, cmap="Oranges", vmin=0, vmax=emax, shading="gouraud")
        for ax in axs[:4]:
            ax.set_aspect("equal")
            ax.axis("off")
        # 5 parity over all steps
        f_all = u.reshape(-1, 2)
        r2x, r2y = r2(o_all[:, 0], f_all[:, 0]), r2(o_all[:, 1], f_all[:, 1])
        rm = np.sqrt(np.mean((o_all - f_all) ** 2, axis=0))
        a5 = axs[4]
        a5.scatter(o_all[:, 0], f_all[:, 0], s=0.6, color="#1565c0", alpha=0.25, lw=0, rasterized=True)
        a5.scatter(o_all[:, 1], f_all[:, 1], s=0.6, color="#c62828", alpha=0.35, lw=0, rasterized=True)
        a5.plot(lim, lim, "k--", lw=0.7)
        a5.set_xlim(lim)
        a5.set_ylim(lim)
        a5.set_box_aspect(1)
        a5.grid(False)
        a5.tick_params(labelsize=6, length=2, pad=1.5)
        a5.set_ylabel(r"$u_{\mathrm{FEM}}$ (mm)", fontsize=7, labelpad=1)
        if i == nr - 1:
            a5.set_xlabel(r"$u_{\mathrm{DIC}}$ (mm)", fontsize=7, labelpad=1)
        else:
            a5.tick_params(labelbottom=False)
        a5.text(0.04, 0.96, rf"$R^2_x = {r2x:.4f}$" + "\n" + rf"$R^2_y = {r2y:.4f}$", transform=a5.transAxes,
                fontsize=6.3, va="top", ha="left")
        a5.text(0.96, 0.04, r"$u_x$", color="#1565c0", fontsize=7, transform=a5.transAxes, ha="right", va="bottom")
        a5.text(0.96, 0.16, r"$u_y$", color="#c62828", fontsize=7, transform=a5.transAxes, ha="right", va="bottom")
        axs[0].text(-0.08, 0.5, r["label"], transform=axs[0].transAxes, rotation=90, ha="right", va="center", fontsize=8.5)
        if i == 0:
            for ax, t in zip(axs, titles):
                ax.set_title(t, fontsize=7.5, pad=3)
        summary[m] = dict(label=r["label"], r2_x=float(r2x), r2_y=float(r2y), rmse_x=float(rm[0]), rmse_y=float(rm[1]),
                          err_mean_plot_step=float(err.mean()), err_max_plot_step=float(err.max()))
        print(f"{r['label']}: R2 x {r2x:.4f} y {r2y:.4f} | RMSE x {rm[0]:.3f} y {rm[1]:.3f} mm | "
              f"error at step {int(steps[k])}: mean {err.mean():.2f}, max {err.max():.2f} mm")
    # colour bars under the field columns (shared by the rows)
    for j, (mappable, lab) in enumerate([(tp, r"$|\mathbf{u}|$ (mm)"), (te, "error (mm)")]):
        cols = slice(1, 3) if j == 0 else slice(3, 4)
        cax = fig.add_subplot(gs[nr, cols])
        cb = fig.colorbar(mappable, cax=cax, orientation="horizontal")
        cb.set_label(lab, fontsize=6.5, labelpad=1)
        cb.ax.tick_params(labelsize=5.5, length=2, pad=1)
    lax = fig.add_subplot(gs[nr, 0])
    lax.axis("off")
    lax.plot([], [], color="#bdbdbd", lw=0.8, label="FEM mesh")
    lax.plot([], [], color="#c62828", lw=0.8, label="measured outline")
    lax.plot([], [], color="k", lw=1.6, label="prescribed end cuts")
    lax.legend(loc="center", fontsize=5.5, frameon=False, handlelength=1.4, borderaxespad=0)
    out = os.path.join(args.fem_dir, "dic_bc_comparison")
    fig.savefig(out + ".pdf", bbox_inches="tight", dpi=300)
    fig.set_dpi(220)
    fig.canvas.print_png(out + ".png")
    plt.close(fig)
    with open(out + ".json", "w") as f:
        json.dump(dict(dataset=args.dataset, plot_step=int(steps[k]), models=summary), f, indent=2)
    print(f"saved {out}.pdf/.png/.json")


if __name__ == "__main__":
    main()

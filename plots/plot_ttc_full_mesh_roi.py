#!/usr/bin/env python3
"""
The DIC data mesh on top of the full TTc specimen mesh, with the region of interest (ROI) highlighted:
(a) reference configuration; (b) deformed at one load step: full-mesh FEM (u_full_<model>.npz from
validation/forward_fem_full_specimen.py) and the measured DIC field, both in the specimen frame.
DIC -> specimen frame: inverse of the affine map X -> X_dic fitted from the mesh (TTc, TTf); TTe meshes are in the DIC
frame. Clamped tab regions (TTf, TTe) are shaded; TTc uses grip edges or clamp lines.

Usage: python3 plots/plot_ttc_full_mesh_roi.py --fem_dir <seed>/fem_validation/ttc_full [--model distilled_mean] [--step 19]
Writes <fem_dir>/full_mesh_with_dic_roi.pdf
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def boundary_edges(cells):
    E = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return u[c == 1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fem_dir", required=True)
    p.add_argument("--model", default="distilled_mean")
    p.add_argument("--step", type=int, default=-1, help="load step index of panel (b) (default: last)")
    p.add_argument("--clamp_bottom_y", type=float, default=None, help="pulled clamp line (default: the bottom edge)")
    p.add_argument("--clamp_top_y", type=float, default=None, help="held clamp line (default: the top edge)")
    p.add_argument("--clamp_distance", type=float, default=None,
                   help="TTe / TTf layouts: clamped regions of this length from each end (default: the mesh's tab regions)")
    p.add_argument("--mesh", default=os.path.join(ROOT, "mesh", "TTc_full_mesh.npz"))
    p.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "preprocessed", "experiment", "ttc_20_steps_ps_incomp.npz"))
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from plots.theme import apply_style, save_figure

    m = np.load(args.mesh, allow_pickle=True)
    X, cells = np.asarray(m["X"], float), np.asarray(m["tris"])
    ds = np.load(args.dataset, allow_pickle=True)
    xd, dcells, u_dic = np.asarray(ds["mesh_pos"])[:, :2].astype(float), np.asarray(ds["cells"]), np.asarray(ds["u_exp"])
    steps = np.asarray(ds["steps"])
    k = args.step if args.step >= 0 else len(steps) + args.step
    tabs = None
    if "X_dic" in m.files:      # TTc / TTf layout: specimen frame (affine X -> X_dic), loading along y
        M = np.linalg.lstsq(np.c_[X, np.ones(len(X))], np.asarray(m["X_dic"], float), rcond=None)[0]
        Ai = np.linalg.inv(M[:2])
        nb, nt = np.asarray(m["nodes_bottom"]), np.asarray(m["nodes_top"])
        xs = (xd - M[2]) @ Ai
        u_meas = u_dic[k] @ Ai
        ax_load, ax_name = 1, "y"
        names = [str(n) for n in np.asarray(m["region_names"])] if "region_names" in m.files else []
        no_lines = args.clamp_bottom_y is None and args.clamp_top_y is None
        if no_lines and (args.clamp_distance is not None or ("tab_bottom" in names and "tab_top" in names)):
            if args.clamp_distance is not None:
                yb, yt = X[:, 1].min() + args.clamp_distance, X[:, 1].max() - args.clamp_distance
            else:
                region = np.asarray(m["region"])
                yb = float(X[np.unique(cells[region == names.index("tab_bottom")]), 1].max())
                yt = float(X[np.unique(cells[region == names.index("tab_top")]), 1].min())
            in_b = np.all(X[cells, 1] <= yb + 1e-6, axis=1)
            in_t = np.all(X[cells, 1] >= yt - 1e-6, axis=1)
            u_last = (u_dic[-1] @ Ai)[:, 1]
            top_pulled = (abs(u_last[xs[:, 1] > xs[:, 1].max() - 1].mean())
                          > abs(u_last[xs[:, 1] < xs[:, 1].min() + 1].mean()))
            tabs = dict(pulled=in_t if top_pulled else in_b, held=in_b if top_pulled else in_t)
            nb = nt = np.array([], dtype=int)
    else:                       # TTe layout: the mesh is in the DIC frame, loading along x, gripped tabs as regions
        xs, u_meas, ax_load, ax_name = xd, u_dic[k], 0, "x"
        region = np.asarray(m["region"])
        names = [str(n) for n in np.asarray(m["region_names"])]
        tab_ids = {n: i for i, n in enumerate(names) if n.startswith("tab")}
        cx = {n: float(X[cells[region == i]].reshape(-1, 2)[:, 0].mean()) for n, i in tab_ids.items()}
        right, left = max(cx, key=cx.get), min(cx, key=cx.get)
        u_last = u_dic[-1, :, 0]
        moves_right = abs(u_last[xd[:, 0] > xd[:, 0].max() - 1].mean()) > abs(u_last[xd[:, 0] < xd[:, 0].min() + 1].mean())
        pulled, held = (right, left) if moves_right else (left, right)
        tabs = dict(pulled=region == tab_ids[pulled], held=region == tab_ids[held])
        if args.clamp_distance is not None:   # elements entirely within the clamp length of each end
            xr, xl = X[:, 0].max() - args.clamp_distance, X[:, 0].min() + args.clamp_distance
            in_r = np.all(X[cells, 0] >= xr - 1e-6, axis=1)
            in_l = np.all(X[cells, 0] <= xl + 1e-6, axis=1)
            tabs = dict(pulled=in_r if moves_right else in_l, held=in_l if moves_right else in_r)
        nb = nt = np.array([], dtype=int)
    fem = np.load(os.path.join(args.fem_dir, f"u_full_{args.model}.npz"), allow_pickle=True)
    u_fem = np.asarray(fem["u_full"])[k]
    label = str(fem["label"])
    grip = float(np.asarray(fem["grip"])[k])

    BE, BD = boundary_edges(cells), boundary_edges(dcells)
    if args.clamp_bottom_y is not None or args.clamp_top_y is not None:
        sys.path.insert(0, os.path.join(ROOT, "validation"))
        from forward_fem_full_specimen import clamp_line_nodes
        if args.clamp_bottom_y is not None:
            nb = clamp_line_nodes(X, cells, args.clamp_bottom_y)[0]
        if args.clamp_top_y is not None:
            nt = clamp_line_nodes(X, cells, args.clamp_top_y)[0]
    nb, nt = nb[np.argsort(X[nb, 0])], nt[np.argsort(X[nt, 0])]
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 9.0) if ax_load == 1 else (13.0, 6.6))
    roi_col, dic_col = "#e65100", "#1565c0"
    for ax, (Xf, Xdic, title) in zip(axes, [
            (X, xs, "(a) Reference configuration"),
            (X + u_fem, xs + u_meas, f"(b) Step {int(steps[k])}: grip displacement {grip:.1f} mm")]):
        ax.triplot(mtri.Triangulation(Xf[:, 0], Xf[:, 1], cells), color="#9e9e9e", lw=0.25, zorder=1)
        for a, b in BE:
            ax.plot(Xf[[a, b], 0], Xf[[a, b], 1], color="#111111", lw=0.9, zorder=2)
        ax.tripcolor(mtri.Triangulation(Xdic[:, 0], Xdic[:, 1], dcells), np.ones(len(Xdic)), cmap="Oranges",
                     vmin=0, vmax=3.0, alpha=0.30, zorder=2)
        ax.triplot(mtri.Triangulation(Xdic[:, 0], Xdic[:, 1], dcells), color=dic_col, lw=0.2, alpha=0.7, zorder=3)
        for a, b in BD:
            ax.plot(Xdic[[a, b], 0], Xdic[[a, b], 1], color=roi_col, lw=1.6, zorder=4)
        if tabs is None:
            ax.plot(Xf[nb, 0], Xf[nb, 1], color="#1b5e20", lw=4, solid_capstyle="butt", zorder=5)
            ax.plot(Xf[nt, 0], Xf[nt, 1], color="#6a1b9a", lw=4, solid_capstyle="butt", zorder=5)
        else:                   # clamped tab regions
            for key, col in (("pulled", "#1b5e20"), ("held", "#6a1b9a")):
                ax.tripcolor(mtri.Triangulation(Xf[:, 0], Xf[:, 1], cells[tabs[key]]), np.ones(len(Xf)),
                             cmap="Greens" if key == "pulled" else "Purples", vmin=0, vmax=2.2, alpha=0.45, zorder=1.5)
        ax.set_aspect("equal")
        ax.set_xlabel("x (mm)")
        ax.set_title(title, fontsize=11)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("y (mm)")
    y_lo, y_hi = xs[:, ax_load].min(), xs[:, ax_load].max()
    axes[0].text(0.02, 0.02, f"ROI: {ax_name} = {y_lo:.1f} ... {y_hi:.1f} mm (length {y_hi - y_lo:.1f} mm)",
                 transform=axes[0].transAxes, fontsize=8.5, color=roi_col, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=10)
    # panel (b): where the ROI ends are in the measurement
    lo_m, hi_m = (xs + u_meas)[:, ax_load].min(), (xs + u_meas)[:, ax_load].max()
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#9e9e9e", lw=1, label=f"full mesh ((b): FEM, {label})"),
               Line2D([], [], color=dic_col, lw=1, label="DIC data mesh ((b): measured)"),
               Line2D([], [], color=roi_col, lw=2, label="ROI outline"),
               Line2D([], [], color="#1b5e20", lw=4, label=("pulled clamp line" if args.clamp_bottom_y is not None else "bottom grip (pulled)")
                      if tabs is None else "clamped tab (pulled)"),
               Line2D([], [], color="#6a1b9a", lw=4, label=("held clamp line" if args.clamp_top_y is not None else "top grip (held)")
                      if tabs is None else "clamped tab (held)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
    axes[1].text(0.02, 0.02, f"measured ROI spans {ax_name} = {lo_m:.1f} ... {hi_m:.1f} mm", transform=axes[1].transAxes,
                 fontsize=8.5, color=roi_col, va="bottom",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85), zorder=10)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = os.path.join(args.fem_dir, "full_mesh_with_dic_roi.pdf")
    save_figure(fig, out)
    plt.close(fig)
    print(out)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Where the DIC region of interest (ROI) sits on the full TTc specimen, in the reference configuration.

The DIC mesh (reference node positions, DIC frame) is mapped into the specimen frame of mesh/TTc_full_mesh.npz with
the mesh's transform x_dic = tx - y, y_dic = x + ty (i.e. x = y_dic - ty, y = tx - x_dic). The DIC boundary nodes on the
specimen sides and around the holes should then lie on the full mesh's boundary; their distance to it measures the
alignment. A rigid correction (rotation about the ROI centre + shift) that minimises those distances is fitted and
reported. The two cut lines that close the ROI at its ends are inside the material and are left out of the fit.

Usage: python3 plots/plot_ttc_roi_alignment.py [--mesh mesh/TTc_full_mesh.npz]
       [--dataset dataset/preprocessed/experiment/ttc_20_steps_ps_incomp.npz] [--out mesh/TTc_roi_alignment]
Writes <out>.pdf/.png and <out>.json.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def boundary_edges(cells):
    E = np.sort(np.vstack([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    u, c = np.unique(E, axis=0, return_counts=True)
    return u[c == 1]


def point_segment_distance(P, A, B):
    """Distance of points P (n, 2) to the nearest of segments A-B (m, 2): (n,) and the nearest point (n, 2)."""
    AB = B - A
    L2 = np.maximum(np.sum(AB ** 2, 1), 1e-30)
    best_d = np.full(len(P), np.inf)
    best_q = np.zeros_like(P)
    for i0 in range(0, len(P), 512):                  # chunked (n x m)
        p = P[i0:i0 + 512, None, :]
        t = np.clip(np.sum((p - A[None]) * AB[None], -1) / L2[None], 0.0, 1.0)
        Q = A[None] + t[..., None] * AB[None]
        d = np.linalg.norm(p - Q, axis=-1)
        j = np.argmin(d, axis=1)
        best_d[i0:i0 + 512] = d[np.arange(len(j)), j]
        best_q[i0:i0 + 512] = Q[np.arange(len(j)), j]
    return best_d, best_q


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mesh", default=os.path.join(ROOT, "mesh", "TTc_full_mesh.npz"))
    p.add_argument("--dataset", default=os.path.join(ROOT, "dataset", "preprocessed", "experiment", "ttc_20_steps_ps_incomp.npz"))
    p.add_argument("--out", default=os.path.join(ROOT, "mesh", "TTc_roi_alignment"))
    args = p.parse_args()

    m = np.load(args.mesh)
    X, cells, tx, ty = np.asarray(m["X"], float), np.asarray(m["tris"]), float(m["tx"]), float(m["ty"])
    nb, nt = np.asarray(m["nodes_bottom"]), np.asarray(m["nodes_top"])
    ds = np.load(args.dataset, allow_pickle=True)
    xd, dcells = np.asarray(ds["mesh_pos"])[:, :2].astype(float), np.asarray(ds["cells"])

    # DIC reference positions in the specimen frame
    to_spec = lambda q: np.stack([q[:, 1] - ty, tx - q[:, 0]], axis=1)
    xs = to_spec(xd)
    y_lo, y_hi = xs[:, 1].min(), xs[:, 1].max()

    # boundaries: full mesh (segments) and DIC mesh (nodes); drop the ROI end cuts (inside the material)
    BE = boundary_edges(cells)
    A, B = X[BE[:, 0]], X[BE[:, 1]]
    dic_b = np.unique(boundary_edges(dcells))
    end_tol = 1.5
    keep = (xs[dic_b, 1] > y_lo + end_tol) & (xs[dic_b, 1] < y_hi - end_tol)
    fit_nodes = dic_b[keep]
    P0 = xs[fit_nodes]
    d0, _ = point_segment_distance(P0, A, B)

    # rigid correction: rotation about the ROI centre + translation, least squares on point-to-boundary distances
    from scipy.optimize import least_squares
    c0 = xs.mean(0)

    def transform(q, th, dx, dy):
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        return (q - c0) @ R.T + c0 + np.array([dx, dy])

    def resid(v):
        return point_segment_distance(transform(P0, *v), A, B)[0]
    fit = least_squares(resid, np.zeros(3), loss="soft_l1", f_scale=0.5)
    th, dx, dy = fit.x
    d1 = resid(fit.x)

    # nodes of the DIC mesh outside the full mesh, before and after the correction
    import matplotlib.tri as mtri
    finder = mtri.Triangulation(X[:, 0], X[:, 1], cells).get_trifinder()
    out0 = finder(xs[:, 0], xs[:, 1]) < 0
    xs1 = transform(xs, th, dx, dy)
    out1 = finder(xs1[:, 0], xs1[:, 1]) < 0
    hole_nodes = np.asarray(m["nodes_holes"])

    summary = dict(
        transform="x = y_dic - ty, y = tx - x_dic", tx=tx, ty=ty,
        roi_reference_specimen_frame=dict(x=[float(xs[:, 0].min()), float(xs[:, 0].max())], y=[float(y_lo), float(y_hi)],
                                          length_y=float(y_hi - y_lo)),
        specimen=dict(x=[float(X[:, 0].min()), float(X[:, 0].max())], y=[float(X[:, 1].min()), float(X[:, 1].max())]),
        free_length_outside_roi=dict(below=float(y_lo - X[:, 1].min()), above=float(X[:, 1].max() - y_hi)),
        boundary_distance_given=dict(mean=float(d0.mean()), median=float(np.median(d0)), p95=float(np.quantile(d0, 0.95)),
                                     max=float(d0.max()), n_nodes=int(len(d0))),
        rigid_correction=dict(rotation_deg=float(np.degrees(th)), shift_x=float(dx), shift_y=float(dy)),
        boundary_distance_corrected=dict(mean=float(d1.mean()), median=float(np.median(d1)),
                                         p95=float(np.quantile(d1, 0.95)), max=float(d1.max())),
        dic_nodes_outside_mesh=dict(given=int(out0.sum()), corrected=int(out1.sum()), total=int(len(xs))))

    # --- figure: (a) whole specimen with the ROI, (b) zoom on the ROI boundary alignment
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from plots.theme import apply_style, save_figure
    apply_style()
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.5, 7.2), gridspec_kw=dict(width_ratios=[0.8, 1.6]))
    tri_full = mtri.Triangulation(X[:, 0], X[:, 1], cells)
    ax0.triplot(tri_full, color="#bdbdbd", lw=0.25)
    for a, b in BE:
        ax0.plot(X[[a, b], 0], X[[a, b], 1], color="#111111", lw=0.9)
    ax0.plot(X[nb, 0], X[nb, 1], color="#1565c0", lw=4, solid_capstyle="butt", label="bottom grip edge (pulled)")
    ax0.plot(X[nt, 0], X[nt, 1], color="#6a1b9a", lw=4, solid_capstyle="butt", label="top grip edge (held)")
    ax0.tripcolor(mtri.Triangulation(xs[:, 0], xs[:, 1], dcells), np.ones(len(xs)), cmap="Oranges", vmin=0, vmax=2.5,
                  alpha=0.55)
    for yv in (y_lo, y_hi):
        ax0.axhline(yv, color="#e65100", ls="--", lw=1.0)
    ax0.text(51.5, 0.5 * (y_lo + y_hi), f"DIC ROI\ny = {y_lo:.1f} ... {y_hi:.1f} mm\n(length {y_hi - y_lo:.1f} mm)",
             color="#e65100", fontsize=8.5, va="center")
    ax0.text(51.5, 0.5 * y_lo, f"{y_lo - X[:, 1].min():.1f} mm\nbelow ROI", fontsize=8, va="center", color="#444")
    ax0.text(51.5, 0.5 * (y_hi + X[:, 1].max()), f"{X[:, 1].max() - y_hi:.1f} mm\nabove ROI", fontsize=8, va="center",
             color="#444")
    ax0.set_aspect("equal")
    ax0.set_xlim(-5, 80)
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("y (mm)")
    ax0.set_title("(a) Full specimen, reference configuration")
    ax0.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, -0.08), frameon=False)

    for a, b in BE:
        ax1.plot(X[[a, b], 0], X[[a, b], 1], color="#111111", lw=1.0, zorder=2)
    ax1.plot(X[hole_nodes, 0], X[hole_nodes, 1], ".", color="#111111", ms=1.5, zorder=2)
    sc = ax1.scatter(P0[:, 0], P0[:, 1], c=d0, s=9, cmap="viridis", vmin=0, vmax=max(1e-3, np.quantile(d0, 0.98)),
                     zorder=4, label="DIC boundary nodes (given transform)")
    ax1.scatter(xs[out0, 0], xs[out0, 1], s=22, facecolors="none", edgecolors="#d32f2f", lw=1.0, zorder=5,
                label=f"DIC nodes outside the full mesh ({int(out0.sum())})")
    ax1.plot(xs[dic_b[~keep], 0], xs[dic_b[~keep], 1], "s", color="#e65100", ms=2, zorder=3, label="ROI end cuts (not fitted)")
    cb = fig.colorbar(sc, ax=ax1, fraction=0.035, pad=0.02)
    cb.set_label("distance to the full-mesh boundary (mm)")
    ax1.set_xlim(xs[:, 0].min() - 3, xs[:, 0].max() + 3)
    ax1.set_ylim(y_lo - 3, y_hi + 3)
    ax1.set_aspect("equal")
    ax1.set_xlabel("x (mm)")
    ax1.set_title("(b) ROI boundary against the full mesh")
    s = summary
    ax1.text(0.01, -0.13, (f"given transform: median distance {s['boundary_distance_given']['median']:.2f} mm, "
                           f"95% {s['boundary_distance_given']['p95']:.2f} mm, max {s['boundary_distance_given']['max']:.2f} mm\n"
                           f"best rigid correction: rotation {s['rigid_correction']['rotation_deg']:.2f} deg, shift "
                           f"({dx:.2f}, {dy:.2f}) mm -> median {s['boundary_distance_corrected']['median']:.2f} mm, "
                           f"95% {s['boundary_distance_corrected']['p95']:.2f} mm"),
             transform=ax1.transAxes, fontsize=8, va="top")
    ax1.legend(fontsize=7.5, loc="upper right", frameon=True, framealpha=0.9)
    fig.tight_layout()
    save_figure(fig, args.out + ".pdf")
    fig.savefig(args.out + ".png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    json.dump(summary, open(args.out + ".json", "w"), indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

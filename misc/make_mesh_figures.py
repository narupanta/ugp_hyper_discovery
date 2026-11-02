"""
Two mesh figures of the notched block (mesh/block_mesh.npz):
  misc/mesh_interior_arrows.png  - scattered arrows on interior nodes only (boundary nodes not drawn), zoom inset
  misc/mesh_boundary_arrows_v2.png - arrows on every boundary node, one colour per side, distributed like the right edge
Usage: python3 misc/make_mesh_figures.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.patches import ConnectionPatch, Rectangle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
d = np.load(os.path.join(ROOT, "mesh", "block_mesh.npz"))
X = d["node_coords"][:, :2]
C = d["cells"]
T = tri.Triangulation(X[:, 0], X[:, 1], C)

# boundary edges (edges used by exactly one triangle) and boundary nodes
E = np.sort(np.vstack([C[:, [0, 1]], C[:, [1, 2]], C[:, [2, 0]]]), axis=1)
uniq, cnt = np.unique(E, axis=0, return_counts=True)
BE = uniq[cnt == 1]
B_nodes = np.unique(BE)
is_b = np.zeros(len(X), bool)
is_b[B_nodes] = True
INT = np.where(~is_b)[0]


def triplot(ax, lw=0.5, color="k", alpha=1.0):
    ax.triplot(T, lw=lw, color=color, alpha=alpha, zorder=1)


# ----------------------------------------------------------------------------- 1) interior arrows, zoom inset
fig = plt.figure(figsize=(13, 6.2))
main = fig.add_axes([0.01, 0.03, 0.52, 0.94])
triplot(main, lw=0.6)
main.plot(X[INT, 0], X[INT, 1], "o", color="#d62728", ms=4.2, zorder=3)
main.set_aspect("equal")
main.axis("off")

# zoom window (data coordinates of the block) and the interior nodes inside it
zc, zw = np.array([0.40, 0.45]), 0.22
zlo, zhi = zc - zw / 2, zc + zw / 2
in_zoom = INT[np.all((X[INT] >= zlo) & (X[INT] <= zhi), axis=1)]
inset = fig.add_axes([0.60, 0.16, 0.38, 0.72])
triplot(inset, lw=1.4)
inset.plot(X[in_zoom, 0], X[in_zoom, 1], "o", color="#d62728", ms=9, zorder=3)
inset.set_xlim(zlo[0], zhi[0])
inset.set_ylim(zlo[1], zhi[1])
inset.set_aspect("equal")
inset.axis("off")
for sp in inset.spines.values():
    sp.set_visible(False)
inset.add_patch(Rectangle(zlo, zw, zw, fill=False, ec="#d62728", lw=2.2, ls=(0, (4, 3)), zorder=5,
                          transform=inset.transData, clip_on=False))

# the zoom window on the main mesh (dashed box, as in the reference)
main.add_patch(Rectangle(zlo, zw, zw, fill=False, ec="#d62728", lw=2.2, ls=(0, (4, 3)), zorder=5))

# scattered arrows on a few interior nodes of the zoom (varied directions, as in the reference)
rng = np.random.default_rng(3)
pick = in_zoom[np.linspace(0, len(in_zoom) - 1, min(6, len(in_zoom))).astype(int)]
angles = np.deg2rad(rng.uniform(-170, 80, len(pick)))
mags = rng.uniform(0.035, 0.06, len(pick))
inset.quiver(X[pick, 0], X[pick, 1], mags * np.cos(angles), mags * np.sin(angles), color="#d62728",
             angles="xy", scale_units="xy", scale=1, width=0.009, headwidth=4.5, zorder=6)

# connectors from the zoom window in the main mesh to the inset frame
for (xa, ya), (xb, yb) in [((zlo[0], zhi[1]), (0, 1)), ((zlo[0], zlo[1]), (0, 0))]:
    pass
ca = [(zlo[0], zhi[1]), (zlo[0], zlo[1])]
cb = [(zlo[0], zhi[1]), (zlo[0], zlo[1])]
for pa, pb in zip(ca, cb):
    fig.add_artist(ConnectionPatch(xyA=pa, coordsA="data", axesA=main, xyB=pb, coordsB="data", axesB=inset,
                                   color="#d62728", lw=1.6, zorder=0))
fig.savefig(os.path.join(ROOT, "misc", "mesh_interior_arrows.png"), dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------------- 2) arrows on boundary nodes
SIDES = {                      # name: (outward normal, colour, membership test)
    "top":    ((0.0, 1.0), "#1f77b4", lambda p: np.isclose(p[:, 1], 1.0)),
    "right":  ((1.0, 0.0), "#2ca02c", lambda p: np.isclose(p[:, 0], 1.0)),
    "left":   ((-1.0, 0.0), "#ff7f0e", lambda p: np.isclose(p[:, 0], 0.0)),
    "bottom": ((0.0, -1.0), "#9467bd", lambda p: np.isclose(p[:, 1], 0.0) & (p[:, 0] > 0.05)),
}
side_nodes = {}
taken = np.zeros(len(X), bool)
for name in ["left", "right", "top", "bottom"]:          # corners go to the first side that claims them
    n, col, test = SIDES[name]
    idx = B_nodes[test(X[B_nodes]) & ~taken[B_nodes]]
    taken[idx] = True
    side_nodes[name] = idx[np.argsort(-X[idx, 1] if name in ("left", "right") else X[idx, 0])]

fig, ax = plt.subplots(figsize=(7.8, 7.8))
triplot(ax, lw=0.6, color="#666666")
L_arrow, offset, band = 0.065, 0.0, 0.028     # arrow length, arrow tail offset from the node (0: on the node), band width
for name in ["left", "right", "top", "bottom"]:
    (nx, ny), col, _ = SIDES[name]
    idx = side_nodes[name]
    P = X[idx]
    # translucent band and edge line, as the reference's right-boundary load
    lo, hi = P.min(0), P.max(0)
    if name in ("left", "right"):
        x0 = 0.0 if name == "left" else 1.0
        ax.add_patch(Rectangle((x0 - band if name == "left" else x0, lo[1]), band, hi[1] - lo[1],
                               fc=col, alpha=0.16, ec="none", zorder=2))
        ax.plot([x0 - band if name == "left" else x0 + band] * 2, [lo[1], hi[1]], color=col, lw=2.2, zorder=3)
    else:
        y0 = 1.0 if name == "top" else 0.0
        ax.add_patch(Rectangle((lo[0], y0 - band if name == "bottom" else y0), hi[0] - lo[0], band,
                               fc=col, alpha=0.16, ec="none", zorder=2))
        ax.plot([lo[0], hi[0]], [y0 - band if name == "bottom" else y0 + band] * 2, color=col, lw=2.2, zorder=3)
    # one arrow per boundary node, pointing out of the body (same length and direction along the side)
    start = P + offset * np.array([nx, ny])
    ax.quiver(start[:, 0], start[:, 1], np.full(len(P), band * nx), np.full(len(P), band * ny),
              color=col, angles="xy", scale_units="xy", scale=1, width=0.0035, headwidth=3.4, headlength=3.8,
              zorder=4)
    ax.plot(P[:, 0], P[:, 1], "o", color=col, ms=2.6, zorder=3)
    # one large arrow at the middle of the coloured line, same direction, as the reference's central load arrow
    mid = 0.5 * (lo + hi)
    if name in ("left", "right"):
        mid = np.array([(0.0 - band if name == "left" else 1.0 + band), mid[1]])
    else:
        mid = np.array([mid[0], (1.0 + band if name == "top" else 0.0 - band)])
    ax.quiver([mid[0]], [mid[1]], [0.09 * nx], [0.09 * ny], color=col, angles="xy", scale_units="xy", scale=1,
              width=0.0085, headwidth=4.4, headlength=4.8, zorder=6)

ax.set_aspect("equal")
ax.axis("off")
ax.set_xlim(-0.14, 1.14)
ax.set_ylim(-0.14, 1.14)
fig.savefig(os.path.join(ROOT, "misc", "mesh_boundary_arrows_v2.png"), dpi=200, bbox_inches="tight")
plt.close(fig)

n_arrows = sum(len(v) for v in side_nodes.values())
print("arrows on boundary nodes:", n_arrows, "| boundary nodes on the four sides:", n_arrows,
      "| notch-arc nodes (traction-free, no arrow):", len(B_nodes) - n_arrows)
print("interior nodes", len(INT), "boundary nodes", len(B_nodes),
      {k: len(v) for k, v in side_nodes.items()}, "zoom interior nodes", len(in_zoom))

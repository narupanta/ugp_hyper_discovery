"""
generate_experiment_domain_diagrams.py

Schematics of the experimental training specimens (TTf and UT, Flaschel et al.) in the style of the synthetic domain
figure (plots/generate_domain_diagrams.py, plain variant): undeformed DIC mesh, outline (holes included), the two
end cuts where the measured displacements are prescribed, dimensions in mm, and a LaTeX summary table (booktabs, as
the synthetic training-domain table).

Usage:
  python3 plots/generate_experiment_domain_diagrams.py [--out_dir plots] [--extract_dir results/<exp>/1/extracted]
Writes <out_dir>/domain_ttf_ut.pdf/.png and <out_dir>/table_ttf_ut.tex.
"""
import argparse
import json
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from plots.theme import apply_style
    apply_style("paper")
except ImportError:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
from plots.generate_domain_diagrams import draw_dim_h, draw_dim_v  # noqa: E402

EXP = "dataset/preprocessed/experiment"
INK, GREY, BOX_EDGE = "#111111", "#808080", "#9e9e9e"
SPECIMENS = {"TTf": f"{EXP}/ttf_20_steps_ps_incomp.npz", "UT": f"{EXP}/ut_20_steps_ps_incomp.npz"}


def boundary_edges(cells):
    """Edges that belong to exactly one triangle (outer outline and hole rims)."""
    e = np.sort(np.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    return uniq[cnt == 1]


LABEL_FS = 15.0
PANEL_H_IN = 5.6          # height of every panel (inches): same relative vertical layout -> aligned X axes
Y_BELOW, Y_ABOVE = 0.42, 0.34   # margins below y = 0 and above y = L_Y, in units of L_Y


def label(ax, x, y, text, ha="center", va="center", fs=LABEL_FS, rot=0):
    return ax.text(x, y, text, ha=ha, va=va, fontsize=fs, color=INK, rotation=rot, zorder=7,
                   bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor=BOX_EDGE, lw=1.2, alpha=0.96))


def text_width_pt(fig, text, fs=LABEL_FS):
    """Rendered width (points) of a boxed label."""
    t = fig.text(0, 0, text, fontsize=fs, bbox=dict(boxstyle="round,pad=0.28", lw=1.2))
    fig.canvas.draw()
    bb = t.get_bbox_patch().get_window_extent(fig.canvas.get_renderer())
    t.remove()
    return bb.width / fig.dpi * 72.0


LEFT_TXT = r"$\partial \Omega_{\mathrm{left}}:\; \mathbf{u} = \mathbf{u}_{\mathrm{DIC,\,left}}$"
RIGHT_TXT = r"$\partial \Omega_{\mathrm{right}}:\; \mathbf{u} = \mathbf{u}_{\mathrm{DIC,\,right}}$"
FREE_TXT = r"$\partial \Omega_{\mathrm{free}}:\; \mathbf{t} = \mathbf{0}$"


def specimen_layout(fig, path):
    """Geometry and the horizontal layout in data units (mm) for a panel of height PANEL_H_IN: label and dimension
    positions follow from the rendered label widths (points) and the panel scale k (points per mm)."""
    d = np.load(path)
    X = np.asarray(d["mesh_pos"], float)[:, :2]
    X = X - X.min(0)                                   # origin at the lower-left corner of the bounding box
    Lx, Ly = X.max(0)
    k = PANEL_H_IN * 72.0 / (Ly * (1 + Y_BELOW + Y_ABOVE))      # points per mm
    pt = 1.0 / k                                                  # mm per point
    wl, wr = text_width_pt(fig, LEFT_TXT), text_width_pt(fig, RIGHT_TXT)
    x_dim = -(10 + wl + 34) * pt                                  # L_Y dimension line left of the left label
    xlim = (x_dim - 30 * pt, Lx + (10 + wr + 6) * pt)
    w_free = text_width_pt(fig, FREE_TXT)
    xlim = (xlim[0], max(xlim[1], (8 + w_free + 6) * pt))
    return dict(d=d, X=X, Lx=Lx, Ly=Ly, k=k, pt=pt, x_dim=x_dim, xlim=xlim, w_free=w_free,
                ylim=(-Y_BELOW * Ly, Ly * (1 + Y_ABOVE)))


def plot_specimen(ax, lay, name, panel_label=None):
    d, X, Lx, Ly, pt = lay["d"], lay["X"], lay["Lx"], lay["Ly"], lay["pt"]
    cells, bcx = np.asarray(d["cells"]), np.asarray(d["bcx"]).ravel()
    ax.triplot(mtri.Triangulation(X[:, 0], X[:, 1], cells), color=GREY, lw=0.35, alpha=0.55, zorder=2)
    for a, b in boundary_edges(cells):
        ax.plot(X[[a, b], 0], X[[a, b], 1], color=INK, lw=1.6, zorder=4, solid_capstyle="round")
    for kk in (1, 2):                                  # the end cuts: thick, prescribed displacement
        n = np.where(bcx == kk)[0]
        n = n[np.argsort(X[n, 1])]
        ax.plot(X[n, 0], X[n, 1], color=INK, lw=4.0, zorder=5, solid_capstyle="butt")
    ym = 0.5 * Ly
    label(ax, -10 * pt, ym, LEFT_TXT, ha="right")
    label(ax, Lx + 10 * pt, ym, RIGHT_TXT, ha="left")
    # centred on the top edge, shifted right if the box would cross the Y axis
    label(ax, max(0.5 * Lx, (8 + 0.5 * lay["w_free"]) * pt), Ly + 12 * pt, FREE_TXT, va="bottom")
    # dimensions (mm)
    yd = -34 * pt
    draw_dim_h(ax, 0.0, Lx, yd, rf"$L_X = {Lx:.1f}\,\mathrm{{mm}}$", offset_text=-14 * pt)
    for x_ in (0.0, Lx):
        ax.plot([x_, x_], [0, yd - 6 * pt], color="#94a3b8", lw=1.0, ls=":")
    xd = lay["x_dim"]
    draw_dim_v(ax, 0.0, Ly, xd, rf"$L_Y = {Ly:.1f}\,\mathrm{{mm}}$", offset_text=-14 * pt)
    for y_ in (0.0, Ly):
        ax.plot([0, xd - 6 * pt], [y_, y_], color="#94a3b8", lw=1.0, ls=":")
    # axes from the origin
    ax.annotate("", xy=(Lx + 28 * pt, 0), xytext=(-10 * pt, 0), annotation_clip=False, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=14, shrinkA=0, shrinkB=0))
    ax.text(Lx + 32 * pt, 0, r"$X$", fontsize=17, ha="left", va="center", color=INK, zorder=9)
    ax.annotate("", xy=(0, Ly + 46 * pt), xytext=(0, -10 * pt), annotation_clip=False, zorder=8,
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=14, shrinkA=0, shrinkB=0))
    ax.text(0, Ly + 50 * pt, r"$Y$", fontsize=17, ha="center", va="bottom", color=INK, zorder=9)
    ax.plot(0, 0, "o", color=INK, ms=4, zorder=9)
    ax.text(-4 * pt, -4 * pt, r"$(0,0)$", fontsize=12, ha="right", va="top", color="#475569", zorder=9)
    ax.text(0.5 * Lx, -66 * pt, name, fontsize=17, ha="center", va="top", color=INK)
    if panel_label:
        ax.text(lay["xlim"][0], Ly + 50 * pt, panel_label, fontsize=18, fontweight="bold", color=INK, va="bottom")
    ax.set_xlim(*lay["xlim"])
    ax.set_ylim(*lay["ylim"])
    ax.set_aspect("equal")
    ax.axis("off")
    return dict(nodes=len(X), cells=len(cells), Lx=Lx, Ly=Ly, n_end=(int((bcx == 1).sum()), int((bcx == 2).sum())),
                steps=np.asarray(d["steps"]), u=np.asarray(d["u_exp"]), bcx=bcx, R=np.asarray(d["reaction_forces"])[:, 0])


def write_table(info, train_idx, out_path, thickness=2.5, noise_rel=0.005, noise_floor=0.02):
    names = list(info)
    st = info[names[0]]["steps"]
    frames = f"{st[0] // 10}--{st[-1] // 10}"
    tr = ", ".join(str(int(st[i] // 10)) for i in train_idx) if train_idx else "all"

    def row(feature, vals):
        return f"{feature} & " + " & ".join(vals) + r" \\"
    d_last = {n: info[n]["u"][-1, info[n]["bcx"] == 2, 0].mean() - info[n]["u"][-1, info[n]["bcx"] == 1, 0].mean()
              for n in names}
    rows = [
        row(r"Number of Nodes ($n_{\mathrm{node}}$)", [f"{info[n]['nodes']}" for n in names]),
        row(r"Number of Cells ($n_c$, TRI3 elements)", [f"{info[n]['cells']}" for n in names]),
        row(r"Dimensions ($L_X \times L_Y$)", [f"${info[n]['Lx']:.1f} \\times {info[n]['Ly']:.1f}$~mm" for n in names]),
        row(r"Thickness", [f"{thickness:g}~mm"] * len(names)),
        row(r"Number of Loadsteps ($n_t$, DIC frames)", [f"{len(st)} (frames {frames})"] * len(names)),
        row(r"Training Loadsteps", [f"{len(train_idx)} (frames {tr})"] * len(names)),
        row(r"End Displacement at Last Step", [f"{d_last[n]:.1f}~mm" for n in names]),
        row(r"Maximum Reaction Force ($R_{\mathrm{max}}$)", [f"{thickness * np.abs(info[n]['R']).max():.0f}~N" for n in names]),
        row(r"Left / Right Boundary ($\partial\Omega_{\mathrm{left}}, \partial\Omega_{\mathrm{right}}$)",
            [r"$\mathbf{u} = \mathbf{u}_{\mathrm{DIC,left/right}}$ (end cuts, " + f"{info[n]['n_end'][0]}/{info[n]['n_end'][1]} nodes)" for n in names]),
        row(r"Free Boundary ($\partial\Omega_{\mathrm{free}}$)", [r"$\mathbf{t} = \mathbf{0}$"] * len(names)),
        row(r"Kinematics", ["incompressible plane stress"] * len(names)),
        row(r"Noise Level", [r"$\boldsymbol{e}_{\mathbf{y}}$: DIC (measured), "
                             + rf"$e_{{R}} = {100 * noise_rel:g}\% \, |R|$"] * len(names)),
    ]
    tex = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\begin{tabular}{l" + "l" * len(names) + "}",
        r"\toprule",
        "Feature / Configuration & " + " & ".join(names) + r" \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Summary of geometric parameters, finite-element discretization (DIC mesh), and boundary conditions "
        r"for the experimental training specimens.}",
        r"\label{tab:experimental_domains}",
        r"\end{table}",
    ])
    with open(out_path, "w") as f:
        f.write(tex + "\n")
    print(f"saved {out_path}")
    print(tex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out_dir", default="plots")
    p.add_argument("--extract_dir", default="results/20261120T094518_hsgp_experimental_None_0.005_None_None_ttf_ut/1/extracted",
                   help="extraction whose training steps go into the table")
    p.add_argument("--thickness", type=float, default=2.5)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    # panels of equal height side by side, each as wide as its (equal-aspect) layout needs
    fig = plt.figure(figsize=(1, 1))
    lays = {name: specimen_layout(fig, path) for name, path in SPECIMENS.items()}
    widths = [(l_["xlim"][1] - l_["xlim"][0]) * l_["k"] / 72.0 for l_ in lays.values()]
    gap, pad_b = 0.5, 0.15
    W, H = sum(widths) + gap * (len(widths) - 1), PANEL_H_IN + 2 * pad_b
    fig.set_size_inches(W, H)
    info, x0 = {}, 0.0
    for (name, lay), w, lab in zip(lays.items(), widths, [r"$\mathbf{(a)}$", r"$\mathbf{(b)}$"]):
        ax = fig.add_axes([x0 / W, pad_b / H, w / W, PANEL_H_IN / H])
        info[name] = plot_specimen(ax, lay, name, lab)
        x0 += w + gap
    out = os.path.join(args.out_dir, "domain_ttf_ut")
    fig.savefig(out + ".pdf", bbox_inches="tight")
    fig.set_dpi(150)
    fig.canvas.print_png(out + ".png")      # the paper theme redirects savefig to PDF
    plt.close(fig)
    print(f"saved {out}.pdf/.png")
    train_idx = []
    mp = os.path.join(args.extract_dir, "extraction_metrics.json")
    if os.path.exists(mp):
        with open(mp) as f:
            train_idx = [int(i) for i in json.load(f).get("train_steps", [])]
    write_table(info, train_idx, os.path.join(args.out_dir, "table_ttf_ut.tex"), thickness=args.thickness)


if __name__ == "__main__":
    main()

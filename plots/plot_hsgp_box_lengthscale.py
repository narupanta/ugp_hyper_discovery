#!/usr/bin/env python3
"""
HSGP box and lengthscales of an extraction (deviatoric GP over (I1_bar, I2_bar)): the observed states, the data box
(half-range S around the data and the reference state, per dimension), the HSGP box (half-width L = c S, c = box
factor), the fitted lengthscales l with their optimisation bounds, and the accuracy conditions of the Hilbert-space
approximation for the squared-exponential kernel (Riutort-Mayol et al. 2023, Stat. Comput. 33:17):
    m >= 1.75 c / (l / S)      (enough basis functions per dimension)
    c >= 3.2 l / S             (box large enough for the lengthscale; c >= 1.2 in any case)

Usage: python3 plots/plot_hsgp_box_lengthscale.py --extract_dir <seed>/extracted
Writes <extract_dir>/hsgp_box_lengthscale.pdf/.png and hsgp_box_lengthscale.json
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--extract_dir", required=True)
    args = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, Rectangle
    from plots.theme import apply_style, save_figure

    d = np.load(os.path.join(args.extract_dir, "hsgp_posterior.npz"), allow_pickle=True)
    info, hyper = json.loads(str(d["info"])), json.loads(str(d["hyper"]))
    c = float(info["box_factor"])
    m = np.max(np.asarray(d["dev_indices"]), axis=0).astype(int)          # basis functions per dimension
    centre, L = np.asarray(d["dev_centre"], float), np.asarray(d["dev_half_width"], float)
    S = L / c
    ls = np.asarray(hyper["dev_ls"], float)
    lb = np.asarray(info["dev_ls_bounds"], float)                        # [[lo...], [hi...]]
    m_req = 1.75 * c * S / ls                                            # = 1.75 L / l
    c_req = np.maximum(3.2 * ls / S, 1.2)
    I = np.load(os.path.join(args.extract_dir, "I_obs_all.npy"))[:, :2]
    names = [r"$\bar I_1$", r"$\bar I_2$"]

    rows = []
    for i in range(2):
        rows.append(dict(dim=["I1_bar", "I2_bar"][i], S=float(S[i]), L=float(L[i]), c=c, l=float(ls[i]),
                         l_min=float(lb[0][i]), l_max=float(lb[1][i]), l_over_S=float(ls[i] / S[i]), m=int(m[i]),
                         m_required=float(m_req[i]), m_ok=bool(m[i] >= m_req[i]), c_required=float(c_req[i]),
                         c_ok=bool(c >= c_req[i]), l_at_upper_bound=bool(np.isclose(ls[i], lb[1][i], rtol=1e-4))))

    apply_style()
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(11.5, 4.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
    a0.scatter(I[:, 0], I[:, 1], s=2, color="#37474f", alpha=0.35, lw=0, rasterized=True, label="observed states")
    a0.plot(3, 3, "k*", ms=10, label="reference state F = I")
    a0.add_patch(Rectangle(centre - S, 2 * S[0], 2 * S[1], fill=False, ec="#1565c0", lw=1.6, label="data box (half-range S)"))
    a0.add_patch(Rectangle(centre - L, 2 * L[0], 2 * L[1], fill=False, ec="#c62828", lw=1.6, ls="--",
                           label=f"HSGP box (L = {c:g} S)"))
    a0.add_patch(Ellipse(centre, 2 * ls[0], 2 * ls[1], fill=False, ec="#2e7d32", lw=1.8, label="fitted lengthscales (1 l)"))
    a0.add_patch(Ellipse(centre, 2 * lb[0][0], 2 * lb[0][1], fill=False, ec="#2e7d32", lw=0.9, ls=":",
                         label="lengthscale lower bound"))
    a0.set_xlim(centre[0] - 1.08 * L[0], centre[0] + 1.08 * L[0])
    a0.set_ylim(centre[1] - 1.08 * L[1], centre[1] + 1.08 * L[1])
    a0.set_aspect("equal")
    a0.set_xlabel(names[0])
    a0.set_ylabel(names[1])
    a0.set_title("(a) Deviatoric input space", fontsize=11)
    a0.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    a0.grid(alpha=0.2)

    a1.axis("off")
    hdr = ["", names[0], names[1]]
    fmt = lambda v: f"{v:.3g}"
    table = [["data half-range S", fmt(S[0]), fmt(S[1])],
             ["box half-width L = c S", fmt(L[0]), fmt(L[1])],
             ["fitted lengthscale l", fmt(ls[0]), fmt(ls[1])],
             ["bounds [l_min, l_max]", f"[{fmt(lb[0][0])}, {fmt(lb[1][0])}]", f"[{fmt(lb[0][1])}, {fmt(lb[1][1])}]"],
             ["l / S", fmt(ls[0] / S[0]), fmt(ls[1] / S[1])],
             ["m used", str(m[0]), str(m[1])],
             ["m required = 1.75 c / (l/S)", fmt(m_req[0]), fmt(m_req[1])],
             ["m condition", "OK" if m[0] >= m_req[0] else "NOT met", "OK" if m[1] >= m_req[1] else "NOT met"],
             [f"c used", fmt(c), fmt(c)],
             ["c required = 3.2 l/S", fmt(c_req[0]), fmt(c_req[1])],
             ["c condition", "OK" if c >= c_req[0] else "NOT met", "OK" if c >= c_req[1] else "NOT met"]]
    t = a1.table(cellText=table, colLabels=hdr, loc="center", cellLoc="center", colWidths=[0.5, 0.25, 0.25])
    t.auto_set_font_size(False)
    t.set_fontsize(9)
    t.scale(1, 1.45)
    for (r, col), cell in t.get_celld().items():
        txt = cell.get_text().get_text()
        if txt == "NOT met":
            cell.set_facecolor("#ffcdd2")
        elif txt == "OK" and col > 0:
            cell.set_facecolor("#c8e6c9")
    a1.set_title("(b) HSGP accuracy conditions (SE kernel)", fontsize=11)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out = os.path.join(args.extract_dir, "hsgp_box_lengthscale")
    save_figure(fig, out + ".pdf")
    fig.set_dpi(180)
    fig.canvas.print_png(out + ".png")
    plt.close(fig)
    json.dump(rows, open(out + ".json", "w"), indent=2)
    for r in rows:
        print(r)
    print(out + ".pdf")


if __name__ == "__main__":
    main()

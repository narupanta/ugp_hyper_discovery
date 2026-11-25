#!/usr/bin/env python3
"""
Discovered models of several test cases in one figure, one row per case (e.g. the median model of each coverage
study):
  (a) violins of the discovered terms with a true value, normalised theta / theta_true (red line at 1): posterior
      mean (white dot) and 95 % interval (black bar);
  (b) violins of the spurious discovered terms (true value 0) on an absolute scale (red line at 0);
  (c) total-order Sobol index S_T of the discovered terms vs the main invariant (deviatoric terms vs I1; if only one
      deviatoric term was discovered, the fibre terms vs I4), the mean S_T as a marker on the left edge;
  (d) energy parity of the GP and the distilled model vs the ground truth at the validation states, R^2 / RMSE /
      EC95 of both (energy_parity_total.npz from plots/plot_distilled_r2_energy.py).
Colours of a term are the same in (a)-(c).

Usage:
  python3 plots/plot_discovery_rows.py <run_dir>:Isihara <run_dir>:NH2 <run_dir>:"Aniso 2F" --out fig.pdf
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from plots.plot_discovery_summary import ANISO_NAMES, find_config, fibre_invariants, plain, tex_name  # noqa: E402
from plots.theme import apply_style  # noqa: E402

TRUE_COL, DIST_COL, GP_COL = "#d62728", "#009E73", "#757575"


def load_case(d):
    from core.material_library import dev16, dev_names, distilled_library
    cfg = find_config(d)
    mp = cfg.get("material_params") or {}
    true_dev = list(mp.get("dev_params", [])) + [0.0] * 10
    true_an = list(mp.get("aniso_params", [])) + [0.0] * 6
    terms = []          # (comp, name, samples, true value, mean S_T)
    sens = {}
    for comp in ("dev", "aniso"):
        sd = os.path.join(d, "output", f"{comp}_sensitivities")
        rp = os.path.join(sd, "relevant_parameters.txt")
        if not os.path.exists(rp):
            continue
        disc = [ln.strip() for ln in open(rp) if ln.strip()]
        st = pd.read_csv(os.path.join(sd, "total_sobol_indices_statistics_output_0.csv"), index_col=0)
        if comp == "dev":
            lib = distilled_library(d)
            smp = dev16(np.load(os.path.join(d, "dev_flow_samples.npy")), lib)
            from core.material_library import dev16_names, distilled_gmr_order
            idx = [dev16_names(distilled_gmr_order(d)).index(plain(p)) for p in disc]
            # dev16 index -> true dev_params index: 0..9 as is, 26 + k (GMR order > 3) -> 10 + k, Ogden: no GMR truth
            tv = [true_dev[i] if i < 10 else (true_dev[i - 16] if i >= 26 and i - 16 < len(true_dev) else 0.0)
                  for i in idx]
        else:
            smp = np.load(os.path.join(d, "aniso_flow_samples.npy"))
            idx = [ANISO_NAMES.index(plain(p)) for p in disc]
            tv = [true_an[i] for i in idx]
        for p, i, t in zip(disc, idx, tv):
            terms.append(dict(comp=comp, name=p, x=smp[:, i], true=float(t), st=float(st.loc["mean", p])))
        sens[comp] = disc
    sc_comp = next((c for c in ("dev", "aniso") if len(sens.get(c, [])) >= 2), None)
    parity = None
    pf = os.path.join(d, "energy_parity_total.npz")
    if os.path.exists(pf):
        parity = dict(np.load(pf))
    return dict(terms=terms, sc_comp=sc_comp, angles=mp.get("angles", [45.0, -45.0]), parity=parity, d=d)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cases", nargs="+", help="<run dir>:<label>")
    ap.add_argument("--out", required=True, help="output .pdf (a .png is written next to it)")
    args = ap.parse_args()
    apply_style()
    from matplotlib.lines import Line2D
    from plots.sensitivity import load_sensitivity_data
    cases = [(c.rsplit(":", 1)[0], c.rsplit(":", 1)[1]) for c in args.cases]
    data = [load_case(d) for d, _ in cases]
    pal = plt.cm.tab10(np.arange(10))
    fs, fs_tick, fs_small = 7.5, 6.5, 6.0

    # layout (inches)
    nr = len(cases)
    h = 1.35                                         # panel height (square panels: S_T, parity)
    n_true_max = max(sum(t["true"] != 0 for t in c["terms"]) for c in data)
    w_v = max(1.2, 0.32 * n_true_max + 0.25)         # violin panel width follows the number of terms
    w_s = 0.55
    m_l, m_r, m_t, m_b = 0.85, 0.08, 0.28, 0.55
    g1, g2, g3, gy = 0.42, 0.55, 0.62, 0.42
    W = m_l + w_v + g1 + w_s + g2 + h + g3 + h + m_r
    H = m_t + nr * h + (nr - 1) * gy + m_b
    fig = plt.figure(figsize=(W, H))

    def rect(x, y, w, hh):
        return [x / W, y / H, w / W, hh / H]

    def style(ax):
        ax.tick_params(labelsize=fs_tick, length=2, pad=1.2, width=0.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_linewidth(0.5)

    def violins(ax, vals, pos, cols):
        vp = ax.violinplot(vals, positions=pos, widths=0.75, showextrema=False)
        for b, c in zip(vp["bodies"], cols):
            b.set_facecolor(c)
            b.set_edgecolor("none")
            b.set_alpha(0.55)
        for v, p in zip(vals, pos):
            lo, hi = np.percentile(v, [2.5, 97.5])
            ax.plot([p, p], [lo, hi], color="k", lw=2.0, solid_capstyle="butt", zorder=3)
            ax.plot(p, np.mean(v), "o", ms=3.2, mfc="white", mec="k", mew=0.5, zorder=4)

    rng = np.random.default_rng(0)
    titles = ["(a) discovered terms", "(b) spurious", r"(c) Sobol index $S_T$", "(d) energy parity"]
    for r, ((d, label), c) in enumerate(zip(cases, data)):
        y0 = m_b + (nr - 1 - r) * (h + gy)
        col = {t["name"]: pal[i % 10] for i, t in enumerate(c["terms"])}
        true_t = [t for t in c["terms"] if t["true"] != 0]
        spur_t = [t for t in c["terms"] if t["true"] == 0]
        x = m_l
        # (a) normalised violins
        ax = fig.add_axes(rect(x, y0, w_v, h))
        if true_t:
            violins(ax, [t["x"] / t["true"] for t in true_t], range(len(true_t)), [col[t["name"]] for t in true_t])
        ax.axhline(1.0, color=TRUE_COL, lw=0.9, zorder=1)
        ax.set_xticks(range(len(true_t)))
        ax.set_xticklabels([tex_name(t["name"]) for t in true_t], fontsize=fs_tick + 0.5)
        ax.set_xlim(-0.6, max(len(true_t), n_true_max) - 0.4)
        ax.set_ylabel(r"$\theta / \theta_{\mathrm{true}}$", fontsize=fs, labelpad=1.5)
        ax.yaxis.set_major_locator(plt.MaxNLocator(4))
        style(ax)
        fig.text(0.12 / W, (y0 + h / 2) / H, label, rotation=90, ha="center", va="center", fontsize=fs + 1)
        if r == 0:
            ax.set_title(titles[0], fontsize=fs, pad=3, loc="left")
        x += w_v + g1
        # (b) spurious terms, absolute
        ax = fig.add_axes(rect(x, y0, w_s, h))
        if spur_t:
            e = int(np.floor(np.log10(max(np.abs(np.percentile(t["x"], 97.5)) for t in spur_t))))
            violins(ax, [t["x"] / 10.0 ** e for t in spur_t], range(len(spur_t)), [col[t["name"]] for t in spur_t])
            ax.set_ylabel(rf"$\theta$ ($10^{{{e}}}$ MPa)", fontsize=fs, labelpad=1.5)
        else:
            ax.text(0.5, 0.5, "none", transform=ax.transAxes, ha="center", va="center", fontsize=fs_small,
                    color="0.4")
            ax.set_yticks([])
        ax.axhline(0.0, color=TRUE_COL, lw=0.9, zorder=1)
        ax.set_xticks(range(len(spur_t)))
        ax.set_xticklabels([tex_name(t["name"]) for t in spur_t], fontsize=fs_tick + 0.5)
        ax.set_xlim(-0.6, max(len(spur_t), 1) - 0.4)
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        style(ax)
        if r == 0:
            ax.set_title(titles[1], fontsize=fs, pad=3, loc="left")
        x += w_s + g2
        # (c) S_T vs the main invariant
        ax = fig.add_axes(rect(x, y0, h, h))
        if c["sc_comp"]:
            df, f3, inv, _ = load_sensitivity_data(d, component=c["sc_comp"], distill_target="sef_split")
            if c["sc_comp"] == "dev":
                v, lb = inv[0], r"$I_1$"
            else:
                v, lb = fibre_invariants(np.asarray(f3), c["angles"])[0], r"$I_4$"
            for t in c["terms"]:
                if t["comp"] != c["sc_comp"]:
                    continue
                ax.scatter(v, df[t["name"]].values, s=1.4, color=col[t["name"]], alpha=0.55, lw=0, rasterized=True)
                ax.plot(0.0, t["st"], marker=">", ms=4.5, color=col[t["name"]], mec="k", mew=0.4,
                        transform=ax.get_yaxis_transform(), clip_on=False, zorder=6)
            ax.set_xlabel(lb, fontsize=fs, labelpad=1.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_ylabel(r"$S_T$", fontsize=fs, labelpad=1.5)
        style(ax)
        if r == 0:
            ax.set_title(titles[2], fontsize=fs, pad=3, loc="left")
        x += h + g3
        # (d) parity
        ax = fig.add_axes(rect(x, y0, h, h))
        if c["parity"] is not None:
            z = c["parity"]
            t_ = z["true"]
            mods = [("GP", z["gp_mean"], z["gp_mean"] - 1.96 * z["gp_std"], z["gp_mean"] + 1.96 * z["gp_std"], GP_COL),
                    ("Dist.", z["dist_mean"], z["dist_q025"], z["dist_q975"], DIST_COL)]
            k = rng.choice(len(t_), min(300, len(t_)), replace=False)
            lines = []
            for zo, (n, mu, lo, hi, cc) in enumerate(mods):
                ax.errorbar(t_[k], mu[k], yerr=[mu[k] - lo[k], hi[k] - mu[k]], fmt="none", ecolor=cc, alpha=0.3,
                            elinewidth=0.4, zorder=1 + 2 * zo)
                ax.scatter(t_, mu, s=1.0, color=cc, alpha=0.5, lw=0, zorder=2 + 2 * zo, rasterized=True)
                r2 = 1 - np.sum((t_ - mu) ** 2) / np.sum((t_ - t_.mean()) ** 2)
                ec = 100 * np.mean((t_ >= lo) & (t_ <= hi))
                lines.append((n, r2, np.sqrt(np.mean((t_ - mu) ** 2)), ec, cc))
            allv = np.concatenate([t_, mods[0][1], mods[1][1]])
            lim = [allv.min(), allv.max()]
            ax.plot(lim, lim, "k--", lw=0.5, zorder=6)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            for j, (n, r2, rm, ec, cc) in enumerate(lines):     # metrics in the empty lower-right half
                ax.text(0.97, 0.05 + 0.12 * (len(lines) - 1 - j),
                        f"{n}: $R^2$ {r2:.4f}, EC {ec:.0f}%", transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=fs_small - 0.6, color=cc)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.set_xlabel(r"$\Psi_{\mathrm{true}}$", fontsize=fs, labelpad=1.2)
        ax.set_ylabel(r"$\Psi_{\mathrm{pred}}$", fontsize=fs, labelpad=1.2)
        style(ax)
        if r == 0:
            ax.set_title(titles[3], fontsize=fs, pad=3, loc="left")
    # legend (bottom)
    handles = [Line2D([], [], color=TRUE_COL, lw=0.9, label="true value"),
               Line2D([], [], marker="o", ls="", ms=3.2, mfc="white", mec="k", mew=0.5, label="posterior mean"),
               Line2D([], [], color="k", lw=2.0, label="95% interval"),
               Line2D([], [], marker=">", ls="", ms=4.5, color="0.6", mec="k", mew=0.4, label=r"mean $S_T$"),
               Line2D([], [], marker="o", ls="", ms=3, color=GP_COL, label="GP (mean, 95%)"),
               Line2D([], [], marker="o", ls="", ms=3, color=DIST_COL, label="distilled (mean, 95%)")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=6, fontsize=fs_small,
               frameon=False, handlelength=1.4, columnspacing=1.2, handletextpad=0.4)
    fig.savefig(args.out, dpi=300)
    fig.set_dpi(250)
    fig.canvas.print_png(args.out[:-4] + ".png")
    plt.close(fig)
    print(f"figure {W * 25.4:.0f} x {H * 25.4:.0f} mm; saved {args.out} (+ .png)")


if __name__ == "__main__":
    main()

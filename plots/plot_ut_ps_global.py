#!/usr/bin/env python3
"""
Global nominal stress of the uniaxial tension (UT) and pure shear (PS) experiments against the distilled model, the
GP posterior and a reference model: one panel per test.

Measured (dataset/experiment/{UT,PS}): per DIC step, the stretch lambda between the two prescribed end cuts (default;
--stretch gauge: area-weighted mean F11 over the central strip |X - X_mid| < gauge_frac * L / 2) and the nominal stress
P11 = R / W0 (end reaction / reference width of the central strip). R is the force on the unit-thickness plate, as in the preprocessed training data (dataset/preprocessed/...),
so P11 has the units of the extracted energies.

Models along the ideal incompressible paths with traction-free faces (F = diag(lam, lam2, 1 / (lam lam2)),
UT lam2 = lam^-1/2, PS lam2 = 1): P11 = dPsi/dlambda (no pressure needed). The GP band is that of the derivative
d/dlambda of the posterior energy (exact: the basis is differentiated, not the marginal stress variances). Beyond the
observed invariant region (plots.training.compute_interpolation_masks) the GP extrapolates: shaded.

Usage:
  python3 plots/plot_ut_ps_global.py --distilled_dir results/<exp>/1/distilled --reference euclid_ttf_ut
Writes <distilled_dir>/ut_ps_global_stress[_ref<key>].pdf/.png and the measured curves ut_ps_global_data.npz (cache).
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from plots.theme import apply_style  # noqa: E402

TESTS = {   # name: (raw folder, in-plane lateral stretch of the ideal path, title)
    "UT": ("dataset/experiment/UT/UT/dataset_600_steps", lambda l: l ** -0.5, "Uniaxial tension"),
    "PS": ("dataset/experiment/PS/PS/dataset_559_steps", lambda l: 0.0 * l + 1.0, "Pure shear"),
}


# last reliable DIC step per test: PS skips frames between 5570 and 5580, then load cell and camera desynchronise and the
# DIC correlation degrades (twice as many elements with implausible area ratio)
LAST_STEP = {"PS": 5570}


def global_curve(folder, gauge_frac=0.3, stretch="ends"):
    """(steps, lambda, lambda2 (mean F22 of the strip), P11 = R / W0) of a raw DIC folder. lambda: "ends" = relative
    displacement of the two prescribed end cuts (bcx 1 / 2) over their reference distance (the specimen-average stretch,
    as in Flaschel et al.); "gauge" = area-weighted mean F11 of the central strip (higher: the middle stretches more)."""
    import pandas as pd
    steps = sorted(int(s) for s in os.listdir(folder) if s.isdigit())
    nodes = pd.read_csv(os.path.join(folder, str(steps[0]), "output_nodes.csv"))
    X = nodes[["x", "y"]].to_numpy()                                   # reference coordinates (same every step)
    el0 = pd.read_csv(os.path.join(folder, str(steps[0]), "output_elements.csv"))
    tri = el0[["node1", "node2", "node3"]].to_numpy()
    xc = X[tri].mean(1)[:, 0]
    x_mid, L = 0.5 * (X[:, 0].min() + X[:, 0].max()), X[:, 0].max() - X[:, 0].min()
    strip = np.abs(xc - x_mid) < 0.5 * gauge_frac * L
    a, b, c = X[tri[:, 0]], X[tri[:, 1]], X[tri[:, 2]]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    w = area[strip] / area[strip].sum()
    bcx = nodes["bcx"].to_numpy()
    e1, e2 = bcx == 1, bcx == 2
    L0 = X[e2, 0].mean() - X[e1, 0].mean()
    W0 = area[strip].sum() / (gauge_frac * L)                          # reference width of the strip
    lam, lam2, P = [], [], []
    for s in steps:
        el = pd.read_csv(os.path.join(folder, str(s), "output_elements.csv"), usecols=["Fxx", "Fyy"])
        R = pd.read_csv(os.path.join(folder, str(s), "output_reactions.csv"))["forces"].to_numpy()
        if stretch == "ends":
            ux = pd.read_csv(os.path.join(folder, str(s), "output_nodes.csv"), usecols=["ux"])["ux"].to_numpy()
            lam.append(float(1.0 + (ux[e2].mean() - ux[e1].mean()) / L0))
        else:
            lam.append(float(w @ el["Fxx"].to_numpy()[strip]))
        lam2.append(float(w @ el["Fyy"].to_numpy()[strip]))
        P.append(0.5 * (abs(R[0]) + abs(R[1])) / W0)                    # the two end reactions (equal and opposite)
    return dict(steps=np.array(steps), lam=np.array(lam), lam2=np.array(lam2), P=np.array(P), W0=W0)


def path_F(lam, lat):
    l2 = lat(lam)
    return jnp.zeros((3, 3)).at[0, 0].set(lam).at[1, 1].set(l2).at[2, 2].set(1.0 / (lam * l2))


def distilled_energy_fn(distilled_dir):
    """psi(theta_row, F) of the distilled dev (+ vol) model and the sample matrix (S, d)."""
    from core.material_library import dev16, distilled_library
    from core.material_models import get_material
    from core.ogden import ogden_dev_energy_jax
    dev = dev16(np.load(os.path.join(distilled_dir, "dev_flow_samples.npy")), distilled_library(distilled_dir))
    vp = os.path.join(distilled_dir, "vol_flow_samples.npy")
    vol = np.load(vp)[:, :3] if os.path.exists(vp) else np.zeros((len(dev), 3))
    vol = np.pad(vol, ((0, 0), (0, 3 - vol.shape[1])))
    theta = np.concatenate([dev, vol], axis=1)
    nd = dev.shape[1]

    def psi(t, F):
        mat = get_material("gmr", dev_params=t[:10], vol_params=t[nd:nd + 3], jit_P=False)
        og = t[10:nd]
        return mat.psi(F) + ogden_dev_energy_jax(F, og[0::2], og[1::2])
    return psi, theta


def model_band_fn(kind, distilled_dir, extract_dir, n_samples, aleatoric_rel=0.0):
    """(label, fn(lam, lat) -> (mean, lo, hi, lo_tot, hi_tot) of P11 = dPsi/dlambda along the path) of the distilled
    model (2.5 / 97.5 percentiles of the parameter samples) or the GP posterior (mean -/+ 1.96 sd of the exact
    dPsi/dlambda). lo/hi: epistemic band; lo_tot/hi_tot: with the force-reading noise of a new measurement,
    sd = aleatoric_rel x P11 (P11 = R / W0, so the relative load-cell tolerance carries over), added in quadrature (GP)
    or as multiplicative noise on each parameter sample (distilled)."""
    if kind == "distilled":
        psi_d, theta = distilled_energy_fn(distilled_dir)
        theta = jnp.asarray(theta[np.random.default_rng(0).choice(len(theta), min(n_samples, len(theta)), replace=False)])

        def fn(lam, lat):
            dpsi = jax.grad(lambda l, th: psi_d(th, path_F(l, lat)))
            Pd = np.asarray(jax.vmap(lambda th: jax.vmap(lambda l: dpsi(l, th))(lam))(theta))
            Pn = Pd * (1.0 + aleatoric_rel * np.random.default_rng(1).standard_normal(Pd.shape))
            return (Pd.mean(0), np.percentile(Pd, 2.5, 0), np.percentile(Pd, 97.5, 0),
                    np.percentile(Pn, 2.5, 0), np.percentile(Pn, 97.5, 0))
        return "Distilled", fn
    from core.teacher import load_hsgp_teacher
    gp = load_hsgp_teacher(extract_dir)

    def fn(lam, lat):
        Fp = jax.vmap(lambda l: path_F(l, lat))(lam)
        dF = jax.vmap(jax.jacfwd(lambda l: path_F(l, lat)))(lam)
        dphi = jax.vmap(lambda F, dF_: jnp.einsum("pij,ij->p", jax.jacfwd(gp.basis)(F), dF_))(Fp, dF)
        m = np.asarray(dphi @ gp.theta_mean)
        sd = np.asarray(jnp.sqrt(jnp.sum((dphi @ gp.theta_factor) ** 2, axis=1)))
        sd_t = np.sqrt(sd ** 2 + (aleatoric_rel * m) ** 2)
        return m, m - 1.96 * sd, m + 1.96 * sd, m - 1.96 * sd_t, m + 1.96 * sd_t
    return "GP", fn


def plot_reference_only(data, ref_key, out_dir, n_points=150, model=None, aleatoric_rel=0.0,
                        figsize=(90 / 25.4, 75 / 25.4), show_ref=None):
    """UT and PS in one compact panel (default single-column size, 90 x 75 mm), one colour per test: measured data
    (circles), the reference model (dashed) and the given models (distilled: solid in the test colour; GP: grey
    dash-dot) with their 95 % bands. Two small legends (line styles / tests); the estimated coverage EC95 (share of the
    measured points inside each model's band) in a box at the lower right."""
    from matplotlib.lines import Line2D
    from plots.plot_reference_energy import ReferenceMaterial
    models = [] if model is None else (model if isinstance(model, list) else [model])
    show_ref = (not models) if show_ref is None else show_ref     # reference curve only when no model is drawn
    ref = ReferenceMaterial(ref_key)
    ref_name = ref.label.replace(" (UT+TTf)", "")
    col = {"UT": "#1565c0", "PS": "#c62828"}
    style = {"GP": dict(ls="-.", lw=1.1, z=7)}
    fig, ax = plt.subplots(figsize=figsize)
    cov_lines, ec_tab, metrics = [], {}, {}
    for t, (_, lat, title) in TESTS.items():
        d = data[t]
        lam = jnp.linspace(1.0, 1.03 * float(d["lam"].max()), n_points)
        Pr = np.asarray(jax.vmap(jax.grad(lambda l: ref.psi(path_F(l, lat))))(lam))
        k = max(1, len(d["lam"]) // 45)                     # ~45 markers per test, so the model lines stay visible
        ax.plot(d["lam"][::k], d["P"][::k], "o", ms=2.3, mfc="none", mew=0.5, color=col[t], zorder=3)
        if show_ref:      # off by default: the published (rounded) coefficients misrepresent the reference on UT
            ax.plot(lam, Pr, "--", color=col[t], lw=1.0, zorder=5)
        curves = [(ref_name, Pr)]
        for name, fn in models:
            mc = "#757575" if name == "GP" else col[t]          # the GP in grey, the distilled model in the test colour
            st = style.get(name, dict(ls="-", lw=1.3, z=6))
            mean, lo, hi, lo_t, hi_t = fn(lam, lat)

            def ec(a, b):
                a_d, b_d = np.interp(d["lam"], np.asarray(lam), a), np.interp(d["lam"], np.asarray(lam), b)
                return 100 * np.mean((d["P"] >= a_d) & (d["P"] <= b_d))
            cov, cov_t = ec(lo, hi), ec(lo_t, hi_t)
            noisy = aleatoric_rel > 0
            ax.fill_between(lam, lo, hi, color=mc, alpha=0.25 if noisy else 0.15, lw=0, zorder=2)
            if noisy:
                ax.fill_between(lam, lo_t, hi_t, color=mc, alpha=0.12, lw=0, zorder=1)
            ax.plot(lam, mean, st["ls"], color=mc, lw=st["lw"], zorder=st["z"])
            cov_lines.append(f"{t} {name}: {cov:.0f}%" + (f" / {cov_t:.0f}%" if noisy else ""))
            curves.append((name, mean))
            ec_tab[(t, name)] = (cov, cov_t if noisy else None)
            print(f"[{t}] {name}: estimated coverage of the 95% band {cov:.1f} % (epistemic)"
                  + (f", {cov_t:.1f} % (+ {100 * aleatoric_rel:g} % force noise)" if noisy else "")
                  + f" ({len(d['P'])} measured points)")
        for nm, curve in curves:
            pred = np.interp(d["lam"], np.asarray(lam), curve)
            m = d["lam"] > 1.02          # relative error: leave out the first points (P11 -> 0 at lambda -> 1)
            rel = 100 * np.abs(pred[m] - d["P"][m]) / d["P"][m]
            r2 = 1 - np.sum((pred - d["P"]) ** 2) / np.sum((d["P"] - d["P"].mean()) ** 2)
            l2 = 100 * np.sum((pred - d["P"]) ** 2) / np.sum(d["P"] ** 2)          # L2(P11) as in Flaschel et al.
            metrics.setdefault(t, {})[nm] = dict(r2=float(r2), rel_mean=float(rel.mean()), rel_max=float(rel.max()),
                                                 l2=float(l2))
            print(f"[{t}] {nm} vs data: R2 {r2:.4f}, rel. error (lambda > 1.02) mean {rel.mean():.1f} %, max {rel.max():.1f} %")
    ax.set_xlim(1.0, None)
    ax.set_ylim(0.0, None)
    ax.set_xlabel(r"Stretch $\lambda$", fontsize=7.5, labelpad=1)
    ax.set_ylabel(r"Nominal stress $P_{11}$", fontsize=7.5, labelpad=1)
    ax.tick_params(labelsize=6.5, length=2.5, pad=1.5, width=0.6)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
    ax.grid(False)
    # legend 1: line styles (black), legend 2: tests (colours)
    h_style = [Line2D([], [], marker="o", ls="", ms=2.8, mfc="none", mew=0.6, color="k", label="data")]
    if show_ref:
        h_style.append(Line2D([], [], ls="--", lw=1.0, color="k", label=ref_name))
    for name, _ in models:
        st = style.get(name, dict(ls="-", lw=1.3))
        h_style.append(Line2D([], [], ls=st["ls"], lw=st["lw"], color="#757575" if name == "GP" else "k",
                              label=f"{name} (mean, 95% CI)"))
    leg1 = ax.legend(handles=h_style, loc="upper left", fontsize=6, frameon=False, handlelength=1.8,
                     borderaxespad=0.3, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(handles=[Line2D([], [], lw=3, color=col[t], label=t) for t in TESTS], loc="upper center",
              bbox_to_anchor=(0.62, 1.0), fontsize=6, frameon=False, handlelength=1.0, borderaxespad=0.3,
              labelspacing=0.3)
    fig.tight_layout(pad=0.2)
    tag = ("only" if not models else "_".join(m_[0].lower() for m_ in models)) + (f"_noise{100 * aleatoric_rel:g}pct" if aleatoric_rel > 0 and model else "")
    out = os.path.join(out_dir, f"ut_ps_global_stress_{ref_key}_{tag}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=300)
    fig.set_dpi(300)
    fig.canvas.print_png(out[:-4] + ".png")
    plt.close(fig)
    print(f"saved {out} (+ .png)")
    write_ut_ps_table(metrics, ec_tab, [n for n, _ in models], ref_name, out[:-4].replace("ut_ps_global_stress", "ut_ps_metrics") + ".tex",
                      aleatoric_rel)


# L2(P11) [%] reported by Flaschel et al. (Table 4, UT + TTf local data) for the reference models: their published
# coefficients are rounded (e.g. 0.0003 for lambda^6.4), which inflates a re-evaluated UT error ~6x, so the table quotes
# the published values for these models
PAPER_L2 = {"EUCLID": {"UT": 0.16, "PS": 0.07}, "EUCLID w/o Ogden": {"UT": 0.17, "PS": 0.11}}


def write_ut_ps_table(metrics, ec_tab, model_names, ref_name, path, aleatoric_rel=0.0):
    """LaTeX table (booktabs): per test and model R^2, mean / max relative error, L2(P11) (Flaschel et al.: 100 sum
    (P_model - P_exp)^2 / sum P_exp^2) and the estimated coverage EC95; the reference models with their published L2
    only (dagger); best value per test and column in bold."""
    label = {"Distilled": "UGP + Distillation", "GP": "GP"}
    refs = [r for r in PAPER_L2]
    rows_out = []
    for t in TESTS:
        names = list(model_names) + refs
        vals = {}
        for n in model_names:
            m = metrics[t][n]
            vals[n] = [m["r2"], m["rel_mean"], m["rel_max"], m["l2"], ec_tab[(t, n)][0] if (t, n) in ec_tab else None]
        for n in refs:
            vals[n] = [None, None, None, PAPER_L2[n][t], None]

        def pick(c, fn):
            cand = [n for n in names if vals[n][c] is not None]
            return fn(cand, key=lambda n: vals[n][c]) if cand else None
        best = [pick(0, max), pick(1, min), pick(2, min), pick(3, min),
                min([n for n in names if vals[n][4] is not None], key=lambda n: abs(vals[n][4] - 95.0), default=None)]
        for j, n in enumerate(names):
            cells = []
            for c, f in enumerate(["{:.4f}", "{:.1f}", "{:.1f}", "{:.2f}", "{:.0f}"]):
                v = vals[n][c]
                txt = "--" if v is None else f.format(v) + (r"$^\dagger$" if c == 3 and n in refs else "")
                cells.append(r"\textbf{" + txt + "}" if v is not None and best[c] == n else txt)
            first = (r"\multirow{" + str(len(names)) + r"}{*}{" + t + "}") if j == 0 else ""
            rows_out.append(f"{first} & {label.get(n, n)} & " + " & ".join(cells) + r" \\")
        rows_out.append(r"\midrule")
    rows_out = rows_out[:-1]
    tex = "\n".join([
        r"\begin{table}[t]", r"\centering", r"\small", r"\begin{tabular}{llccccc}", r"\toprule",
        r"Test & Model & $R^2$ & $\bar{\epsilon}_{\mathrm{rel}}$ (\%) & $\max \epsilon_{\mathrm{rel}}$ (\%) & "
        r"$\mathcal{L}^2(P_{11})$ (\%) & EC$_{95}$ (\%) \\", r"\midrule", *rows_out, r"\bottomrule", r"\end{tabular}",
        r"\caption{Global nominal stress $P_{11}$ of the uniaxial tension (UT, training data) and pure shear (PS, not "
        r"used in training) tests against the model predictions along the ideal incompressible deformation paths. "
        r"$R^2$ over all measured points; relative error $\epsilon_{\mathrm{rel}} = 100\,|P_{11}^{\mathrm{model}} - "
        r"P_{11}^{\mathrm{exp}}|\,/\,P_{11}^{\mathrm{exp}}$ for $\lambda > 1.02$; "
        r"$\mathcal{L}^2(P_{11}) = 100 \sum (P_{11}^{\mathrm{model}} - P_{11}^{\mathrm{exp}})^2 / \sum (P_{11}^{\mathrm{exp}})^2$ "
        r"as in Flaschel et al.; EC$_{95}$: fraction of the measured curve (all DIC frames) inside the 95\,\% band of the "
        r"model (epistemic). $^\dagger$Value reported by Flaschel et al.\ (their published coefficients are rounded, "
        r"so their errors are not re-evaluated here). Best value per test in bold (EC$_{95}$: closest to 95\,\%).}",
        r"\label{tab:ut_ps_metrics}", r"\end{table}"])
    with open(path, "w") as f:
        f.write(tex + "\n")
    print(f"saved {path}")
    print(tex)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--distilled_dir", required=True)
    p.add_argument("--extract_dir", default=None, help="GP extraction (default: dev_source_extraction_dir.txt)")
    p.add_argument("--out_dir", default=None, help="output folder (default: --distilled_dir)")
    p.add_argument("--reference", default="euclid_ttf_ut", help="key of plots/plot_reference_energy.REFERENCES ('' = none)")
    p.add_argument("--n_samples", type=int, default=512, help="distilled parameter samples for the band")
    p.add_argument("--gauge_frac", type=float, default=0.3, help="length fraction of the central strip (measured stretch)")
    p.add_argument("--stretch", choices=["ends", "gauge"], default="ends",
                   help="measured stretch: ends = between the two end cuts (paper convention), gauge = central strip F11")
    p.add_argument("--n_points", type=int, default=150)
    p.add_argument("--train_ut_steps", default="dataset/preprocessed/experiment/ut_20_steps_ps_incomp.npz",
                   help="preprocessed UT training set: its steps are marked as training data ('' = no marking)")
    p.add_argument("--model", nargs="+", choices=["distilled", "gp", "none"], default=["distilled"],
                   help="--ref_only: the model drawn with the reference (mean, 95%% band, estimated coverage)")
    p.add_argument("--aleatoric_rel", type=float, default=0.0,
                   help="--ref_only: relative force-reading noise of a new measurement (e.g. 0.005), drawn as a second, "
                        "lighter band (epistemic + aleatoric) with its own coverage")
    p.add_argument("--ref_only", action="store_true",
                   help="one panel: the reference model along UT and PS with both measured curves (no GP / distilled)")
    args = p.parse_args()
    apply_style()
    dd = os.path.abspath(args.distilled_dir)
    ex = args.extract_dir
    if ex is None:
        for f in ("dev_source_extraction_dir.txt", "source_extraction_dir.txt"):
            if os.path.exists(os.path.join(dd, f)):
                ex = open(os.path.join(dd, f)).read().strip()
                break
    if ex and not os.path.isdir(ex):           # recorded on the cluster: the sibling extraction folder
        ex = os.path.join(os.path.dirname(dd), "extracted")

    # --- measured global curves (cached)
    cache = os.path.join(dd, "ut_ps_global_data.npz")
    data = {}
    if os.path.exists(cache):
        z = np.load(cache)
        data = {t: {k[len(t) + 1:]: z[k] for k in z.files if k.startswith(t + "_")} for t in TESTS}
    for t, (folder, _, _) in TESTS.items():
        if (not data.get(t) or float(data[t].get("gauge_frac", -1)) != args.gauge_frac
                or str(data[t].get("stretch", "gauge")) != args.stretch):
            print(f"[{t}] reading {folder} ...")
            data[t] = global_curve(os.path.join(ROOT, folder), args.gauge_frac, args.stretch)
            data[t]["gauge_frac"] = args.gauge_frac
            data[t]["stretch"] = args.stretch
        d = data[t]
        print(f"[{t}] {len(d['steps'])} steps, W0 = {float(d['W0']):.2f} mm, lambda max {d['lam'].max():.3f} "
              f"(lateral {d['lam2'][np.argmax(d['lam'])]:.3f}), P11 max {d['P'].max():.3f}")
    np.savez(cache, **{f"{t}_{k}": v for t, d in data.items() for k, v in d.items()})
    for t, last in LAST_STEP.items():       # drop unreliable final frames (see LAST_STEP)
        keep = data[t]["steps"] <= last
        data[t] = {k_: (v[keep] if np.ndim(v) and len(v) == len(keep) else v) for k_, v in data[t].items()}
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else dd
    os.makedirs(out_dir, exist_ok=True)
    if args.ref_only:
        model = [model_band_fn(m_, dd, ex, args.n_samples, args.aleatoric_rel) for m_ in args.model if m_ != "none"] or None
        plot_reference_only(data, args.reference or "euclid_ttf_ut", out_dir, args.n_points, model, args.aleatoric_rel)
        return

    # --- models
    from core.teacher import load_hsgp_teacher
    from plots.training import compute_interpolation_masks
    gp = load_hsgp_teacher(ex)
    psi_d, theta = distilled_energy_fn(dd)
    theta = theta[np.random.default_rng(0).choice(len(theta), min(args.n_samples, len(theta)), replace=False)]
    ref = None
    if args.reference:
        from plots.plot_reference_energy import ReferenceMaterial
        ref = ReferenceMaterial(args.reference)
    I_obs_f = os.path.join(ex, "I_obs_all.npy")
    I_obs = np.load(I_obs_f) if os.path.exists(I_obs_f) else None

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    dist_col, gp_col, ref_col = "#009E73", "gray", "black"
    data_col = {"UT": "#1565c0", "PS": "#c62828"}
    train_steps = set()
    if args.train_ut_steps and os.path.exists(os.path.join(ROOT, args.train_ut_steps)):
        train_steps = set(np.load(os.path.join(ROOT, args.train_ut_steps))["steps"].tolist())
    for ax, (t, (_, lat, title)) in zip(axes, TESTS.items()):
        d = data[t]
        lam = jnp.linspace(1.0, 1.03 * float(d["lam"].max()), args.n_points)
        Fp = jax.vmap(lambda l: path_F(l, lat))(lam)
        dF = jax.vmap(jax.jacfwd(lambda l: path_F(l, lat)))(lam)                     # dF/dlambda (N, 3, 3)

        # distilled: P11 = dPsi/dlambda per sample
        dpsi = jax.grad(lambda l, th: psi_d(th, path_F(l, lat)))
        Pd = np.asarray(jax.vmap(lambda th: jax.vmap(lambda l: dpsi(l, th))(lam))(jnp.asarray(theta)))
        # GP: derivative of the basis along the path -> exact mean and variance of dPsi/dlambda
        dphi = jax.vmap(lambda F, dF_: jnp.einsum("pij,ij->p", jax.jacfwd(gp.basis)(F), dF_))(Fp, dF)
        gm = np.asarray(dphi @ gp.theta_mean)
        gs = np.asarray(jnp.sqrt(jnp.sum((dphi @ gp.theta_factor) ** 2, axis=1)))
        inside = np.asarray(compute_interpolation_masks(gp, Fp[None], I_obs)["tot"][0])
        out_k = np.where(~inside[1:])[0]
        l_ex = float(lam[out_k[0] + 1]) if len(out_k) else None

        ax.fill_between(lam, gm - 1.96 * gs, gm + 1.96 * gs, color=gp_col, alpha=0.3, lw=0, label="GP 95% CI", zorder=2)
        ax.plot(lam, gm, color=gp_col, lw=1.5, label="GP mean", zorder=4)
        ax.fill_between(lam, np.percentile(Pd, 2.5, 0), np.percentile(Pd, 97.5, 0), color=dist_col, alpha=0.22, lw=0,
                        label="Distilled 95% CI", zorder=3)
        ax.plot(lam, Pd.mean(0), color=dist_col, lw=2.0, label="Distilled mean", zorder=5)
        if ref is not None:
            Pr = np.asarray(jax.vmap(jax.grad(lambda l: ref.psi(path_F(l, lat))))(lam))
            ax.plot(lam, Pr, "--", color=ref_col, lw=1.5, label=f"{ref.label} (reference)", zorder=6)
        is_tr = np.array([s in train_steps for s in d["steps"]]) if t == "UT" else np.zeros(len(d["steps"]), bool)
        ax.plot(d["lam"], d["P"], "o", ms=2.2, mfc="none", mew=0.6, color=data_col[t], alpha=0.6, zorder=7,
                label=f"{t} global data")
        if is_tr.any():
            ax.plot(d["lam"][is_tr], d["P"][is_tr], "o", ms=4.5, color=data_col[t], mec="white", mew=0.6, zorder=8,
                    label="UT training steps")
        if l_ex is not None:
            ax.axvspan(l_ex, float(lam[-1]), color="lightgrey", alpha=0.45, lw=0, zorder=1, label="GP extrapolation")
            ax.axvline(l_ex, color="black", ls=":", lw=1.1, zorder=4)
        ax.set_xlim(1.0, float(lam[-1]))
        ymax = 1.15 * max(float(d["P"].max()), float(Pd.mean(0).max()))
        ax.set_ylim(0.0, ymax)
        ax.set_title(f"{title} ({t}{', training' if is_tr.any() else ', not used in training'})", fontsize=10.5)
        ax.set_xlabel(r"Stretch $\lambda$")
        ax.set_ylabel(r"Nominal stress $P_{11}$")
        ax.grid(alpha=0.2)
        # summary: distilled / GP mean vs the measured points
        for nm, curve in (("distilled", Pd.mean(0)), ("GP", gm)):
            pred = np.interp(d["lam"], np.asarray(lam), curve)
            m = d["lam"] > 1.02
            print(f"[{t}] {nm} mean vs data: RMSE {np.sqrt(np.mean((pred - d['P']) ** 2)):.4f}, "
                  f"MAPE (lambda > 1.02) {100 * np.mean(np.abs(pred[m] - d['P'][m]) / d['P'][m]):.1f} %")
        if l_ex is not None:
            print(f"[{t}] GP extrapolates beyond lambda = {l_ex:.3f}")

    h, lab = [], []
    for ax in axes:
        for x_, y_ in zip(*ax.get_legend_handles_labels()):
            if y_ not in lab:
                h.append(x_)
                lab.append(y_)
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8.8, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=[0, 0.12, 1, 1])
    out = os.path.join(out_dir, f"ut_ps_global_stress{'_ref' + args.reference if args.reference else ''}.pdf")
    fig.savefig(out, bbox_inches="tight", dpi=200)
    fig.set_dpi(200)
    fig.canvas.print_png(out[:-4] + ".png")
    plt.close(fig)
    print(f"saved {out} (+ .png)")


if __name__ == "__main__":
    main()

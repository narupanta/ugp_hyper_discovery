#!/usr/bin/env python3
"""
Summary tables over several coverage studies (one per test case), as Markdown and LaTeX:
  1. discovered parameters: per candidate term, the true value, the lower and upper 95% bounds (mean +- std over
     runs of each run's bound) and the per-term
     rate (true term: interval covers the true value; absent term: filtered out, i.e. the interval contains 0);
  2. strain-energy accuracy of the GP posterior (one per noise seed) and the distilled model (one per seed x init):
     RMSE, R^2 and 95% coverage (EC) on the validation load steps, mean +- std; plus the median distilled model.

Needs each study's coverage_summary.json (validation/distillation_coverage_study.py) and the runs' flow samples and
validation_metrics.json (plots/plot_distilled_r2_energy.py).

Usage: python3 plots/make_coverage_comparison_tables.py <study_dir>:<label> [...] --out <basename>
Writes <basename>.md and <basename>.tex
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize_coverage_metrics import read  # noqa: E402
from make_coverage_latex_table import tex_term  # noqa: E402


def run_samples(run_dir, library, aniso):
    """Full parameter samples of one run (columns as coverage_summary.json 'parameters')."""
    from core.material_library import full_param_vectors, sort_ogden_terms
    dev = sort_ogden_terms(np.load(os.path.join(run_dir, "dev_flow_samples.npy")), library)
    vol = np.zeros((dev.shape[0], 3))
    full = full_param_vectors(dev, vol, library)
    if aniso:
        an = np.load(os.path.join(run_dir, "aniso_flow_samples.npy"))
        n = min(len(full), len(an))
        full = np.concatenate([full[:n], np.zeros((n, max(0, 19 - full.shape[1])))], axis=1)
        full[:, 13:13 + min(6, an.shape[1])] = an[:n, :6]
    return full


def load_case(study, level=0.95):
    summ = json.load(open(os.path.join(study, "coverage_summary.json")))
    names, truth = summ["parameters"], np.asarray(summ["truth"], float)
    terms = summ["candidate_terms"]
    aniso = any(n in names for n in ("C42", "C62"))
    k = max(summ["checkpoints"], key=int)
    per = summ["checkpoints"][k]["per_parameter"]
    a = 0.5 * (1.0 - level)
    rows = {n: dict(mean=[], lo=[], hi=[], sel=[]) for n in terms["true"] + terms["absent"]}
    metrics = {}
    for s in summ["noise_seeds"]:
        for m in range(summ["n_inits"]):
            rd = os.path.join(study, f"seed{s}", f"init{m}")
            if not os.path.exists(os.path.join(rd, "dev_flow_samples.npy")):
                continue
            full = run_samples(rd, summ.get("library", "gmr"), aniso)
            lo, hi, mu = np.quantile(full, a, axis=0), np.quantile(full, 1 - a, axis=0), full.mean(0)
            for n in rows:
                i = names.index(n)
                rows[n]["mean"].append(mu[i])
                rows[n]["lo"].append(lo[i])
                rows[n]["hi"].append(hi[i])
                rows[n]["sel"].append(bool(lo[i] > 1e-12 or hi[i] < -1e-12))
            r = read(rd, "total")
            if r is not None:
                metrics[(s, m)] = r
    keys = sorted(metrics)
    gp = {}
    for s in sorted({s for s, _ in keys}):
        gp[s] = metrics[min(k_ for k_ in keys if k_[0] == s)]["gp"]
    dist = [metrics[k_]["dist"] for k_ in keys]
    rm, r2, ec = (np.array([d[q] for d in dist]) for q in ("rmse", "r2", "coverage"))
    rank = lambda x: np.argsort(np.argsort(x, kind="stable"), kind="stable")
    score = (rank(rm) + rank(-r2) + rank(np.abs(ec - 95.0))) / 3.0
    med = keys[int(np.argsort(score, kind="stable")[(len(keys) - 1) // 2])]
    return dict(names=names, truth=truth, terms=terms, per=per, rows=rows, n_runs=len(rows[terms["true"][0]]["mean"]),
                gp=list(gp.values()), dist=dist, median=med, median_vals=metrics[med]["dist"])


def g(x):
    return f"{x:.3g}" if abs(x) >= 1e-3 or x == 0 else f"{x:.1e}"


def gt(x):
    """g() for LaTeX math: 2.9e-05 -> 2.9 \\times 10^{-5}."""
    t = g(x)
    if "e" in t:
        m, e = t.split("e")
        return f"{m} \\times 10^{{{int(e)}}}"
    return t


def ms(v, d):
    v = np.asarray(v, float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def coverage_confusion(c):
    """Pooled over runs x candidate terms: true terms covered / not; absent terms filtered (covers 0) / spurious."""
    n = c["n_runs"]
    tp = sum(round(c["per"][t] / 100 * n) for t in c["terms"]["true"])
    tn = sum(round(c["per"][t] / 100 * n) for t in c["terms"]["absent"])
    nt, na = n * len(c["terms"]["true"]), n * len(c["terms"]["absent"])
    return dict(tp=tp, fn=nt - tp, tn=tn, fp=na - tn, nt=nt, na=na)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cases", nargs="+", help="<study_dir>:<label>")
    p.add_argument("--out", required=True, help="output basename (.md and .tex are added)")
    args = p.parse_args()
    cases = []
    for c in args.cases:
        path, _, label = c.rpartition(":")
        print(f"loading {label} ({path}) ...", flush=True)
        cases.append((label, load_case(os.path.abspath(path))))

    # ---------------- Markdown
    L = ["# Coverage studies: discovered parameters and energy accuracy", ""]
    L += ["Runs per case: " + ", ".join(f"{lab} {c['n_runs']} (10 noise seeds x inits)" for lab, c in cases) + ".", "",
          "## 1. Discovered parameters", "",
          "95% lower / upper bound: mean ± std over runs of each run's 2.5% / 97.5% posterior quantile. Rate: true term = interval covers "
          "the true value; absent term = filtered out (interval contains 0). Candidate terms disabled in every run "
          "are left out.", "",
          "| Case | Term | True value | 95% lower bound | 95% upper bound | Covered / filtered (%) |",
          "|---|---|---|---|---|---|"]
    for lab, c in cases:
        first = True
        for n in c["terms"]["true"] + c["terms"]["absent"]:
            r, t = c["rows"][n], c["truth"][c["names"].index(n)]
            (lm, ls), (hm, hs) = ms(r["lo"], 3), ms(r["hi"], 3)
            kind = "covered" if t != 0 else "filtered"
            L.append(f"| {lab if first else ''} | {n} | {g(t)} | {g(lm)} ± {g(ls)} | {g(hm)} ± {g(hs)} | "
                     f"{c['per'][n]:.0f} ({kind}) |")
            first = False
        mean_t = np.mean([c["per"][n] for n in c["terms"]["true"] + c["terms"]["absent"]])
        L.append(f"| | *mean over terms* | | | | *{mean_t:.1f}* |")
    L += ["", "## 2. Strain-energy accuracy (validation load steps)", "",
          "GP posterior: one per noise seed; distilled model: one per seed x init. Mean ± std. Median model: middle "
          "of the ranking by RMSE, R² and |EC − 95| (ranks averaged).", "",
          "| Case | Model | RMSE | R² | 95% EC (%) |", "|---|---|---|---|---|"]
    for lab, c in cases:
        for nm, lst in (("GP posterior", c["gp"]), ("Distilled", c["dist"])):
            vals = [ms([d[q] for d in lst], 0) for q in ("rmse", "r2", "coverage")]
            L.append(f"| {lab if nm == 'GP posterior' else ''} | {nm} | {vals[0][0]:.4f} ± {vals[0][1]:.4f} | "
                     f"{vals[1][0]:.4f} ± {vals[1][1]:.4f} | {vals[2][0]:.1f} ± {vals[2][1]:.1f} |")
        v, (s, m) = c["median_vals"], c["median"]
        L.append(f"| | Distilled, median model (seed {s}, init {m}) | {v['rmse']:.4f} | {v['r2']:.4f} | {v['coverage']:.1f} |")
    L += ["", "## 3. Confusion matrix by interval coverage", "",
          "Pooled over runs x candidate terms. Covered = the 95% interval contains the true value; for a term not in the "
          "true model the true value is 0, so covered = filtered out and not covered = spurious.", "",
          "| Case | | Interval covers the true value | Interval misses it | Rate (%) |", "|---|---|---|---|---|"]
    for lab, c in cases:
        q = coverage_confusion(c)
        L.append(f"| {lab} | In the true model | {q['tp']} (found and covered) | {q['fn']} (missed or not covered) | "
                 f"{100 * q['tp'] / q['nt']:.1f} |")
        L.append(f"| | Not in the true model | {q['tn']} (filtered) | {q['fp']} (spurious) | {100 * q['tn'] / q['na']:.1f} |")
    open(args.out + ".md", "w").write("\n".join(L) + "\n")

    # ---------------- LaTeX
    T = ["% generated by plots/make_coverage_comparison_tables.py; needs \\usepackage{booktabs,multirow}",
         "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}",
         "\\begin{tabular}{llrrrr}", "\\toprule",
         "Case & Term & True & 95\\% lower bound & 95\\% upper bound & Cov./filt. (\\%) \\\\", "\\midrule"]
    for j, (lab, c) in enumerate(cases):
        tl = c["terms"]["true"] + c["terms"]["absent"]
        for i, n in enumerate(tl):
            r, t = c["rows"][n], c["truth"][c["names"].index(n)]
            (lm, ls), (hm, hs) = ms(r["lo"], 3), ms(r["hi"], 3)
            cell = f"\\multirow{{{len(tl) + 1}}}{{*}}{{{lab}}}" if i == 0 else ""
            rate = f"${c['per'][n]:.0f}" + ("$" if t != 0 else "^\\dagger$")
            T.append(f"{cell} & {tex_term(n)} & ${gt(t)}$ & ${gt(lm)} \\pm {gt(ls)}$ & ${gt(hm)} \\pm {gt(hs)}$ & "
                     f"{rate} \\\\")
        mean_t = np.mean([c["per"][n] for n in tl])
        T.append(f" & \\multicolumn{{4}}{{l}}{{\\emph{{mean over terms}}}} & ${mean_t:.1f}$ \\\\")
        if j < len(cases) - 1:
            T.append("\\midrule")
    T += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Discovered parameters over all runs (noise seeds $\\times$ flow initialisations). 95\\% lower / upper "
          "bound: mean $\\pm$ std over the runs of each run's 2.5\\% / 97.5\\% posterior quantile; Cov./filt.: true terms, runs whose interval covers the true value; terms not in the "
          "true model ($^\\dagger$), runs in which the term is filtered out.}",
          "\\label{tab:coverage_parameters}", "\\end{table}", "",
          "\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{llccc}", "\\toprule",
          "Case & Model & RMSE & $R^2$ & EC$_{95}$ (\\%) \\\\", "\\midrule"]
    for j, (lab, c) in enumerate(cases):
        for i, (nm, lst) in enumerate((("GP posterior", c["gp"]), ("Distilled", c["dist"]))):
            vals = [ms([d[q] for d in lst], 0) for q in ("rmse", "r2", "coverage")]
            cell = f"\\multirow{{3}}{{*}}{{{lab}}}" if i == 0 else ""
            T.append(f"{cell} & {nm} & ${vals[0][0]:.4f} \\pm {vals[0][1]:.4f}$ & "
                     f"${vals[1][0]:.4f} \\pm {vals[1][1]:.4f}$ & ${vals[2][0]:.1f} \\pm {vals[2][1]:.1f}$ \\\\")
        v, (s, m) = c["median_vals"], c["median"]
        T.append(f" & Median model (seed ${s}$, init ${m}$) & ${v['rmse']:.4f}$ & ${v['r2']:.4f}$ & ${v['coverage']:.1f}$ \\\\")
        if j < len(cases) - 1:
            T.append("\\midrule")
    T += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Strain-energy accuracy on the validation load steps: GP posterior (one per noise seed) and "
          "distilled model (one per noise seed and initialisation), mean $\\pm$ std; median model: the middle run of "
          "the ranking by RMSE, $R^2$ and $|\\mathrm{EC}_{95} - 95|$ (ranks averaged).}",
          "\\label{tab:coverage_metrics}", "\\end{table}", "",
          "\\begin{table}[t]", "\\centering", "\\small", "\\begin{tabular}{llccc}", "\\toprule",
          "Case & & Interval covers true value & Interval misses true value & Rate (\\%) \\\\", "\\midrule"]
    for j, (lab, c) in enumerate(cases):
        q = coverage_confusion(c)
        T.append(f"\\multirow{{2}}{{*}}{{{lab}}} & In true model & ${q['tp']}$ & ${q['fn']}$ & "
                 f"${100 * q['tp'] / q['nt']:.1f}$ \\\\")
        T.append(f" & Not in true model & ${q['tn']}$ & ${q['fp']}$ & ${100 * q['tn'] / q['na']:.1f}$ \\\\")
        if j < len(cases) - 1:
            T.append("\\midrule")
    T += ["\\bottomrule", "\\end{tabular}",
          "\\caption{Confusion matrix by interval coverage, pooled over all runs and candidate terms. A term in the "
          "true model counts as covered when its 95\\% interval contains the true value (otherwise missed or not "
          "covered); a term not in the true model has true value 0, so it is covered when it is filtered out (interval "
          "contains 0) and misses when it is spurious. Rate: covered share of each row.}",
          "\\label{tab:coverage_confusion}", "\\end{table}"]
    open(args.out + ".tex", "w").write("\n".join(T) + "\n")
    print("\n".join(L))
    print(f"\n{args.out}.md\n{args.out}.tex")


if __name__ == "__main__":
    main()

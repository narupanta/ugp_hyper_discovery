#!/usr/bin/env python3
"""
Energy data of an experimental run (no ground truth) for plots/plot_discovery_summary.py, with a published reference
model in place of the truth:
  energy_parity_total.npz        GP and distilled energy vs the reference at the validation load steps of every
                                 experiment of the recipe (random subset of the element states);
  mode_energy_g<max_gamma>.npz   the six standard modes (definitions and data-region test as plots/plot_split_summary.py)
                                 with the reference energy as psi_true.
Both files carry ref_label (the reference name), which the summary figure uses for its labels.

Usage:
  python3 plots/export_reference_energy.py --distilled_dir results/<exp>/1/distilled --reference euclid_ttf_ut
"""
import argparse
import os
import sys

import jax
import jax.numpy as jnp
import numpy as np
import yaml

jax.config.update("jax_enable_x64", True)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--distilled_dir", required=True)
    ap.add_argument("--extract_dir", default=None, help="default: <distilled_dir>/../extracted")
    ap.add_argument("--reference", default="euclid_ttf_ut")
    ap.add_argument("--max_gamma", type=float, default=2.0)
    ap.add_argument("--n_states", type=int, default=4000, help="validation states used for the parity (random subset)")
    ap.add_argument("--n_samples", type=int, default=256, help="distilled parameter samples")
    args = ap.parse_args()
    from core.teacher import load_hsgp_teacher
    from core.utils import load_f3x3_from_dataset
    from plots.plot_reference_energy import ReferenceMaterial
    from plots.plot_ut_ps_global import distilled_energy_fn
    d = os.path.abspath(args.distilled_dir)
    ex = args.extract_dir or os.path.join(os.path.dirname(d), "extracted")
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(d)), "config.yaml")))
    ref = ReferenceMaterial(args.reference)
    ref_label = ref.label.replace(" (UT+TTf)", "")
    gp = load_hsgp_teacher(ex)
    psi_d, theta = distilled_energy_fn(d)
    rng = np.random.default_rng(0)
    theta = jnp.asarray(theta[rng.choice(len(theta), min(args.n_samples, len(theta)), replace=False)])

    # 1. parity at the validation states of every experiment
    val = [int(v) for v in cfg.get("val_load_steps_indices", [])]
    exps = cfg.get("experiments") or [dict(dataset_path=cfg["dataset_path"])]
    F = []
    for e in exps:
        prep = dict(np.load(os.path.join(ROOT, e["dataset_path"]), allow_pickle=True))
        F3 = np.asarray(load_f3x3_from_dataset(prep))
        steps = [s for s in val if s < F3.shape[0]] or list(range(F3.shape[0]))
        F.append(F3[steps].reshape(-1, 3, 3))
    F = np.concatenate(F)
    F = jnp.asarray(F[rng.choice(len(F), min(args.n_states, len(F)), replace=False)])
    t = np.asarray(jax.vmap(ref.psi)(F))
    g = gp.psi_dist(F)
    S = np.asarray(jax.vmap(lambda th: jax.vmap(lambda f: psi_d(th, f))(F))(theta))
    np.savez(os.path.join(d, "energy_parity_total.npz"), true=t, gp_mean=np.asarray(g.mean),
             gp_std=np.sqrt(np.maximum(np.asarray(g.var), 0.0)), dist_mean=S.mean(0),
             dist_q025=np.percentile(S, 2.5, 0), dist_q975=np.percentile(S, 97.5, 0), ref_label=np.array(ref_label),
             val_steps=np.asarray(val))
    print(f"parity: {len(t)} validation states vs {ref_label}")

    # 2. standard modes, as plots/plot_split_summary.py (same mode definitions and data-region test), with the
    #    reference energy as psi_true
    from plots.training import compute_interpolation_masks, standard_mode_F
    me = os.path.join(d, f"mode_energy_g{args.max_gamma:g}.npz")
    gamma = jnp.linspace(0.0, args.max_gamma, 100)
    Fm = standard_mode_F(gp, gamma)                                   # (6, 100, 3, 3)
    I_obs = np.load(os.path.join(ex, "I_obs_all.npy")) if os.path.exists(os.path.join(ex, "I_obs_all.npy")) else None
    inside = np.asarray(compute_interpolation_masks(gp, Fm, I_obs)["tot"])
    gm, gs, dm, d1, d2, pt = [], [], [], [], [], []
    for k in range(Fm.shape[0]):
        g = gp.psi_dist(Fm[k])
        gm.append(np.asarray(g.mean))
        gs.append(np.sqrt(np.maximum(np.asarray(g.var), 0.0)))
        Sm = np.asarray(jax.vmap(lambda th: jax.vmap(lambda f: psi_d(th, f))(Fm[k]))(theta))
        dm.append(Sm.mean(0))
        d1.append(np.percentile(Sm, 2.5, 0))
        d2.append(np.percentile(Sm, 97.5, 0))
        pt.append(np.asarray(jax.vmap(ref.psi)(Fm[k])))
    np.savez(me, gamma=np.asarray(gamma), F=np.asarray(Fm), mode_names=np.array(["UT", "ET", "PS", "UC", "EC", "SS"]),
             inside=inside, psi_true=np.stack(pt), gp_mean=np.stack(gm), gp_std=np.stack(gs), dist_mean=np.stack(dm),
             dist_q025=np.stack(d1), dist_q975=np.stack(d2), ref_label=np.array(ref_label))
    print(f"modes: {me} with psi_true = {ref_label}")


if __name__ == "__main__":
    main()

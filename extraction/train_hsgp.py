"""
HSGP extraction (closed-form reduced-rank GP posterior under the EIV likelihood, core/hsgp_*.py) for one seed.

Writes the same layout as extraction/train_unsupervised.py into --batch_dir:
    hsgp_posterior.npz, hsgp_history.json, config.json, I_obs_all.npy, extraction_metrics.json and the plots.

--phase fit|plot|all: the fit and the plots run in separate processes on small machines (peak memory adds up
otherwise); scripts/run_hsgp_pipeline.sh calls the two phases one after the other.

HSGP settings are read from the recipe (keys below) and can be overridden on the command line:
    hsgp_likelihood: eiv            # eiv | residual | residual_learned
    hsgp_prior_mean: none           # none | linear_elastic
    hsgp_envelope: 1                # amplitude envelope s_dev, s_vol from the linear-elastic stage
    hsgp_warp: 1                    # GP inputs log(I_bar - 2), log J
    hsgp_continuation: 1            # load-step continuation over the training steps
    hsgp_num_basis_dev: 24          # eigenfunctions per deviatoric dimension
    hsgp_num_basis_vol: 32
    hsgp_box_factor: 8.0
    hsgp_max_outer: 60              # GP-stage iteration cap per continuation stage
    hsgp_amplitude_prior_scale: 1.0 # log-normal hyperprior widths (log units)
    hsgp_lengthscale_prior_scale: 1.0
    hsgp_reaction_weight: 1.0       # residual_learned only
    hsgp_sigma_cap: 10.0            # GP amplitudes capped at this multiple of their prior centre
    hsgp_stability_search: 1        # Gauss-Newton updates must keep the mean tangent positive definite
    hsgp_hyper_restarts: 1          # multi-start hyperparameter optimisation (prior centre, lengthscales x0.5, x2)
    hsgp_continuation_stages:       # optional explicit ladder of cumulative load-step sets (recipe only), e.g.
      - [1, 2, 3]                   #   default: [first two training steps], then one more step per stage
      - [1, 2, 3, 5]
      - [1, 2, 3, 5, 7]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from core.datasetclass import DatasetFactory
from core.hsgp_inference import fit_hsgp, eiv_reaction_prediction
from core.hsgp_model import HSGPHyperelasticity
from core.loss_function import ellipticity_penalty
from core.material_models import get_material
from core.utils import load_f3x3_from_dataset
from plots.theme import apply_style, save_figure

HSGP_DEFAULTS = dict(hsgp_likelihood="eiv", hsgp_prior_mean="none", hsgp_envelope=1, hsgp_warp=1, hsgp_continuation=1,
                     hsgp_num_basis_dev=24, hsgp_num_basis_vol=32, hsgp_box_factor=8.0, hsgp_max_outer=60,
                     hsgp_amplitude_prior_scale=1.0, hsgp_lengthscale_prior_scale=1.0, hsgp_reaction_weight=1.0,
                     hsgp_sigma_cap=10.0, hsgp_stability_search=1, hsgp_hyper_restarts=1)


def parse_args():
    p = argparse.ArgumentParser(description="HSGP strain-energy extraction (one seed).")
    p.add_argument("--recipe", required=True, help="Recipe / experiment config.yaml")
    p.add_argument("--dataset_path", required=True, help="Dataset spec 'clean.npz?seed=..&disp_noise=..&load_noise=..' or file")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch_dir", required=True, help="Output directory (<experiment>/<seed>/extracted)")
    p.add_argument("--train_load_steps_indices", type=int, nargs="+", default=None)
    p.add_argument("--val_load_steps_indices", type=int, nargs="+", default=None)
    p.add_argument("--test_load_steps_indices", type=int, nargs="*", default=None)
    p.add_argument("--phase", choices=["fit", "plot", "all"], default="all")
    for k, v in HSGP_DEFAULTS.items():   # --hsgp_warp 0 etc. override the recipe
        p.add_argument(f"--{k}", type=type(v), default=None)
    return p.parse_args()


def external_work_density(prep, node_type, last_step):
    """Data energy scale: external work up to last_step / specimen area (as in extraction/train_unsupervised.py)."""
    u = np.asarray(prep["u_obs"] if "u_obs" in prep else prep["u"])[: last_step + 1]
    area = float(np.sum(np.asarray(prep["dA"])))
    R = np.asarray(prep["reaction_forces"])[: last_step + 1]
    nt = np.asarray(node_type)
    ux = u[:, nt[:, 3] == 1, 0].mean(axis=1) if np.any(nt[:, 3] == 1) else np.zeros(len(u))
    uy = u[:, nt[:, 4] == 1, 1].mean(axis=1) if np.any(nt[:, 4] == 1) else np.zeros(len(u))
    dW = 0.5 * (R[1:, 0] + R[:-1, 0]) * np.diff(ux) + 0.5 * (R[1:, 1] + R[:-1, 1]) * np.diff(uy)
    e = float(np.sum(dW)) / area
    return e if np.isfinite(e) and e > 0 else 1.0


def plot_reactions(model, F_steps, prep, train_steps, save_path, has_truth):
    """Reaction forces per load step: epistemic band vs true reactions (synthetic), predictive band vs measured."""
    corr = model.info.get("likelihood", "eiv") not in ("residual", "residual_learned")
    mean, cov = eiv_reaction_prediction(model, F_steps, prep["cells"], prep["node_type"], prep["dNdX"], prep["dA"], correction=corr)
    sd_epi = np.sqrt(np.stack([cov[:, 0, 0], cov[:, 1, 1]], axis=1))
    sig_R = np.maximum(np.asarray(prep["load_noise_std_steps"]), 1e-3)
    sd_pred = np.sqrt(sd_epi ** 2 + sig_R ** 2)
    R_obs = np.asarray(prep["reaction_forces"])
    panels = [("predictive", sd_pred, R_obs, "measured reaction")]
    if has_truth:
        panels.insert(0, ("epistemic", sd_epi, np.asarray(prep["reaction_forces_true"]), "true reaction"))
    steps = np.arange(F_steps.shape[0])
    is_tr = np.isin(steps, train_steps)
    apply_style()
    fig, axes = plt.subplots(2, len(panels), figsize=(7.0 * len(panels), 10.0), sharex=True, squeeze=False)
    out = {}
    for i, comp in enumerate(["x", "y"]):
        for j, (kind, sd, ref, lab) in enumerate(panels):
            ax = axes[i, j]
            m, s, r = mean[:, i], sd[:, i], ref[:, i]
            hit = np.abs(r - m) <= 1.96 * s
            ax.fill_between(steps, m - 1.96 * s, m + 1.96 * s, color="tab:blue", alpha=0.25, label="95% band")
            ax.plot(steps, m, color="tab:blue", lw=2, label="posterior mean")
            ax.plot(steps[~is_tr], r[~is_tr], "o", color="k", ms=5, label=f"{lab} (validation step)")
            ax.plot(steps[is_tr], r[is_tr], "s", color="tab:red", ms=7, label=f"{lab} (training step)")
            for k in np.where(~hit)[0]:
                ax.plot(steps[k], r[k], "x", color="tab:orange", ms=11, mew=2)
            r2 = 1 - np.sum((r - m) ** 2) / np.sum((r - r.mean()) ** 2)
            cov_all = hit.mean() * 100
            cov_val = hit[~is_tr].mean() * 100 if np.any(~is_tr) else float("nan")
            ax.set_title(f"$R_{comp}$ {kind}: coverage {cov_all:.0f}% (val {cov_val:.0f}%), $R^2$ {r2:.4f}")
            ax.set_ylabel(f"$R_{comp}$")
            if i == 1:
                ax.set_xlabel("load step")
            if i == 0 and j == 0:
                ax.legend(fontsize=9)
            out[f"{kind}_{comp}"] = dict(coverage=float(cov_all), coverage_val=float(cov_val), r2=float(r2),
                                         z=((r - m) / s).tolist())
    fig.suptitle("Reaction forces (" + ("EIV-corrected prediction" if corr else "residual model, no state correction")
                 + " at the observed displacements)")
    save_figure(fig, os.path.join(save_path, "reaction_force_uncertainty.pdf"))
    return out


def plot_convergence(history, save_path):
    apply_style()
    it = [h["iteration"] for h in history]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(it, [h["log_evidence"] for h in history], "o-"); ax[0].set_title("log evidence"); ax[0].set_xlabel("outer iteration")
    ax[1].semilogy(it, [h["force_change"] for h in history], "o-"); ax[1].set_title("relative change of internal forces")
    ax[2].semilogy(it, [h["sigma_u_x"] for h in history], "o-", label=r"$\sigma_{u,x}$")
    ax[2].semilogy(it, [h["sigma_u_y"] for h in history], "s-", label=r"$\sigma_{u,y}$"); ax[2].legend(); ax[2].set_title("displacement noise")
    stage_starts = [h["iteration"] for k, h in enumerate(history)
                    if k > 0 and h.get("continuation_stage", 0) != history[k - 1].get("continuation_stage", 0)]
    for a in ax:
        for x in stage_starts:
            a.axvline(x - 0.5, color="grey", ls=":")
    save_figure(fig, os.path.join(save_path, "hsgp_convergence.pdf"))


def main():
    args = parse_args()
    rec = yaml.safe_load(open(args.recipe)) or {}
    opt = {k: (getattr(args, k) if getattr(args, k) is not None else type(v)(rec.get(k, v))) for k, v in HSGP_DEFAULTS.items()}
    train_steps = args.train_load_steps_indices or rec.get("train_load_steps_indices")
    val_steps = args.val_load_steps_indices or rec.get("val_load_steps_indices", [])
    test_steps = args.test_load_steps_indices if args.test_load_steps_indices is not None else rec.get("test_load_steps_indices", [])
    save_path = args.batch_dir
    os.makedirs(save_path, exist_ok=True)

    prep = DatasetFactory.create("dataset/precomputed_vfm", data_path=args.dataset_path).get_data()
    has_truth = bool(rec.get("has_ground_truth", True)) and "reaction_forces_true" in prep
    mat_p = rec.get("material_params", {}) or {}
    mat = get_material(rec.get("material_model_name", "isihara"), dev_params=mat_p.get("dev_params"),
                       vol_params=mat_p.get("vol_params")) if has_truth else None
    F_all = load_f3x3_from_dataset(prep, material_model=mat)                       # (n_steps, C, 3, 3) observed
    energy_scale = external_work_density(prep, prep["node_type"], max(train_steps))
    lin_prior_scale = float(rec.get("linear_elastic_prior_scale", 100.0 * energy_scale))
    disp_noise = float(rec.get("disp_noise", prep.get("disp_noise", 0.0)))
    post_path = os.path.join(save_path, "hsgp_posterior.npz")
    print(f"[HSGP] seed {args.seed} | steps {train_steps} | energy scale {energy_scale:.4g} | options {opt}", flush=True)

    if args.phase in ("fit", "all"):
        t0 = time.time()
        model, hist = fit_hsgp(
            F_all[jnp.array(train_steps)], np.asarray(prep["reaction_forces"])[train_steps],
            np.asarray(prep["load_noise_std_steps"])[train_steps], prep["cells"], prep["node_type"], prep["dNdX"], prep["dA"],
            energy_scale=energy_scale, lin_prior_scale=lin_prior_scale,
            num_basis_dev=opt["hsgp_num_basis_dev"], num_basis_vol=opt["hsgp_num_basis_vol"], box_factor=opt["hsgp_box_factor"],
            amplitude_prior_scale=opt["hsgp_amplitude_prior_scale"], lengthscale_prior_scale=opt["hsgp_lengthscale_prior_scale"],
            prior_mean=opt["hsgp_prior_mean"], envelope=bool(opt["hsgp_envelope"]), likelihood=opt["hsgp_likelihood"],
            sigma_u_known=disp_noise, reaction_weight=opt["hsgp_reaction_weight"], warp=bool(opt["hsgp_warp"]),
            continuation=bool(opt["hsgp_continuation"]), max_outer=opt["hsgp_max_outer"], step_labels=train_steps,
            sigma_cap=opt["hsgp_sigma_cap"], stability_search=bool(opt["hsgp_stability_search"]),
            hyper_restarts=bool(opt["hsgp_hyper_restarts"]), continuation_stages=rec.get("hsgp_continuation_stages"))
        fit_time = time.time() - t0
        model.save(post_path)
        feats = jax.vmap(model.feature_extractor.extract)(F_all[jnp.array(train_steps)].reshape(-1, 3, 3))
        I_obs = np.concatenate([np.asarray(feats[0]), np.asarray(feats[1])], axis=-1)
        np.save(os.path.join(save_path, "I_obs_all.npy"), I_obs)
        cfg_out = dict(rec, extraction_method="hsgp", dataset_path=args.dataset_path, seed=args.seed, batch_dir=save_path,
                       train_load_steps_indices=train_steps, val_load_steps_indices=val_steps, test_load_steps_indices=test_steps,
                       data_energy_scale=energy_scale, hsgp_continuation_stages=rec.get("hsgp_continuation_stages"), linear_elastic_prior_scale=lin_prior_scale, hsgp_options=opt,
                       hsgp=dict(model.info, hyper={k: np.asarray(v).tolist() for k, v in model.hyper.items()}))
        json.dump(cfg_out, open(os.path.join(save_path, "config.json"), "w"), indent=4, default=float)
        json.dump(hist, open(os.path.join(save_path, "hsgp_history.json"), "w"), indent=2)
        print(f"[seed {args.seed}] fit done in {fit_time:.0f}s, converged {model.info['converged']}", flush=True)
        if args.phase == "fit":
            return
    else:
        model = HSGPHyperelasticity.load(post_path)
        hist = json.load(open(os.path.join(save_path, "hsgp_history.json")))
        I_obs = np.load(os.path.join(save_path, "I_obs_all.npy"))
        fit_time = model.info.get("fit_time", float("nan"))

    t0 = time.time()
    plot_convergence(hist, save_path)
    reac = plot_reactions(model, F_all, prep, train_steps, save_path, has_truth)
    _, frac = ellipticity_penalty(model.psi_det, F_all[jnp.array(train_steps)], 8)
    metrics = dict(seed=args.seed, extraction_method="hsgp", extraction_time=fit_time, train_steps=train_steps,
                   val_steps=val_steps, log_evidence=model.info["log_evidence"], converged=model.info["converged"],
                   outer_iterations=model.info["outer_iterations"],
                   all_stages_converged=model.info.get("all_stages_converged"), stage_status=model.info.get("stage_status"),
                   sigma_u_x=model.noise["sigma_u_x"], sigma_u_y=model.noise["sigma_u_y"],
                   linear_elastic_mean=model.linear_elastic_summary(), small_strain_moduli=model.small_strain_moduli(),
                   fraction_unstable=float(frac), reaction_forces=reac,
                   hyper={k: np.asarray(v).tolist() for k, v in model.hyper.items()})
    if has_truth:   # synthetic benchmark: energy and stress against the true material
        from plots.training import (plot_training_r2, plot_training_r2_piola, plot_combined_validation,
                                    plot_energy_decomposition_validation)
        r2_res = plot_training_r2(model, mat, F_all, save_path, train_steps=train_steps, val_steps=val_steps, test_steps=test_steps)
        plot_training_r2_piola(model, mat, F_all, save_path, train_steps=train_steps, val_steps=val_steps,
                               test_steps=test_steps, n_pws_samples=64)
        plot_combined_validation(model, mat, save_path, 0, I_obs=I_obs)
        plot_energy_decomposition_validation(model, mat, save_path, I_obs=I_obs)
        metrics.update(r2=r2_res[0], rmse=r2_res[1], ec=r2_res[2],
                       r2_train=r2_res.train_metrics.get("r2"), ec_train=r2_res.train_metrics.get("ec"),
                       r2_val=r2_res.val_metrics.get("r2"), rmse_val=r2_res.val_metrics.get("rmse"),
                       ec_val=r2_res.val_metrics.get("ec"))
    json.dump(metrics, open(os.path.join(save_path, "extraction_metrics.json"), "w"), indent=4, default=float)
    print(f"[seed {args.seed}] plots and metrics done in {time.time() - t0:.0f}s -> {save_path} | "
          f"R2 val {metrics.get('r2_val', float('nan')):.4f} EC val {metrics.get('ec_val', float('nan')):.1f} | "
          f"moduli {metrics['small_strain_moduli']}", flush=True)


if __name__ == "__main__":
    main()

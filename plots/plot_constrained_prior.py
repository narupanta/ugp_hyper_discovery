#!/usr/bin/env python3
"""
Samples of the constrained HSGP prior theta = M(eta) gamma, gamma ~ N(0, I), along the six standard deformation modes:
energy psi and stress |P|_F against the mode amplitude, with samples of the same prior WITHOUT the reference
constraints (theta = Lambda^1/2 z) optionally overlaid for comparison. Basis boxes and hyperparameters eta are taken from a fitted
posterior, so the prior is the one that fit used. Prints max |psi(I)| and max |P(I)| over the samples.

Usage: python3 plots/plot_constrained_prior.py <experiment>/<seed>/extracted [--n_samples 20 --gamma_max 0.5]
       [--show_unconstrained]
Writes <extracted>/constrained_prior_modes.pdf.
"""
import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt

from core.hsgp_basis import se_spectral_density
from core.hsgp_inference import _Constraints, constrained_transform
from core.hsgp_model import HSGPHyperelasticity
from plots.theme import apply_style, save_figure
from plots.training import standard_mode_F

MODE_TITLES = ["uniaxial tension", "equibiaxial tension", "pure shear", "uniaxial compression",
               "equibiaxial compression", "simple shear"]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("extracted_dir")
    p.add_argument("--n_samples", type=int, default=20)
    p.add_argument("--gamma_max", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--show_unconstrained", action="store_true",
                   help="overlay samples of the prior without the reference constraints (much larger scale)")
    args = p.parse_args()

    model = HSGPHyperelasticity.load(os.path.join(args.extracted_dir, "hsgp_posterior.npz"))
    ips = getattr(model, "incompressible_plane_stress", False)
    hyper = {k: jnp.asarray(v) for k, v in model.hyper.items()}
    if ips:   # volumetric GP off in the fit: any positive amplitude, its block is zeroed by the transform
        hyper["vol_sig"] = jnp.asarray(1.0)
    M = constrained_transform(model, _Constraints(model), hyper, ips)                       # (P, P)
    # same prior without the reference projection: theta = Lambda^1/2 z (linear-elastic block zero)
    sq_dev = jnp.sqrt(se_spectral_density(model.dev_box, hyper["dev_sig"], hyper["dev_ls"]))
    sq_vol = jnp.sqrt(se_spectral_density(model.vol_box, hyper["vol_sig"], hyper["vol_ls"])) * (0.0 if ips else 1.0)
    D = jnp.concatenate([jnp.zeros(model.n_lin), sq_dev, sq_vol])

    key_c, key_u = jax.random.split(jax.random.PRNGKey(args.seed))
    P_ = model.num_params
    th_con = (M @ jax.random.normal(key_c, (P_, args.n_samples))).T                        # (S, P)
    th_unc = (D[:, None] * jax.random.normal(key_u, (P_, args.n_samples))).T

    gamma = jnp.linspace(0.0, args.gamma_max, 101)
    F = standard_mode_F(model, gamma)                                                       # (6, G, 3, 3)
    Ff = F.reshape(-1, 3, 3)
    phi = jax.vmap(model.basis)(Ff)                                                         # (6G, P)
    dphi = jax.vmap(jax.jacfwd(model.basis))(Ff)                                            # (6G, P, 3, 3)

    def evaluate(theta):
        psi = (phi @ theta.T).T.reshape(theta.shape[0], 6, -1)
        P = jnp.einsum("npij,sp->snij", dphi, theta)
        Pn = jnp.sqrt(jnp.sum(P ** 2, axis=(-2, -1))).reshape(theta.shape[0], 6, -1)
        return np.asarray(psi), np.asarray(Pn)

    psi_c, P_c = evaluate(th_con)
    psi_u, P_u = evaluate(th_unc)
    g = np.asarray(gamma)

    print(f"prior from {args.extracted_dir}: dev_sig {float(hyper['dev_sig']):.3g}, vol_sig "
          f"{0.0 if ips else float(hyper['vol_sig']):.3g}, {args.n_samples} samples")
    print(f"  at F = I (gamma = 0), max over samples and modes:")
    print(f"    constrained:   |psi| {np.abs(psi_c[:, :, 0]).max():.3e}   |P|_F {P_c[:, :, 0].max():.3e}")
    print(f"    unconstrained: |psi| {np.abs(psi_u[:, :, 0]).max():.3e}   |P|_F {P_u[:, :, 0].max():.3e}")

    apply_style()
    fig, axes = plt.subplots(2, 6, figsize=(24, 8), sharex=True)
    for m in range(6):
        for row, (yc, yu, lab) in enumerate([(psi_c, psi_u, r"$\Psi$"), (P_c, P_u, r"$\|\mathbf{P}\|_F$")]):
            ax = axes[row, m]
            for s in range(args.n_samples if args.show_unconstrained else 0):
                ax.plot(g, yu[s, m], color="0.75", lw=0.8, label="unconstrained prior" if s == 0 else None)
            for s in range(args.n_samples):
                ax.plot(g, yc[s, m], color="tab:blue", lw=1.0, alpha=0.8, label="constrained prior" if s == 0 else None)
            ax.axhline(0.0, color="k", lw=0.6)
            ax.axvline(0.0, color="k", lw=0.6, ls=":")
            if row == 0:
                ax.set_title(f"{MODE_TITLES[m]}\nmax $|\\Psi(I)|$ = {np.abs(yc[:, m, 0]).max():.1e}")
            else:
                ax.set_title(f"max $\\|P(I)\\|$ = {yc[:, m, 0].max():.1e}")
                ax.set_xlabel(r"mode amplitude $\gamma$")
            if m == 0:
                ax.set_ylabel(lab)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("HSGP prior samples along the standard deformation modes (F = I at $\\gamma = 0$)")
    out = os.path.join(args.extracted_dir, "constrained_prior_modes.pdf")
    save_figure(fig, out)
    print(f"saved {out}")


if __name__ == "__main__":
    main()

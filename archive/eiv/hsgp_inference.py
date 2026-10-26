"""
Deterministic posterior of the HSGP strain-energy model under the errors-in-variables (EIV) likelihood.

Model (displacement control, observed state u_obs = z + eps on the free DOFs, load-cell reactions R_obs):
  free DOFs:  eps_hat_t(theta) = K_ff(theta)^{-1} f_free(u_obs; theta)                  ~ N(0, sigma_u,d^2 I)
  reactions:  R_t(theta) = R(u_obs; theta) - K_rf(theta) eps_hat_t(theta)               ~ N(R_obs,t, sigma_R,t^2)
f_int(u_obs; theta) = B_t theta and K(theta) are exactly linear in theta (B_t: FE-assembled basis stresses, computed
once). eps_hat is homogeneous of degree 0 in theta (equilibrium carries no stress magnitude) and R_t of degree 1, so the
Gauss-Newton linearisation at the current mean theta_bar is
  eps_hat_t(theta) ~= eps_bar_t + J_t theta,   J_t = K_ff^{-1} (B_t,free - D_t,free),   J_t theta_bar = 0,
  R_t(theta)       ~= J_R,t theta,             J_R,t = B_t,R - K_rf J_t - D_t,R        (exact along theta_bar),
with D_t = dK/dtheta . eps_bar_t (one JVP of B_t along the correction). Freezing K instead would break the scale
invariance and reward theta -> 0. With J fixed the model is linear-Gaussian, so the posterior over theta and the
evidence are closed form; the outer loop re-linearises (Gauss-Newton / Laplace approximation at the MAP) and
re-optimises the hyperparameters by MAP-II on the evidence:
  hyper = (sigma_dev, l_dev, sigma_vol, l_vol, sigma_u_x, sigma_u_y),  log-normal priors on sigma and l,
  l bounded to the range the basis resolves (core/hsgp_basis.recommended_num_basis), flat prior on log sigma_u.

GP prior conditioned exactly on the reference constraints A theta = 0 (zero energy and volumetric stress at F = I):
  w = Lambda^{1/2} Pi u,  u ~ N(0, I),  Pi = I - At^T (At At^T)^{-1} At,  At = A Lambda^{1/2},
which is the Gaussian N(0, Lambda) conditioned on A w = 0, smooth in the hyperparameters and well conditioned.
"""
import time
from typing import Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from core.features import IsotropicFeatureExtractor
from core.hsgp_basis import Box, make_box, eigenfunctions, se_spectral_density
from core.hsgp_model import HSGPHyperelasticity, DEV_REFERENCE, VOL_REFERENCE
from core.loss_function import build_eiv_indices, assemble_internal_force_and_tangent

jax.config.update("jax_enable_x64", True)

SIGMA_U_FLOOR = 1e-9         # keeps the evidence finite for (nearly) noise-free displacements
LS_MAX_FRACTION = 1.0 / 2.67   # l_max = box_factor * S / 2.67 (constrained-kernel error < 1e-3 at c = 8, see tests)


def lengthscale_bounds(box: Box, box_factor: float):
    """Per-dimension [l_min, l_max] the basis resolves: l_min from m >= 1.75 c S / l, l_max ~ c S / 2.67."""
    S = box.half_width / box_factor
    m = box.indices.max(axis=0)
    return 1.75 * box_factor * S / m, LS_MAX_FRACTION * box_factor * S


def basis_force_matrix(model: HSGPHyperelasticity, f3x3_cells, cells, n_nodes: int, dNdX, dA):
    """B (2n, P): internal nodal forces of each basis function at one load step (out-of-plane F held fixed)."""
    def dphi(f3):
        return jax.jacfwd(lambda f2: model.basis(f3.at[:2, :2].set(f2)))(f3[:2, :2])     # (P, 2, 2)
    D = jax.vmap(dphi)(f3x3_cells)                                                         # (C, P, 2, 2)
    f_cell = jnp.einsum("cpij,caj->caip", D, dNdX) * dA[:, None, None, None]               # (C, 3, 2, P)
    return jnp.zeros((n_nodes, 2, model.num_params)).at[cells].add(f_cell).reshape(2 * n_nodes, -1)


def eiv_step_linearisation(model: HSGPHyperelasticity, theta, f_t, cells, n_nodes: int, dNdX, dA, eiv, damping: float = 0.0):
    """
    EIV state correction of one load step and its exact Gauss-Newton Jacobians at theta (free and reaction DOFs).
    damping = 0: eps = K^-1 f (Newton step), J = K^-1 (B - D), D_k = K_k eps.
    damping = delta > 0: eps = (K^2 + mu^2 I)^-1 K f, mu^2 = delta^2 ||K||_F^2 / m (degree 2 in theta, so eps stays
    degree 0 and J theta = 0 exactly). Identical to the Newton step where |lambda(K)| >> mu; components of f along
    eigenvalues near zero or negative (indefinite tangent of a poor mean far from the data) are damped instead of
    exploding or flipping sign. Its Jacobian, with A = K^2 + mu^2 I and r = f - K eps = mu^2 A^-1 f:
        d eps / d theta_k = A^-1 [K B_k - K D_k + K_k r - eps d(mu^2)/d theta_k],
    K_k r and K_k eps are JVPs of the basis-force matrix, d(mu^2)/d theta from one reverse pass of ||K(theta)||_F^2.
    Reactions R = R(u) - K_RF eps give J_R = B_R - K_RF J - D_R in both cases.
    Returns eps (m,), J (m, P), J_R (2, P).
    """
    free, rx, ry = eiv["dof_free"], eiv["dof_rx"], eiv["dof_ry"]
    force_fn = lambda f: basis_force_matrix(model, f, cells, n_nodes, dNdX, dA)

    def tangent(th):
        f_int, K = assemble_internal_force_and_tangent(lambda f: model.basis(f) @ th, f_t, cells, n_nodes, dNdX, dA, eiv)
        return f_int, K

    def grad_field(v):                                     # free-DOF vector -> deformation-gradient perturbation
        v_nodes = jnp.zeros(2 * n_nodes).at[free].set(v).reshape(n_nodes, 2)
        return jnp.zeros_like(f_t).at[:, :2, :2].set(jnp.einsum("cai,caj->cij", v_nodes[cells], dNdX))

    f_int, K = tangent(theta)
    Kff = K[free][:, free]
    Krf = jnp.stack([K[rx][:, free].sum(0), K[ry][:, free].sum(0)])
    f = f_int[free]

    def newton():
        eps = jnp.linalg.solve(Kff, f)
        B, D = jax.jvp(force_fn, (f_t,), (grad_field(eps),))                            # D = dK/dtheta . eps
        return eps, B, D, jnp.linalg.solve(Kff, B[free] - D[free])

    def damped():
        m = Kff.shape[0]
        lam, V = jnp.linalg.eigh(0.5 * (Kff + Kff.T))
        mu2 = damping ** 2 * jnp.sum(Kff ** 2) / m
        a_inv = 1.0 / (lam ** 2 + mu2)
        eps = V @ (lam * a_inv * (V.T @ f))
        r = f - Kff @ eps
        B, D = jax.jvp(force_fn, (f_t,), (grad_field(eps),))
        _, E = jax.jvp(force_fn, (f_t,), (grad_field(r),))                              # E = dK/dtheta . r
        dmu2 = damping ** 2 / m * jax.grad(lambda th: jnp.sum(tangent(th)[1][free][:, free] ** 2))(theta)
        rhs = Kff @ (B[free] - D[free]) + E[free] - eps[:, None] * dmu2[None, :]
        return eps, B, D, V @ (a_inv[:, None] * (V.T @ rhs))

    if isinstance(damping, (int, float)):
        eps, B, D, J = newton() if damping <= 0.0 else damped()
    else:                                                  # traced damping (annealed inside one compiled function)
        eps, B, D, J = jax.lax.cond(damping > 0.0, damped, newton)
    D_R = jnp.stack([D[rx].sum(0), D[ry].sum(0)])
    B_R = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
    return eps, J, B_R - Krf @ J - D_R


class _Constraints:
    """Reference-constraint rows on the GP weights, through the model's (possibly warped) bases at the raw reference.
    Built before the envelope is set; the envelope is 1 with zero slope at the reference, so the rows do not change."""
    def __init__(self, model: HSGPHyperelasticity):
        self.A_dev = model.dev_basis(jnp.asarray(DEV_REFERENCE))[None, :]
        vol_fn = lambda j: model.vol_basis(j)
        self.A_vol = jnp.stack([vol_fn(jnp.asarray(VOL_REFERENCE)), jax.jacfwd(vol_fn)(jnp.asarray(VOL_REFERENCE))[:, 0]])


def _projected_sqrt(A, lam):
    sq = jnp.sqrt(jnp.maximum(lam, 0.0))
    At = A * sq[None, :]
    Pi = jnp.eye(lam.shape[0]) - At.T @ jnp.linalg.solve(At @ At.T, At)
    return sq[:, None] * Pi


def constrained_transform(model, cons: _Constraints, hyper, lin_scale: float, lin_only: bool = False):
    """M (P, P): theta = M gamma with gamma ~ N(0, I) gives the constrained prior."""
    Md = _projected_sqrt(cons.A_dev, se_spectral_density(model.dev_box, hyper["dev_sig"], hyper["dev_ls"]))
    Mv = _projected_sqrt(cons.A_vol, se_spectral_density(model.vol_box, hyper["vol_sig"], hyper["vol_ls"]))
    if lin_only:
        Md, Mv = 0.0 * Md, 0.0 * Mv
    if getattr(model, "incompressible_plane_stress", False):   # J = 1: no volumetric GP
        Mv = 0.0 * Mv
    return jax.scipy.linalg.block_diag(lin_scale * jnp.eye(model.n_lin), Md, Mv)


def fit_hsgp(f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA, *, energy_scale: float,
             lin_prior_scale: float, num_basis_dev: int = 24, num_basis_vol: int = 32, box_factor: float = 8.0,
             amplitude_factor: float = 10.0, amplitude_prior_scale: float = 1.0,
             lengthscale_factor: float = 1.5, lengthscale_prior_scale: float = 1.0,
             prior_mean: str = "linear_elastic", envelope: bool = False,
             likelihood: str = "eiv", sigma_u_known: Optional[float] = None, reaction_weight: float = 1.0, warp: bool = False, max_outer: int = 30, lin_max: int = 15, lin_tol: float = 1e-4,
             tol: float = 1e-5, sigma_cap: float = 10.0, max_hyper_step: float = 2.0, continuation: bool = False,
             stability_search: bool = True, hyper_restarts: bool = True, continuation_stages=None,
             incompressible_plane_stress: bool = False,
             step_labels=None, eiv_damping: float = 0.0, eiv_damping_decay: float = 0.1,
             adaptive: bool = False, hyper_iters_per_stage: int = 0, ls_floor_fraction: float = 0.0,
             tol_intermediate: Optional[float] = None, hyper_delay: int = 0, cap_accept_tol: float = 0.0,
             sigma_f_known: Optional[float] = None, nigp_warmup: str = "residual", eiv_warmup: str = "none",
             sigma_f_rel: Optional[float] = None, sigma_f_cv: bool = False, cv_folds: int = 2, cv_grid=None,
             sigma_f_evidence: bool = False, reaction_weight_rule: str = "fixed", sigma_f_from_reaction: bool = False,
             cv_tol: float = 0.0, sigma_f_joint: bool = False, joint_refit_unweighted: bool = False,
             hyper_steps: int = 60, verbose: bool = True, log_fn=print):
    """
    f3x3_steps: (T, C, 3, 3) observed deformation gradients of the training steps; R_obs, sigma_R: (T, 2).
    prior_mean: 'linear_elastic' (psi = mu h_mu + kappa h_kappa + GP, coefficients ~ N(0, lin_prior_scale^2)), 'none'
    (zero-mean GP prior: the linear-elastic fit only initialises the first tangent and the coefficients are fixed at 0), or
    'linear_elastic_fixed' (fixed mean function: the coefficients are frozen at the linear-elastic stage values mu_hat,
    kappa_hat and the GP models only the departure from that material). With free coefficients the linear-elastic terms and
    the GP are not identifiable: e.g. kappa -> 20x its value inside the data, cancelled by a short-lengthscale GP, leaves
    the excess stiffness alone outside the data. The fixed mean reverts to the fitted linear-elastic material there.
    envelope: multiply the GP bases by s_dev = 1 + mu h_mu / e_bar and s_vol = 1 + kappa h_kappa / e_bar, with (mu, kappa)
    from the linear-elastic stage and e_bar = energy_scale (see hsgp_model); for energies spanning orders of magnitude.
    likelihood: 'eiv' (module docstring) or 'residual': no state correction, free-node forces B_free theta ~ N(0, sigma_f^2)
    and reactions B_R theta ~ N(R_obs, sigma_R^2), exactly linear in theta. sigma_f is NOT learned (the evidence is
    unbounded as sigma_f -> 0: the residual space is larger than the basis); it is the force-equivalent noise of the known
    displacement noise, sigma_f,i = sigma_u_known * sqrt(sum_j K_ij^2) over free j, with K the tangent of a linear-elastic
    material whose (mu, kappa) are fitted to the measured reactions alone (2-parameter least squares), fixed for the whole
    fit. Letting sigma_f follow the posterior mean collapses theta -> 0: the free term then always prefers a smaller energy.
    likelihood: 'residual_learned' reproduces the SVGP residual setup: per-direction sigma_f learned by the variational
    (mean-field) update sigma_f,d^2 = (||B_f,d m||^2 + tr(B_f,d S B_f,d^T)) / N_d, i.e. from the EXPECTED residual under the
    posterior (the ELBO optimum), and the reaction log-likelihood multiplied by reaction_weight (the old reaction_loss_weight).
    Its evidence has no maximum (theta = 0 makes the free residual exactly zero), so sigma_f is not part of MAP-II.
    likelihood: 'residual_exact': the 'residual' likelihood (same fixed force noise, needs sigma_u_known) with the exact
    Bayesian update. The model is linear in theta and the statistics do not depend on it, so for given hyperparameters
    the posterior is exactly Gaussian and closed form: all training steps in one stage (no continuation), and each stage
    (linear-elastic, then GP) is a single solve: hyperparameters optimised to convergence on the exact evidence, then
    the closed-form posterior (no relaxation, no stability line search, no cap on the hyperparameter step). The
    stability of the result is reported, not enforced. With prior_mean 'none' the linear-elastic stage is skipped (its
    (mu, kappa) would be discarded by the zero-mean GP anyway).
    Without sigma_u_known and sigma_f_known, residual_exact infers the per-direction force-residual noise sigma_f by EM:
    sigma_f,d^2 = (||B_f,d m||^2 + tr(B_f,d S B_f,d^T)) / N_d from the exact posterior (m, S), alternated with the exact
    posterior until both are stationary (the evidence itself is unbounded as sigma_f -> 0, see 'residual_learned').
    Caveat: with the amplitudes on the evidence this can collapse (synthetic Isihara: sigma_f -> floor, deviatoric
    amplitude -> 0), since a linear model of exact-zero targets can always shrink its residual; the EIV likelihood
    has non-zero targets (-eps_bar) and a scale-invariant eps, which is what makes its noise identifiable.
    eiv_warmup='residual_exact' (likelihood 'eiv'): the first GP iteration is the exact residual Gaussian (closed form,
    force noise from sigma_f_known, or from sigma_u_known and a linear-elastic tangent; reactions as in 'residual'),
    replacing the linear-elastic stage when prior_mean is 'none'; its mean and hyperparameters start the EIV iterations,
    which infer sigma_u as usual. Best at low displacement noise, where the residual solution is nearly unbiased.
    likelihood: 'nigp': warm-up with the exact residual Gaussian (force noise from sigma_u_known and a linear-elastic
    tangent), then NIGP-style input-noise iterations: the displacement noise reaches the free forces as
    K(theta_bar) Sigma_u K(theta_bar)^T with the tangent frozen at the current mean, i.e. design K^-1 B_f with target 0,
    and the reactions conditioned on the free forces, R ~ N((B_R - K_Rf K^-1 B_f) theta, Sigma_R); repeated (with the
    usual relaxation and stability line search) until the mean is stationary. nigp_warmup='linear' skips the residual
    GP solve and starts the NIGP iterations from the linear-elastic stage (a stable tangent; the residual solution can
    be unstable, and K^-1 of an indefinite tangent is meaningless). CAVEAT (tested on synthetic Isihara): both variants
    collapse to theta -> 0. With K frozen, theta = c theta_bar scales the free term c^2 ||eps_bar||^2 / sigma_u^2 ~ c^2 N_f,
    so each update shrinks the scale by I_R / (N_f + I_R) < 1 (I_R: reaction information) and there is no fixed point at
    the true scale. EIV keeps the derivative of K (J = K^-1 (B - D), J theta_bar = 0), which makes eps scale-invariant;
    kept as a diagnostic of that difference. Needs sigma_u_known: with zero targets
    the input noise is not identifiable (the evidence is unbounded as sigma_u -> 0), unlike the EIV likelihood.
    sigma_f_rel / sigma_f_cv (residual likelihoods): per-step force-residual noise sigma_f,t = s |R_t| (the residual of
    the true material on noisy data scales with the load: s is nearly constant over the steps). sigma_f_rel gives s;
    sigma_f_cv chooses s by cross-validation over the TRAINING steps ('residual_exact', zero-mean prior): interleaved
    sigma_f_from_reaction: sigma_f,t,d = (N_BC / N_free) sigma_R,t,d per load step and direction (sigma_R from the
    load-cell model): the force-residual noise tied to the reaction noise.
    reaction_weight_rule='euclid' sets reaction_weight = N_free / N_BC (free and boundary DOFs), the EUCLID balancing of
    the reaction term against the free-residual term (residual likelihoods; the EIV likelihood is unweighted).
    sigma_f_evidence chooses s by the log evidence of the reaction-weighted likelihood on all training steps: with
    reaction_weight >= 10 it has an interior maximum near the true residual level (Isihara 1e-5: s ~ 2-4e-4 for weights
    10..1000), while at weight 1 it grows without bound as s -> 0. Cross-validation:
    cv_tol > 0: instead of the best score, the smallest s within cv_tol nats of it (the CV curve has a flat plateau at
    large s, on which the maximum is noisy; the smallest s on it uses the free residuals the most).
    folds, fit on the others, held-out log predictive density of the REACTIONS (the free residuals have zero
    targets and would select a vanishing energy, as the evidence does), log grid
    (cv_grid, default 1e-5..1, 11 points) refined around the optimum; the validation steps stay untouched.
    sigma_f_joint ('residual_exact', zero-mean prior): sigma_f,t = s |R_t| with log s a hyperparameter, optimised with the
    kernel hyperparameters on the (reaction-weighted) evidence by the same L-BFGS (one solve; as sigma_f_evidence, but by
    gradient instead of a grid). Needs reaction_weight >~ 10 for an interior maximum (weight 1: s -> 0).
    joint_refit_unweighted (with sigma_f_joint): the weighted evidence only selects s; s is then fixed and the kernel
    hyperparameters and the posterior are recomputed with reaction weight 1, so that the reactions count at their
    actual noise level (a weight w makes the posterior treat each reading as w readings: bands ~ 1/sqrt(w) too narrow).
    sigma_f_known (residual likelihoods): iid noise std of the free-node force residual, set directly (force units);
    replaces the force-equivalent noise from sigma_u_known, so no reaction-only linear-elastic material is fitted.
    warp: GP inputs log(I_bar - 2), log J (see hsgp_model); boxes, spans and lengthscales live in the warped space.
    The linear-elastic stage runs until its relative force change < lin_tol (at most lin_max iterations), then the GP
    stage runs at most max_outer iterations. Robustness guards: amplitudes capped at sigma_cap x the prior centre
    (sigma_cap <= 0: no cap), and
    each hyperparameter moves at most max_hyper_step (log units) per outer iteration.
    continuation: load-step continuation for large deformations. The steps (in the given order, smallest first) are fitted
    as cumulative sets [0, 1], [0, 1, 2], ..., [0..T-1], each stage warm-started from the previous posterior, so every
    Gauss-Newton problem starts close to its solution; the linear-elastic stage runs on the first set only.
    step_labels: the dataset load-step numbers of f3x3_steps, used only in logs and in the history/info records.
    stability_search: stability margin = min over active load steps of lambda_min(K_ff) / mean diag(K_ff) of the
    posterior-mean energy. A Gauss-Newton update is accepted only if it keeps a stable mean stable (margin > 0), or, for a
    mean that is not yet stable (e.g. at a newly added continuation step), does not make it less stable; otherwise the
    step is halved (down to 1/64), and if no admissible step exists the stage ends at the current mean. Without it an unstable mean can make the EIV correction K^-1 f vanish and
    drive the evidence to a spurious optimum (sigma_u -> 0, most states unstable).
    continuation_stages: explicit ladder as a list of cumulative sets of load-step labels (step_labels), e.g.
    [[1, 2, 3], [1, 2, 3, 5], [1, 2, 3, 5, 7], ...]; each set must contain the previous one and the last must be all
    training steps. Default (continuation=True): [labels[:2], labels[:3], ..., labels].
    hyper_restarts: in the GP stage, optimise the hyperparameters from the current point and from three restarts (prior
    centre, lengthscales x 0.5 and x 2) and keep the highest evidence: the evidence is multimodal in the lengthscales,
    and a single start lets tiny numerical differences (CPU vs GPU) pick different optima.
    eiv_damping: relative damping delta of the EIV correction (eiv_step_linearisation), eps = (K^2 + mu^2)^-1 K f with
    mu^2 = delta^2 ||K||_F^2 / m; 0 = plain Newton step K^-1 f. Keeps the correction bounded when the linearisation
    point is far from the data (indefinite tangent), e.g. a full-range first stage. A fixed delta would bias the converged
    posterior (the noise correction lives in the softest tangent modes, lambda_min / rms(lambda) ~ 2e-4 at large
    stretches), so it is annealed: delta_0 = eiv_damping at the first GP iteration of each continuation stage, multiplied
    by eiv_damping_decay per iteration and set to 0 below 1e-8; a stage can only converge undamped.
    adaptive: adaptive load-step continuation with rollback (like adaptive load stepping in nonlinear FE). Training
    steps are indexed in the given order (smallest load first). The first stage tries the full range coarsely
    {first, middle, last}; every later stage either extends the covered range (adding the midpoint and the end of the
    new interval) or, once the full range is covered, refines it (adding the midpoints of all gaps). A stage that does
    not converge (stability stop, iteration cap or non-finite) is rolled back to the last converged state (mean, factor,
    hyperparameters) and retried with half the extension, or half the refinement points. If no smaller stage exists the
    fit stops and returns the last converged posterior; info['valid_load_steps'] lists the steps it is conditioned on
    (info['truncated'] is True if that is not all training steps). Overrides continuation / continuation_stages.
    hyper_iters_per_stage: > 0 optimises the hyperparameters only in the first k GP iterations of each continuation
    stage, then keeps them fixed and iterates the Gauss-Newton fixed point for theta alone (no flip-flop between
    near-tied evidence modes). 0 = optimise at every iteration.
    ls_floor_fraction: > 0 adds a one-sided log-normal penalty (width 0.1 log units) on lengthscales below
    ls_floor_fraction x the span of the active stage's data (per GP input dimension): a stage that has only seen a narrow
    range cannot select features much finer than that range, which it would extrapolate explosively. 0 = off.
    hyper_delay: > 0 keeps the hyperparameters fixed in the first k GP iterations of every stage after the first GP
    stage, so the mean absorbs the newly added load steps before the evidence is re-optimised. Right after steps are
    added the linearised misfit at them is large, and optimising the evidence then makes the GP much larger and rougher
    (in the 40-step runs, failed stages started with a 3.5-7x amplitude jump; converged stages with none). With
    hyper_iters_per_stage > 0 the optimisation window is iterations [hyper_delay, hyper_delay + hyper_iters_per_stage).
    cap_accept_tol: > 0 (adaptive only) counts a stage that reaches the iteration cap with its last force change below
    cap_accept_tol as converged instead of rolling it back (Gauss-Newton jitter just above tol at the noise floor).
    tol_intermediate: convergence tolerance of the intermediate continuation stages (they only pass on the
    linearisation point and the hyperparameters); default tol. The stage on all training steps always uses tol.
    Returns (HSGPHyperelasticity with the posterior, history list of dicts).
    """
    # residual_exact without a given noise: sigma_f inferred per direction by EM (expected residual), exact posterior
    exact_learn = likelihood == "residual_exact" and not (sigma_u_known or sigma_f_known or sigma_f_rel or sigma_f_cv
                                                          or sigma_f_evidence or sigma_f_from_reaction or sigma_f_joint)
    nigp = likelihood == "nigp"
    # per-step force-residual noise sigma_f,t = s |R_t| (s given, chosen by cross-validation / evidence grid, or a
    # hyperparameter of the evidence optimisation)
    force_rel = bool(sigma_f_rel) or bool(sigma_f_cv) or bool(sigma_f_evidence) or bool(sigma_f_joint)
    if sigma_f_joint and (sigma_f_cv or sigma_f_evidence or sigma_f_rel):
        raise ValueError("sigma_f_joint excludes sigma_f_rel / sigma_f_cv / sigma_f_evidence (one rule for s).")
    fnoise = dict(s=float(sigma_f_rel) if sigma_f_rel else 1.0, cv=None)
    if nigp_warmup not in ("residual", "linear"):
        raise ValueError(f"nigp_warmup must be 'residual' or 'linear', got {nigp_warmup!r}")
    # 'linear': the warm-up is the linear-elastic stage alone (stable tangent); the GP stages are NIGP from the start
    if eiv_warmup not in ("none", "residual_exact"):
        raise ValueError(f"eiv_warmup must be 'none' or 'residual_exact', got {eiv_warmup!r}")
    # warm-up by the exact residual Gaussian (first GP iteration), then the likelihood's own iterations
    eiv_warm = likelihood == "eiv" and eiv_warmup == "residual_exact"
    warmup_residual = eiv_warm or (nigp and nigp_warmup == "residual")
    nigp_state = dict(warm_done=not warmup_residual)
    t0 = time.time()
    f3x3_steps = jnp.asarray(f3x3_steps, dtype=jnp.float64)
    cells_j, dNdX, dA = jnp.asarray(cells), jnp.asarray(dNdX), jnp.asarray(dA)
    node_type, cells = np.asarray(node_type), np.asarray(cells)
    n_nodes = node_type.shape[0]
    eiv = build_eiv_indices(node_type, cells)
    free, rx, ry = eiv["dof_free"], eiv["dof_rx"], eiv["dof_ry"]
    bc_free_ratio = float(2 * n_nodes - len(free)) / float(len(free))     # N_BC / N_free (DOFs)
    if reaction_weight_rule == "euclid":   # EUCLID balancing of the reaction term: N_free / N_BC (DOFs)
        reaction_weight = float(len(free)) / float(2 * n_nodes - len(free))
        log_fn(f"[HSGP] reaction weight (EUCLID, N_free / N_BC = {len(free)} / {2 * n_nodes - len(free)}): {reaction_weight:.3g}")
    elif reaction_weight_rule != "fixed":
        raise ValueError(f"reaction_weight_rule must be 'fixed' or 'euclid', got {reaction_weight_rule!r}")
    dirs = free % 2
    R_obs, sigma_R = jnp.asarray(R_obs), jnp.maximum(jnp.asarray(sigma_R), 1e-3)
    T = f3x3_steps.shape[0]

    # basis boxes from the training features (+ reference)
    from core.features import make_feature_extractor
    feats = jax.vmap(make_feature_extractor(incompressible_plane_stress).extract)(f3x3_steps.reshape(-1, 3, 3))
    dev_x, vol_x = np.asarray(feats[0]), np.asarray(feats[1])
    if warp:   # GP input space (same map as HSGPHyperelasticity.gp_inputs)
        dev_x, vol_x = np.log(np.maximum(dev_x - 2.0, 1e-12)), np.log(np.maximum(vol_x, 1e-12))
        ref_d, ref_v = np.zeros(2), np.zeros(1)
    else:
        ref_d, ref_v = DEV_REFERENCE, VOL_REFERENCE
    dev_x_steps, vol_x_steps = dev_x.reshape(T, -1, dev_x.shape[-1]), vol_x.reshape(T, -1, vol_x.shape[-1])
    dev_box = make_box(dev_x, ref_d, box_factor, num_basis_dev)
    if incompressible_plane_stress:
        # J = 1 everywhere: the volumetric features have no spread; a nominal box keeps the (inactive) volumetric
        # basis well-scaled. Its contribution is the constant phi_vol(1), which the reference conditions remove.
        vol_box = make_box(np.array([[0.5], [1.5]]) if not warp else np.log(np.array([[0.5], [1.5]])), ref_v,
                           1.0, num_basis_vol)
    else:
        vol_box = make_box(vol_x, ref_v, box_factor, num_basis_vol)
    P = 2 + dev_box.num_basis + vol_box.num_basis
    theta0 = jnp.zeros(P).at[0].set(1.0).at[1].set(1.0)
    model = HSGPHyperelasticity(dev_box, vol_box, theta0, jnp.zeros((P, 1)), warp=warp,
                                incompressible_plane_stress=incompressible_plane_stress)
    cons = _Constraints(model)

    # hyperparameter bounds, priors and initial values
    lsd_lo, lsd_hi = lengthscale_bounds(dev_box, box_factor)
    lsv_lo, lsv_hi = lengthscale_bounds(vol_box, box_factor)
    span_d = np.maximum(dev_x.max(0) - dev_x.min(0), 1e-6)
    span_v = np.maximum(vol_x.max(0) - vol_x.min(0), 1e-6)
    # prior centre: lengthscale_factor x data span, kept half a log-unit inside the resolvable range
    centre_d = np.clip(lengthscale_factor * span_d, lsd_lo * np.exp(0.5), lsd_hi * np.exp(-0.5))
    centre_v = np.clip(lengthscale_factor * span_v, lsv_lo * np.exp(0.5), lsv_hi * np.exp(-0.5))
    sig_centre = float(np.log(amplitude_factor * energy_scale))
    bounds = dict(dev=(jnp.log(lsd_lo), jnp.log(lsd_hi)), vol=(jnp.log(lsv_lo), jnp.log(lsv_hi)))

    def to_raw(log_l, lo, hi):
        # few basis functions per input (m < ~7) leave too narrow a range for the half-log-unit margin of the
        # initial value: keep it strictly inside the bounds
        s = jnp.clip((log_l - lo) / (hi - lo), 0.05, 0.95)
        return jnp.log(s / (1 - s))

    def unpack(eta):
        lo_d, hi_d = bounds["dev"]
        lo_v, hi_v = bounds["vol"]
        cap = sig_centre + np.log(sigma_cap) if sigma_cap and sigma_cap > 0 else np.inf   # sigma_cap <= 0: no cap
        h = dict(dev_sig=jnp.exp(jnp.minimum(eta["log_sig_dev"], cap)), vol_sig=jnp.exp(jnp.minimum(eta["log_sig_vol"], cap)),
                 dev_ls=jnp.exp(lo_d + (hi_d - lo_d) * jax.nn.sigmoid(eta["raw_ls_dev"])),
                 vol_ls=jnp.exp(lo_v + (hi_v - lo_v) * jax.nn.sigmoid(eta["raw_ls_vol"])),
                 su_x=SIGMA_U_FLOOR + jnp.exp(eta["log_su"][0]), su_y=SIGMA_U_FLOOR + jnp.exp(eta["log_su"][1]))
        if incompressible_plane_stress:
            # J = 1: the volumetric basis has no force contribution, so the evidence is flat in its hyperparameters
            # (they would only drift through the restarts). Fixed values here; constrained_transform switches the
            # volumetric GP off (theta_vol = 0 exactly), and vol_sig is reported as 0.
            h["vol_sig"], h["vol_ls"] = jnp.exp(jnp.asarray(sig_centre)), jnp.asarray(centre_v)
        if sigma_f_joint:
            h["s"] = jnp.exp(eta["log_s"])
        return h

    eta = dict(log_sig_dev=jnp.asarray(sig_centre), log_sig_vol=jnp.asarray(sig_centre),
               raw_ls_dev=to_raw(jnp.log(centre_d), *bounds["dev"]), raw_ls_vol=to_raw(jnp.log(centre_v), *bounds["vol"]),
               log_su=jnp.log(jnp.array([1e-3, 1e-3])))
    if sigma_f_joint:
        eta["log_s"] = jnp.log(jnp.asarray(1e-3))

    eta_init = dict(eta)

    def with_lengthscales(eta_, factor):
        """eta with all lengthscales multiplied by factor (kept inside their bounds)."""
        out = dict(eta_)
        for key, comp in (("raw_ls_dev", "dev"), ("raw_ls_vol", "vol")):
            lo, hi = bounds[comp]
            log_l = lo + (hi - lo) * jax.nn.sigmoid(eta_[key]) + np.log(factor)
            out[key] = to_raw(jnp.clip(log_l, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo)), lo, hi)
        return out

    def log_hyperprior(h, stats=None):
        lp = 0.0
        if ls_floor_fraction > 0 and stats is not None:   # one-sided penalty below the stage's data-range floor
            lp += -0.5 * (jnp.sum((jnp.minimum(jnp.log(h["dev_ls"]) - jnp.log(stats["ls_floor_dev"]), 0.0) / 0.1) ** 2)
                          + jnp.sum((jnp.minimum(jnp.log(h["vol_ls"]) - jnp.log(stats["ls_floor_vol"]), 0.0) / 0.1) ** 2))
        if amplitude_prior_scale > 0:
            lp += -0.5 * ((jnp.log(h["dev_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
            if not incompressible_plane_stress:
                lp += -0.5 * ((jnp.log(h["vol_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
        if lengthscale_prior_scale > 0:
            lp += -0.5 * (jnp.sum(((jnp.log(h["dev_ls"]) - jnp.log(centre_d)) / lengthscale_prior_scale) ** 2)
                          + jnp.sum(((jnp.log(h["vol_ls"]) - jnp.log(centre_v)) / lengthscale_prior_scale) ** 2))
        return lp

    # basis internal forces (exactly linear in theta), computed once per basis (again after setting the envelope)
    force_fn = lambda f: basis_force_matrix(model, f, cells_j, n_nodes, dNdX, dA)
    active = list(range(T))                    # load steps in the current continuation stage
    n_free_x, n_free_y = int(np.sum(dirs == 0)), int(np.sum(dirs == 1))
    N_x, N_y = n_free_x * T, n_free_y * T

    def _step_stats(theta, f_t, R_t, w, damping):
        eps, J, JR = eiv_step_linearisation(model, theta, f_t, cells_j, n_nodes, dNdX, dA, eiv, damping)
        y = -eps                                                                          # eps_bar + J theta ~ 0
        Jx, Jy, yx, yy = J[dirs == 0], J[dirs == 1], y[dirs == 0], y[dirs == 1]
        return dict(Gx=Jx.T @ Jx, Gy=Jy.T @ Jy, cx=Jx.T @ yx, cy=Jy.T @ yy, sx=yx @ yx, sy=yy @ yy,
                    GR=JR.T @ (w[:, None] * JR), bR=JR.T @ (w * R_t))

    def _step_stats_res(theta_noise, f_t, R_t, w, damping):
        """Residual likelihood: free forces B_f theta ~ N(0, diag sigma_f^2), reactions B_R theta ~ N(R_obs, sigma_R^2)."""
        if sigma_f_from_reaction:     # sigma_f,t,d = (N_BC / N_free) sigma_R,t,d (w = 1 / sigma_R,t^2 per direction)
            w_f = w[dirs] / bc_free_ratio ** 2
        elif force_rel:            # sigma_f,t = s |R_t|: unit-s weights here, scaled by 1/s^2 in linearise
            w_f = jnp.full(free.shape[0], 1.0 / jnp.maximum(jnp.sum(R_t ** 2), 1e-30))
        elif sigma_f_known:                                                                # force noise given directly
            w_f = jnp.full(free.shape[0], 1.0 / sigma_f_known ** 2)
        else:
            psi = lambda f: model.basis(f) @ theta_noise
            _, K = assemble_internal_force_and_tangent(psi, f_t, cells_j, n_nodes, dNdX, dA, eiv)
            w_f = 1.0 / (sigma_u_known ** 2 * jnp.sum(K[free][:, free] ** 2, axis=1))     # force-equivalent noise
        B = force_fn(f_t)
        Bf = B[free]
        B_R_t = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        zero = jnp.zeros(P)
        wr = reaction_weight * w                                                            # tempered reactions
        return dict(Gx=Bf.T @ (w_f[:, None] * Bf), Gy=jnp.zeros((P, P)), cx=zero, cy=zero, sx=0.0, sy=0.0,
                    GR=B_R_t.T @ (wr[:, None] * B_R_t), bR=B_R_t.T @ (wr * R_t), logdetW_f=jnp.sum(jnp.log(w_f)))

    def _step_stats_res_learned(theta_noise, f_t, R_t, w, damping):
        """Residual likelihood with learned per-direction force noise: unweighted free Gram matrices per direction."""
        B = force_fn(f_t)
        Bx, By = B[free][dirs == 0], B[free][dirs == 1]
        B_R_t = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        zero = jnp.zeros(P)
        wr = reaction_weight * w
        return dict(Gx=Bx.T @ Bx, Gy=By.T @ By, cx=zero, cy=zero, sx=0.0, sy=0.0,
                    GR=B_R_t.T @ (wr[:, None] * B_R_t), bR=B_R_t.T @ (wr * R_t))

    def _step_stats_nigp(theta, f_t, R_t, w, damping):
        """NIGP-style input noise: the displacement noise Sigma_u reaches the free forces as K Sigma_u K^T, with the
        tangent K frozen at theta (the current mean). Whitened: K^-1 B_f theta ~ N(0, Sigma_u) (design K^-1 B_f, target 0);
        reactions conditioned on the free forces (Schur complement): R ~ N((B_R - K_Rf K^-1 B_f) theta, Sigma_R)."""
        psi = lambda f: model.basis(f) @ theta
        _, K = assemble_internal_force_and_tangent(psi, f_t, cells_j, n_nodes, dNdX, dA, eiv)
        Kff = K[free][:, free]
        Krf = jnp.stack([K[rx][:, free].sum(0), K[ry][:, free].sum(0)])
        B = force_fn(f_t)
        J = jnp.linalg.solve(Kff, B[free])                                                    # (m, P)
        JR = jnp.stack([B[rx].sum(0), B[ry].sum(0)]) - Krf @ J
        Jx, Jy = J[dirs == 0], J[dirs == 1]
        wu = 1.0 / sigma_u_known ** 2
        zero = jnp.zeros(P)
        wr = reaction_weight * w
        return dict(Gx=wu * Jx.T @ Jx, Gy=wu * Jy.T @ Jy, cx=zero, cy=zero, sx=0.0, sy=0.0,
                    GR=JR.T @ (wr[:, None] * JR), bR=JR.T @ (wr * R_t),
                    logdetW_f=jnp.asarray(free.shape[0] * np.log(wu)))

    def compile_basis():
        """(Re)build the jitted functions that close over the basis (it changes when the envelope is set). jax.jit
        caches compiled code per function object, so fresh wrappers are needed to force a re-trace with the new basis."""
        fn = jax.jit(lambda f: force_fn(f))
        stats_fn = {"residual": _step_stats_res, "nigp": _step_stats_nigp,
                    "residual_exact": _step_stats_res_learned if exact_learn else _step_stats_res,
                    "residual_learned": _step_stats_res_learned}.get(likelihood, _step_stats)
        return [fn(f3x3_steps[t]) for t in range(T)], jax.jit(lambda *a: stats_fn(*a))

    B_steps, step_stats = compile_basis()
    nigp_warm_stats = jax.jit(lambda *a: _step_stats_res(*a))   # NIGP warm-up: exact residual Gaussian

    def _kff_margin(theta, f_t):
        """Stability margin of the energy theta at one load step: smallest eigenvalue of the free-DOF tangent divided by
        its mean diagonal (> 0: positive definite)."""
        _, K = assemble_internal_force_and_tangent(lambda f: model.basis(f) @ theta, f_t, cells_j, n_nodes, dNdX, dA, eiv)
        Kff = 0.5 * (K[free][:, free] + K[free][:, free].T)
        return jnp.linalg.eigvalsh(Kff)[0] / jnp.maximum(jnp.mean(jnp.abs(jnp.diag(Kff))), 1e-300)

    def make_stability_check():
        margin_fn = jax.jit(lambda th, f: _kff_margin(th, f))   # fresh object: re-traced after the envelope changes the basis
        def margin(theta):
            vals = [float(margin_fn(theta, f3x3_steps[t])) for t in active]
            return min(vals) if all(np.isfinite(vals)) else -np.inf
        return margin

    stability_margin = make_stability_check()

    stats_cache = {}
    rw_scale = dict(v=1.0)        # multiplies the reaction weight the step statistics were assembled with

    def linearise(theta_mean, damping=0.0):
        """Sufficient statistics of the Gauss-Newton-linearised EIV model at theta_mean (summed over load steps), with
        EIV damping delta = damping; for the residual likelihood, of the exact linear model with the force noise from the
        tangent at theta_noise."""
        warm = warmup_residual and not nigp_state["warm_done"]
        theta_free = residual and (warm or not nigp)          # statistics independent of theta_mean
        th = theta_noise if theta_free else theta_mean
        fn = nigp_warm_stats if warm else step_stats
        d = jnp.asarray(damping, dtype=jnp.float64)
        key = (tuple(active), warm)
        if theta_free and key in stats_cache:
            stats = dict(stats_cache[key])
        else:
            out = [fn(th, f3x3_steps[t], R_obs[t], 1.0 / sigma_R[t] ** 2, d) for t in active]
            stats = {k: sum(o[k] for o in out) for k in out[0]}
            if theta_free:
                stats_cache[key] = dict(stats)
        if rw_scale["v"] != 1.0:             # reaction weight changed after the statistics were built (linear in it)
            stats = dict(stats, GR=stats["GR"] * rw_scale["v"], bR=stats["bR"] * rw_scale["v"])
        if force_rel and theta_free and not sigma_f_joint:   # sigma_f,t = s |R_t|: free-force weights / s^2
            # (sigma_f_joint: unit-s statistics here, s applied in posterior_factors / neg_log_post)
            s2 = fnoise["s"] ** 2
            stats = dict(stats, Gx=stats["Gx"] / s2, Gy=stats["Gy"] / s2,
                         logdetW_f=stats["logdetW_f"] - free.shape[0] * len(active) * np.log(s2))
        w_all = (reaction_weight * rw_scale["v"] if (learned_noise or residual) else 1.0) / sigma_R[np.asarray(active)] ** 2
        stats.update(yWy_R=jnp.sum(w_all * R_obs[np.asarray(active)] ** 2), logdetW_R=jnp.sum(jnp.log(w_all)),
                     theta_fix=theta_fix, ls_floor_dev=ls_floor[0], ls_floor_vol=ls_floor[1])
        # square-root factors of the Gram blocks (G = R^T R, negative round-off eigenvalues clipped), so that the
        # posterior factor comes from a QR of [I; C] instead of a Cholesky of I + M^T G M (no squared condition number)
        for g in ("Gx", "Gy", "GR"):
            d, U = jnp.linalg.eigh(0.5 * (stats[g] + stats[g].T))
            stats["R" + g[1:]] = jnp.sqrt(jnp.maximum(d, 0.0))[:, None] * U.T
        return stats

    if likelihood not in ("eiv", "residual", "residual_learned", "residual_exact", "nigp"):
        raise ValueError(f"likelihood must be 'eiv', 'residual', 'residual_learned', 'residual_exact' or 'nigp', "
                         f"got {likelihood!r}")
    if nigp and not sigma_u_known:
        raise ValueError("likelihood='nigp' needs sigma_u_known (displacement noise std): with zero equilibrium targets "
                         "the input noise is not identifiable from the evidence.")
    learned_noise = likelihood == "residual_learned" or exact_learn
    if (likelihood == "residual" or eiv_warm) and not (sigma_u_known or sigma_f_known or force_rel or sigma_f_from_reaction):
        raise ValueError(f"likelihood={likelihood!r} needs sigma_u_known (displacement noise std) or sigma_f_known "
                         f"(force-residual noise std) to set the force noise.")
    # fixed noise weights (EIV with a residual warm-up: only during the warm-up iteration)
    residual = (likelihood in ("residual", "residual_exact", "nigp") and not exact_learn) or eiv_warm
    exact = likelihood == "residual_exact"
    if exact:   # closed-form posterior: one stage, full steps, no step control (see docstring)
        continuation, adaptive, continuation_stages = False, False, None
        stability_search, max_hyper_step, hyper_delay, hyper_iters_per_stage = False, np.inf, 0, 0
        hyper_steps = max(hyper_steps, 300)    # the single solve needs fully optimised hyperparameters
    single_solve = exact and not exact_learn   # learned sigma_f: alternate exact posterior and EM noise update

    if prior_mean not in ("linear_elastic", "none", "linear_elastic_fixed"):
        raise ValueError(f"prior_mean must be 'linear_elastic', 'none' or 'linear_elastic_fixed', got {prior_mean!r}")

    def posterior_factors(eta_, stats, lin_only):
        h = unpack(eta_)
        lin_scale = lin_prior_scale if (lin_only or prior_mean == "linear_elastic") else 0.0
        M = constrained_transform(model, cons, h, lin_scale, lin_only)
        wx, wy = (1.0, 1.0) if residual else (1.0 / h["su_x"] ** 2, 1.0 / h["su_y"] ** 2)   # residual: weights in stats
        if sigma_f_joint:                    # unit-s free-force statistics, weighted by 1 / s^2 here
            wx = wy = 1.0 / h["s"] ** 2
        b = wx * stats["cx"] + wy * stats["cy"] + stats["bR"]
        yWy = wx * stats["sx"] + wy * stats["sy"] + stats["yWy_R"]
        # fixed mean function theta = theta_fix + M gamma: shift the data by the linear prediction of theta_fix
        th_f = stats["theta_fix"]
        G_th = wx * (stats["Gx"] @ th_f) + wy * (stats["Gy"] @ th_f) + stats["GR"] @ th_f
        yWy = yWy - 2.0 * th_f @ b + th_f @ G_th
        b = b - G_th
        # A = I + M^T G M = R^T R from the QR of [I; sqrt(wx) R_x M; sqrt(wy) R_y M; R_R M]; L = R^T (lower)
        C = jnp.concatenate([jnp.eye(P), jnp.sqrt(wx) * (stats["Rx"] @ M), jnp.sqrt(wy) * (stats["Ry"] @ M), stats["RR"] @ M])
        R = jnp.linalg.qr(C, mode="r")
        L = R.T
        v = jax.scipy.linalg.solve_triangular(L, M.T @ b, lower=True)
        return h, M, L, v, yWy

    def neg_log_post(eta_, stats, lin_only):
        h, M, L, v, yWy = posterior_factors(eta_, stats, lin_only)
        noise_logdet = 0.5 * stats["logdetW_f"] if residual else -N_x * jnp.log(h["su_x"]) - N_y * jnp.log(h["su_y"])
        if sigma_f_joint:                    # log |W_f| at s: the unit-s value minus n_free log s^2
            noise_logdet = noise_logdet - (N_x + N_y) * jnp.log(h["s"])
        log_z = (-0.5 * yWy + 0.5 * v @ v - jnp.sum(jnp.log(jnp.abs(jnp.diag(L)))) + noise_logdet + 0.5 * stats["logdetW_R"]
                 - 0.5 * (N_x + N_y + 2 * len(active)) * jnp.log(2 * jnp.pi))
        return -(log_z + log_hyperprior(h, stats)), log_z

    def _optimise_hyper(eta_, stats, lin_only):
        if learned_noise:   # sigma_f is set by the variational update, not by the (unbounded) evidence
            fun = lambda e: neg_log_post(dict(e, log_su=jax.lax.stop_gradient(e["log_su"])), stats, lin_only)[0]
        else:
            fun = lambda e: neg_log_post(e, stats, lin_only)[0]
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(fun)

        def body(_, carry):
            e, s = carry
            val, g = vg(e, state=s)
            u, s = opt.update(g, s, e, value=val, grad=g, value_fn=fun)
            return optax.apply_updates(e, u), s
        return jax.lax.fori_loop(0, hyper_steps, body, (eta_, opt.init(eta_)))[0]

    def make_optimiser():
        """Fresh jitted optimiser: it closes over the active step set (N_x, N_y), which changes between stages."""
        return jax.jit(lambda e, st, lo: _optimise_hyper(e, st, lo), static_argnums=2)

    def posterior(eta_, stats, lin_only):
        h, M, L, v, _ = posterior_factors(eta_, stats, lin_only)
        m_gamma = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)
        L_inv_T = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(P), lower=False)   # Cov(gamma) = L^-T L^-1
        return stats["theta_fix"] + M @ m_gamma, M @ L_inv_T

    def force_change(th_new, th_old):
        num = sum(float(jnp.sum((B_steps[t] @ (th_new - th_old)) ** 2)) for t in active)
        den = sum(float(jnp.sum((B_steps[t] @ th_new) ** 2)) for t in active)
        return np.sqrt(num / max(den, 1e-300))

    history, theta_mean, converged = [], theta0, False
    theta_fix = jnp.zeros(P)                   # fixed mean function (prior_mean='linear_elastic_fixed', after the linear stage)

    def stage_ls_floor(idx):
        """Lengthscale floors (dev (2,), vol (1,)) from the span of the active steps' data in GP input space."""
        d, v = dev_x_steps[np.asarray(idx)].reshape(-1, dev_x.shape[-1]), vol_x_steps[np.asarray(idx)].reshape(-1, vol_x.shape[-1])
        frac = max(ls_floor_fraction, 0.0)
        return (jnp.asarray(frac * np.maximum(d.max(0) - d.min(0), 1e-6)), jnp.asarray(frac * np.maximum(v.max(0) - v.min(0), 1e-6)))
    ls_floor = stage_ls_floor(list(range(T)))
    factor, relax, prev_change, eta_prev = jnp.zeros((P, 1)), 1.0, np.inf, eta
    theta_noise = None
    if residual and not sigma_f_known and not force_rel and not sigma_f_from_reaction:
        # force-noise material: linear-elastic (mu, kappa) from the reactions alone, R_t ~ B_R,t[:, :2] (mu, kappa)
        X = jnp.concatenate([jnp.stack([B[rx].sum(0), B[ry].sum(0)])[:, :2] for B in B_steps])          # (2T, 2)
        w = (1.0 / sigma_R ** 2).reshape(-1)
        lin = jnp.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * R_obs.reshape(-1)))
        theta_noise = theta0.at[0].set(lin[0]).at[1].set(lin[1])
        log_fn(f"[HSGP] residual likelihood: force noise from the reaction-only linear-elastic fit mu {float(lin[0]):.4g}, "
               f"kappa {float(lin[1]):.4g}, sigma_u {sigma_u_known:.3g}")
    labels = list(map(int, step_labels)) if step_labels is not None else list(range(T))
    if continuation_stages:
        pos = {lab: i for i, lab in enumerate(labels)}
        stages = []
        for st in continuation_stages:
            missing = [x for x in st if int(x) not in pos]
            if missing:
                raise ValueError(f"continuation stage {st} uses load steps {missing} that are not training steps {labels}")
            idx = sorted(pos[int(x)] for x in st)
            if stages and not set(stages[-1]) <= set(idx):
                raise ValueError(f"continuation stages must be cumulative: {st} does not contain the previous stage")
            stages.append(idx)
        if sorted(stages[-1]) != list(range(T)):
            stages.append(list(range(T)))     # always finish on all training steps
    else:
        stages = [list(range(k)) for k in range(2, T + 1)] if (continuation and T > 2) else [list(range(T))]

    # adaptive continuation state: covered (last converged) step set, extension target, refinement batch size
    ad = dict(D=[], r_try=T - 1, refine_n=None, snapshot=None, stop_reason=None, skipped=[], cap=T - 1, range_note=None)

    def adaptive_proposal():
        D = ad["D"]
        if not D:
            return sorted({0, ad["r_try"] // 2, ad["r_try"]})
        r = max(D)
        if r < ad["cap"]:
            return sorted(set(D) | {(r + ad["r_try"]) // 2, ad["r_try"]})
        new = []
        for a, b in zip(D, D[1:]):          # per gap: the free step closest to its midpoint (skipped steps excluded)
            free_in_gap = [t for t in range(a + 1, b) if t not in ad["skipped"]]
            if free_in_gap:
                new.append(min(free_in_gap, key=lambda t: abs(t - (a + b) / 2)))
        if not new:
            return None
        return sorted(set(D) | set(new[:ad["refine_n"]] if ad["refine_n"] else new))

    if adaptive:
        stages = [adaptive_proposal()] if T > 2 else [list(range(T))]
    all_idx = list(range(T))
    it, aborted, stage_status, lin_done, si, gp_done = -1, False, [], False, -1, False
    if (exact or warmup_residual) and prior_mean == "none":
        lin_done = True                         # zero-mean GP: the linear-elastic (mu, kappa) would be discarded
        log_fn(f"[HSGP] {'exact residual warm-up' if warmup_residual else 'exact update'} with a zero-mean prior: "
               f"linear-elastic stage skipped")
    if sigma_f_joint:
        if not exact or prior_mean != "none":
            raise ValueError("sigma_f_joint needs likelihood 'residual_exact' and prior_mean 'none'.")
        log_fn(f"[HSGP] force noise sigma_f,t = s |R_t|: s optimised with the kernel hyperparameters on the evidence "
               f"(reaction weight {reaction_weight:.3g})")
        if joint_refit_unweighted:
            # step 1: s from the weighted evidence on all training steps; step 2 (the main solve below): s fixed,
            # reaction weight 1, hyperparameters and posterior recomputed
            active = list(all_idx)
            N_x, N_y = n_free_x * len(active), n_free_y * len(active)
            ls_floor = stage_ls_floor(active)
            st, sel_opt, best = linearise(theta_mean), make_optimiser(), None
            for e0 in [eta_init] + ([with_lengthscales(eta_init, 0.5), with_lengthscales(eta_init, 2.0)]
                                    if hyper_restarts else []):
                cand = sel_opt(e0, st, False)
                val, lz = neg_log_post(cand, st, False)
                if np.isfinite(float(val)) and (best is None or float(val) < best[0]):
                    best = (float(val), float(lz), cand)
            if best is None:
                raise RuntimeError("joint s selection: no finite evidence")
            s_sel = float(unpack(best[2])["s"])
            log_fn(f"[HSGP] step 1: s = {s_sel:.3g} (weighted evidence {best[1]:.2f}, reaction weight "
                   f"{reaction_weight:.3g}); step 2: s fixed, reaction weight 1")
            fnoise["s"] = s_sel
            fnoise["cv"] = dict(method="joint evidence, refit unweighted", reaction_weight=float(reaction_weight),
                                s=s_sel, log_evidence_weighted=best[1])
            sigma_f_joint = False                 # from here on s is fixed (statistics scaled by 1 / s^2)
            rw_scale["v"] = 1.0 / float(reaction_weight)
            active = None
    elif joint_refit_unweighted:
        raise ValueError("joint_refit_unweighted needs sigma_f_joint.")
    if sigma_f_cv or sigma_f_evidence:
        if not exact or prior_mean != "none":
            raise ValueError("sigma_f_cv / sigma_f_evidence need likelihood 'residual_exact' and prior_mean 'none'.")
        select_how = "cross-validation" if sigma_f_cv else "evidence"

        def heldout_score(m, Fc, t, s_rel):
            """Held-out reactions: log N(R_t; B_R m, Sigma_R,t + B_R S B_R^T), S = Fc Fc^T. The held-out free residuals are
            NOT scored: their target is zero, and a vanishing energy (s -> 0) predicts them perfectly, so that score is
            unbounded like the evidence; a vanishing energy predicts zero reactions instead."""
            B = B_steps[t]
            BR = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
            UR = BR @ Fc
            C = jnp.diag(sigma_R[t] ** 2) + UR @ UR.T
            d = R_obs[t] - BR @ m
            lp_r = -0.5 * (d @ jnp.linalg.solve(C, d) + jnp.linalg.slogdet(C)[1] + 2 * jnp.log(2 * jnp.pi))
            return float(lp_r)

        folds = [all_idx[k::max(2, int(cv_folds))] for k in range(max(2, int(cv_folds)))]
        cv_opt = make_optimiser()

        def cv_score(s_rel):
            nonlocal active, N_x, N_y, ls_floor
            fnoise["s"], total = float(s_rel), 0.0
            if sigma_f_evidence:     # log evidence of the (reaction-weighted) likelihood on all training steps
                active = list(all_idx)
                N_x, N_y = n_free_x * len(active), n_free_y * len(active)
                ls_floor = stage_ls_floor(active)
                st = linearise(theta_mean)
                best = None
                for e0 in [eta_init] + ([with_lengthscales(eta_init, 0.5), with_lengthscales(eta_init, 2.0)]
                                        if hyper_restarts else []):
                    cand = cv_opt(e0, st, False)
                    val, lz = neg_log_post(cand, st, False)
                    if np.isfinite(float(val)) and (best is None or float(val) < best[0]):
                        best = (float(val), float(lz))
                return best[1] if best is not None else -np.inf
            for hold in folds:
                active = [t for t in all_idx if t not in hold]
                N_x, N_y = n_free_x * len(active), n_free_y * len(active)
                ls_floor = stage_ls_floor(active)
                st = linearise(theta_mean)
                best = None
                for e0 in [eta_init] + ([with_lengthscales(eta_init, 0.5), with_lengthscales(eta_init, 2.0)]
                                        if hyper_restarts else []):
                    cand = cv_opt(e0, st, False)
                    val = float(neg_log_post(cand, st, False)[0])
                    if np.isfinite(val) and (best is None or val < best[0]):
                        best = (val, cand)
                if best is None:
                    return -np.inf
                m, Fc = posterior(best[1], st, False)
                total += sum(heldout_score(m, Fc, t, s_rel) for t in hold)
            return total

        grid = sorted(float(g) for g in (cv_grid if cv_grid is not None else np.logspace(-5, 0, 11)))
        scores = {}
        for g in grid:
            scores[g] = cv_score(g)
            log_fn(f"[HSGP] force-noise {select_how}: s = {g:.3g} (sigma_f,t = s |R_t|): score {scores[g]:.2f}")
        g_best = max(scores, key=scores.get)
        for fct in (10 ** -0.25, 10 ** 0.25, 10 ** -0.125, 10 ** 0.125):     # refine around the grid optimum
            g = g_best * fct
            scores[g] = cv_score(g)
            log_fn(f"[HSGP] force-noise {select_how}: s = {g:.3g}: score {scores[g]:.2f}")
            g_best = max(scores, key=scores.get)
        g_max = g_best
        if sigma_f_cv and cv_tol > 0:   # plateau rule: smallest s within cv_tol nats of the best held-out score
            g_best = min(g for g, v in scores.items() if v >= scores[g_max] - cv_tol)
            log_fn(f"[HSGP] force-noise {select_how}: best s = {g_max:.3g}; smallest s within {cv_tol:g} nats: {g_best:.3g}")
        fnoise["s"] = g_best
        fnoise["cv"] = dict(method=select_how, reaction_weight=float(reaction_weight), s=g_best, s_best_score=g_max,
                            tol=float(cv_tol),
                            folds=[[labels[t] for t in f] for f in folds] if sigma_f_cv else None,
                            curve={f"{k:.6g}": v for k, v in sorted(scores.items())})
        if sigma_f_evidence and g_best in (grid[0], grid[-1]):
            log_fn(f"[HSGP] WARNING: the evidence is maximal at the grid edge (s = {g_best:.3g}): no interior optimum. "
                   f"With reaction_weight {reaction_weight:g} the evidence may be unbounded as s -> 0 (use >= 10).")
        log_fn(f"[HSGP] force-noise {select_how}: selected s = {g_best:.3g} (sigma_f,t = s |R_t|)")
        active = None
    while si + 1 < len(stages):
        si += 1
        active = stages[si]
        aborted = False
        N_x, N_y = n_free_x * len(active), n_free_y * len(active)
        ls_floor = stage_ls_floor(active)
        optimise_hyper = make_optimiser()
        if adaptive or len(stages) > 1:
            log_fn(f"[HSGP] continuation stage {si}: load steps {[labels[t] for t in active]}")
        stage_lin, n_lin, n_full, converged, unstable_stop = not lin_done, 0, 0, False, False
        stage_tol = tol if (sorted(active) == all_idx or tol_intermediate is None) else tol_intermediate
        relax, prev_change = 1.0, np.inf
        while n_full < max_outer:
            it += 1
            lin_only = stage_lin
            # annealed EIV damping: delta_0 at the first GP iteration of each stage, x eiv_damping_decay per iteration,
            # exactly 0 below 1e-8 (a stage converges only undamped, so the final posterior is the exact EIV one)
            damping = 0.0 if lin_only else eiv_damping * eiv_damping_decay ** n_full
            damping = damping if damping >= 1e-8 else 0.0
            stats = linearise(theta_mean, damping)
            eta_prev_iter = eta              # hyperparameters before this iteration's update (stability fallback)
            clip_step = lambda e: jax.tree_util.tree_map(
                lambda a, b: a + jnp.clip(b - a, -max_hyper_step, max_hyper_step), eta, e)   # max_hyper_step per iteration
            starts = [eta] + ([] if (lin_only or not hyper_restarts) else
                              [clip_step(eta_init), with_lengthscales(eta, 0.5), with_lengthscales(eta, 2.0)])
            delayed = (not lin_only) and gp_done and hyper_delay > 0 and n_full < hyper_delay
            frozen = delayed or ((not lin_only) and hyper_iters_per_stage > 0
                                 and n_full >= (hyper_delay if gp_done else 0) + hyper_iters_per_stage)
            best = (float(neg_log_post(eta, stats, lin_only)[0]), eta) if frozen else None
            for e0 in ([] if frozen else starts):
                cand = clip_step(optimise_hyper(e0, stats, lin_only))
                val = float(neg_log_post(cand, stats, lin_only)[0])
                if np.isfinite(val) and (best is None or val < best[0]):
                    best = (val, cand)
            eta_new = best[1] if best is not None else eta
            new_mean, new_factor = posterior(eta_new, stats, lin_only)
            _, log_z = neg_log_post(eta_new, stats, lin_only)
            if not (np.isfinite(float(log_z)) and bool(jnp.all(jnp.isfinite(new_mean)))):
                # hyperparameter step diverged: keep the previous hyperparameters for this linearisation
                _, log_z_old = neg_log_post(eta, stats, lin_only)
                log_fn(f"[HSGP] it {it}: hyperparameter step non-finite (logZ at previous hyper {float(log_z_old):.2f}); "
                       f"stats finite: {all(bool(jnp.all(jnp.isfinite(v))) for v in stats.values())}; keeping previous hyperparameters.")
                eta_new = eta
                new_mean, new_factor = posterior(eta_new, stats, lin_only)
                _, log_z = neg_log_post(eta_new, stats, lin_only)
            if not (np.isfinite(float(log_z)) and bool(jnp.all(jnp.isfinite(new_mean)))):
                log_fn(f"[HSGP] it {it}: non-finite evidence or posterior; stopping at the last finite iterate.")
                aborted = True
                break
            eta, factor = eta_new, new_factor
            if sigma_f_joint:
                fnoise["s"] = float(unpack(eta)["s"])
            if learned_noise:   # variational update of the per-direction force noise from the expected residual
                S_theta = factor @ factor.T
                su_new = [jnp.sqrt((new_mean @ stats[g] @ new_mean + jnp.trace(stats[g] @ S_theta)) / n) for g, n in (("Gx", N_x), ("Gy", N_y))]
                eta = dict(eta, log_su=jnp.log(jnp.maximum(jnp.stack(su_new) - SIGMA_U_FLOOR, 1e-300)))
            change = force_change(new_mean, theta_mean)
            # Gauss-Newton relaxation: halve the step when the update grows (oscillation), recover slowly otherwise
            relax = 1.0 if (lin_only or exact) else (max(relax * 0.5, 0.125) if change > prev_change else min(relax * 1.5, 1.0))
            prev_change = np.inf if lin_only else change
            h = unpack(eta)
            rec = dict(iteration=it, continuation_stage=si, load_steps=[labels[t] for t in active], stage="linear" if lin_only else "full", log_evidence=float(log_z), force_change=change, eiv_damping=damping,
                       relaxation=relax, mu=float(new_mean[0]), kappa=float(new_mean[1]),
                       sigma_u_x=float(h["su_x"]), sigma_u_y=float(h["su_y"]),
                       dev_sig=float(h["dev_sig"]), vol_sig=0.0 if incompressible_plane_stress else float(h["vol_sig"]),
                       dev_ls=np.asarray(h["dev_ls"]).tolist(), vol_ls=np.asarray(h["vol_ls"]).tolist(), time=time.time() - t0)
            history.append(rec)
            if verbose:
                if exact_learn:
                    noise_txt = f"sigma_f {rec['sigma_u_x']:.2e},{rec['sigma_u_y']:.2e} (EM)"
                elif residual:   # the noise is given, not inferred
                    noise_txt = (f"sigma_f = {bc_free_ratio:.3g} x sigma_R,t" if sigma_f_from_reaction else
                                 f"sigma_f {fnoise['s']:.2e} x |R_t|" if force_rel else
                                 f"sigma_f {sigma_f_known:.2e} (given)" if sigma_f_known else f"sigma_u {sigma_u_known:.2e} (given)")
                else:
                    noise_txt = f"sigma_u {rec['sigma_u_x']:.2e},{rec['sigma_u_y']:.2e}"
                log_fn(f"[HSGP] s{si} it {it:2d} ({rec['stage']:6s}) logZ {rec['log_evidence']:12.2f} | dF/F {change:.2e} (relax {relax:.2f}) | "
                       f"mu {rec['mu']:.3f} kappa {rec['kappa']:.3f} | {noise_txt} | "
                       f"sig dev/vol {rec['dev_sig']:.3g}/{rec['vol_sig']:.3g} | l dev {np.round(rec['dev_ls'], 3)} vol {np.round(rec['vol_ls'], 3)} "
                       f"{f'| damping {damping:.0e} ' if damping > 0 else ''}({rec['time']:.0f}s)")
            # stability line search: only blocks moves from a stable mean into instability. A mean that is already
            # unstable at the active steps (e.g. the previous stage's fit at a newly added, larger load step) must be
            # allowed to move: every small step from it is unstable too, and blocking it would freeze the stage.
            warm_iter = warmup_residual and not nigp_state["warm_done"]
            if stability_search and not lin_only and not warm_iter:   # the warm-up takes the full exact solution
                m0 = stability_margin(theta_mean)
                admissible = (lambda m: m > 0.0) if m0 > 0.0 else (lambda m: m >= m0)   # stable: stay stable; else: no worse
                step = relax
                while step >= 1.0 / 64 and not admissible(stability_margin(theta_mean + step * (new_mean - theta_mean))):
                    step *= 0.5
                if step < 1.0 / 64:
                    # fallback: the hyperparameter update made the posterior too flexible (typical at a newly added, larger
                    # load step); recondition with the previous hyperparameters, which stays closer to the current mean
                    fb_mean, fb_factor = posterior(eta_prev_iter, stats, lin_only)
                    step = relax
                    while step >= 1.0 / 64 and not admissible(stability_margin(theta_mean + step * (fb_mean - theta_mean))):
                        step *= 0.5
                    if step >= 1.0 / 64:
                        log_fn(f"[HSGP] it {it}: hyperparameter update rejected by the stability line search (margin {m0:.3g}); "
                               f"step {step:.3g} with the previous hyperparameters")
                        eta, factor, new_mean = eta_prev_iter, fb_factor, fb_mean
                        change = force_change(new_mean, theta_mean)
                if step < 1.0 / 64:
                    log_fn(f"[HSGP] it {it}: no update keeps the mean tangent {'positive definite' if m0 > 0 else 'from losing stability'} "
                           f"(margin {m0:.3g}); keeping the current mean and ending continuation stage {si}.")
                    unstable_stop = True
                    break
                if step < relax:
                    log_fn(f"[HSGP] it {it}: step reduced {relax:.3g} -> {step:.3g} by the stability line search (margin {m0:.3g})")
                relax = step
            theta_mean = theta_mean + relax * (new_mean - theta_mean)
            if stage_lin:
                n_lin += 1
                if change < lin_tol or n_lin >= lin_max or single_solve:   # exact: one closed-form solve
                    stage_lin, lin_done = False, True
                    log_fn(f"[HSGP] linear-elastic stage {'solved (exact)' if single_solve else 'converged' if change < lin_tol else 'stopped (lin_max)'} after {n_lin} "
                           f"iterations: mu {float(theta_mean[0]):.4g}, kappa {float(theta_mean[1]):.4g}")
                    if prior_mean == "linear_elastic_fixed":
                        theta_fix = jnp.zeros(P).at[0].set(theta_mean[0]).at[1].set(theta_mean[1])
                        log_fn(f"[HSGP] fixed mean function: mu {float(theta_mean[0]):.4g}, kappa {float(theta_mean[1]):.4g}")
                    if envelope:
                        mu_e, kappa_e = float(theta_mean[0]), float(theta_mean[1])
                        model.envelope = dict(mu=mu_e, kappa=kappa_e, scale=float(energy_scale))
                        B_steps, step_stats = compile_basis()
                        stats_cache.clear()                              # the basis changed
                        nigp_warm_stats = jax.jit(lambda *a: _step_stats_res(*a))
                        stability_margin = make_stability_check()
                        log_fn(f"[HSGP] envelope set: mu {mu_e:.4g}, kappa {kappa_e:.4g}, scale {energy_scale:.4g}")
                    ad["snapshot"] = dict(theta=theta_mean, eta=eta, factor=factor, D=[])   # rollback point before any GP stage
                continue
            n_full += 1
            if warmup_residual and not nigp_state["warm_done"]:   # warm-up done: switch to the likelihood's iterations
                nigp_state["warm_done"] = True
                prev_change, relax = np.inf, 1.0
                if eiv_warm:   # EIV from here on: displacement noise sigma_u inferred again, new jitted optimiser
                    residual = False
                    optimise_hyper = make_optimiser()
                    log_fn("[HSGP] exact residual warm-up done; EIV iterations start from its mean and hyperparameters")
                else:
                    log_fn(f"[HSGP] NIGP warm-up (exact residual Gaussian) done; next iterations use the input noise "
                           f"K(theta) Sigma_u K(theta)^T with the tangent at the current mean")
                continue
            if (change < stage_tol or single_solve) and damping == 0.0:   # exact: one closed-form solve
                if not stability_search or stability_margin(new_mean) >= min(stability_margin(theta_mean), 0.0):
                    theta_mean = new_mean
                converged = True
                break

        near_converged = False
        if (adaptive and not converged and not unstable_stop and not aborted and cap_accept_tol > 0
                and n_full >= max_outer and history and history[-1]["force_change"] < cap_accept_tol):
            near_converged = True
            log_fn(f"[HSGP] stage {si} reached the iteration cap with force change {history[-1]['force_change']:.2e} "
                   f"< {cap_accept_tol:g}: accepted as converged")
        stage_status.append(dict(stage=si, load_steps=[labels[t] for t in active], converged=converged,
                                 iterations=n_full, aborted=aborted, stopped_for_stability=unstable_stop,
                                 accepted_at_cap=near_converged))
        converged = converged or near_converged
        gp_done = gp_done or (converged and lin_done)
        if not converged and not aborted and not unstable_stop:
            log_fn(f"[HSGP] WARNING: continuation stage {si} (load steps {[labels[t] for t in active]}) stopped at the "
                   f"iteration cap max_outer={max_outer} without converging; later stages start from this point and "
                   f"the result can depend on it. Increase max_outer.")
        if adaptive:
            if converged:
                r_old, r_new = (max(ad["D"]) if ad["D"] else 0), max(active)
                # next extension: twice the last successful jump (adaptive load stepping); the full range first
                r_next = T - 1 if not ad["D"] else min(T - 1, r_new + 2 * max(r_new - r_old, 1))
                ad.update(D=list(active), r_try=min(r_next, ad["cap"]), refine_n=None,
                          snapshot=dict(theta=theta_mean, eta=eta, factor=factor, D=list(active)))
            else:
                snap = ad["snapshot"]
                if snap is None:          # failed before the linear-elastic stage finished: nothing to roll back to
                    ad["stop_reason"] = "failed before the first rollback point"
                    break
                theta_mean, eta, factor = snap["theta"], snap["eta"], snap["factor"]
                D = ad["D"]
                extending = (not D) or max(active) > max(D)
                if extending:
                    r = max(D) if D else 0
                    if ad["r_try"] - r <= 1:
                        if not D:
                            ad["stop_reason"] = "no GP stage converged"
                        else:   # range limited: keep refining inside the covered range, then stop
                            ad["cap"] = r
                            ad["range_note"] = f"range limited to load step {labels[r]} (adding {labels[ad['r_try']]} failed)"
                            log_fn(f"[HSGP] {ad['range_note']}")
                    else:
                        ad["r_try"] = (r + ad["r_try"]) // 2
                else:
                    n_new = len(active) - len(D)
                    if n_new <= 1:   # a single interior step that cannot be added: leave it out and continue
                        ad["skipped"] += [t for t in active if t not in D]
                        ad["refine_n"] = None
                    else:
                        ad["refine_n"] = n_new // 2
                log_fn(f"[HSGP] stage {si} failed; rolled back to load steps {[labels[t] for t in D]}"
                       + (f"; stopping: {ad['stop_reason']}" if ad["stop_reason"] else ""))
                if ad["stop_reason"]:
                    break
            nxt = adaptive_proposal()
            if nxt is not None:
                stages.append(nxt)
            continue
        if aborted:
            break

    valid = sorted(ad["D"]) if adaptive else all_idx
    h = unpack(eta)
    hyper = {k: np.asarray(v) for k, v in h.items() if k not in ("su_x", "su_y")}
    if incompressible_plane_stress:
        hyper["vol_sig"] = np.asarray(0.0)          # volumetric GP switched off (theta_vol = 0)
    info = dict(prior_mean=prior_mean, eiv_damping=float(eiv_damping), eiv_damping_decay=float(eiv_damping_decay), envelope=model.envelope, likelihood=likelihood, sigma_u_known=sigma_u_known, sigma_f_known=sigma_f_known,
                sigma_f_rel=fnoise["s"] if force_rel else None, force_noise_cv=fnoise["cv"],
                reaction_weight=reaction_weight, warp=warp, continuation=continuation, continuation_stages=[[labels[t] for t in a] for a in stages],
                adaptive=adaptive, valid_load_steps=[labels[t] for t in valid], truncated=valid != all_idx,
                adaptive_stop_reason=ad["stop_reason"] or ad["range_note"], skipped_load_steps=[labels[t] for t in ad["skipped"]], hyper_iters_per_stage=hyper_iters_per_stage,
                ls_floor_fraction=ls_floor_fraction, tol_intermediate=tol_intermediate, hyper_delay=hyper_delay,
                cap_accept_tol=cap_accept_tol,
                stage_status=stage_status, sigma_cap=sigma_cap, stability_search=stability_search, hyper_restarts=hyper_restarts, all_stages_converged=(valid == all_idx) if adaptive else (bool(stage_status) and all(x["converged"] for x in stage_status)), converged=converged, outer_iterations=len(history), log_evidence=history[-1]["log_evidence"],
                box_factor=box_factor, num_basis_dev=num_basis_dev, num_basis_vol=num_basis_vol,
                dev_ls_bounds=[np.asarray(lsd_lo).tolist(), np.asarray(lsd_hi).tolist()],
                vol_ls_bounds=[np.asarray(lsv_lo).tolist(), np.asarray(lsv_hi).tolist()],
                lengthscale_prior_centre=dict(dev=centre_d.tolist(), vol=centre_v.tolist()),
                amplitude_prior_centre=float(np.exp(sig_centre)), fit_time=time.time() - t0)
    model = HSGPHyperelasticity(dev_box, vol_box, theta_mean, factor, hyper=hyper,
                                noise=(dict(sigma_u_x=float(sigma_u_known), sigma_u_y=float(sigma_u_known)) if residual   # the given noise
                                       else dict(sigma_u_x=float(h["su_x"]), sigma_u_y=float(h["su_y"]))), info=info,
                                envelope=model.envelope, warp=warp,
                                incompressible_plane_stress=incompressible_plane_stress)
    return model, history


def eiv_reaction_prediction(model: HSGPHyperelasticity, f3x3_steps, cells, node_type, dNdX, dA, correction: bool = True,
                            damping: Optional[float] = None):
    """
    Reaction forces predicted by the posterior at observed states, with the EIV correction of the displacement noise:
    R_t(theta) ~= J_R,t theta (exact along the posterior mean, see module docstring), linearised at the posterior mean.
    correction=False: plain reactions at the observed state, R = B_R theta (residual-likelihood model).
    damping: EIV damping (eiv_step_linearisation); default 0, as the fit converges undamped.
    Returns mean (T, 2) and epistemic covariance (T, 2, 2); add the load-cell variance for the predictive distribution.
    """
    damping = 0.0 if damping is None else float(damping)
    f3x3_steps = jnp.asarray(f3x3_steps, dtype=jnp.float64)
    node_type, cells_np = np.asarray(node_type), np.asarray(cells)
    cells_j, dNdX, dA = jnp.asarray(cells_np), jnp.asarray(dNdX), jnp.asarray(dA)
    n_nodes = node_type.shape[0]
    eiv = build_eiv_indices(node_type, cells_np)
    free, rx, ry = eiv["dof_free"], eiv["dof_rx"], eiv["dof_ry"]
    theta = model.theta_mean
    force_fn = lambda f: basis_force_matrix(model, f, cells_j, n_nodes, dNdX, dA)

    @jax.jit
    def jac_R(f_t):
        if not correction:
            B = force_fn(f_t)
            return jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        return eiv_step_linearisation(model, theta, f_t, cells_j, n_nodes, dNdX, dA, eiv, damping)[2]

    JR = jnp.stack([jac_R(f3x3_steps[t]) for t in range(f3x3_steps.shape[0])])            # (T, 2, P)
    G = JR @ model.theta_factor                                                             # (T, 2, Q)
    return np.asarray(JR @ theta), np.asarray(G @ jnp.swapaxes(G, 1, 2))

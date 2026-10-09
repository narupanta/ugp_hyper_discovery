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

from .features import IsotropicFeatureExtractor
from .hsgp_basis import Box, make_box, eigenfunctions, se_spectral_density
from .hsgp_model import HSGPHyperelasticity, DEV_REFERENCE, VOL_REFERENCE
from .loss_function import build_eiv_indices, assemble_internal_force_and_tangent

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
    return jax.scipy.linalg.block_diag(lin_scale * jnp.eye(model.n_lin), Md, Mv)


def fit_hsgp(f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA, *, energy_scale: float,
             lin_prior_scale: float, num_basis_dev: int = 24, num_basis_vol: int = 32, box_factor: float = 8.0,
             amplitude_factor: float = 10.0, amplitude_prior_scale: float = 1.0,
             lengthscale_factor: float = 1.5, lengthscale_prior_scale: float = 1.0,
             prior_mean: str = "linear_elastic", envelope: bool = False,
             likelihood: str = "eiv", sigma_u_known: Optional[float] = None, reaction_weight: float = 1.0, warp: bool = False, max_outer: int = 30, lin_max: int = 15, lin_tol: float = 1e-4,
             tol: float = 1e-5, sigma_cap: float = 10.0, max_hyper_step: float = 2.0, continuation: bool = False,
             stability_search: bool = True, hyper_restarts: bool = True, continuation_stages=None,
             step_labels=None,
             hyper_steps: int = 60, verbose: bool = True, log_fn=print):
    """
    f3x3_steps: (T, C, 3, 3) observed deformation gradients of the training steps; R_obs, sigma_R: (T, 2).
    prior_mean: 'linear_elastic' (psi = mu h_mu + kappa h_kappa + GP, coefficients ~ N(0, lin_prior_scale^2)) or 'none'
    (zero-mean GP prior: the linear-elastic fit only initialises the first tangent and the coefficients are fixed at 0).
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
    warp: GP inputs log(I_bar - 2), log J (see hsgp_model); boxes, spans and lengthscales live in the warped space.
    The linear-elastic stage runs until its relative force change < lin_tol (at most lin_max iterations), then the GP
    stage runs at most max_outer iterations. Robustness guards: amplitudes capped at sigma_cap x the prior centre, and
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
    Returns (HSGPHyperelasticity with the posterior, history list of dicts).
    """
    t0 = time.time()
    f3x3_steps = jnp.asarray(f3x3_steps, dtype=jnp.float64)
    cells_j, dNdX, dA = jnp.asarray(cells), jnp.asarray(dNdX), jnp.asarray(dA)
    node_type, cells = np.asarray(node_type), np.asarray(cells)
    n_nodes = node_type.shape[0]
    eiv = build_eiv_indices(node_type, cells)
    free, rx, ry = eiv["dof_free"], eiv["dof_rx"], eiv["dof_ry"]
    dirs = free % 2
    R_obs, sigma_R = jnp.asarray(R_obs), jnp.maximum(jnp.asarray(sigma_R), 1e-3)
    T = f3x3_steps.shape[0]

    # basis boxes from the training features (+ reference)
    feats = jax.vmap(IsotropicFeatureExtractor().extract)(f3x3_steps.reshape(-1, 3, 3))
    dev_x, vol_x = np.asarray(feats[0]), np.asarray(feats[1])
    if warp:   # GP input space (same map as HSGPHyperelasticity.gp_inputs)
        dev_x, vol_x = np.log(np.maximum(dev_x - 2.0, 1e-12)), np.log(np.maximum(vol_x, 1e-12))
        ref_d, ref_v = np.zeros(2), np.zeros(1)
    else:
        ref_d, ref_v = DEV_REFERENCE, VOL_REFERENCE
    dev_box = make_box(dev_x, ref_d, box_factor, num_basis_dev)
    vol_box = make_box(vol_x, ref_v, box_factor, num_basis_vol)
    P = 2 + dev_box.num_basis + vol_box.num_basis
    theta0 = jnp.zeros(P).at[0].set(1.0).at[1].set(1.0)
    model = HSGPHyperelasticity(dev_box, vol_box, theta0, jnp.zeros((P, 1)), warp=warp)
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
        s = (log_l - lo) / (hi - lo)
        return jnp.log(s / (1 - s))

    def unpack(eta):
        lo_d, hi_d = bounds["dev"]
        lo_v, hi_v = bounds["vol"]
        cap = sig_centre + np.log(sigma_cap)
        return dict(dev_sig=jnp.exp(jnp.minimum(eta["log_sig_dev"], cap)), vol_sig=jnp.exp(jnp.minimum(eta["log_sig_vol"], cap)),
                    dev_ls=jnp.exp(lo_d + (hi_d - lo_d) * jax.nn.sigmoid(eta["raw_ls_dev"])),
                    vol_ls=jnp.exp(lo_v + (hi_v - lo_v) * jax.nn.sigmoid(eta["raw_ls_vol"])),
                    su_x=SIGMA_U_FLOOR + jnp.exp(eta["log_su"][0]), su_y=SIGMA_U_FLOOR + jnp.exp(eta["log_su"][1]))

    eta = dict(log_sig_dev=jnp.asarray(sig_centre), log_sig_vol=jnp.asarray(sig_centre),
               raw_ls_dev=to_raw(jnp.log(centre_d), *bounds["dev"]), raw_ls_vol=to_raw(jnp.log(centre_v), *bounds["vol"]),
               log_su=jnp.log(jnp.array([1e-3, 1e-3])))

    eta_init = dict(eta)

    def with_lengthscales(eta_, factor):
        """eta with all lengthscales multiplied by factor (kept inside their bounds)."""
        out = dict(eta_)
        for key, comp in (("raw_ls_dev", "dev"), ("raw_ls_vol", "vol")):
            lo, hi = bounds[comp]
            log_l = lo + (hi - lo) * jax.nn.sigmoid(eta_[key]) + np.log(factor)
            out[key] = to_raw(jnp.clip(log_l, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo)), lo, hi)
        return out

    def log_hyperprior(h):
        lp = 0.0
        if amplitude_prior_scale > 0:
            lp += -0.5 * (((jnp.log(h["dev_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
                          + ((jnp.log(h["vol_sig"]) - sig_centre) / amplitude_prior_scale) ** 2)
        if lengthscale_prior_scale > 0:
            lp += -0.5 * (jnp.sum(((jnp.log(h["dev_ls"]) - jnp.log(centre_d)) / lengthscale_prior_scale) ** 2)
                          + jnp.sum(((jnp.log(h["vol_ls"]) - jnp.log(centre_v)) / lengthscale_prior_scale) ** 2))
        return lp

    # basis internal forces (exactly linear in theta), computed once per basis (again after setting the envelope)
    force_fn = lambda f: basis_force_matrix(model, f, cells_j, n_nodes, dNdX, dA)
    active = list(range(T))                    # load steps in the current continuation stage
    n_free_x, n_free_y = int(np.sum(dirs == 0)), int(np.sum(dirs == 1))
    N_x, N_y = n_free_x * T, n_free_y * T

    def _step_stats(theta, f_t, R_t, w):
        psi = lambda f: model.basis(f) @ theta
        f_int, K = assemble_internal_force_and_tangent(psi, f_t, cells_j, n_nodes, dNdX, dA, eiv)
        Kff = K[free][:, free]
        Krf = jnp.stack([K[rx][:, free].sum(0), K[ry][:, free].sum(0)])
        eps = jnp.linalg.solve(Kff, f_int[free])                                         # EIV correction at theta_bar
        eps_nodes = jnp.zeros(2 * n_nodes).at[free].set(eps).reshape(n_nodes, 2)
        dF = jnp.zeros_like(f_t).at[:, :2, :2].set(jnp.einsum("cai,caj->cij", eps_nodes[cells_j], dNdX))
        B, D = jax.jvp(force_fn, (f_t,), (dF,))                                          # D = dK/dtheta . eps
        D_R = jnp.stack([D[rx].sum(0), D[ry].sum(0)])
        B_R_t = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        J = jnp.linalg.solve(Kff, B[free] - D[free])                                     # (m, P)
        JR = B_R_t - Krf @ J - D_R                                                        # (2, P)
        y = -eps                                                                          # eps_bar + J theta ~ 0
        Jx, Jy, yx, yy = J[dirs == 0], J[dirs == 1], y[dirs == 0], y[dirs == 1]
        return dict(Gx=Jx.T @ Jx, Gy=Jy.T @ Jy, cx=Jx.T @ yx, cy=Jy.T @ yy, sx=yx @ yx, sy=yy @ yy,
                    GR=JR.T @ (w[:, None] * JR), bR=JR.T @ (w * R_t))

    def _step_stats_res(theta_noise, f_t, R_t, w):
        """Residual likelihood: free forces B_f theta ~ N(0, diag sigma_f^2), reactions B_R theta ~ N(R_obs, sigma_R^2)."""
        psi = lambda f: model.basis(f) @ theta_noise
        _, K = assemble_internal_force_and_tangent(psi, f_t, cells_j, n_nodes, dNdX, dA, eiv)
        w_f = 1.0 / (sigma_u_known ** 2 * jnp.sum(K[free][:, free] ** 2, axis=1))         # force-equivalent noise
        B = force_fn(f_t)
        Bf = B[free]
        B_R_t = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        zero = jnp.zeros(P)
        return dict(Gx=Bf.T @ (w_f[:, None] * Bf), Gy=jnp.zeros((P, P)), cx=zero, cy=zero, sx=0.0, sy=0.0,
                    GR=B_R_t.T @ (w[:, None] * B_R_t), bR=B_R_t.T @ (w * R_t), logdetW_f=jnp.sum(jnp.log(w_f)))

    def _step_stats_res_learned(theta_noise, f_t, R_t, w):
        """Residual likelihood with learned per-direction force noise: unweighted free Gram matrices per direction."""
        B = force_fn(f_t)
        Bx, By = B[free][dirs == 0], B[free][dirs == 1]
        B_R_t = jnp.stack([B[rx].sum(0), B[ry].sum(0)])
        zero = jnp.zeros(P)
        wr = reaction_weight * w
        return dict(Gx=Bx.T @ Bx, Gy=By.T @ By, cx=zero, cy=zero, sx=0.0, sy=0.0,
                    GR=B_R_t.T @ (wr[:, None] * B_R_t), bR=B_R_t.T @ (wr * R_t))

    def compile_basis():
        """(Re)build the jitted functions that close over the basis (it changes when the envelope is set). jax.jit
        caches compiled code per function object, so fresh wrappers are needed to force a re-trace with the new basis."""
        fn = jax.jit(lambda f: force_fn(f))
        stats_fn = {"residual": _step_stats_res, "residual_learned": _step_stats_res_learned}.get(likelihood, _step_stats)
        return [fn(f3x3_steps[t]) for t in range(T)], jax.jit(lambda *a: stats_fn(*a))

    B_steps, step_stats = compile_basis()

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

    def linearise(theta_mean):
        """Sufficient statistics of the Gauss-Newton-linearised EIV model at theta_mean (summed over load steps); for the
        residual likelihood, of the exact linear model with the force noise from the tangent at theta_noise."""
        th = theta_noise if residual else theta_mean
        out = [step_stats(th, f3x3_steps[t], R_obs[t], 1.0 / sigma_R[t] ** 2) for t in active]
        stats = {k: sum(o[k] for o in out) for k in out[0]}
        w_all = (reaction_weight if learned_noise else 1.0) / sigma_R[np.asarray(active)] ** 2
        stats.update(yWy_R=jnp.sum(w_all * R_obs[np.asarray(active)] ** 2), logdetW_R=jnp.sum(jnp.log(w_all)))
        # square-root factors of the Gram blocks (G = R^T R, negative round-off eigenvalues clipped), so that the
        # posterior factor comes from a QR of [I; C] instead of a Cholesky of I + M^T G M (no squared condition number)
        for g in ("Gx", "Gy", "GR"):
            d, U = jnp.linalg.eigh(0.5 * (stats[g] + stats[g].T))
            stats["R" + g[1:]] = jnp.sqrt(jnp.maximum(d, 0.0))[:, None] * U.T
        return stats

    if likelihood not in ("eiv", "residual", "residual_learned"):
        raise ValueError(f"likelihood must be 'eiv', 'residual' or 'residual_learned', got {likelihood!r}")
    learned_noise = likelihood == "residual_learned"
    if likelihood == "residual" and not sigma_u_known:
        raise ValueError("likelihood='residual' needs sigma_u_known (displacement noise std) to set the force noise.")
    residual = likelihood == "residual"

    if prior_mean not in ("linear_elastic", "none"):
        raise ValueError(f"prior_mean must be 'linear_elastic' or 'none', got {prior_mean!r}")

    def posterior_factors(eta_, stats, lin_only):
        h = unpack(eta_)
        lin_scale = lin_prior_scale if (lin_only or prior_mean == "linear_elastic") else 0.0
        M = constrained_transform(model, cons, h, lin_scale, lin_only)
        wx, wy = (1.0, 1.0) if residual else (1.0 / h["su_x"] ** 2, 1.0 / h["su_y"] ** 2)   # residual: weights in stats
        b = wx * stats["cx"] + wy * stats["cy"] + stats["bR"]
        yWy = wx * stats["sx"] + wy * stats["sy"] + stats["yWy_R"]
        # A = I + M^T G M = R^T R from the QR of [I; sqrt(wx) R_x M; sqrt(wy) R_y M; R_R M]; L = R^T (lower)
        C = jnp.concatenate([jnp.eye(P), jnp.sqrt(wx) * (stats["Rx"] @ M), jnp.sqrt(wy) * (stats["Ry"] @ M), stats["RR"] @ M])
        R = jnp.linalg.qr(C, mode="r")
        L = R.T
        v = jax.scipy.linalg.solve_triangular(L, M.T @ b, lower=True)
        return h, M, L, v, yWy

    def neg_log_post(eta_, stats, lin_only):
        h, M, L, v, yWy = posterior_factors(eta_, stats, lin_only)
        noise_logdet = 0.5 * stats["logdetW_f"] if residual else -N_x * jnp.log(h["su_x"]) - N_y * jnp.log(h["su_y"])
        log_z = (-0.5 * yWy + 0.5 * v @ v - jnp.sum(jnp.log(jnp.abs(jnp.diag(L)))) + noise_logdet + 0.5 * stats["logdetW_R"]
                 - 0.5 * (N_x + N_y + 2 * len(active)) * jnp.log(2 * jnp.pi))
        return -(log_z + log_hyperprior(h)), log_z

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
        return M @ m_gamma, M @ L_inv_T

    def force_change(th_new, th_old):
        num = sum(float(jnp.sum((B_steps[t] @ (th_new - th_old)) ** 2)) for t in active)
        den = sum(float(jnp.sum((B_steps[t] @ th_new) ** 2)) for t in active)
        return np.sqrt(num / max(den, 1e-300))

    history, theta_mean, converged = [], theta0, False
    factor, relax, prev_change, eta_prev = jnp.zeros((P, 1)), 1.0, np.inf, eta
    theta_noise = None
    if residual:
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
    it, aborted, stage_status = -1, False, []
    for si, active in enumerate(stages):
        N_x, N_y = n_free_x * len(active), n_free_y * len(active)
        optimise_hyper = make_optimiser()
        if len(stages) > 1:
            log_fn(f"[HSGP] continuation stage {si}: load steps {[labels[t] for t in active]}")
        stage_lin, n_lin, n_full, converged, unstable_stop = si == 0, 0, 0, False, False
        relax, prev_change = 1.0, np.inf
        while n_full < max_outer:
            it += 1
            lin_only = stage_lin
            stats = linearise(theta_mean)
            eta_prev_iter = eta              # hyperparameters before this iteration's update (stability fallback)
            clip_step = lambda e: jax.tree_util.tree_map(
                lambda a, b: a + jnp.clip(b - a, -max_hyper_step, max_hyper_step), eta, e)   # max_hyper_step per iteration
            starts = [eta] + ([] if (lin_only or not hyper_restarts) else
                              [clip_step(eta_init), with_lengthscales(eta, 0.5), with_lengthscales(eta, 2.0)])
            best = None
            for e0 in starts:
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
            if learned_noise:   # variational update of the per-direction force noise from the expected residual
                S_theta = factor @ factor.T
                su_new = [jnp.sqrt((new_mean @ stats[g] @ new_mean + jnp.trace(stats[g] @ S_theta)) / n) for g, n in (("Gx", N_x), ("Gy", N_y))]
                eta = dict(eta, log_su=jnp.log(jnp.maximum(jnp.stack(su_new) - SIGMA_U_FLOOR, 1e-300)))
            change = force_change(new_mean, theta_mean)
            # Gauss-Newton relaxation: halve the step when the update grows (oscillation), recover slowly otherwise
            relax = 1.0 if lin_only else (max(relax * 0.5, 0.125) if change > prev_change else min(relax * 1.5, 1.0))
            prev_change = np.inf if lin_only else change
            h = unpack(eta)
            rec = dict(iteration=it, continuation_stage=si, load_steps=[labels[t] for t in active], stage="linear" if lin_only else "full", log_evidence=float(log_z), force_change=change,
                       relaxation=relax, mu=float(new_mean[0]), kappa=float(new_mean[1]),
                       sigma_u_x=float(h["su_x"]), sigma_u_y=float(h["su_y"]),
                       dev_sig=float(h["dev_sig"]), vol_sig=float(h["vol_sig"]),
                       dev_ls=np.asarray(h["dev_ls"]).tolist(), vol_ls=np.asarray(h["vol_ls"]).tolist(), time=time.time() - t0)
            history.append(rec)
            if verbose:
                log_fn(f"[HSGP] s{si} it {it:2d} ({rec['stage']:6s}) logZ {rec['log_evidence']:12.2f} | dF/F {change:.2e} (relax {relax:.2f}) | "
                       f"mu {rec['mu']:.3f} kappa {rec['kappa']:.3f} | sigma_u {rec['sigma_u_x']:.2e},{rec['sigma_u_y']:.2e} | "
                       f"sig dev/vol {rec['dev_sig']:.3g}/{rec['vol_sig']:.3g} | l dev {np.round(rec['dev_ls'], 3)} vol {np.round(rec['vol_ls'], 3)} "
                       f"({rec['time']:.0f}s)")
            # stability line search: only blocks moves from a stable mean into instability. A mean that is already
            # unstable at the active steps (e.g. the previous stage's fit at a newly added, larger load step) must be
            # allowed to move: every small step from it is unstable too, and blocking it would freeze the stage.
            if stability_search and not lin_only:
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
                if change < lin_tol or n_lin >= lin_max:
                    stage_lin = False
                    log_fn(f"[HSGP] linear-elastic stage {'converged' if change < lin_tol else 'stopped (lin_max)'} after {n_lin} "
                           f"iterations: mu {float(theta_mean[0]):.4g}, kappa {float(theta_mean[1]):.4g}")
                    if envelope:
                        mu_e, kappa_e = float(theta_mean[0]), float(theta_mean[1])
                        model.envelope = dict(mu=mu_e, kappa=kappa_e, scale=float(energy_scale))
                        B_steps, step_stats = compile_basis()
                        stability_margin = make_stability_check()
                        log_fn(f"[HSGP] envelope set: mu {mu_e:.4g}, kappa {kappa_e:.4g}, scale {energy_scale:.4g}")
                continue
            n_full += 1
            if change < tol:
                if not stability_search or stability_margin(new_mean) >= min(stability_margin(theta_mean), 0.0):
                    theta_mean = new_mean
                converged = True
                break

        stage_status.append(dict(stage=si, load_steps=[labels[t] for t in active], converged=converged,
                                 iterations=n_full, aborted=aborted, stopped_for_stability=unstable_stop))
        if not converged and not aborted and not unstable_stop:
            log_fn(f"[HSGP] WARNING: continuation stage {si} (load steps {[labels[t] for t in active]}) stopped at the "
                   f"iteration cap max_outer={max_outer} without converging; later stages start from this point and "
                   f"the result can depend on it. Increase max_outer.")
        if aborted:
            break

    h = unpack(eta)
    hyper = {k: np.asarray(v) for k, v in h.items() if k not in ("su_x", "su_y")}
    info = dict(prior_mean=prior_mean, envelope=model.envelope, likelihood=likelihood, sigma_u_known=sigma_u_known,
                reaction_weight=reaction_weight, warp=warp, continuation=continuation, continuation_stages=[[labels[t] for t in a] for a in stages],
                stage_status=stage_status, sigma_cap=sigma_cap, stability_search=stability_search, hyper_restarts=hyper_restarts, all_stages_converged=bool(stage_status) and all(x["converged"] for x in stage_status), converged=converged, outer_iterations=len(history), log_evidence=history[-1]["log_evidence"],
                box_factor=box_factor, num_basis_dev=num_basis_dev, num_basis_vol=num_basis_vol,
                dev_ls_bounds=[np.asarray(lsd_lo).tolist(), np.asarray(lsd_hi).tolist()],
                vol_ls_bounds=[np.asarray(lsv_lo).tolist(), np.asarray(lsv_hi).tolist()],
                lengthscale_prior_centre=dict(dev=centre_d.tolist(), vol=centre_v.tolist()),
                amplitude_prior_centre=float(np.exp(sig_centre)), fit_time=time.time() - t0)
    model = HSGPHyperelasticity(dev_box, vol_box, theta_mean, factor, hyper=hyper,
                                noise=dict(sigma_u_x=float(h["su_x"]), sigma_u_y=float(h["su_y"])), info=info,
                                envelope=model.envelope, warp=warp)
    return model, history


def eiv_reaction_prediction(model: HSGPHyperelasticity, f3x3_steps, cells, node_type, dNdX, dA, correction: bool = True):
    """
    Reaction forces predicted by the posterior at observed states, with the EIV correction of the displacement noise:
    R_t(theta) ~= J_R,t theta (exact along the posterior mean, see module docstring), linearised at the posterior mean.
    correction=False: plain reactions at the observed state, R = B_R theta (residual-likelihood model).
    Returns mean (T, 2) and epistemic covariance (T, 2, 2); add the load-cell variance for the predictive distribution.
    """
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
        psi = lambda f: model.basis(f) @ theta
        f_int, K = assemble_internal_force_and_tangent(psi, f_t, cells_j, n_nodes, dNdX, dA, eiv)
        Kff = K[free][:, free]
        Krf = jnp.stack([K[rx][:, free].sum(0), K[ry][:, free].sum(0)])
        eps = jnp.linalg.solve(Kff, f_int[free])
        eps_nodes = jnp.zeros(2 * n_nodes).at[free].set(eps).reshape(n_nodes, 2)
        dF = jnp.zeros_like(f_t).at[:, :2, :2].set(jnp.einsum("cai,caj->cij", eps_nodes[cells_j], dNdX))
        B, D = jax.jvp(force_fn, (f_t,), (dF,))
        J = jnp.linalg.solve(Kff, B[free] - D[free])
        return jnp.stack([B[rx].sum(0), B[ry].sum(0)]) - Krf @ J - jnp.stack([D[rx].sum(0), D[ry].sum(0)])

    JR = jnp.stack([jac_R(f3x3_steps[t]) for t in range(f3x3_steps.shape[0])])            # (T, 2, P)
    G = JR @ model.theta_factor                                                             # (T, 2, Q)
    return np.asarray(JR @ theta), np.asarray(G @ jnp.swapaxes(G, 1, 2))

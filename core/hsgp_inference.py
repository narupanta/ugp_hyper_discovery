"""
Exact Gaussian posterior of the HSGP strain-energy model under the residual likelihood (FE-VFM residual evaluated at
the measured displacements).

Model, per training load step t (u = measured displacements, so the residual is exactly linear in the weights):
  free DOFs:   B_free,t theta ~ N(0, sigma_free,t^2 I),   sigma_free,t = s |R_t|
  reactions:   B_R,t theta    ~ N(R_t, Sigma_R,t)         (load-cell noise, diagonal)
B_t = internal nodal forces of each basis function at the measured state (assembled once). In the paper's notation
this is one linear-Gaussian pseudo-observation per step, q_t = H_t theta + eps_t (pseudo_observation), and several
steps / experiments are stacked, q = [q_1; ...; q_T] (stack_observations; the fit sums their gaussian_statistics,
which is the same). With the constrained GP
prior theta = M(eta) gamma, gamma ~ N(0, I), the posterior over theta and the evidence are closed form for given
(eta, s). The kernel hyperparameters eta = (sigma_dev, l_dev, sigma_vol, l_vol) are set by MAP-II on the evidence
(log-normal hyperpriors optional, lengthscales bounded to the range the basis resolves, amplitudes capped).

The free-residual scale s cannot come from the evidence: the free residuals have zero targets, and the evidence grows
without bound as s -> 0 (a vanishing energy fits them exactly). It is either given (sigma_f_rel) or chosen by K-fold
cross-validation over the training steps, scoring only the held-out REACTIONS by their log predictive density
log N(R_t; B_R,t m, Sigma_R,t + B_R,t S B_R,t^T) (a vanishing energy predicts zero reactions and is penalised).

GP prior conditioned exactly on the reference constraints A theta = 0 (zero energy and volumetric stress at F = I):
  w = Lambda^{1/2} Pi u,  u ~ N(0, I),  Pi = I - At^T (At At^T)^{-1} At,  At = A Lambda^{1/2}.
Anisotropic option (fibres a0, a1): an additive 2D GP over (I4_bar, I6_bar) with its own amplitude and lengthscales,
conditioned on psi_aniso(1, 1) = 0 and zero slopes in I4 and I6 there (zero energy and stress at F = I).
The two linear-elastic coefficients of HSGPHyperelasticity (theta[0], theta[1]) are kept at zero (zero-mean prior).
"""
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .features import make_feature_extractor
from .hsgp_basis import Box, make_box, se_spectral_density
from .hsgp_model import HSGPHyperelasticity, DEV_REFERENCE, VOL_REFERENCE, ANISO_REFERENCE
from .mechanics import dof_sets

jax.config.update("jax_enable_x64", True)

LS_MAX_FRACTION = 1.0 / 3.2    # l_max = box_factor * S / 3.2: c >= 3.2 l / S (Riutort-Mayol et al. 2023, SE kernel)


def lengthscale_bounds(box: Box, box_factor: float):
    """Per-dimension [l_min, l_max] where the HSGP approximates the SE kernel well (Riutort-Mayol et al. 2023):
    l_min from m >= 1.75 c S / l, l_max from c >= 3.2 l / S."""
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


# ----------------------------------------------------------------------------------------------------------------------
# Pseudo-observation model (paper notation): per load step t
#     q_t = H_t theta + eps_t,   eps_t ~ N(0, diag(sigma_t^2)),
#     q_t = [0 (free DOFs); R_t (loaded-edge reactions); S_t (support reactions, where measured); (0 global equil.)].
# Several load steps / experiments: q = [q_1; ...; q_T], H = [H_1; ...; H_T] (independent noise). The posterior only
# needs H^T W H, H^T W q, q^T W q and log det W (W = diag(sigma^-2)), which are sums over the stacked blocks, so the
# fit stacks per-step STATISTICS (gaussian_statistics) instead of the tall matrices themselves (same result, memory
# O(P^2) per step instead of O(n_free P)).
# ----------------------------------------------------------------------------------------------------------------------
class Observation(NamedTuple):
    """One block of the pseudo-observations: q = H theta + eps, eps ~ N(0, diag(sigma^2))."""
    H: jnp.ndarray        # (n, P)
    q: jnp.ndarray        # (n,)
    sigma: jnp.ndarray    # (n,)


def pseudo_observation(B, dofs, R_t, sigma_R_t, S_t=None, sigma_S_t=None, *, global_equilibrium: float = 0.0):
    """Pseudo-observation blocks of one load step from its basis force matrix B_t (2n, P) (basis_force_matrix).

      free:        H = B_t[free],           q = 0,     sigma = |R_t|        (x s: the free-residual scale, see below)
      reaction:    one row per boundary with a measured reaction:
                   loaded edges (x, y): H = sum of B_t over their DOFs,  q = R_t,  sigma = sigma_R,t
                   supports (fixed x, fixed y): H = sum over their DOFs,  q = S_t,  sigma = sigma_S,t
                   (S_t, sigma_S_t: (2,) measured support reactions; NaN or None = not measured, no row; a direction
                   without fixed DOFs has no row either)
      equilibrium: (global_equilibrium = c > 0) H = sum of B_t[free] per direction, q = 0, sigma = c |R_t|
    The free block is returned at s = 1: its noise is sigma_free,t = s |R_t| and s is chosen outside (scale_noise).
    """
    R_t, sigma_R_t = jnp.asarray(R_t), jnp.asarray(sigma_R_t)
    r = jnp.sqrt(jnp.maximum(jnp.sum(R_t ** 2), 1e-30))
    free = np.asarray(dofs["free"])
    obs = dict(free=Observation(B[free], jnp.zeros(len(free)), jnp.full(len(free), r)))
    H_R, q_R, s_R = [B[dofs["rx"]].sum(0), B[dofs["ry"]].sum(0)], [R_t[0], R_t[1]], [sigma_R_t[0], sigma_R_t[1]]
    for comp, key in ((0, "fx"), (1, "fy")):
        if S_t is not None and len(dofs[key]) and np.isfinite(float(S_t[comp])):
            H_R.append(B[dofs[key]].sum(0))
            q_R.append(jnp.asarray(S_t[comp]))
            s_R.append(jnp.maximum(jnp.asarray(sigma_S_t[comp]), 1e-3))
    obs["reaction"] = Observation(jnp.stack(H_R), jnp.stack(q_R), jnp.stack(s_R))
    if global_equilibrium > 0:
        H_eq = jnp.stack([B[free[free % 2 == 0]].sum(0), B[free[free % 2 == 1]].sum(0)])
        obs["equilibrium"] = Observation(H_eq, jnp.zeros(2), jnp.full(2, float(global_equilibrium) * r))
    return obs


def stack_observations(blocks):
    """[q_1; q_2; ...] = [H_1; H_2; ...] theta + eps: the joint observation of independent blocks."""
    return Observation(*(jnp.concatenate([getattr(b, k) for b in blocks]) for k in Observation._fields))


@dataclass(frozen=True)
class GaussianStats:
    """Sufficient statistics of an observation block: G = H^T W H, b = H^T W q, qWq = q^T W q, logdetW, n rows.
    Statistics of stacked blocks are the sums of the blocks' statistics (stats(stack(a, b)) == stats(a) + stats(b))."""
    G: jnp.ndarray
    b: jnp.ndarray
    qWq: jnp.ndarray
    logdetW: jnp.ndarray
    n: float

    def __add__(self, o):
        if isinstance(o, int) and o == 0:      # sum() starts from 0
            return self
        return GaussianStats(self.G + o.G, self.b + o.b, self.qWq + o.qWq, self.logdetW + o.logdetW, self.n + o.n)

    __radd__ = __add__


def gaussian_statistics(obs: Observation) -> GaussianStats:
    w = 1.0 / obs.sigma ** 2
    return GaussianStats(obs.H.T @ (w[:, None] * obs.H), obs.H.T @ (w * obs.q), jnp.sum(w * obs.q ** 2),
                         jnp.sum(jnp.log(w)), float(obs.q.shape[0]))


def scale_noise(st: GaussianStats, s: float) -> GaussianStats:
    """Statistics of the same block with its noise sigma multiplied by s."""
    s2 = float(s) ** 2
    return GaussianStats(st.G / s2, st.b / s2, st.qWq / s2, st.logdetW - st.n * np.log(s2), st.n)


def sqrt_factor(G):
    """R with R^T R = G (symmetric eigendecomposition, PSD-safe)."""
    d, U = jnp.linalg.eigh(0.5 * (G + G.T))
    return jnp.sqrt(jnp.maximum(d, 0.0))[:, None] * U.T


def log_predictive_density(obs: Observation, mean, factor):
    """log N(q; H m, diag(sigma^2) + H S H^T) with the posterior theta ~ N(m, S), S = factor factor^T."""
    U = obs.H @ factor
    C = jnp.diag(obs.sigma ** 2) + U @ U.T
    d = obs.q - obs.H @ mean
    return float(-0.5 * (d @ jnp.linalg.solve(C, d) + jnp.linalg.slogdet(C)[1] + d.shape[0] * jnp.log(2 * jnp.pi)))


class _Constraints:
    """Reference-constraint rows on the GP weights: zero deviatoric energy, zero volumetric energy and pressure."""
    def __init__(self, model: HSGPHyperelasticity):
        self.A_dev = model.dev_basis(jnp.asarray(DEV_REFERENCE))[None, :]
        vol_fn = lambda j: model.vol_basis(j)
        self.A_vol = jnp.stack([vol_fn(jnp.asarray(VOL_REFERENCE)), jax.jacfwd(vol_fn)(jnp.asarray(VOL_REFERENCE))[:, 0]])
        if model.is_anisotropic:   # energy and both fibre slopes at I4 = I6 = 1
            an_fn = lambda x: model.aniso_basis(x)
            ref = jnp.asarray(ANISO_REFERENCE)
            self.A_aniso = jnp.concatenate([an_fn(ref)[None, :], jax.jacfwd(an_fn)(ref).T])     # (3, K_aniso)


def _projected_sqrt(A, lam):
    sq = jnp.sqrt(jnp.maximum(lam, 0.0))
    At = A * sq[None, :]
    Pi = jnp.eye(lam.shape[0]) - At.T @ jnp.linalg.solve(At @ At.T, At)
    return sq[:, None] * Pi


def constrained_transform(model, cons: _Constraints, hyper, incompressible_plane_stress: bool = False):
    """M (P, P): theta = M gamma with gamma ~ N(0, I) gives the constrained prior (linear-elastic block zero)."""
    Md = _projected_sqrt(cons.A_dev, se_spectral_density(model.dev_box, hyper["dev_sig"], hyper["dev_ls"]))
    Mv = _projected_sqrt(cons.A_vol, se_spectral_density(model.vol_box, hyper["vol_sig"], hyper["vol_ls"]))
    if incompressible_plane_stress:   # J = 1: no volumetric GP
        Mv = 0.0 * Mv
    blocks = [jnp.zeros((model.n_lin, model.n_lin)), Md, Mv]
    if model.is_anisotropic:
        blocks.append(_projected_sqrt(cons.A_aniso, se_spectral_density(model.aniso_box, hyper["aniso_sig"], hyper["aniso_ls"])))
    return jax.scipy.linalg.block_diag(*blocks)


def reaction_prediction(model: HSGPHyperelasticity, f3x3_steps, cells, node_type, dNdX, dA):
    """Reactions R_t = B_R,t theta at the measured states: mean (T, 2) and epistemic covariance (T, 2, 2)."""
    f3x3_steps = jnp.asarray(f3x3_steps, dtype=jnp.float64)
    node_type, cells_np = np.asarray(node_type), np.asarray(cells)
    cells_j, dNdX, dA = jnp.asarray(cells_np), jnp.asarray(dNdX), jnp.asarray(dA)
    n_nodes = node_type.shape[0]
    idx = dof_sets(node_type)
    rx, ry = idx["rx"], idx["ry"]

    @jax.jit
    def B_R(f_t):
        B = basis_force_matrix(model, f_t, cells_j, n_nodes, dNdX, dA)
        return jnp.stack([B[rx].sum(0), B[ry].sum(0)])

    BR = jnp.stack([B_R(f3x3_steps[t]) for t in range(f3x3_steps.shape[0])])             # (T, 2, P)
    G = BR @ model.theta_factor
    return np.asarray(BR @ model.theta_mean), np.asarray(G @ jnp.swapaxes(G, 1, 2))


def fit_hsgp(f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA, *, energy_scale: float,
             num_basis_dev: int = 24, num_basis_vol: int = 32, box_factor: float = 8.0,
             amplitude_factor: float = 10.0, amplitude_prior_scale: float = 0.0,
             lengthscale_factor: float = 1.5, lengthscale_prior_scale: float = 0.0,
             sigma_cap: float = 10.0, hyper_restarts: bool = True, incompressible_plane_stress: bool = False,
             sigma_f_rel: Optional[float] = None, cv_folds: int = 2, cv_grid=None, cv_tol: float = 0.0,
             cv_data: Optional[dict] = None, fibers=None, num_basis_aniso: Optional[int] = None,
             step_labels=None, hyper_steps: int = 300, hyper_gtol: float = 1e-6, hyper_ftol: float = 1e-12,
             global_equilibrium: float = 0.0, support_R=None, support_sigma=None,
             reaction_noise_grid=None, reaction_sigma_fn=None, cv_by_experiment: bool = False,
             verbose: bool = True, log_fn=print):
    """
    f3x3_steps: (T, C, 3, 3) measured deformation gradients of the training steps; R_obs, sigma_R: (T, 2).
    Several experiments (specimens with their own meshes) at once: pass f3x3_steps, R_obs, sigma_R, cells, node_type,
    dNdX, dA and step_labels as LISTS with one entry per experiment. The likelihood is a sum over load steps, so the
    steps of all experiments are simply pooled (one GP, one s; the folds interleave the pooled steps); step labels
    become '<experiment index>:<step>' unless given as strings.
    sigma_f_rel: free-residual scale s (sigma_free,t = s |R_t|); None = chosen by cross-validation:
      cv_folds interleaved folds of the training steps; log grid cv_grid (default 1e-5..1, 11 points) refined around
      the best score; cv_tol > 0: the smallest s within cv_tol nats of the best held-out score (plateau rule).
      cv_data: a separate validation set instead of the folds, dict(f3x3 (V, C, 3, 3), R (V, 2), sigma_R (V, 2),
      labels, optional support_R / support_sigma (V, 2)): each s is fitted on ALL training steps and scored on these steps' reactions only (cv_folds ignored);
      they never enter the fit.
    fibers: (a0, a1) in-plane fibre vectors (a1 may be None) for the anisotropic GP over (I4_bar, I6_bar), with
    num_basis_aniso functions per fibre invariant (default num_basis_dev); None: isotropic.
    global_equilibrium c > 0: per step and direction one more observation, sum of the free-DOF internal forces = 0
    with sigma_eq,t = c |R_t| (the body is in equilibrium, so the free residuals cannot all push one way; with one load
    cell this is "support reaction = -load-cell reaction" without counting the load cell twice). The free residuals
    are otherwise independent with sigma_f = s |R_t|, whose sum is only bounded by sqrt(n_free) s |R_t|; this row is an
    extra Gaussian observation of the same rows (an approximation of a zero-sum residual model). 0: off.
    support_R, support_sigma: (T, 2) measured reactions of the supports (fixed-x / fixed-y boundaries) and their noise,
    as given by the data (NaN: not measured); each measured one is one more reaction row, like the loaded edges. None
    (e.g. experiments with one load cell): loaded-edge reactions only. Lists per experiment for several experiments.
    reaction_noise_grid, reaction_sigma_fn: cross-validation of the reaction noise as well: for each candidate r of
      the grid, the loaded-edge reaction rows get sigma_R = reaction_sigma_fn(r) ((T, 2), or a list per experiment, of
      the training steps; e.g. max(r |R_t|, floor)), s is cross-validated as above, and the pair (r, s) with the best
      held-out reaction score is taken (the plateau rule cv_tol applies to s at that r). The held-out score uses the
      same r for its own noise, so a larger r is only preferred where the misfit of the held-out reactions needs it.
      With sigma_f_rel given, only r is cross-validated (at that s). Not with cv_data.
    cv_by_experiment: several experiments: one fold per experiment (leave one experiment out) instead of the
      interleaved steps; held-out reactions of a whole specimen measure the error that is shared by its steps.
    hyper_steps, hyper_gtol, hyper_ftol: L-BFGS of the hyperparameters stops at |grad| < hyper_gtol or a relative change
    of the objective < hyper_ftol, after at most hyper_steps iterations.
    amplitude_factor x energy_scale: prior centre of the amplitudes; sigma_cap: amplitudes capped at sigma_cap x that
    centre (<= 0: no cap); amplitude/lengthscale_prior_scale > 0: log-normal hyperpriors of that width (log units).
    Returns (HSGPHyperelasticity with the posterior, history: one record of the final solve).
    """
    t0 = time.time()
    multi = isinstance(cells, (list, tuple))
    if not multi:
        f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA = ([x] for x in (f3x3_steps, R_obs, sigma_R, cells,
                                                                               node_type, dNdX, dA))
        step_labels = None if step_labels is None else [step_labels]
        support_R, support_sigma = [support_R], [support_sigma]
    if support_R is None:
        support_R = support_sigma = [None] * len(f3x3_steps)
    if multi and cv_data is not None:
        raise ValueError("cv_data (validation-step selection of s) is not supported for several experiments; use the folds")
    meshes = []                      # per experiment: mesh arrays and the DOF index sets
    for c, nt, dn, da in zip(cells, node_type, dNdX, dA):
        meshes.append(dict(cells=jnp.asarray(np.asarray(c)), n_nodes=np.asarray(nt).shape[0], dNdX=jnp.asarray(dn),
                           dA=jnp.asarray(da), **dof_sets(nt)))
    step_exp, step_F, labels = [], [], []        # pooled steps: experiment index, F (C_e, 3, 3), label
    for e, Fe in enumerate(f3x3_steps):
        Fe = jnp.asarray(Fe, dtype=jnp.float64)
        lab_e = list(step_labels[e]) if step_labels is not None and step_labels[e] is not None else list(range(Fe.shape[0]))
        for t in range(Fe.shape[0]):
            step_exp.append(e)
            step_F.append(Fe[t])
            labels.append((lab_e[t] if isinstance(lab_e[t], str) else f"{e}:{int(lab_e[t])}") if multi else int(lab_e[t]))
    S_obs = np.concatenate([np.full((np.asarray(R).shape[0], 2), np.nan) if S is None else np.asarray(S, float)
                            for R, S in zip(R_obs, support_R)])
    sigma_S = np.concatenate([np.full((np.asarray(R).shape[0], 2), np.nan) if S is None else np.asarray(S, float)
                              for R, S in zip(R_obs, support_sigma)])
    R_obs = jnp.concatenate([jnp.asarray(r) for r in R_obs])
    sigma_R = jnp.maximum(jnp.concatenate([jnp.asarray(s) for s in sigma_R]), 1e-3)
    T = len(step_F)
    all_idx = list(range(T))

    # basis boxes from the training features (+ reference)
    aniso_on = fibers is not None
    if aniso_on:
        from .features import AnisotropicFeatureExtractor
        extractor = AnisotropicFeatureExtractor(jnp.asarray(fibers[0], dtype=jnp.float64),
                                                a1=None if fibers[1] is None else jnp.asarray(fibers[1], dtype=jnp.float64),
                                                incompressible_plane_stress=incompressible_plane_stress)
    else:
        extractor = make_feature_extractor(incompressible_plane_stress)
    feats = jax.vmap(extractor.extract)(jnp.concatenate([F.reshape(-1, 3, 3) for F in step_F]))
    dev_x, vol_x = np.asarray(feats[0]), np.asarray(feats[1])
    an_x = np.asarray(feats[2])[:, :2] if aniso_on else None
    dev_box = make_box(dev_x, DEV_REFERENCE, box_factor, num_basis_dev)
    if incompressible_plane_stress:
        # J = 1 everywhere: a nominal box keeps the (inactive) volumetric basis well-scaled
        vol_box = make_box(np.array([[0.5], [1.5]]), VOL_REFERENCE, 1.0, num_basis_vol)
    else:
        vol_box = make_box(vol_x, VOL_REFERENCE, box_factor, num_basis_vol)
    an_box = make_box(an_x, ANISO_REFERENCE, box_factor, int(num_basis_aniso or num_basis_dev)) if aniso_on else None
    fib = dict(aniso_box=an_box, a0=fibers[0], a1=fibers[1]) if aniso_on else {}
    P = 2 + dev_box.num_basis + vol_box.num_basis + (an_box.num_basis if aniso_on else 0)
    model = HSGPHyperelasticity(dev_box, vol_box, jnp.zeros(P), jnp.zeros((P, 1)),
                                incompressible_plane_stress=incompressible_plane_stress, **fib)
    cons = _Constraints(model)

    # hyperparameter bounds, priors and initial values
    lsd_lo, lsd_hi = lengthscale_bounds(dev_box, box_factor)
    lsv_lo, lsv_hi = lengthscale_bounds(vol_box, box_factor)
    span_d = np.maximum(dev_x.max(0) - dev_x.min(0), 1e-6)
    span_v = np.maximum(vol_x.max(0) - vol_x.min(0), 1e-6)
    if aniso_on:
        lsa_lo, lsa_hi = lengthscale_bounds(an_box, box_factor)
        span_a = np.maximum(an_x.max(0) - an_x.min(0), 1e-6)
        centre_a = np.clip(lengthscale_factor * span_a, lsa_lo * np.exp(0.5), lsa_hi * np.exp(-0.5))
    centre_d = np.clip(lengthscale_factor * span_d, lsd_lo * np.exp(0.5), lsd_hi * np.exp(-0.5))
    centre_v = np.clip(lengthscale_factor * span_v, lsv_lo * np.exp(0.5), lsv_hi * np.exp(-0.5))
    sig_centre = float(np.log(amplitude_factor * energy_scale))
    bounds = dict(dev=(jnp.log(lsd_lo), jnp.log(lsd_hi)), vol=(jnp.log(lsv_lo), jnp.log(lsv_hi)))
    if aniso_on:
        bounds["aniso"] = (jnp.log(lsa_lo), jnp.log(lsa_hi))
    ls_keys = (("raw_ls_dev", "dev"), ("raw_ls_vol", "vol")) + ((("raw_ls_aniso", "aniso"),) if aniso_on else ())
    cap = sig_centre + np.log(sigma_cap) if sigma_cap and sigma_cap > 0 else np.inf

    def to_raw(log_l, lo, hi):
        s = jnp.clip((log_l - lo) / (hi - lo), 0.05, 0.95)
        return jnp.log(s / (1 - s))

    def unpack(eta):
        lo_d, hi_d = bounds["dev"]
        lo_v, hi_v = bounds["vol"]
        h = dict(dev_sig=jnp.exp(jnp.minimum(eta["log_sig_dev"], cap)), vol_sig=jnp.exp(jnp.minimum(eta["log_sig_vol"], cap)),
                 dev_ls=jnp.exp(lo_d + (hi_d - lo_d) * jax.nn.sigmoid(eta["raw_ls_dev"])),
                 vol_ls=jnp.exp(lo_v + (hi_v - lo_v) * jax.nn.sigmoid(eta["raw_ls_vol"])))
        if incompressible_plane_stress:   # volumetric GP off: fixed nominal values (the evidence is flat in them)
            h["vol_sig"], h["vol_ls"] = jnp.exp(jnp.asarray(sig_centre)), jnp.asarray(centre_v)
        if aniso_on:
            lo_a, hi_a = bounds["aniso"]
            h["aniso_sig"] = jnp.exp(jnp.minimum(eta["log_sig_aniso"], cap))
            h["aniso_ls"] = jnp.exp(lo_a + (hi_a - lo_a) * jax.nn.sigmoid(eta["raw_ls_aniso"]))
        return h

    eta_init = dict(log_sig_dev=jnp.asarray(sig_centre), log_sig_vol=jnp.asarray(sig_centre),
                    raw_ls_dev=to_raw(jnp.log(centre_d), *bounds["dev"]), raw_ls_vol=to_raw(jnp.log(centre_v), *bounds["vol"]))
    if aniso_on:
        eta_init.update(log_sig_aniso=jnp.asarray(sig_centre), raw_ls_aniso=to_raw(jnp.log(centre_a), *bounds["aniso"]))

    def with_lengthscales(eta_, factor):
        out = dict(eta_)
        for key, comp in ls_keys:
            lo, hi = bounds[comp]
            log_l = lo + (hi - lo) * jax.nn.sigmoid(eta_[key]) + np.log(factor)
            out[key] = to_raw(jnp.clip(log_l, lo + 1e-3 * (hi - lo), hi - 1e-3 * (hi - lo)), lo, hi)
        return out

    starts = [eta_init] + ([with_lengthscales(eta_init, 0.5), with_lengthscales(eta_init, 2.0)] if hyper_restarts else [])

    def log_hyperprior(h):
        lp = 0.0
        if amplitude_prior_scale > 0:
            lp += -0.5 * ((jnp.log(h["dev_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
            if not incompressible_plane_stress:
                lp += -0.5 * ((jnp.log(h["vol_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
            if aniso_on:
                lp += -0.5 * ((jnp.log(h["aniso_sig"]) - sig_centre) / amplitude_prior_scale) ** 2
        if lengthscale_prior_scale > 0:
            lp += -0.5 * (jnp.sum(((jnp.log(h["dev_ls"]) - jnp.log(centre_d)) / lengthscale_prior_scale) ** 2)
                          + jnp.sum(((jnp.log(h["vol_ls"]) - jnp.log(centre_v)) / lengthscale_prior_scale) ** 2))
            if aniso_on:
                lp += -0.5 * jnp.sum(((jnp.log(h["aniso_ls"]) - jnp.log(centre_a)) / lengthscale_prior_scale) ** 2)
        return lp

    # pseudo-observations of every training step, reduced at once to their statistics (the tall H_t is not kept):
    # step_stats[t] = (free block at s = 1, reaction [+ equilibrium] blocks); reaction blocks kept for the CV scores
    force_fns = [jax.jit(lambda f, m=m: basis_force_matrix(model, f, m["cells"], m["n_nodes"], m["dNdX"], m["dA"]))
                 for m in meshes]
    def step_observation(F_t, R_t, sR_t, S_t=None, sS_t=None, e=0):
        return pseudo_observation(force_fns[e](F_t), meshes[e], R_t, sR_t, S_t, sS_t,
                                  global_equilibrium=global_equilibrium)

    step_stats, reaction_obs, reaction_stats = [], [], []
    for t in all_idx:
        obs = step_observation(step_F[t], R_obs[t], sigma_R[t], S_obs[t], sigma_S[t], step_exp[t])
        step_stats.append((gaussian_statistics(obs["free"]),
                           sum(gaussian_statistics(o) for k, o in obs.items() if k not in ("free", "reaction"))))
        reaction_obs.append(obs["reaction"])
        reaction_stats.append(gaussian_statistics(obs["reaction"]))
        del obs

    def set_reaction_sigma(sig):
        """New noise of the loaded-edge reaction rows (T, 2) of every training step (support rows unchanged)."""
        sig = jnp.maximum(jnp.asarray(sig, dtype=jnp.float64), 1e-3)
        for t in all_idx:
            o = reaction_obs[t]
            reaction_obs[t] = Observation(o.H, o.q, jnp.concatenate([sig[t], o.sigma[2:]]))
            reaction_stats[t] = gaussian_statistics(reaction_obs[t])
    if multi:
        log_fn(f"[HSGP] {len(meshes)} experiments pooled: steps per experiment "
               f"{[step_exp.count(e) for e in range(len(meshes))]}, free DOFs {[len(m['free']) for m in meshes]}")
    log_fn(f"[HSGP] residual likelihood: {T} load steps, {'+'.join(str(len(m['free'])) for m in meshes)} free DOFs, P = {P} weights "
           f"(dev {dev_box.num_basis}, vol {vol_box.num_basis}" + (f", aniso {an_box.num_basis}" if aniso_on else "")
           + f"); basis forces assembled in {time.time() - t0:.0f}s"
           + (f"; global equilibrium rows sigma = {global_equilibrium:g} |R_t|" if global_equilibrium > 0 else "")
           + f"; reaction rows per step: {sorted(set(int(o.q.shape[0]) for o in reaction_obs))} "
             f"(2 loaded edges + measured supports)")

    def stats_for(active, s):
        """Statistics of the stacked pseudo-observations of the active steps at free-residual scale s, as arrays for
        the jitted evidence. The free and the other blocks keep separate square-root factors (stacked in the QR below),
        which avoids losing the reaction Gram next to the large free Gram G_f / s^2 when s is small."""
        free = scale_noise(sum(step_stats[t][0] for t in active), s)
        rest = sum(step_stats[t][1] + reaction_stats[t] for t in active)
        tot = free + rest
        return dict(roots=jnp.concatenate([sqrt_factor(free.G), sqrt_factor(rest.G)]), b=tot.b, qWq=tot.qWq,
                    logdetW=tot.logdetW, n_obs=tot.n)

    def posterior_factors(eta, st):
        """theta = M gamma, gamma ~ N(0, I): posterior precision of gamma A = I + M^T H^T W H M = L L^T, from the QR of
        [I; W^1/2 H M] (no squared condition number); v = L^-1 M^T H^T W q."""
        h = unpack(eta)
        M = constrained_transform(model, cons, h, incompressible_plane_stress)
        L = jnp.linalg.qr(jnp.concatenate([jnp.eye(P), st["roots"] @ M]), mode="r").T
        v = jax.scipy.linalg.solve_triangular(L, M.T @ st["b"], lower=True)
        return h, M, L, v

    def neg_log_post(eta, st):
        """log Z = log N(q; 0, H M M^T H^T + W^-1) (evidence of the stacked pseudo-observations) + hyperprior."""
        h, M, L, v = posterior_factors(eta, st)
        log_z = (-0.5 * st["qWq"] + 0.5 * v @ v - jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
                 + 0.5 * st["logdetW"] - 0.5 * st["n_obs"] * jnp.log(2 * jnp.pi))
        return -(log_z + log_hyperprior(h)), log_z

    opt_log = []    # (iterations, converged) of every hyperparameter optimisation (all starts, all CV fits)

    @jax.jit
    def optimise(eta0, st):
        fun = lambda e: neg_log_post(e, st)[0]
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(fun)

        # L-BFGS until converged: |grad| < hyper_gtol or a relative objective change < hyper_ftol, at most hyper_steps.
        # Without the stop, iterating past convergence lets a lengthscale at its bound walk its raw (sigmoid)
        # parameter outwards with a vanishing gradient, the inverse-Hessian estimate blows up and one huge step can
        # throw the hyperparameters past the amplitude cap (zero gradient there: stuck at a much worse evidence).
        def cond(carry):
            return (carry[2] < hyper_steps) & ~carry[4]

        def body(carry):
            e, s, i, v_prev, _ = carry
            val, g = vg(e, state=s)
            conv = ((optax.tree_utils.tree_l2_norm(g) < hyper_gtol)
                    | (jnp.abs(v_prev - val) <= hyper_ftol * jnp.maximum(1.0, jnp.abs(val))))
            u, s_new = opt.update(g, s, e, value=val, grad=g, value_fn=fun)
            e_new = jax.tree_util.tree_map(lambda a, b: jnp.where(conv, a, b), e, optax.apply_updates(e, u))
            return e_new, s_new, i + 1, val, conv
        e, _, n_it, _, conv = jax.lax.while_loop(cond, body, (eta0, opt.init(eta0), 0, jnp.inf, False))
        return e, n_it, conv

    def posterior(eta, st):
        _, M, L, v = posterior_factors(eta, st)
        m_gamma = jax.scipy.linalg.solve_triangular(L.T, v, lower=False)
        L_inv_T = jax.scipy.linalg.solve_triangular(L.T, jnp.eye(P), lower=False)   # Cov(gamma) = L^-T L^-1
        return M @ m_gamma, M @ L_inv_T

    def solve(active, s):
        """MAP-II hyperparameters (multi-start) and the posterior on the active steps at scale s."""
        st = {k: v for k, v in stats_for(active, s).items()}
        best = None
        for e0 in starts:
            cand, n_it, conv = optimise(e0, st)
            val, lz = neg_log_post(cand, st)
            opt_log.append((int(n_it), bool(conv)))
            if np.isfinite(float(val)) and (best is None or float(val) < best[0]):
                best = (float(val), float(lz), cand, int(n_it), bool(conv))
        if best is None:
            return None
        m, Fc = posterior(best[2], st)
        return dict(eta=best[2], log_z=best[1], mean=m, factor=Fc, n_iter=best[3], converged=best[4])

    cv = None
    r_grid = [float(r) for r in (reaction_noise_grid or [])]
    if r_grid and reaction_sigma_fn is None:
        raise ValueError("reaction_noise_grid needs reaction_sigma_fn")
    if r_grid and cv_data is not None:
        raise ValueError("reaction_noise_grid is not supported with cv_data; use the folds")

    def sigma_for(r):
        sig = reaction_sigma_fn(r)
        return np.concatenate([np.asarray(x, float) for x in sig]) if isinstance(sig, (list, tuple)) else np.asarray(sig, float)
    s_fixed = sigma_f_rel
    if sigma_f_rel is None or r_grid:
        K = max(2, int(cv_folds))
        folds = [all_idx[k::K] for k in range(K)]
        if cv_by_experiment and multi:
            folds = [[t for t in all_idx if step_exp[t] == e] for e in range(len(meshes))]
            log_fn(f"[HSGP] cross-validation: leave one experiment out ({len(folds)} folds)")
        if cv_data is not None:      # separate validation steps: fit on all training steps, score these reactions
            v_F = jnp.asarray(cv_data["f3x3"], dtype=jnp.float64)
            v_R = jnp.asarray(cv_data["R"])
            v_sR = jnp.maximum(jnp.asarray(cv_data["sigma_R"]), 1e-3)
            v_lab = [int(x) for x in cv_data.get("labels", range(v_F.shape[0]))]
            if v_F.shape[0] == 0:
                raise ValueError("cv_data has no validation steps")
            v_S, v_sS = cv_data.get("support_R"), cv_data.get("support_sigma")
            v_obs = [step_observation(v_F[v], v_R[v], v_sR[v], None if v_S is None else v_S[v],
                                      None if v_sS is None else v_sS[v])["reaction"] for v in range(v_F.shape[0])]
            folds = []
            log_fn(f"[HSGP] cross-validation: fit on training steps {labels}, score validation steps {v_lab}")

        def cv_score(s):
            """Held-out reactions only: sum of log N(q_R; H_R m, Sigma_R + H_R S H_R^T) over the held-out steps."""
            splits = ([(all_idx, v_obs)] if cv_data is not None else
                      [([t for t in all_idx if t not in hold], [reaction_obs[t] for t in hold]) for hold in folds])
            total = 0.0
            for train, held_out in splits:
                res = solve(train, s)
                if res is None:
                    return -np.inf
                total += sum(log_predictive_density(o, res["mean"], res["factor"]) for o in held_out)
            return total

        def s_scan(tag=""):
            """Held-out scores over the s grid (refined around its optimum); a fixed s: that s only."""
            if s_fixed is not None:
                sc = {float(s_fixed): cv_score(float(s_fixed))}
                log_fn(f"[HSGP] cross-validation{tag}: s = {float(s_fixed):.3g} (fixed): held-out reaction score "
                       f"{sc[float(s_fixed)]:.2f}")
                return sc
            grid = sorted(float(g) for g in (cv_grid if cv_grid is not None else np.logspace(-5, 0, 11)))
            sc = {}
            for g in grid:
                sc[g] = cv_score(g)
                log_fn(f"[HSGP] cross-validation{tag}: s = {g:.3g} (sigma_free,t = s |R_t|): held-out reaction score {sc[g]:.2f}")
            gb = max(sc, key=sc.get)
            for fct in (10 ** -0.25, 10 ** 0.25, 10 ** -0.125, 10 ** 0.125):     # refine around the grid optimum
                g = gb * fct
                sc[g] = cv_score(g)
                log_fn(f"[HSGP] cross-validation{tag}: s = {g:.3g}: held-out reaction score {sc[g]:.2f}")
                gb = max(sc, key=sc.get)
            return sc

        r_curve, r_best = None, None
        if r_grid:
            r_scores = {}
            for r in r_grid:
                set_reaction_sigma(sigma_for(r))
                r_scores[r] = s_scan(f" [reaction noise r = {r:g}]")
                b = max(r_scores[r], key=r_scores[r].get)
                log_fn(f"[HSGP] cross-validation: reaction noise r = {r:g}: best s = {b:.3g}, held-out reaction score "
                       f"{r_scores[r][b]:.2f}")
            r_best = max(r_grid, key=lambda r: max(r_scores[r].values()))
            set_reaction_sigma(sigma_for(r_best))
            scores = r_scores[r_best]
            r_curve = {f"{r:g}": {f"{k:.6g}": v for k, v in sorted(r_scores[r].items())} for r in r_grid}
            log_fn(f"[HSGP] cross-validation: selected reaction noise r = {r_best:g}")
        else:
            scores = s_scan()
        g_best = max(scores, key=scores.get)
        s_best_score = g_best
        if cv_tol > 0:   # plateau rule: the smallest s within cv_tol nats of the best held-out score
            g_best = min(g for g, v in scores.items() if v >= scores[s_best_score] - cv_tol)
            log_fn(f"[HSGP] cross-validation: best s = {s_best_score:.3g}; smallest s within {cv_tol:g} nats: {g_best:.3g}")
        sigma_f_rel = g_best
        cv = dict(method="cross-validation",
                  holdout="validation" if cv_data is not None else ("experiments" if cv_by_experiment and multi else "folds"),
                  reaction_noise_rel=r_best, reaction_noise_curve=r_curve, s_fixed=s_fixed is not None,
                  validation_steps=v_lab if cv_data is not None else None, s=g_best, s_best_score=s_best_score, tol=float(cv_tol),
                  folds=[[labels[t] for t in f] for f in folds], curve={f"{k:.6g}": v for k, v in sorted(scores.items())})
        log_fn(f"[HSGP] cross-validation: selected s = {g_best:.3g}")

    res = solve(all_idx, float(sigma_f_rel))
    if res is None:
        raise RuntimeError("HSGP residual fit: no finite evidence for any hyperparameter start")
    n_unconv = sum(not c for _, c in opt_log)
    log_fn(f"[HSGP] hyperparameter L-BFGS: final fit {res['n_iter']} iterations "
           f"({'converged' if res['converged'] else f'NOT converged in {hyper_steps}'}); "
           f"all {len(opt_log)} optimisations: median {int(np.median([n for n, _ in opt_log]))} iterations, "
           f"{n_unconv} not converged")
    h = unpack(res["eta"])
    hyper = {k: np.asarray(v) for k, v in h.items()}
    hyper["s"] = np.asarray(float(sigma_f_rel))
    if incompressible_plane_stress:
        hyper["vol_sig"] = np.asarray(0.0)          # volumetric GP switched off (theta_vol = 0)
    rec = dict(iteration=0, log_evidence=res["log_z"], s=float(sigma_f_rel), dev_sig=float(hyper["dev_sig"]),
               vol_sig=float(hyper["vol_sig"]), dev_ls=np.asarray(h["dev_ls"]).tolist(),
               vol_ls=np.asarray(h["vol_ls"]).tolist(), time=time.time() - t0)
    if aniso_on:
        rec.update(aniso_sig=float(hyper["aniso_sig"]), aniso_ls=np.asarray(h["aniso_ls"]).tolist())
    if verbose:
        log_fn(f"[HSGP] posterior: logZ {rec['log_evidence']:.2f} | sigma_free = {sigma_f_rel:.3g} x |R_t| | "
               f"sig dev/vol {rec['dev_sig']:.3g}/{rec['vol_sig']:.3g} | l dev {np.round(rec['dev_ls'], 3)} "
               f"vol {np.round(rec['vol_ls'], 3)}"
               + (f" | aniso sig {rec['aniso_sig']:.3g} l {np.round(rec['aniso_ls'], 3)}" if aniso_on else "")
               + f" ({rec['time']:.0f}s)")
    info = dict(likelihood="residual_exact", sigma_f_rel=float(sigma_f_rel), force_noise_cv=cv, converged=bool(res["converged"]),
                hyper_iterations=int(res["n_iter"]), hyper_unconverged=int(n_unconv), hyper_optimisations=len(opt_log),
                global_equilibrium=float(global_equilibrium), support_reaction_rows=int(reaction_obs[0].q.shape[0]) - 2,
                outer_iterations=1, log_evidence=res["log_z"], valid_load_steps=labels, truncated=False,
                sigma_cap=sigma_cap, hyper_restarts=hyper_restarts, box_factor=box_factor,
                num_basis_dev=num_basis_dev, num_basis_vol=num_basis_vol,
                dev_ls_bounds=[np.asarray(lsd_lo).tolist(), np.asarray(lsd_hi).tolist()],
                vol_ls_bounds=[np.asarray(lsv_lo).tolist(), np.asarray(lsv_hi).tolist()],
                anisotropic=aniso_on, num_basis_aniso=int(an_box.num_basis) if aniso_on else 0,
                aniso_ls_bounds=[np.asarray(lsa_lo).tolist(), np.asarray(lsa_hi).tolist()] if aniso_on else None,
                lengthscale_prior_centre=dict(dev=centre_d.tolist(), vol=centre_v.tolist()),
                amplitude_prior_centre=float(np.exp(sig_centre)), fit_time=time.time() - t0)
    model = HSGPHyperelasticity(dev_box, vol_box, res["mean"], res["factor"], hyper=hyper,
                                noise=dict(sigma_f_rel=float(sigma_f_rel)), info=info,
                                incompressible_plane_stress=incompressible_plane_stress, **fib)
    return model, [rec]

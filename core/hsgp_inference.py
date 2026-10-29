"""
Exact Gaussian posterior of the HSGP strain-energy model under the residual likelihood (FE-VFM residual evaluated at
the measured displacements).

Model, per training load step t (u = measured displacements, so the residual is exactly linear in the weights):
  free DOFs:   B_free,t theta ~ N(0, sigma_free,t^2 I),   sigma_free,t = s |R_t|
  reactions:   B_R,t theta    ~ N(R_t, Sigma_R,t)         (load-cell noise, diagonal)
B_t = internal nodal forces of each basis function at the measured state (assembled once). With the constrained GP
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
The EIV likelihood and the other noise models are in archive/eiv/hsgp_inference.py.
"""
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .features import make_feature_extractor
from .hsgp_basis import Box, make_box, se_spectral_density
from .hsgp_model import HSGPHyperelasticity, DEV_REFERENCE, VOL_REFERENCE, ANISO_REFERENCE
from .loss_function import build_eiv_indices

jax.config.update("jax_enable_x64", True)

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
    idx = build_eiv_indices(node_type, cells_np)
    rx, ry = idx["dof_rx"], idx["dof_ry"]

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
             step_labels=None, hyper_steps: int = 300, verbose: bool = True, log_fn=print):
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
      labels): each s is fitted on ALL training steps and scored on these steps' reactions only (cv_folds ignored);
      they never enter the fit.
    fibers: (a0, a1) in-plane fibre vectors (a1 may be None) for the anisotropic GP over (I4_bar, I6_bar), with
    num_basis_aniso functions per fibre invariant (default num_basis_dev); None: isotropic.
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
    if multi and cv_data is not None:
        raise ValueError("cv_data (validation-step selection of s) is not supported for several experiments; use the folds")
    meshes = []                      # per experiment: mesh arrays and the DOF index sets
    for c, nt, dn, da in zip(cells, node_type, dNdX, dA):
        nt, c = np.asarray(nt), np.asarray(c)
        ix = build_eiv_indices(nt, c)
        meshes.append(dict(cells=jnp.asarray(c), n_nodes=nt.shape[0], dNdX=jnp.asarray(dn), dA=jnp.asarray(da),
                           free=ix["dof_free"], rx=ix["dof_rx"], ry=ix["dof_ry"]))
    step_exp, step_F, labels = [], [], []        # pooled steps: experiment index, F (C_e, 3, 3), label
    for e, Fe in enumerate(f3x3_steps):
        Fe = jnp.asarray(Fe, dtype=jnp.float64)
        lab_e = list(step_labels[e]) if step_labels is not None and step_labels[e] is not None else list(range(Fe.shape[0]))
        for t in range(Fe.shape[0]):
            step_exp.append(e)
            step_F.append(Fe[t])
            labels.append((lab_e[t] if isinstance(lab_e[t], str) else f"{e}:{int(lab_e[t])}") if multi else int(lab_e[t]))
    R_obs = jnp.concatenate([jnp.asarray(r) for r in R_obs])
    sigma_R = jnp.maximum(jnp.concatenate([jnp.asarray(s) for s in sigma_R]), 1e-3)
    # single experiment: the names used below
    free, rx, ry, n_free = meshes[0]["free"], meshes[0]["rx"], meshes[0]["ry"], len(meshes[0]["free"])
    T = len(step_F)
    all_idx = list(range(T))

    # basis boxes from the training features (+ reference)
    aniso_on = fibers is not None
    if aniso_on:
        from .features import AnisotropicFeatureExtractor
        extractor = AnisotropicFeatureExtractor(jnp.asarray(fibers[0], dtype=jnp.float64),
                                                a1=None if fibers[1] is None else jnp.asarray(fibers[1], dtype=jnp.float64))
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

    # per-step statistics at unit s (computed once): free Gram B_f^T B_f / |R_t|^2, reaction Gram and right-hand side
    force_fns = [jax.jit(lambda f, m=m: basis_force_matrix(model, f, m["cells"], m["n_nodes"], m["dNdX"], m["dA"]))
                 for m in meshes]
    force_fn = force_fns[0]
    B_R_steps, step_stats = [], []
    for t in all_idx:
        m = meshes[step_exp[t]]
        B = force_fns[step_exp[t]](step_F[t])
        Bf, BR = B[m["free"]], jnp.stack([B[m["rx"]].sum(0), B[m["ry"]].sum(0)])
        nf = len(m["free"])
        r2 = jnp.maximum(jnp.sum(R_obs[t] ** 2), 1e-30)
        w = 1.0 / sigma_R[t] ** 2
        step_stats.append(dict(Gf=Bf.T @ Bf / r2, logdetW_f=-nf * jnp.log(r2), GR=BR.T @ (w[:, None] * BR),
                               bR=BR.T @ (w * R_obs[t]), yWy=jnp.sum(w * R_obs[t] ** 2), logdetW_R=jnp.sum(jnp.log(w)),
                               n_free=float(nf)))
        B_R_steps.append(BR)
    if multi:
        log_fn(f"[HSGP] {len(meshes)} experiments pooled: steps per experiment "
               f"{[step_exp.count(e) for e in range(len(meshes))]}, free DOFs {[len(m['free']) for m in meshes]}")
    log_fn(f"[HSGP] residual likelihood: {T} load steps, {'+'.join(str(len(m['free'])) for m in meshes)} free DOFs, P = {P} weights "
           f"(dev {dev_box.num_basis}, vol {vol_box.num_basis}" + (f", aniso {an_box.num_basis}" if aniso_on else "")
           + f"); basis forces assembled in {time.time() - t0:.0f}s")

    def sqrt_factor(G):
        d, U = jnp.linalg.eigh(0.5 * (G + G.T))
        return jnp.sqrt(jnp.maximum(d, 0.0))[:, None] * U.T

    def stats_for(active, s):
        """Summed statistics of the active steps at free-residual scale s, with square-root factors of the Grams."""
        st = {k: sum(step_stats[t][k] for t in active) for k in step_stats[0]}
        nf_tot = st.pop("n_free")                          # free DOFs summed over the active steps (per mesh)
        st["Gf"], st["logdetW_f"] = st["Gf"] / s ** 2, st["logdetW_f"] - nf_tot * np.log(s ** 2)
        st["Rf"], st["RR"] = sqrt_factor(st["Gf"]), sqrt_factor(st["GR"])
        st["n_obs"] = float(nf_tot + 2 * len(active))
        return st

    def posterior_factors(eta, st):
        h = unpack(eta)
        M = constrained_transform(model, cons, h, incompressible_plane_stress)
        # A = I + M^T (G_f + G_R) M = R^T R from the QR of [I; R_f M; R_R M] (no squared condition number)
        C = jnp.concatenate([jnp.eye(P), st["Rf"] @ M, st["RR"] @ M])
        L = jnp.linalg.qr(C, mode="r").T
        v = jax.scipy.linalg.solve_triangular(L, M.T @ st["bR"], lower=True)   # free targets are zero
        return h, M, L, v

    def neg_log_post(eta, st):
        h, M, L, v = posterior_factors(eta, st)
        log_z = (-0.5 * st["yWy"] + 0.5 * v @ v - jnp.sum(jnp.log(jnp.abs(jnp.diag(L))))
                 + 0.5 * st["logdetW_f"] + 0.5 * st["logdetW_R"] - 0.5 * st["n_obs"] * jnp.log(2 * jnp.pi))
        return -(log_z + log_hyperprior(h)), log_z

    @jax.jit
    def optimise(eta0, st):
        fun = lambda e: neg_log_post(e, st)[0]
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(fun)

        def body(_, carry):
            e, s = carry
            val, g = vg(e, state=s)
            u, s = opt.update(g, s, e, value=val, grad=g, value_fn=fun)
            return optax.apply_updates(e, u), s
        return jax.lax.fori_loop(0, hyper_steps, body, (eta0, opt.init(eta0)))[0]

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
            cand = optimise(e0, st)
            val, lz = neg_log_post(cand, st)
            if np.isfinite(float(val)) and (best is None or float(val) < best[0]):
                best = (float(val), float(lz), cand)
        if best is None:
            return None
        m, Fc = posterior(best[2], st)
        return dict(eta=best[2], log_z=best[1], mean=m, factor=Fc)

    def heldout_score(m, Fc, t, BR=None, R=None, sR=None):
        """log N(R_t; B_R m, Sigma_R,t + B_R S B_R^T), S = Fc Fc^T (held-out reactions only); BR/R/sR: a step
        outside the training set."""
        BR, R, sR = (B_R_steps[t], R_obs[t], sigma_R[t]) if BR is None else (BR, R, sR)
        UR = BR @ Fc
        C = jnp.diag(sR ** 2) + UR @ UR.T
        d = R - BR @ m
        return float(-0.5 * (d @ jnp.linalg.solve(C, d) + jnp.linalg.slogdet(C)[1] + 2 * jnp.log(2 * jnp.pi)))

    cv = None
    if sigma_f_rel is None:
        K = max(2, int(cv_folds))
        folds = [all_idx[k::K] for k in range(K)]
        if cv_data is not None:      # separate validation steps: fit on all training steps, score these reactions
            v_F = jnp.asarray(cv_data["f3x3"], dtype=jnp.float64)
            v_R = jnp.asarray(cv_data["R"])
            v_sR = jnp.maximum(jnp.asarray(cv_data["sigma_R"]), 1e-3)
            v_lab = [int(x) for x in cv_data.get("labels", range(v_F.shape[0]))]
            if v_F.shape[0] == 0:
                raise ValueError("cv_data has no validation steps")
            v_BR = []
            for v in range(v_F.shape[0]):
                B = force_fn(v_F[v])
                v_BR.append(jnp.stack([B[rx].sum(0), B[ry].sum(0)]))
            folds = []
            log_fn(f"[HSGP] cross-validation: fit on training steps {labels}, score validation steps {v_lab}")

        def cv_score(s):
            if cv_data is not None:
                res = solve(all_idx, s)
                if res is None:
                    return -np.inf
                return sum(heldout_score(res["mean"], res["factor"], None, v_BR[v], v_R[v], v_sR[v])
                           for v in range(len(v_BR)))
            total = 0.0
            for hold in folds:
                res = solve([t for t in all_idx if t not in hold], s)
                if res is None:
                    return -np.inf
                total += sum(heldout_score(res["mean"], res["factor"], t) for t in hold)
            return total

        grid = sorted(float(g) for g in (cv_grid if cv_grid is not None else np.logspace(-5, 0, 11)))
        scores = {}
        for g in grid:
            scores[g] = cv_score(g)
            log_fn(f"[HSGP] cross-validation: s = {g:.3g} (sigma_free,t = s |R_t|): held-out reaction score {scores[g]:.2f}")
        g_best = max(scores, key=scores.get)
        for fct in (10 ** -0.25, 10 ** 0.25, 10 ** -0.125, 10 ** 0.125):     # refine around the grid optimum
            g = g_best * fct
            scores[g] = cv_score(g)
            log_fn(f"[HSGP] cross-validation: s = {g:.3g}: held-out reaction score {scores[g]:.2f}")
            g_best = max(scores, key=scores.get)
        s_best_score = g_best
        if cv_tol > 0:   # plateau rule: the smallest s within cv_tol nats of the best held-out score
            g_best = min(g for g, v in scores.items() if v >= scores[s_best_score] - cv_tol)
            log_fn(f"[HSGP] cross-validation: best s = {s_best_score:.3g}; smallest s within {cv_tol:g} nats: {g_best:.3g}")
        sigma_f_rel = g_best
        cv = dict(method="cross-validation", holdout="validation" if cv_data is not None else "folds",
                  validation_steps=v_lab if cv_data is not None else None, s=g_best, s_best_score=s_best_score, tol=float(cv_tol),
                  folds=[[labels[t] for t in f] for f in folds], curve={f"{k:.6g}": v for k, v in sorted(scores.items())})
        log_fn(f"[HSGP] cross-validation: selected s = {g_best:.3g}")

    res = solve(all_idx, float(sigma_f_rel))
    if res is None:
        raise RuntimeError("HSGP residual fit: no finite evidence for any hyperparameter start")
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
    info = dict(likelihood="residual_exact", sigma_f_rel=float(sigma_f_rel), force_noise_cv=cv, converged=True,
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

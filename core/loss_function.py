import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, Any

# Enforce mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from .utils import deformation_gradient_element, transformation_jacobian, fto3x3
from .model import SparseHyperelasticityGP


def build_eiv_indices(node_type: np.ndarray, cells: np.ndarray) -> dict:
    """
    Static (numpy) DOF bookkeeping for the errors-in-variables likelihood (displacement control).
    Flat DOF index = 2 * node + direction.
      dof_free:  (m,) DOFs carrying displacement measurement noise (not Dirichlet).
      dof_rx/ry: DOFs whose internal forces sum to the measured x/y reaction.
      k_rows/k_cols: (C*36,) scatter indices of element stiffness blocks into the global matrix.
    """
    node_type = np.asarray(node_type)
    cells = np.asarray(cells)
    is_fix_x, is_fix_y = node_type[:, 1] == 1, node_type[:, 2] == 1
    is_loaded_x, is_loaded_y = node_type[:, 3] == 1, node_type[:, 4] == 1
    free_mask = np.stack([~(is_fix_x | is_loaded_x), ~(is_fix_y | is_loaded_y)], axis=-1).reshape(-1)
    elem_dofs = (2 * cells[:, :, None] + np.arange(2)[None, None, :]).reshape(cells.shape[0], 6)  # (C, 6)
    return dict(
        dof_free=np.where(free_mask)[0],
        dof_rx=2 * np.where(is_loaded_x)[0],
        dof_ry=2 * np.where(is_loaded_y)[0] + 1,
        k_rows=np.repeat(elem_dofs[:, :, None], 6, axis=2).reshape(-1),
        k_cols=np.repeat(elem_dofs[:, None, :], 6, axis=1).reshape(-1),
    )


def assemble_internal_force_and_tangent(psi_fn: Any, f3x3_cells: jnp.ndarray, cells: jnp.ndarray, n_nodes: int,
                                        dNdX: jnp.ndarray, dA: jnp.ndarray, eiv: dict):
    """
    Internal nodal forces and tangent stiffness K = d f_int / d u for one load step and energy psi(F).
    Out-of-plane entries of F (F33 = 1 or the dataset's lambda3) are held fixed, consistent with the residual.
    f3x3_cells: (C, 3, 3).  Returns f_flat: (2n,), K: (2n, 2n) with flat DOF index 2 * node + direction.
    """
    def psi_inplane(f2, f3):
        return psi_fn(f3.at[:2, :2].set(f2))

    piola_2d = jax.grad(psi_inplane)                     # (2,2)
    tangent_2d = jax.jacfwd(piola_2d)                    # (2,2,2,2) = dP_ij / dF_kl
    f2_cells = f3x3_cells[:, :2, :2]
    P = jax.vmap(piola_2d)(f2_cells, f3x3_cells)          # (C,2,2)
    A = jax.vmap(tangent_2d)(f2_cells, f3x3_cells)        # (C,2,2,2,2)

    f_int_cell = jnp.einsum("cij,caj->cai", P, dNdX) * dA[:, None, None]                        # (C,3,2)
    f_flat = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell).reshape(-1)  # (2n,)
    k_cell = jnp.einsum("caj,cijkl,cbl->caibk", dNdX, A, dNdX) * dA[:, None, None, None, None]  # (C,3,2,3,2)
    K = jnp.zeros((2 * n_nodes, 2 * n_nodes), dtype=jnp.float64).at[eiv["k_rows"], eiv["k_cols"]].add(k_cell.reshape(-1))
    return f_flat, K


def internal_force(psi_fn: Any, f3x3_cells: jnp.ndarray, cells: jnp.ndarray, n_nodes: int,
                   dNdX: jnp.ndarray, dA: jnp.ndarray) -> jnp.ndarray:
    """Internal nodal forces f_int (2n,) for one load step, out-of-plane F entries held fixed."""
    P = jax.vmap(jax.grad(lambda f2, f3: psi_fn(f3.at[:2, :2].set(f2))))(f3x3_cells[:, :2, :2], f3x3_cells)
    f_int_cell = jnp.einsum("cij,caj->cai", P, dNdX) * dA[:, None, None]
    return jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell).reshape(-1)


def eiv_linearisation(mean_psi_fn: Any, f3x3_cells: jnp.ndarray, cells: jnp.ndarray, n_nodes: int,
                      dNdX: jnp.ndarray, dA: jnp.ndarray, eiv: dict):
    """
    Tangent blocks of the posterior-mean energy at the observed state of one load step.
    Returns K_ff: (m, m) free-free stiffness and K_rf: (2, m) sensitivity of the x/y reactions to free DOFs.
    """
    _, K = assemble_internal_force_and_tangent(mean_psi_fn, f3x3_cells, cells, n_nodes, dNdX, dA, eiv)
    dof_free = eiv["dof_free"]
    K_rf = jnp.stack([K[eiv["dof_rx"]][:, dof_free].sum(axis=0), K[eiv["dof_ry"]][:, dof_free].sum(axis=0)])
    return K[dof_free][:, dof_free], K_rf


def eiv_force_equivalent_sigma(psi_fn: Any, f3x3: jnp.ndarray, cells: jnp.ndarray, n_nodes: int,
                               dNdX: jnp.ndarray, dA: jnp.ndarray, eiv: dict, sigma_u_dof: jnp.ndarray):
    """
    Nodal force-residual noise implied by displacement noise: Cov(r_f) = K_ff diag(sigma_u^2) K_ff^T.
    Returns the per-free-DOF std averaged (in variance) over load steps f3x3: (T, C, 3, 3) -> (m,).
    Lets downstream tools that expect a force-residual sigma_free keep working under the EIV likelihood.
    """
    dof_free = eiv["dof_free"]

    def step_var(f_step):
        _, K = assemble_internal_force_and_tangent(psi_fn, f_step, cells, n_nodes, dNdX, dA, eiv)
        K_ff = K[dof_free][:, dof_free]
        return jnp.sum(K_ff**2 * sigma_u_dof[None, :]**2, axis=1)

    return jnp.sqrt(jnp.mean(jax.lax.map(step_var, f3x3), axis=0))


def ellipticity_penalty(psi_fn: Any, f3x3: jnp.ndarray, n_dirs: int = 8):
    """
    Strong-ellipticity (material stability) prior on an energy psi at the given deformation states.
    For each state and direction n the in-plane acoustic tensor Q_ik(n) = A_ijkl n_j n_l (A = dP/dF, out-of-plane
    F entries held fixed) must be positive definite. Its smallest eigenvalue is normalised by the mean stiffness
    (so the prior does not depend on the stress scale) and only negative values are penalised:
        penalty = sum_{states, n} relu(-lambda_min(Q) / s_ref)^2,   zero for any stable material.
    f3x3: (..., 3, 3). Returns (penalty, fraction of (state, direction) pairs that violate ellipticity).
    """
    def tangent(f3):
        return jax.jacfwd(jax.grad(lambda f2: psi_fn(f3.at[:2, :2].set(f2))))(f3[:2, :2])   # (2,2,2,2)

    A = jax.vmap(tangent)(f3x3.reshape(-1, 3, 3))
    theta = jnp.arange(n_dirs) * jnp.pi / n_dirs
    n = jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)                          # (D, 2)
    Q = jnp.einsum("eijkl,dj,dl->edik", A, n, n)
    a, c = Q[..., 0, 0], Q[..., 1, 1]
    b = 0.5 * (Q[..., 0, 1] + Q[..., 1, 0])
    lam_min = 0.5 * (a + c) - jnp.sqrt(0.25 * (a - c) ** 2 + b ** 2 + 1e-30)
    s_ref = jax.lax.stop_gradient(jnp.abs(jnp.mean(0.5 * (a + c))) + 1e-12)
    r = lam_min / s_ref
    return jnp.sum(jax.nn.relu(-r) ** 2), jnp.mean(r < 0)


def damped_newton_step(K: jnp.ndarray, r: jnp.ndarray, damping: float = 0.0) -> jnp.ndarray:
    """
    eps = K^{-1} r, or with damping > 0 the Tikhonov-damped step eps = (K^T K + mu^2 I)^{-1} K^T r,
    mu = damping * rms singular value of K (scale-invariant). The undamped step has poles where K is singular,
    i.e. where the mean energy loses stability, which makes the likelihood an infinite barrier between unstable
    and stable materials; damping bounds the step there and leaves it unchanged where K is well conditioned.
    """
    if damping <= 0:
        return jnp.linalg.solve(K, r)
    KtK = K.T @ K
    mu2 = (damping ** 2) * jnp.trace(KtK) / K.shape[0]
    return jnp.linalg.solve(KtK + mu2 * jnp.eye(K.shape[0], dtype=K.dtype), K.T @ r)


def eiv_log_likelihood(eps_hat: jnp.ndarray, reaction_res: jnp.ndarray, dof_dir: np.ndarray,
                       sigma_fix_x: jnp.ndarray, sigma_fix_y: jnp.ndarray, reaction_loss_weight: float = 1.0,
                       nodal_noise: bool = False, sigma_global: jnp.ndarray = None, prior_dof: float = 4.0):
    """
    Log-likelihood of one GP path under the errors-in-variables model, with the displacement noise
    variance integrated out analytically (no noise parameter has to chase the residual scale):
      constant noise, Jeffreys prior p(sigma^2) ~ 1/sigma^2, per direction d with N_d residuals:
          log p = -N_d/2 * (log(2 pi s_d / N_d) + 1),    s_d = sum eps_hat^2
        (equal to the Gaussian log-likelihood at the profile estimate sigma^2 = s_d / N_d).
      per-DOF noise, conjugate hierarchical prior sigma_i^2 ~ InvGamma(a, a sigma_global^2), a = prior_dof / 2:
          log p_i = log Student-t marginal of the T residuals of DOF i (bounded gradient in log sigma_global).
    Reaction residuals keep a Gaussian likelihood with the load-cell noise sigma_fix.
    eps_hat: (T, m), reaction_res: (T, 2), dof_dir: (m,) 0 for x / 1 for y.
    """
    T = eps_hat.shape[0]
    if nodal_noise:
        a = 0.5 * prior_dof
        b = a * sigma_global**2
        s_i = jnp.sum(eps_hat**2, axis=0)                                                       # (m,)
        ll_i = (jax.scipy.special.gammaln(a + 0.5 * T) - jax.scipy.special.gammaln(a) + a * jnp.log(b)
                - (a + 0.5 * T) * jnp.log(b + 0.5 * s_i) - 0.5 * T * jnp.log(2 * jnp.pi))
        free_x_ll = jnp.sum(jnp.where(dof_dir == 0, ll_i, 0.0))
        free_y_ll = jnp.sum(jnp.where(dof_dir == 1, ll_i, 0.0))
    else:
        def profile_ll(mask):
            n_d = T * int(np.sum(mask))
            s_d = jnp.sum(jnp.where(mask[None, :], eps_hat**2, 0.0))
            return -0.5 * n_d * (jnp.log(2 * jnp.pi * s_d / n_d) + 1.0)
        free_x_ll = profile_ll(dof_dir == 0)
        free_y_ll = profile_ll(dof_dir == 1)

    sx = jnp.maximum(sigma_fix_x, 1e-3)
    sy = jnp.maximum(sigma_fix_y, 1e-3)
    fix_x_ll = reaction_loss_weight * jnp.sum(-0.5 * reaction_res[:, 0]**2 / sx**2 - 0.5 * jnp.log(2 * jnp.pi * sx**2))
    fix_y_ll = reaction_loss_weight * jnp.sum(-0.5 * reaction_res[:, 1]**2 / sy**2 - 0.5 * jnp.log(2 * jnp.pi * sy**2))

    total = free_x_ll + free_y_ll + fix_x_ll + fix_y_ll
    return total, (free_x_ll, free_y_ll, fix_x_ll, fix_y_ll, jnp.sum(eps_hat**2), jnp.sum(reaction_res**2))


def eiv_noise_estimate(eps_hat: jnp.ndarray, dof_dir: np.ndarray, nodal_noise: bool = False,
                       sigma_global: jnp.ndarray = None, prior_dof: float = 4.0) -> jnp.ndarray:
    """
    Point estimate of the displacement noise std per free DOF from residuals eps_hat: (T, m):
    profile estimate sqrt(mean eps^2) per direction, or the InvGamma posterior mode per DOF.
    """
    T = eps_hat.shape[0]
    s_i = jnp.sum(eps_hat**2, axis=0)
    if nodal_noise:
        a = 0.5 * prior_dof
        return jnp.sqrt((a * sigma_global**2 + 0.5 * s_i) / (a + 0.5 * T + 1.0))
    sig = [jnp.sqrt(jnp.sum(jnp.where(dof_dir == d, s_i, 0.0)) / (T * np.sum(dof_dir == d))) for d in (0, 1)]
    return jnp.where(dof_dir == 0, sig[0], sig[1])


def noise_log_prior(params: Any, is_free_x: np.ndarray, is_free_y: np.ndarray, prior_dof: float) -> jnp.ndarray:
    """
    Hierarchical prior for explicitly learned per-node noise ('residual' likelihood, nodal mode):
        sigma_i^2 ~ InvGamma(a, a * sigma_global^2),  a = prior_dof / 2,
    the same conjugate prior that the 'eiv' likelihood integrates out. Nodes are shrunk towards the learned
    shared level instead of being fitted from a handful of load steps. Density is taken in log(sigma^2) space
    to match the log parameterisation. Returns 0 for scalar noise or prior_dof <= 0.
    """
    sx, sy = params.sigma_free_x, params.sigma_free_y
    if prior_dof is None or prior_dof <= 0 or sx is None or jnp.ndim(sx) == 0 or params.sigma_global is None:
        return jnp.zeros((), dtype=jnp.float64)
    a = 0.5 * prior_dof
    b = a * params.sigma_global**2
    var = jnp.concatenate([sx[is_free_x], sy[is_free_y]])**2
    return jnp.sum(a * jnp.log(b) - jax.scipy.special.gammaln(a) - a * jnp.log(var) - b / var)


def total_stochastic_loss(p: Any, model: SparseHyperelasticityGP, f3x3: jnp.ndarray, cells: jnp.ndarray,
                          n_nodes: int, f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, dNdX: jnp.ndarray,
                          dA: jnp.ndarray, key: jnp.ndarray, n_s: int, normalize_ell: int = 0,
                          vfm_mode: str = "linear_triangle", V_basis: jnp.ndarray = None,
                          control_mode: str = "force", loads: jnp.ndarray = None,
                          reaction_loss_weight: float = 1.0, likelihood: str = "residual",
                          eiv: dict = None, noise_prior_dof: float = 4.0, eiv_damping: float = 0.0,
                          stability_weight: float = 0.0, stability_dirs: int = 8) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    """
    Computes the variational stochastic VFM loss and KL divergence ELBO objective.
    Strictly preserves functional purity without mutating stateful class instance attributes.
    Supports vfm_mode: 'linear_triangle', 'global_vf', or 'mix'.
    Supports control_mode: 'force' or 'displacement'.
    likelihood: 'residual' (iid Gaussian nodal force residuals) or 'eiv' (errors-in-variables,
    displacement-space likelihood; displacement control only, requires `eiv` from build_eiv_indices).
    stability_weight > 0 adds the strong-ellipticity prior on the posterior-mean energy (ellipticity_penalty).
    Returns aux = (ell, kl, free_x, free_y, fix_x, fix_y, sum_free_loss, sum_fix_loss, noise_log_prior,
                   stability_penalty, fraction_unstable).
    """
    params = model.load_params(p)
    gpweight = model.precompute_weights_from_loaded(params)
    sigma_fix_x = params.sigma_fix_x
    sigma_fix_y = params.sigma_fix_y

    main_key = jr.split(key, n_s + 1)
    subkey = main_key[1:]

    node_type_np = np.asarray(node_type)
    is_fix_x, is_fix_y = node_type_np[:, 1] == 1, node_type_np[:, 2] == 1
    if control_mode == "displacement":
        is_free_x = ~(is_fix_x | (node_type_np[:, 3] == 1))
        is_free_y = ~(is_fix_y | (node_type_np[:, 4] == 1))
    else:
        is_free_x, is_free_y = ~is_fix_x, ~is_fix_y
    nodal_noise = params.sigma_free_x is not None and jnp.ndim(params.sigma_free_x) > 0
    log_prior = noise_log_prior(params, is_free_x, is_free_y, noise_prior_dof) if likelihood != "eiv" else jnp.zeros(())
    if stability_weight > 0:
        stab_pen, frac_unstable = ellipticity_penalty(lambda f: model.psi_det(f, params=params, weights=gpweight), f3x3, stability_dirs)
    else:
        stab_pen, frac_unstable = jnp.zeros(()), jnp.zeros(())

    if likelihood == "eiv":
        dof_free = eiv["dof_free"]
        dof_dir = dof_free % 2

        # Errors-in-variables: with u_obs = z + eps on the free DOFs, linearising equilibrium f_free(z) = 0 at
        # u_obs gives one Gauss-Newton step eps_hat = K_ff^{-1} r_f, and the reaction at the corrected state is
        # R(u_obs) - K_rf eps_hat. Both are invariant to rescaling psi, so the free-DOF term carries no
        # information on the stress magnitude (equilibrium cannot identify it) and the measured reaction sets it.
        # The tangent is the posterior-mean one (delta-method noise propagation): per-path tangents of unstable
        # GP samples are near-singular, which makes E[eps_hat^2] heavy-tailed and the ELBO estimator unusable.
        mean_psi = lambda f: model.psi_det(f, params=params, weights=gpweight)
        # Batched over load steps and MC samples (one fused GPU computation, as in the residual likelihood):
        # only the mean energy is differentiated twice; sampled paths need first derivatives only.
        K_ff, K_rf = jax.vmap(lambda f_step: eiv_linearisation(mean_psi, f_step, cells, n_nodes, dNdX, dA, eiv))(f3x3)

        def sample_forces(k):
            psi_fn = model.get_path_psi_fn(k, params=params, weights=gpweight)
            return jax.vmap(lambda f_step: internal_force(psi_fn, f_step, cells, n_nodes, dNdX, dA))(f3x3)

        f_int = jax.vmap(sample_forces)(subkey)                                        # (S, T, 2n)
        eps_hat = jax.vmap(lambda K_t, r_t: damped_newton_step(K_t, r_t.T, eiv_damping).T, in_axes=(0, 1), out_axes=1)(
            K_ff, f_int[..., dof_free])                                                # (S, T, m)
        R_obs = jnp.stack([f_int[..., eiv["dof_rx"]].sum(-1), f_int[..., eiv["dof_ry"]].sum(-1)], axis=-1)  # (S, T, 2)
        reaction_res = R_obs - jnp.einsum("tim,stm->sti", K_rf, eps_hat) - loads[None, :, :2]

        ell_, terms = jax.vmap(lambda e, r: eiv_log_likelihood(
            e, r, dof_dir, sigma_fix_x, sigma_fix_y, reaction_loss_weight,
            nodal_noise=nodal_noise, sigma_global=params.sigma_global, prior_dof=noise_prior_dof))(eps_hat, reaction_res)
        kl_div = model.kl_divergence(params=params, weights=gpweight)
        total_loss = -jnp.mean(ell_) + kl_div - log_prior + stability_weight * stab_pen
        return total_loss, (jnp.mean(ell_), kl_div) + tuple(jnp.mean(t) for t in terms) + (log_prior, stab_pen, frac_unstable)

    piola2x2 = lambda f, k: model.piola(f, k, params=params, weights=gpweight)[:2, :2]
    piola_cells = jax.vmap(piola2x2, in_axes=(0, None))
    piola_steps = jax.vmap(piola_cells, in_axes=(0, None))
    piola_sampling = jax.vmap(piola_steps, in_axes=(None, 0))
    piola2x2_cells = piola_sampling(f3x3, subkey)

    # vmapped_ell maps over Monte Carlo samples
    vmapped_ell = jax.vmap(ell, in_axes=(None, None, None, None, None, None, None, 0, None, None, None, None, None, None, None, None))
    ell_, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_free_loss, sum_fix_loss) = vmapped_ell(
        params, sigma_fix_x, sigma_fix_y, cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA, normalize_ell, vfm_mode, V_basis, control_mode, loads, reaction_loss_weight
    )
    
    kl_div = model.kl_divergence(params=params, weights=gpweight)

    total_loss = -jnp.mean(ell_) + kl_div - log_prior + stability_weight * stab_pen
    return total_loss, (jnp.mean(ell_), kl_div, jnp.mean(free_x_log_likelihood), jnp.mean(free_y_log_likelihood),
                        jnp.mean(fix_x_log_likelihood), jnp.mean(fix_y_log_likelihood), jnp.mean(sum_free_loss), jnp.mean(sum_fix_loss),
                        log_prior, stab_pen, frac_unstable)


def ell(p: Any, sigma_fix_x: jnp.ndarray, sigma_fix_y: jnp.ndarray, cells: jnp.ndarray, n_nodes: int, 
        f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, piola2x2_cells: jnp.ndarray, dNdX: jnp.ndarray, dA: jnp.ndarray,
        normalize_ell: int = 0, vfm_mode: str = "linear_triangle", V_basis: jnp.ndarray = None,
        control_mode: str = "force", loads: jnp.ndarray = None,
        reaction_loss_weight: float = 1.0):
    sigma_free_x = jnp.maximum(p.sigma_free_x, 1e-6)
    sigma_free_y = jnp.maximum(p.sigma_free_y, 1e-6)
    sigma_fix_x = jnp.maximum(sigma_fix_x, 1e-3)
    sigma_fix_y = jnp.maximum(sigma_fix_y, 1e-3)

    sigma_global = getattr(p, "sigma_global", None)
    if sigma_global is None:
        sigma_global = 0.5 * (jnp.mean(sigma_free_x) + jnp.mean(sigma_free_y))
    sigma_global = jnp.maximum(sigma_global, 1e-6)

    # vmap over load steps for the VFM loss
    n_steps = f_neu_nodes.shape[0] if f_neu_nodes is not None else piola2x2_cells.shape[0]
    loads_schedule = loads if loads is not None else jnp.zeros((n_steps, 2), dtype=jnp.float64)

    free_x_loss, free_y_loss, fix_x_loss, fix_y_loss, global_loss = jax.vmap(
        vfm_loss, in_axes=(None, None, 0, None, 0, None, None, None, None, 0)
    )(cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA, V_basis, control_mode, loads_schedule)

    n_steps = free_x_loss.shape[0]
    n_freedofs_x = free_x_loss.shape[1]
    n_freedofs_y = free_y_loss.shape[1]
    n_vfs = global_loss.shape[1]

    # Reaction force log-likelihoods (boundary traction equilibrium) scaled by reaction_loss_weight
    if normalize_ell == 1:
        fix_x_log_likelihood = reaction_loss_weight * (1.0 / n_steps) * jnp.sum(- (1.0 / (2 * (sigma_fix_x**2))) * (fix_x_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_x**2)))
        fix_y_log_likelihood = reaction_loss_weight * (1.0 / n_steps) * jnp.sum(- (1.0 / (2 * (sigma_fix_y**2))) * (fix_y_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_y**2)))
    else:
        fix_x_log_likelihood = reaction_loss_weight * jnp.sum(- (1.0 / (2 * (sigma_fix_x**2))) * (fix_x_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_x**2)))
        fix_y_log_likelihood = reaction_loss_weight * jnp.sum(- (1.0 / (2 * (sigma_fix_y**2))) * (fix_y_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_y**2)))

    # Identify free nodes masks for nodal-diagonal variance resolution
    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    if control_mode == "displacement":
        is_free_x = ~(is_fix_x | (node_type[:, 3] == 1))
        is_free_y = ~(is_fix_y | (node_type[:, 4] == 1))
    else:
        is_free_x = ~is_fix_x
        is_free_y = ~is_fix_y

    def _extract_active_sigma(sigma, is_free_mask):
        if sigma.ndim == 0 or sigma.size == 1:
            return jnp.squeeze(sigma)
        if sigma.shape[0] == is_free_mask.shape[0]:
            return sigma[is_free_mask]
        return sigma

    sig_x = _extract_active_sigma(sigma_free_x, is_free_x)
    sig_y = _extract_active_sigma(sigma_free_y, is_free_y)

    # Nodal residuals log-likelihood (supports both constant scalar and diagonal nodal vectors)
    if sig_x.ndim == 0:
        if normalize_ell == 1:
            n_free_total_x = n_steps * n_freedofs_x
            free_x_log_likelihood = - (1.0 / (2 * (sig_x**2))) * (jnp.sum(free_x_loss**2) / n_free_total_x) - 0.5 * jnp.log(2 * jnp.pi * (sig_x**2))
        else:
            free_x_log_likelihood = - (1.0 / (2 * (sig_x**2))) * jnp.sum(free_x_loss**2) - (n_steps * n_freedofs_x) / 2.0 * jnp.log(2 * jnp.pi * (sig_x**2))
    else:
        node_sq_res_x = jnp.sum(free_x_loss**2, axis=0)  # (n_freedofs_x,)
        quad_terms_x = - 0.5 * (node_sq_res_x / (sig_x**2))
        log_terms_x = - 0.5 * n_steps * jnp.log(2.0 * jnp.pi * (sig_x**2))
        if normalize_ell == 1:
            n_free_total_x = n_steps * n_freedofs_x
            free_x_log_likelihood = jnp.sum(quad_terms_x) / n_free_total_x + jnp.sum(log_terms_x) / n_free_total_x
        else:
            free_x_log_likelihood = jnp.sum(quad_terms_x + log_terms_x)

    if sig_y.ndim == 0:
        if normalize_ell == 1:
            n_free_total_y = n_steps * n_freedofs_y
            free_y_log_likelihood = - (1.0 / (2 * (sig_y**2))) * (jnp.sum(free_y_loss**2) / n_free_total_y) - 0.5 * jnp.log(2 * jnp.pi * (sig_y**2))
        else:
            free_y_log_likelihood = - (1.0 / (2 * (sig_y**2))) * jnp.sum(free_y_loss**2) - (n_steps * n_freedofs_y) / 2.0 * jnp.log(2 * jnp.pi * (sig_y**2))
    else:
        node_sq_res_y = jnp.sum(free_y_loss**2, axis=0)  # (n_freedofs_y,)
        quad_terms_y = - 0.5 * (node_sq_res_y / (sig_y**2))
        log_terms_y = - 0.5 * n_steps * jnp.log(2.0 * jnp.pi * (sig_y**2))
        if normalize_ell == 1:
            n_free_total_y = n_steps * n_freedofs_y
            free_y_log_likelihood = jnp.sum(quad_terms_y) / n_free_total_y + jnp.sum(log_terms_y) / n_free_total_y
        else:
            free_y_log_likelihood = jnp.sum(quad_terms_y + log_terms_y)

    # Split global virtual field residuals into X and Y equations:
    Mx = n_vfs // 2
    My = n_vfs - Mx
    global_x_loss = global_loss[:, :Mx]
    global_y_loss = global_loss[:, Mx:]

    # Global virtual fields log-likelihood (separate X and Y equations)
    sig_gx = jnp.mean(sig_x) if sig_x.ndim > 0 else sig_x
    sig_gy = jnp.mean(sig_y) if sig_y.ndim > 0 else sig_y
    if normalize_ell == 1:
        n_global_total_x = n_steps * Mx
        n_global_total_y = n_steps * My
        global_x_log_likelihood = - (1.0 / (2 * (sig_gx**2))) * (jnp.sum(global_x_loss**2) / n_global_total_x) - 0.5 * jnp.log(2 * jnp.pi * (sig_gx**2))
        global_y_log_likelihood = - (1.0 / (2 * (sig_gy**2))) * (jnp.sum(global_y_loss**2) / n_global_total_y) - 0.5 * jnp.log(2 * jnp.pi * (sig_gy**2))
    else:
        global_x_log_likelihood = - (1.0 / (2 * (sig_gx**2))) * jnp.sum(global_x_loss**2) - (n_steps * Mx) / 2.0 * jnp.log(2 * jnp.pi * (sig_gx**2))
        global_y_log_likelihood = - (1.0 / (2 * (sig_gy**2))) * jnp.sum(global_y_loss**2) - (n_steps * My) / 2.0 * jnp.log(2 * jnp.pi * (sig_gy**2))

    sum_nodal_loss = jnp.sum(free_x_loss**2) + jnp.sum(free_y_loss**2)
    sum_global_loss = jnp.sum(global_loss**2)
    sum_fix_loss = jnp.sum(fix_x_loss**2) + jnp.sum(fix_y_loss**2)

    if vfm_mode == "linear_triangle":
        expected_log_likelihood = free_x_log_likelihood + free_y_log_likelihood + (fix_x_log_likelihood + fix_y_log_likelihood)
        return expected_log_likelihood, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_nodal_loss, sum_fix_loss)
    elif vfm_mode == "global_vf":
        expected_log_likelihood = global_x_log_likelihood + global_y_log_likelihood + (fix_x_log_likelihood + fix_y_log_likelihood)
        return expected_log_likelihood, (global_x_log_likelihood, global_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_global_loss, sum_fix_loss)
    elif vfm_mode == "mix":
        expected_log_likelihood = global_x_log_likelihood + global_y_log_likelihood + free_x_log_likelihood + free_y_log_likelihood + (fix_x_log_likelihood + fix_y_log_likelihood)
        return expected_log_likelihood, (global_x_log_likelihood + free_x_log_likelihood, global_y_log_likelihood + free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_global_loss, sum_nodal_loss)
    else:
        raise ValueError(f"Unknown vfm_mode: {vfm_mode}. Expected 'linear_triangle', 'global_vf', or 'mix'.")


def vfm_loss(cells: jnp.ndarray, n_nodes: int, f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, 
             piola2x2: jnp.ndarray, dNdx: jnp.ndarray, dA: jnp.ndarray,
             V_basis: jnp.ndarray = None, control_mode: str = "force",
             load_step: jnp.ndarray = None):
    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)  # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2) using explicit float64 precision
    f_int_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell)

    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    is_loaded_x = (node_type[:, 3] == 1)
    is_loaded_y = (node_type[:, 4] == 1)

    if control_mode == "displacement":
        # Constrained DOFs are both zero-fixed and prescribed-displacement boundaries
        is_free_x = ~(is_fix_x | is_loaded_x)
        is_free_y = ~(is_fix_y | is_loaded_y)

        # Free node residuals (tractions are zero on free nodes)
        free_x_loss = f_int_nodes[is_free_x, 0]
        free_y_loss = f_int_nodes[is_free_y, 1]

        # Reaction force equilibrium on the loaded boundary (against load-cell measurement)
        f_pred_x = jnp.sum(f_int_nodes[is_loaded_x, 0])
        f_pred_y = jnp.sum(f_int_nodes[is_loaded_y, 1])

        target_x = load_step[0] if (load_step is not None and load_step.shape[0] > 0) else 0.0
        target_y = load_step[1] if (load_step is not None and load_step.shape[0] > 1) else 0.0

        fix_x_loss = f_pred_x - target_x
        fix_y_loss = f_pred_y - target_y
        R_nodes = f_int_nodes
    else:
        # Force control:
        # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
        R_nodes = f_int_nodes - f_neu_nodes
        free_x_loss = R_nodes[~is_fix_x, 0]
        free_y_loss = R_nodes[~is_fix_y, 1]
        
        # Global equilibrium loss (sum of reactions + sum of external forces on free nodes)
        fix_x_loss = jnp.sum(R_nodes[is_fix_x, 0]) + jnp.sum(f_neu_nodes[~is_fix_x, 0])
        fix_y_loss = jnp.sum(R_nodes[is_fix_y, 1]) + jnp.sum(f_neu_nodes[~is_fix_y, 1])

    if V_basis is not None:
        global_loss = jnp.einsum("mid,id->m", V_basis, R_nodes)
    else:
        global_loss = jnp.zeros((1,), dtype=jnp.float64)

    return free_x_loss, free_y_loss, fix_x_loss, fix_y_loss, global_loss


def neumann_cell_force(coords_el: jnp.ndarray, onehot_types_el: jnp.ndarray, t3: float, t4: float):
    """
    onehot_types_el: (3, 5) array - one-hot encoded types for 3 nodes
    Columns: [0: Internal, 1: FixX, 2: FixY, 3: Right(t3), 4: Top(t4)]
    """
    edges = jnp.array([[0, 1], [1, 2], [2, 0]])
    f_cell = jnp.zeros((3, 2), dtype=jnp.float64)

    for idx in range(3):
        i, j = edges[idx]
        is_right = (onehot_types_el[i, 3] == 1) & (onehot_types_el[j, 3] == 1)
        is_top = (onehot_types_el[i, 4] == 1) & (onehot_types_el[j, 4] == 1)
        L = jnp.linalg.norm(coords_el[j] - coords_el[i])

        f_cell = f_cell.at[i, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        f_cell = f_cell.at[j, 0].add(jnp.where(is_right, 0.5 * L * t3, 0.0))
        f_cell = f_cell.at[i, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))
        f_cell = f_cell.at[j, 1].add(jnp.where(is_top, 0.5 * L * t4, 0.0))

    return f_cell


def physical_loss_per_loadstep_force_controlled(u: jnp.ndarray, load: jnp.ndarray, piola_func: Any, 
                                                coords: jnp.ndarray, cells: jnp.ndarray, node_type: jnp.ndarray):
    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2

    f = jax.vmap(fto3x3)(F)
    piola = jax.vmap(piola_func)(f)
    piola2x2 = piola[:, :2, :2]

    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)

    f_int_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell)
    
    t3, t4 = load
    types = node_type[cells]
    f_neu_cells = jax.vmap(neumann_cell_force, in_axes=(0, 0, None, None))(coord_cells, types, t3, t4)
    f_neu_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_neu_cells)

    R_nodes = f_int_nodes - f_neu_nodes
    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    
    is_free_x = ~is_fix_x
    is_free_y = ~is_fix_y

    free_x_loss = R_nodes[is_free_x, 0]
    free_y_loss = R_nodes[is_free_y, 1]
    
    # Evaluate global equilibrium for the fixed DOFs using the applied traction DOFs
    neu_nodes_right = (node_type[:, 3] == 1)
    neu_nodes_top = (node_type[:, 4] == 1)
    total_traction_force = f_neu_nodes[neu_nodes_right | neu_nodes_top].sum(axis=0)
    fixed_nodes_loss1 = jnp.sum(R_nodes[is_fix_x, 0]) + total_traction_force[0]
    fixed_nodes_loss2 = jnp.sum(R_nodes[is_fix_y, 1]) + total_traction_force[1]

    free_loss = jnp.stack([free_x_loss, free_y_loss], axis=-1)
    fix_loss = jnp.stack([fixed_nodes_loss1, fixed_nodes_loss2])
    return free_loss, fix_loss


def physical_loss_displacement_controlled(u: jnp.ndarray, loads: jnp.ndarray, piola_func: Any, 
                                          coords: jnp.ndarray, cells: jnp.ndarray, node_type: jnp.ndarray):
    u_cells = u[cells]
    coord_cells = coords[cells]
    n_nodes = coords.shape[0]
    F, dNdx = deformation_gradient_element(coord_cells, u_cells)
    dA = jnp.linalg.det(transformation_jacobian(coord_cells)) / 2
    f = jax.vmap(fto3x3)(F)

    piola = jax.vmap(piola_func)(f)[:, :2, :2]
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)

    f_int_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell)

    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    is_loaded_x = (node_type[:, 3] == 1)
    is_loaded_y = (node_type[:, 4] == 1)

    free_x = ~(is_fix_x | is_loaded_x)
    free_y = ~(is_fix_y | is_loaded_y)

    free_r_x = jnp.sum(f_int_nodes[free_x, 0] ** 2)
    free_r_y = jnp.sum(f_int_nodes[free_y, 1] ** 2)
    free_r_total = free_r_x + free_r_y

    f_pred_x = jnp.sum(f_int_nodes[is_loaded_x, 0])
    f_pred_y = jnp.sum(f_int_nodes[is_loaded_y, 1])

    target_x = loads[0] if (loads is not None and loads.shape[0] > 0) else 0.0
    target_y = loads[1] if (loads is not None and loads.shape[0] > 1) else 0.0

    reaction_loss_x = (f_pred_x - target_x) ** 2
    reaction_loss_y = (f_pred_y - target_y) ** 2
    reaction_loss = reaction_loss_x + reaction_loss_y

    return free_r_total, reaction_loss

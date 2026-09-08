"""
gp_component.py: Centralized Functional Gaussian Process Mathematics for Invariant Subspaces

Provides reusable, strictly functional routines for:
1. Component parameter transformations (load_component_params, inv_softplus).
2. Covariance Gram matrix assembly & weight precomputation (compute_component_weights).
3. Physics-informed pathwise sampling via Random Fourier Features and Matheron's rule (sample_component_path).
4. Deterministic and stochastic posterior predictive distributions (mean, variance, joint covariance).
5. Variational KL divergence calculations (compute_component_kl).
6. Initial inducing point selection (FPS) and variational parameter initialization (init_component_raw_params).
7. Anchor and parameter gradient-freezing logic (freeze_component_anchor).
"""

from typing import NamedTuple, Tuple, Optional, Callable, Dict, Any
import numpy as np
import jax
import jax.numpy as jnp
from jax import random, vmap

# Enforce mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from core.kernel import rbf
from core.utils import farthest_point_sampling_with_fixed_point


# ==============================================================================
# 1. Data Structures
# ==============================================================================

class ComponentRawParams(NamedTuple):
    raw_ls: jnp.ndarray
    raw_sig: jnp.ndarray
    raw_u_mean: jnp.ndarray
    raw_u_var: jnp.ndarray
    raw_z: jnp.ndarray


class ComponentParams(NamedTuple):
    ls: jnp.ndarray
    sig: jnp.ndarray
    u_mean: jnp.ndarray
    u_var: jnp.ndarray
    z: jnp.ndarray


class ComponentWeights(NamedTuple):
    Kzz: jnp.ndarray
    Kzz_inv: jnp.ndarray
    v: jnp.ndarray
    trace_term: jnp.ndarray
    mahalanobis_term: jnp.ndarray
    M_mat: jnp.ndarray
    logterm: jnp.ndarray


# ==============================================================================
# 2. Parameter Transformations & Anchoring
# ==============================================================================

def inv_softplus(y: jnp.ndarray) -> jnp.ndarray:
    """Computes initial unconstrained raw parameters from physical positive values."""
    y_safe = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), 1e-15)
    return jnp.where(y_safe > 20.0, y_safe, jnp.log(jnp.expm1(y_safe)))


def get_full_cov(raw: jnp.ndarray) -> jnp.ndarray:
    """Builds a positive-definite covariance matrix from unconstrained lower-triangular raw parameters."""
    L_raw = jnp.tril(raw, k=-1)
    L_diag = jnp.diag(jax.nn.softplus(jnp.diag(raw)))
    L = L_raw + L_diag
    return L @ L.T


def load_component_params(
    raw_ls: jnp.ndarray,
    raw_sig: jnp.ndarray,
    raw_u_mean: jnp.ndarray,
    raw_u_var: jnp.ndarray,
    raw_z: jnp.ndarray,
    anchor: jnp.ndarray,
    covariance_mode: str = "diag",
    max_val: Optional[jnp.ndarray] = None,
    u_var_anchor: float = 1e-12,
    is_dev: bool = False
) -> ComponentParams:
    """
    Applies physical positivity and structural anchor constraints to unconstrained raw parameters.
    Reference anchor (index 0) is enforced to have zero mean and fixed tiny variance (u_var_anchor).
    """
    to_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)
    anchor = to_f64(anchor)

    # Inducing points: softplus ensures positive offset from reference anchor for dev
    if is_dev:
        z = to_f64(jax.nn.softplus(raw_z)) + anchor
    else:
        z = to_f64(jax.nn.softplus(raw_z))
    z = z.at[0].set(anchor)

    # Inducing mean: softplus ensures positive energy, anchored to 0.0 at reference
    u_mean = to_f64(jax.nn.softplus(raw_u_mean)).at[0].set(0.0)

    # Inducing covariance: diagonal softplus or full Cholesky
    if "full" in covariance_mode:
        u_var = to_f64(get_full_cov(raw_u_var))
        u_var = u_var.at[0, :].set(0.0).at[:, 0].set(0.0).at[0, 0].set(u_var_anchor)
    else:
        u_var = to_f64(jax.nn.softplus(raw_u_var)).at[0].set(u_var_anchor)

    # Lengthscale & signal variance
    if "full" not in covariance_mode and max_val is not None:
        ls = to_f64(jnp.mean(max_val) * 2.0 * jax.nn.sigmoid(raw_ls))
    else:
        ls = to_f64(jax.nn.softplus(raw_ls))
    sig = to_f64(jnp.exp(raw_sig))

    return ComponentParams(ls=ls, sig=sig, u_mean=u_mean, u_var=u_var, z=z)


# ==============================================================================
# 3. Kernel Gram Assembly & Weight Precomputation
# ==============================================================================

def compute_component_weights(
    z: jnp.ndarray,
    u_mean: jnp.ndarray,
    u_var: jnp.ndarray,
    ls: jnp.ndarray,
    sig: jnp.ndarray,
    covariance_mode: str = "diag",
    kzz_jitter: float = 1e-8
) -> ComponentWeights:
    """
    Precomputes Gram matrices Kzz, Kzz_inv, projection matrix M, and KL divergence terms.
    Strictly assumes a zero-mean prior for the energy component.
    """
    M_pts = z.shape[0]
    Kzz = rbf(z, z, sig, ls) + kzz_jitter * jnp.eye(M_pts, dtype=jnp.float64)
    K_inv = jnp.linalg.solve(Kzz, jnp.eye(M_pts, dtype=jnp.float64))

    U_cov = u_var if "full" in covariance_mode else jnp.diag(u_var)

    if "whitened" in covariance_mode:
        L_z = jnp.linalg.cholesky(Kzz)
        v_diff = L_z @ u_mean
        U_cov_true = L_z @ U_cov @ L_z.T

        M_mat = K_inv @ (Kzz - U_cov_true) @ K_inv.T
        trace_term = jnp.trace(U_cov)
        mahalanobis_term = jnp.sum(u_mean ** 2)
        log_term = -jnp.log(jnp.linalg.det(U_cov))
    else:
        v_diff = u_mean
        U_cov_true = U_cov

        M_mat = K_inv @ (Kzz - U_cov_true) @ K_inv.T
        trace_term = jnp.trace(K_inv @ U_cov_true)
        mahalanobis_term = v_diff.T @ K_inv @ v_diff
        log_term = jnp.log(jnp.linalg.det(Kzz)) - jnp.log(jnp.linalg.det(U_cov_true))

    return ComponentWeights(
        Kzz=Kzz, Kzz_inv=K_inv, v=v_diff, trace_term=trace_term,
        mahalanobis_term=mahalanobis_term, M_mat=M_mat, logterm=log_term
    )


# ==============================================================================
# 4. Pathwise Sampling (RFF + Matheron's Rule)
# ==============================================================================

def sample_component_path(
    k_prior_w: jnp.ndarray,
    k_W: jnp.ndarray,
    k_b: jnp.ndarray,
    k_u: jnp.ndarray,
    z: jnp.ndarray,
    u_mean: jnp.ndarray,
    u_var: jnp.ndarray,
    ls: jnp.ndarray,
    sig: jnp.ndarray,
    Kzz: jnp.ndarray,
    L: int = 200,
    covariance_mode: str = "diag"
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """
    Constructs an exact differentiable sample path function f(x) via Random Fourier Features
    conditioned on inducing points using Matheron's rule: f(x) = f_prior(x) + k(x, z) Kzz^-1 (u - f_prior(z)).
    """
    D = z.shape[-1]
    w_prior = random.normal(k_prior_w, (L,), dtype=jnp.float64)
    W = random.normal(k_W, (D, L), dtype=jnp.float64)
    b = random.uniform(k_b, (L,), dtype=jnp.float64) * 2.0 * jnp.pi

    ls_col = ls[:, None] if ls.ndim == 1 else ls.reshape(D, 1)

    def f_prior(x):
        phi = jnp.sqrt(2.0 * sig**2 / L) * jnp.cos(jnp.dot(x, W / ls_col) + b)
        return jnp.dot(phi, w_prior)

    U_cov = u_var if "full" in covariance_mode else jnp.diag(u_var)
    u_sample = random.multivariate_normal(k_u, u_mean, U_cov, dtype=jnp.float64)

    if "whitened" in covariance_mode:
        u_sample = jnp.linalg.cholesky(Kzz) @ u_sample

    prior_at_z = vmap(f_prior)(z)
    v_corr = jnp.linalg.solve(Kzz, u_sample - prior_at_z)

    def path_fn(x):
        k_xz = rbf(x, z, sig, ls)
        return (f_prior(x) + jnp.dot(k_xz, v_corr)).squeeze()

    return path_fn


# ==============================================================================
# 5. Predictions & Moments (Mean, Variance, Covariance, KL)
# ==============================================================================

def predict_component_mean(x: jnp.ndarray, z: jnp.ndarray, sig: jnp.ndarray, ls: jnp.ndarray, Kzz_inv: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Evaluates posterior GP mean: mu(x) = k(x, z) @ Kzz_inv @ v."""
    k_xz = rbf(x, z, sig, ls)
    return k_xz @ Kzz_inv @ v


def predict_component_var(x: jnp.ndarray, z: jnp.ndarray, sig: jnp.ndarray, ls: jnp.ndarray, M_mat: jnp.ndarray) -> jnp.ndarray:
    """Evaluates posterior marginal variance: sigma^2(x) = sig^2 - diag(k(x, z) @ M @ k(z, x))."""
    k_xz = rbf(x, z, sig, ls)
    var = sig**2 - jnp.sum((k_xz @ M_mat) * k_xz, axis=-1)
    return jnp.maximum(var, 1e-8)


def predict_component_joint_cov(x1: jnp.ndarray, x2: jnp.ndarray, z: jnp.ndarray, sig: jnp.ndarray, ls: jnp.ndarray, M_mat: jnp.ndarray) -> jnp.ndarray:
    """Evaluates dense joint covariance: Cov(x1, x2) = k(x1, x2) - k(x1, z) @ M @ k(z, x2)."""
    k_12 = rbf(x1, x2, sig, ls)
    k_1z = rbf(x1, z, sig, ls)
    k_z2 = rbf(z, x2, sig, ls)
    cov = k_12 - k_1z @ M_mat @ k_z2
    return 0.5 * (cov + cov.T)


def compute_component_kl(mahalanobis: jnp.ndarray, logterm: jnp.ndarray, trace: jnp.ndarray, num_z: int) -> jnp.ndarray:
    """Computes component-wise KL divergence: 0.5 * (logterm - M + trace + mahalanobis)."""
    return 0.5 * (logterm - float(num_z) + trace + mahalanobis)


# ==============================================================================
# 6. Inducing Point & Raw Parameter Initializers
# ==============================================================================

def init_component_raw_params(
    key: jnp.ndarray,
    flat_feats: jnp.ndarray,
    n_ip: int,
    anchor: jnp.ndarray,
    covariance_mode: str = "diag",
    u_var_anchor: float = 1e-12,
    is_dev: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    Initializes inducing points via FPS and generates raw parameters (Normal(0,1) and inv_softplus).
    Returns a dict with keys: [z, raw_z, raw_ls, raw_sig, raw_u_mean, raw_u_var].
    """
    k1, k2 = random.split(key)
    dim = flat_feats.shape[-1]
    anchor = jnp.asarray(anchor, dtype=jnp.float64)

    # 1. Farthest Point Sampling with reference anchor fixed at index 0
    z = farthest_point_sampling_with_fixed_point(flat_feats, n_ip, anchor)

    # 2. Raw inducing coordinates (inverted softplus)
    if is_dev:
        raw_z = inv_softplus(z - anchor)
    else:
        raw_z = inv_softplus(z)

    # 3. Variational mean (Normal(0, 1) with anchor set to 0.0)
    raw_u_mean = random.normal(k2, (n_ip,), dtype=jnp.float64).at[0].set(0.0)

    # 4. Variational variance
    if "full" in covariance_mode:
        raw_u_var = random.normal(k2, (n_ip, n_ip), dtype=jnp.float64) * 0.1
        raw_u_var = raw_u_var.at[jnp.diag_indices(n_ip)].set(inv_softplus(u_var_anchor))
    else:
        raw_u_var = random.normal(k2, (n_ip,), dtype=jnp.float64).at[0].set(inv_softplus(u_var_anchor))

    # 5. Hyper-parameters
    raw_ls = random.normal(k1, (dim,), dtype=jnp.float64)
    raw_sig = random.normal(k1, (), dtype=jnp.float64)

    return {
        "z": z,
        "raw_z": raw_z,
        "raw_ls": raw_ls,
        "raw_sig": raw_sig,
        "raw_u_mean": raw_u_mean,
        "raw_u_var": raw_u_var
    }


def freeze_component_anchor(
    grads: Any,
    prefix: str,
    covariance_mode: str = "diag",
    is_fixed_z: bool = False
) -> Dict[str, jnp.ndarray]:
    """
    Zeroes out gradients for anchor index 0 (u_mean=0, u_var=u_var_anchor, z=anchor)
    and optionally freezes all inducing point positions.
    """
    raw_z_name = f"raw_{prefix}_z"
    raw_u_mean_name = f"raw_{prefix}_u_mean"
    raw_u_var_name = f"raw_{prefix}_u_var"

    grad_z = getattr(grads, raw_z_name)
    grad_u_mean = getattr(grads, raw_u_mean_name)
    grad_u_var = getattr(grads, raw_u_var_name)

    # Anchor index 0
    if covariance_mode == "full":
        new_u_var = grad_u_var.at[0, :].set(0.0).at[:, 0].set(0.0)
    else:
        new_u_var = grad_u_var.at[0].set(0.0)

    new_u_mean = grad_u_mean.at[0].set(0.0)

    if is_fixed_z:
        new_z = jnp.zeros_like(grad_z)
    else:
        new_z = grad_z.at[0].set(0.0)

    return {
        raw_z_name: new_z,
        raw_u_mean_name: new_u_mean,
        raw_u_var_name: new_u_var
    }

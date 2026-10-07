from typing import NamedTuple
import jax.numpy as jnp

class EnergyDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

class StressDist(NamedTuple):
    mean: jnp.ndarray
    var: jnp.ndarray

class GPRawParams(NamedTuple):
    raw_dev_ls: jnp.ndarray
    raw_dev_sig: jnp.ndarray
    raw_dev_u_mean: jnp.ndarray
    raw_dev_u_var: jnp.ndarray
    raw_dev_z: jnp.ndarray

    raw_vol_ls: jnp.ndarray
    raw_vol_sig: jnp.ndarray
    raw_vol_u_mean: jnp.ndarray
    raw_vol_u_var: jnp.ndarray
    raw_vol_z: jnp.ndarray

    log_sigma_free_x: jnp.ndarray
    log_sigma_free_y: jnp.ndarray
    log_sigma_fix_x: jnp.ndarray
    log_sigma_fix_y: jnp.ndarray

    raw_aniso_ls: jnp.ndarray = None
    raw_aniso_sig: jnp.ndarray = None
    raw_aniso_u_mean: jnp.ndarray = None
    raw_aniso_u_var: jnp.ndarray = None
    raw_aniso_z: jnp.ndarray = None
    raw_aniso_theta_mean: jnp.ndarray = None
    raw_aniso_theta_var: jnp.ndarray = None
    log_sigma_global: jnp.ndarray = None
    log_kzz_noise: jnp.ndarray = None
    # None (all parameter files saved before 2026-10-07) = legacy transforms: lengthscale bound 2*max(feature),
    # absolute Kzz jitter. 2 = lengthscale bound 1x the feature range, jitter relative to sig^2.
    # 3 = as 2 with the lengthscale bound 2x the feature range.
    param_version: jnp.ndarray = None
    # Optional linear-elastic explicit basis (prior mean): psi += b_mu * (I1_bar - 3)/2 + b_kappa * (J - 1)^2/2 with
    # q(b) = N(lin_mean, L L^T), L from lin_chol_raw = [raw L11, L21, raw L22] (softplus on the diagonal).
    lin_mean: jnp.ndarray = None
    lin_chol_raw: jnp.ndarray = None

class GPParams(NamedTuple) :
    dev_ls: jnp.ndarray = None
    dev_sig: jnp.ndarray = None
    dev_u_mean: jnp.ndarray = None
    dev_u_var: jnp.ndarray = None
    dev_z: jnp.ndarray = None

    vol_ls: jnp.ndarray = None
    vol_sig: jnp.ndarray = None
    vol_u_mean: jnp.ndarray = None
    vol_u_var: jnp.ndarray = None
    vol_z: jnp.ndarray = None

    sigma_free_x: jnp.ndarray = None
    sigma_free_y: jnp.ndarray = None
    sigma_fix_x: jnp.ndarray = None
    sigma_fix_y: jnp.ndarray = None
    sigma_global: jnp.ndarray = None

    aniso_ls: jnp.ndarray = None
    aniso_sig: jnp.ndarray = None
    aniso_u_mean: jnp.ndarray = None
    aniso_u_var: jnp.ndarray = None
    aniso_theta_mean: jnp.ndarray = None
    aniso_theta_var: jnp.ndarray = None
    aniso_z: jnp.ndarray = None
    kzz_noise: jnp.ndarray = None
    lin_mean: jnp.ndarray = None      # posterior mean of (mu, kappa) of the linear-elastic basis
    lin_cov: jnp.ndarray = None       # posterior covariance (2, 2)

class GPWeights(NamedTuple) :
    dev_Kzz: jnp.ndarray
    dev_v: jnp.ndarray
    dev_trace_term: jnp.ndarray
    dev_mahalanobis_term: jnp.ndarray
    dev_M_mat: jnp.ndarray
    dev_Kzz_inv: jnp.ndarray
    dev_logterm: jnp.ndarray

    vol_Kzz: jnp.ndarray
    vol_v: jnp.ndarray
    vol_trace_term: jnp.ndarray
    vol_mahalanobis_term: jnp.ndarray
    vol_M_mat: jnp.ndarray
    vol_Kzz_inv: jnp.ndarray
    vol_logterm: jnp.ndarray

    aniso_Kzz: jnp.ndarray = None
    aniso_v: jnp.ndarray = None
    aniso_trace_term: jnp.ndarray = None
    aniso_mahalanobis_term: jnp.ndarray = None
    aniso_M_mat: jnp.ndarray = None
    aniso_Kzz_inv: jnp.ndarray = None
    aniso_logterm: jnp.ndarray = None


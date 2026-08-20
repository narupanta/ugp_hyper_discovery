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
    raw_vol_kappa: jnp.ndarray

    log_sigma_free_x: jnp.ndarray
    log_sigma_free_y: jnp.ndarray
    log_sigma_fix_x: jnp.ndarray
    log_sigma_fix_y: jnp.ndarray

    raw_aniso_ls: jnp.ndarray = None
    raw_aniso_sig: jnp.ndarray = None
    raw_aniso_u_mean: jnp.ndarray = None
    raw_aniso_u_var: jnp.ndarray = None
    raw_aniso_z: jnp.ndarray = None
    raw_aniso_kappa: jnp.ndarray = None
    raw_aniso_theta_mean: jnp.ndarray = None
    raw_aniso_theta_var: jnp.ndarray = None

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
    vol_kappa: jnp.ndarray = None

    sigma_free_x: jnp.ndarray = None
    sigma_free_y: jnp.ndarray = None
    sigma_fix_x: jnp.ndarray = None
    sigma_fix_y: jnp.ndarray = None

    aniso_ls: jnp.ndarray = None
    aniso_sig: jnp.ndarray = None
    aniso_u_mean: jnp.ndarray = None
    aniso_u_var: jnp.ndarray = None
    aniso_theta_mean: jnp.ndarray = None
    aniso_theta_var: jnp.ndarray = None
    aniso_z: jnp.ndarray = None
    aniso_kappa: jnp.ndarray = None

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


class SyntheticData(NamedTuple) :
    load_array: jnp.ndarray
    u_array: jnp.ndarray
    coords: jnp.ndarray
    cells: jnp.ndarray
    node_type: jnp.ndarray
    dev_inv_array: jnp.ndarray
    vol_inv_array: jnp.ndarray
    invariants_array: jnp.ndarray
    disp_noise_level: float
    load_noise_level: float

class TrainingSetup(NamedTuple) :
    dataset_dir: str
    material_model: str
    save_path: str
    training_mode: str
    n_dev_ip: int
    n_vol_ip: int
    true_material_parameters: jnp.ndarray


class PrecomputedVFMData(NamedTuple) :
    f_neu: jnp.ndarray
    node_type: jnp.ndarray
    F: jnp.ndarray
    dNdX: jnp.ndarray
    dA: jnp.ndarray
import jax
import jax.numpy as jnp
import jax.random as jr
from typing import Tuple, Any

# Enforce mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from .utils import deformation_gradient_element, transformation_jacobian, fto3x3
from .model import SparseHyperelasticityGP


def total_stochastic_loss(p: Any, model: SparseHyperelasticityGP, f3x3: jnp.ndarray, cells: jnp.ndarray, 
                          n_nodes: int, f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, dNdX: jnp.ndarray, 
                          dA: jnp.ndarray, key: jnp.ndarray, n_s: int) -> Tuple[jnp.ndarray, Tuple[jnp.ndarray, ...]]:
    """
    Computes the variational stochastic VFM loss and KL divergence ELBO objective.
    Strictly preserves functional purity without mutating stateful class instance attributes.
    """
    params = model.load_params(p)
    gpweight = model.precompute_weights_from_loaded(params)
    sigma_fix_x = params.sigma_fix_x
    sigma_fix_y = params.sigma_fix_y

    main_key = jr.split(key, n_s + 1)
    subkey = main_key[1:]

    piola2x2 = lambda f, k: model.piola(f, k, params=params, weights=gpweight)[:2, :2]
    piola_cells = jax.vmap(piola2x2, in_axes=(0, None))
    piola_steps = jax.vmap(piola_cells, in_axes=(0, None))
    piola_sampling = jax.vmap(piola_steps, in_axes=(None, 0))
    piola2x2_cells = piola_sampling(f3x3, subkey)

    # vmapped_ell maps over Monte Carlo samples
    vmapped_ell = jax.vmap(ell, in_axes=(None, None, None, None, None, None, None, 0, None, None))
    ell_, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, sum_free_loss, sum_fix_loss) = vmapped_ell(
        params, sigma_fix_x, sigma_fix_y, cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA
    )
    
    kl_div = model.kl_divergence(params=params, weights=gpweight)
    
    total_loss = -jnp.mean(ell_) + kl_div
    return total_loss, (jnp.mean(ell_), kl_div, jnp.mean(free_x_log_likelihood), jnp.mean(free_y_log_likelihood), 
                        jnp.mean(fix_x_log_likelihood), jnp.mean(fix_y_log_likelihood), jnp.mean(sum_free_loss), jnp.mean(sum_fix_loss))


def ell(p: Any, sigma_fix_x: jnp.ndarray, sigma_fix_y: jnp.ndarray, cells: jnp.ndarray, n_nodes: int, 
        f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, piola2x2_cells: jnp.ndarray, dNdX: jnp.ndarray, dA: jnp.ndarray):
    sigma_free_x = jnp.maximum(p.sigma_free_x, 1e-6)
    sigma_free_y = jnp.maximum(p.sigma_free_y, 1e-6)
    sigma_fix_x = jnp.maximum(sigma_fix_x, 1e-3)
    sigma_fix_y = jnp.maximum(sigma_fix_y, 1e-3)

    # vmap over load steps for the VFM loss
    free_x_loss, free_y_loss, fix_x_loss, fix_y_loss = jax.vmap(vfm_loss, in_axes=(None, None, 0, None, 0, None, None))(
        cells, n_nodes, f_neu_nodes, node_type, piola2x2_cells, dNdX, dA
    )

    n_steps = free_x_loss.shape[0]
    n_freedofs_x = free_x_loss.shape[1]
    n_freedofs_y = free_y_loss.shape[1]

    free_x_log_likelihood = - (1.0 / (2 * (sigma_free_x**2))) * jnp.sum(free_x_loss**2) - (n_steps * n_freedofs_x) / 2.0 * jnp.log(2 * jnp.pi * (sigma_free_x**2))
    free_y_log_likelihood = - (1.0 / (2 * (sigma_free_y**2))) * jnp.sum(free_y_loss**2) - (n_steps * n_freedofs_y) / 2.0 * jnp.log(2 * jnp.pi * (sigma_free_y**2))
    
    fix_x_log_likelihood = jnp.sum(- (1.0 / (2 * (sigma_fix_x**2))) * (fix_x_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_x**2)))
    fix_y_log_likelihood = jnp.sum(- (1.0 / (2 * (sigma_fix_y**2))) * (fix_y_loss**2) - 0.5 * jnp.log(2 * jnp.pi * (sigma_fix_y**2)))

    expected_log_likelihood = free_x_log_likelihood + free_y_log_likelihood + (fix_x_log_likelihood + fix_y_log_likelihood)
    return expected_log_likelihood, (free_x_log_likelihood, free_y_log_likelihood, fix_x_log_likelihood, fix_y_log_likelihood, jnp.sum(free_x_loss**2) + jnp.sum(free_y_loss**2), jnp.sum(fix_x_loss**2) + jnp.sum(fix_y_loss**2))


def vfm_loss(cells: jnp.ndarray, n_nodes: int, f_neu_nodes: jnp.ndarray, node_type: jnp.ndarray, 
             piola2x2: jnp.ndarray, dNdx: jnp.ndarray, dA: jnp.ndarray):
    # internal element nodal forces: (C,3,2)
    f_int_cell = jnp.einsum("cij, cnj -> cin", piola2x2, dNdx) * dA[:, None, None]
    f_int_cell = jnp.swapaxes(f_int_cell, 1, 2)  # (C,3,2)

    # assemble into global internal force vector (n_nodes, 2) using explicit float64 precision
    f_int_nodes = jnp.zeros((n_nodes, 2), dtype=jnp.float64).at[cells].add(f_int_cell)

    # --- Residual R = int(grad v : P) dx  -  int(v·T) ds(Neumann)
    R_nodes = f_int_nodes - f_neu_nodes
    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    
    free_x_loss = R_nodes[~is_fix_x, 0]
    free_y_loss = R_nodes[~is_fix_y, 1]
    
    # Global equilibrium loss (sum of reactions + sum of external forces on free nodes)
    fix_x_loss = jnp.sum(R_nodes[is_fix_x, 0]) + jnp.sum(f_neu_nodes[~is_fix_x, 0])
    fix_y_loss = jnp.sum(R_nodes[is_fix_y, 1]) + jnp.sum(f_neu_nodes[~is_fix_y, 1])

    return free_x_loss, free_y_loss, fix_x_loss, fix_y_loss


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


def total_physical_loss(u_array: jnp.ndarray, loads: jnp.ndarray, piola_func: Any, 
                        coords: jnp.ndarray, cells: jnp.ndarray, node_type: jnp.ndarray):
    plpl = jax.vmap(physical_loss_per_loadstep_force_controlled, in_axes=(0, 0, None, None, None, None))
    free_node_residual, reaction_loss = plpl(u_array, loads, piola_func, coords, cells, node_type)
    return free_node_residual, reaction_loss


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
    # is_fix_x is node_type[:, 1]
    # is_fix_y is node_type[:, 2]
    
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

    free_r0 = jnp.sum(f_int_nodes[node_type == 0] ** 2)
    free_r1 = jnp.sum(f_int_nodes[node_type == 1, 1] ** 2)
    free_r2 = jnp.sum(f_int_nodes[node_type == 2, 0] ** 2)
    free_r3 = jnp.sum(f_int_nodes[node_type == 3, 1] ** 2)
    free_r4 = jnp.sum(f_int_nodes[node_type == 4, 0] ** 2)
    free_r_total = free_r0 + free_r1 + free_r2 + free_r3 + free_r4

    fnl_left = (jnp.sum(f_int_nodes[node_type == 1, 0]) - loads[0])**2
    fnl_bottom = (jnp.sum(f_int_nodes[node_type == 2, 1]) - loads[1])**2
    fnl_right = (jnp.sum(f_int_nodes[node_type == 3, 0]) - loads[2])**2
    fnl_top = (jnp.sum(f_int_nodes[node_type == 4, 1]) - loads[3])**2

    reaction_loss = fnl_left + fnl_bottom + fnl_right + fnl_top
    return free_r_total, reaction_loss

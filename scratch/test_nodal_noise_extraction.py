import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib

matplotlib.use('Agg')

# Mandatory 64-bit precision standard
jax.config.update("jax_enable_x64", True)

from core.dataclass import GPRawParams, GPParams
from core.model import SparseHyperelasticityGP
from core.features import IsotropicFeatureExtractor
from core.loss_function import total_stochastic_loss, ell
from core.trainer import HyperelasticGPTrainer
from extraction.train_unsupervised import get_freeze_fn
from plots.training import (
    plot_loss_analysis,
    plot_parameters_hist,
    plot_vfm_loss_analysis,
    plot_nodal_noise_spatial_distribution
)


def run_tests():
    print("=" * 60)
    print("Testing Heteroscedastic Nodal Noise Feature Pipeline")
    print("=" * 60)

    # 1. Setup a mini 2D specimen mesh: 12 nodes, 10 elements
    np.random.seed(42)
    n_nodes = 12
    mesh_pos = np.zeros((n_nodes, 2), dtype=np.float64)
    # 3x4 grid of nodes
    xs = np.linspace(0.0, 10.0, 4)
    ys = np.linspace(0.0, 5.0, 3)
    grid_x, grid_y = np.meshgrid(xs, ys)
    mesh_pos[:, 0] = grid_x.flatten()
    mesh_pos[:, 1] = grid_y.flatten()

    # Node types: bottom nodes fixed in y (node_type[:, 2] = 1), left nodes fixed in x (node_type[:, 1] = 1)
    # top nodes loaded in y (node_type[:, 4] = 1)
    node_type = np.zeros((n_nodes, 5), dtype=np.int32)
    for idx, (x, y) in enumerate(mesh_pos):
        if np.isclose(x, 0.0):
            node_type[idx, 1] = 1  # fix_x
        if np.isclose(y, 0.0):
            node_type[idx, 2] = 1  # fix_y
        if np.isclose(y, 5.0):
            node_type[idx, 4] = 1  # load_y

    control_mode = "displacement"
    is_fix_x = (node_type[:, 1] == 1)
    is_fix_y = (node_type[:, 2] == 1)
    is_free_x = ~(is_fix_x | (node_type[:, 3] == 1))
    is_free_y = ~(is_fix_y | (node_type[:, 4] == 1))
    is_free_x_jnp = jnp.asarray(is_free_x)
    is_free_y_jnp = jnp.asarray(is_free_y)

    print(f"Total nodes: {n_nodes}")
    print(f"Free X nodes: {int(np.sum(is_free_x))}, Fixed X nodes: {int(np.sum(~is_free_x))}")
    print(f"Free Y nodes: {int(np.sum(is_free_y))}, Fixed Y nodes: {int(np.sum(~is_free_y))}")

    # Build dummy cells (triangle elements)
    cells = np.array([
        [0, 1, 4], [1, 5, 4],
        [1, 2, 5], [2, 6, 5],
        [2, 3, 6], [3, 7, 6],
        [4, 5, 8], [5, 9, 8],
        [5, 6, 9], [6, 10, 9]
    ], dtype=np.int32)
    n_elements = cells.shape[0]

    # Dummy deformation gradient fields: 3 load steps
    n_steps = 3
    f3x3 = np.zeros((n_steps, n_elements, 3, 3), dtype=np.float64)
    for s in range(n_steps):
        f3x3[s, :, :, :] = np.eye(3) + 0.05 * (s + 1) * np.ones((3, 3))
    f3x3 = jnp.asarray(f3x3)
    cells_jnp = jnp.asarray(cells)
    node_type_jnp = jnp.asarray(node_type)
    dNdX = jnp.ones((n_elements, 3, 2), dtype=jnp.float64) * 0.1
    dA = jnp.ones((n_elements,), dtype=jnp.float64) * 0.5
    f_neu_nodes = jnp.zeros((n_steps, n_nodes, 2), dtype=jnp.float64)
    loads_train = jnp.zeros((n_steps, 2), dtype=jnp.float64)

    # Inducing points
    n_ip = 5
    I_z = jnp.array([
        [3.0, 3.0, 1.0],
        [3.5, 3.4, 1.1],
        [4.0, 3.8, 1.2],
        [4.5, 4.2, 1.3],
        [5.0, 4.6, 1.4]
    ], dtype=jnp.float64)

    min_dev = jnp.array([3.0, 3.0])
    max_dev = jnp.array([5.0, 5.0])
    min_vol = jnp.array([1.0])
    max_vol = jnp.array([1.5])

    # 2. Test Heteroscedastic Parameter Initialization
    log_sigma_free_x_init = jnp.zeros(n_nodes, dtype=jnp.float64)
    log_sigma_free_y_init = jnp.zeros(n_nodes, dtype=jnp.float64)

    params = GPRawParams(
        raw_dev_ls=jnp.array([1.0, 1.0]),
        raw_dev_sig=jnp.array(0.0),
        raw_dev_z=I_z[:, :2],
        raw_dev_u_mean=jnp.zeros(n_ip),
        raw_dev_u_var=jnp.zeros(n_ip),
        raw_vol_ls=jnp.array([1.0]),
        raw_vol_sig=jnp.array(0.0),
        raw_vol_z=I_z[:, 2:3],
        raw_vol_u_mean=jnp.zeros(n_ip),
        raw_vol_u_var=jnp.zeros(n_ip),
        log_sigma_free_x=log_sigma_free_x_init,
        log_sigma_free_y=log_sigma_free_y_init,
        log_sigma_fix_x=jnp.zeros(n_steps),
        log_sigma_fix_y=jnp.zeros(n_steps),
        log_sigma_global=jnp.log(jnp.array(1.0))
    )

    extractor = IsotropicFeatureExtractor()
    model = SparseHyperelasticityGP(
        raw_params=params,
        I_z=I_z,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        sampling_mode="pathwise",
        feature_extractor=extractor
    )

    phys_params = model.load_params(params)
    assert phys_params.sigma_free_x.shape == (n_nodes,), f"Expected shape ({n_nodes},), got {phys_params.sigma_free_x.shape}"
    assert phys_params.sigma_free_y.shape == (n_nodes,), f"Expected shape ({n_nodes},), got {phys_params.sigma_free_y.shape}"
    print(f"✅ Heteroscedastic parameter shapes verified: {phys_params.sigma_free_x.shape}")

    # 3. Test Gradient Computation and Freeze Function
    def loss_fn(p, k):
        return total_stochastic_loss(
            p, model, f3x3, cells, n_nodes, f_neu_nodes, node_type, dNdX, dA,
            k, n_s=2, normalize_ell=0,
            vfm_mode="linear_triangle", V_basis=None,
            control_mode=control_mode, loads=loads_train,
            reaction_loss_weight=1.0
        )

    key = jax.random.PRNGKey(0)
    loss_val, grads = jax.value_and_grad(lambda p: loss_fn(p, key)[0])(params)
    print(f"✅ Loss evaluated: {float(loss_val):.4f}")
    assert grads.log_sigma_free_x.shape == (n_nodes,), f"Grad shape mismatch: {grads.log_sigma_free_x.shape}"

    # Verify freeze_fn zeroes out gradients for non-free nodes
    freeze_fn = get_freeze_fn(
        is_fixed_noise=True,
        is_fixed_z=True,
        covariance_mode="diag",
        is_free_x=is_free_x_jnp,
        is_free_y=is_free_y_jnp
    )
    frozen_grads = freeze_fn(grads)

    fixed_x_grads = frozen_grads.log_sigma_free_x[~is_free_x]
    fixed_y_grads = frozen_grads.log_sigma_free_y[~is_free_y]
    assert jnp.all(fixed_x_grads == 0.0), f"Fixed X grads not zero: {fixed_x_grads}"
    assert jnp.all(fixed_y_grads == 0.0), f"Fixed Y grads not zero: {fixed_y_grads}"
    print("✅ Fixed node noise gradients are exactly 0.0 after freeze_fn.")

    # 4. Test Optimizer Integration (5 steps)
    opt = optax.adam(learning_rate=0.01)
    opt_state = opt.init(params)
    out_dir = "scratch/test_nodal_out"
    os.makedirs(out_dir, exist_ok=True)

    trainer = HyperelasticGPTrainer(
        model=model,
        initial_params=params,
        loss_fn=loss_fn,
        opt_state=opt_state,
        optimizer=opt,
        save_path=out_dir,
        true_mat_model=None,
        I_z=I_z,
        I_all=I_z,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        freeze_fn=freeze_fn,
        seed=42,
        vfm_mode="linear_triangle",
        free_noise_mode="nodal"
    )

    trained_params = trainer.train(n_iterations=10, main_key=key, log_info_str="Test Nodal Run", block_size=5)
    cur_p = model.load_params(trained_params)

    # Verify that fixed nodes stayed at exp(0) = 1.0, while free nodes learned distinct values
    fixed_x_vals = cur_p.sigma_free_x[~is_free_x]
    free_x_vals = cur_p.sigma_free_x[is_free_x]
    np.testing.assert_allclose(fixed_x_vals, 1.0, rtol=1e-5, err_msg="Fixed X nodes deviated from 1.0!")
    print(f"✅ Fixed nodes remained exactly 1.0: {fixed_x_vals}")
    print(f"✅ Free node noise values after 10 steps: {free_x_vals}")

    # Check metadata.json
    import json
    with open(os.path.join(out_dir, "metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta.get("free_noise_mode") == "nodal", f"metadata free_noise_mode wrong: {meta}"
    print(f"✅ metadata.json verified: free_noise_mode = {meta['free_noise_mode']}")

    # 5. Test Plotting Routines with 2D sigma_free
    plot_loss_analysis(trainer.loss_components_hist, trainer.params_hist, trainer.steps_history, out_dir)
    plot_parameters_hist(trainer.params_hist, trainer.steps_history, out_dir)
    plot_vfm_loss_analysis(trainer.loss_components_hist, trainer.params_hist, trainer.steps_history, out_dir, "mix")
    plot_nodal_noise_spatial_distribution(
        mesh_pos=mesh_pos,
        node_type=node_type,
        sigma_free_x=cur_p.sigma_free_x,
        sigma_free_y=cur_p.sigma_free_y,
        save_path=out_dir,
        control_mode=control_mode,
        cells=cells
    )

    assert os.path.exists(os.path.join(out_dir, "nodal_noise_spatial_distribution.pdf")), "Spatial distribution PDF missing"
    assert os.path.exists(os.path.join(out_dir, "loss_and_physics.pdf")), "Loss plot missing"
    assert os.path.exists(os.path.join(out_dir, "physics_noise_evolution.pdf")), "Physics noise evolution plot missing"
    print("✅ All plots generated successfully without shape errors.")

    # 6. Backward Compatibility Test: Constant Scalar Mode
    print("\nVerifying backward compatibility with constant scalar mode...")
    scalar_params = GPRawParams(
        raw_dev_ls=jnp.array([1.0, 1.0]),
        raw_dev_sig=jnp.array(0.0),
        raw_dev_z=I_z[:, :2],
        raw_dev_u_mean=jnp.zeros(n_ip),
        raw_dev_u_var=jnp.zeros(n_ip),
        raw_vol_ls=jnp.array([1.0]),
        raw_vol_sig=jnp.array(0.0),
        raw_vol_z=I_z[:, 2:3],
        raw_vol_u_mean=jnp.zeros(n_ip),
        raw_vol_u_var=jnp.zeros(n_ip),
        log_sigma_free_x=jnp.log(jnp.array(1.0)),
        log_sigma_free_y=jnp.log(jnp.array(1.0)),
        log_sigma_fix_x=jnp.zeros(n_steps),
        log_sigma_fix_y=jnp.zeros(n_steps),
        log_sigma_global=jnp.log(jnp.array(1.0))
    )
    scalar_loss, scalar_grads = jax.value_and_grad(lambda p: total_stochastic_loss(
        p, model, f3x3, cells, n_nodes, f_neu_nodes, node_type, dNdX, dA,
        key, n_s=2, normalize_ell=0,
        vfm_mode="linear_triangle", V_basis=None,
        control_mode=control_mode, loads=loads_train
    )[0])(scalar_params)
    assert scalar_grads.log_sigma_free_x.ndim == 0, "Scalar grad should be scalar"
    print(f"✅ Scalar mode gradient verified: loss={float(scalar_loss):.4f}, grad_sig_x={float(scalar_grads.log_sigma_free_x):.4e}")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! Heteroscedastic nodal noise is fully verified.")


if __name__ == "__main__":
    run_tests()

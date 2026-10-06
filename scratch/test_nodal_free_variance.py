import jax
import jax.numpy as jnp
import jax.random as jr
import optax

jax.config.update("jax_enable_x64", True)

from core.dataclass import GPRawParams, GPParams
from core.model import SparseHyperelasticityGP
from core.features import IsotropicFeatureExtractor
from core.loss_function import total_stochastic_loss, ell, vfm_loss

def test_nodal_free_variance():
    print("=== TEST: Heteroscedastic Nodal Free Variance Mode ===")
    
    n_nodes = 6
    n_steps = 3
    cells = jnp.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])
    
    # Boundary conditions: node 0 is fixed in x and y
    node_type = jnp.zeros((n_nodes, 5))
    node_type = node_type.at[0, 1].set(1) # fix x
    node_type = node_type.at[0, 2].set(1) # fix y
    
    f_neu_nodes = jnp.zeros((n_steps, n_nodes, 2))
    dNdX = jnp.ones((cells.shape[0], 3, 2)) * 0.1
    dA = jnp.ones(cells.shape[0]) * 0.5
    f3x3 = jnp.tile(jnp.eye(3)[None, :, :], (n_steps, cells.shape[0], 1, 1))

    # Nodal noise initialized to log(1.0) for each node (shape: (n_nodes,))
    log_sigma_free_x_nodal = jnp.zeros(n_nodes, dtype=jnp.float64)
    log_sigma_free_y_nodal = jnp.zeros(n_nodes, dtype=jnp.float64)

    p_nodal = GPRawParams(
        raw_dev_ls=jnp.ones(2),
        raw_dev_sig=jnp.array(0.0),
        raw_dev_u_mean=jnp.zeros(5),
        raw_dev_u_var=jnp.ones(5),
        raw_dev_z=jnp.zeros((5, 2)),
        raw_vol_ls=jnp.ones(1),
        raw_vol_sig=jnp.array(0.0),
        raw_vol_u_mean=jnp.zeros(5),
        raw_vol_u_var=jnp.ones(5),
        raw_vol_z=jnp.zeros((5, 1)),
        log_sigma_free_x=log_sigma_free_x_nodal,
        log_sigma_free_y=log_sigma_free_y_nodal,
        log_sigma_fix_x=jnp.zeros(n_steps),
        log_sigma_fix_y=jnp.zeros(n_steps),
        log_kzz_noise=jnp.log(jnp.array(1e-8, dtype=jnp.float64)),
    )

    model = SparseHyperelasticityGP(
        raw_params=p_nodal,
        I_z=jnp.zeros((5, 3)),
        min_dev=jnp.zeros(2), min_vol=jnp.zeros(1),
        max_dev=jnp.ones(2) * 5.0, max_vol=jnp.ones(1) * 2.0,
        kzz_jitter=1e-8,
    )
    loaded_params = model.load_params(p_nodal)
    assert loaded_params.sigma_free_x.shape == (n_nodes,)
    assert loaded_params.sigma_free_y.shape == (n_nodes,)
    print(f"✓ GPParams loaded with nodal shape: {loaded_params.sigma_free_x.shape}")

    # Now let's see how ell() evaluates it
    key = jr.PRNGKey(42)
    def loss_wrapper(p, k):
        return total_stochastic_loss(
            p, model, f3x3, cells, n_nodes, f_neu_nodes, node_type, dNdX, dA,
            k, n_s=2, normalize_ell=0
        )

    (loss, aux), grads = jax.value_and_grad(loss_wrapper, has_aux=True)(p_nodal, key)
    print(f"✓ Loss: {float(loss):.4f}")
    print(f"✓ grads.log_sigma_free_x shape: {grads.log_sigma_free_x.shape}")
    print(f"✓ grads.log_sigma_free_x values: {grads.log_sigma_free_x}")

if __name__ == "__main__":
    test_nodal_free_variance()


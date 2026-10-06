import jax
import jax.numpy as jnp
import jax.random as jr
import optax

jax.config.update("jax_enable_x64", True)

from core.dataclass import GPRawParams, GPParams
from core.model import SparseHyperelasticityGP
from core.features import IsotropicFeatureExtractor
from core.loss_function import total_stochastic_loss, ell

def run_tests():
    print("=== TEST 1: Default / Legacy GPRawParams backward compatibility ===")
    p_legacy = GPRawParams(
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
        log_sigma_free_x=jnp.array(0.0),
        log_sigma_free_y=jnp.array(0.0),
        log_sigma_fix_x=jnp.zeros(2),
        log_sigma_fix_y=jnp.zeros(2),
    )
    I_z = jnp.zeros((5, 3))
    min_dev, max_dev = jnp.zeros(2), jnp.ones(2) * 5.0
    min_vol, max_vol = jnp.zeros(1), jnp.ones(1) * 2.0

    model = SparseHyperelasticityGP(
        raw_params=p_legacy,
        I_z=I_z,
        min_dev=min_dev,
        min_vol=min_vol,
        max_dev=max_dev,
        max_vol=max_vol,
        kzz_jitter=1e-8,
    )
    loaded_legacy = model.load_params(p_legacy)
    assert loaded_legacy.kzz_noise is None
    assert abs(model._effective_kzz_jitter(loaded_legacy) - 1e-8) < 1e-15
    print("✓ Legacy mode OK, effective jitter = 1e-8")

    print("\n=== TEST 2: Trainable kzz_noise initialization ===")
    p_trainable = p_legacy._replace(log_kzz_noise=jnp.log(jnp.array(1e-6, dtype=jnp.float64)))
    loaded_trainable = model.load_params(p_trainable)
    assert loaded_trainable.kzz_noise is not None
    assert jnp.isclose(loaded_trainable.kzz_noise, 1e-6)
    eff_jitter = model._effective_kzz_jitter(loaded_trainable)
    assert jnp.isclose(eff_jitter, 1e-6 + 1e-12)
    print(f"✓ Trainable mode OK, kzz_noise = {float(loaded_trainable.kzz_noise):.2e}, eff_jitter = {float(eff_jitter):.2e}")

    print("\n=== TEST 3: Simultaneous separate training of kzz_noise and sigma_free ===")
    assert loaded_trainable.sigma_free_x is not None
    assert loaded_trainable.sigma_free_y is not None
    assert loaded_trainable.kzz_noise is not None
    print(f"✓ Parameters are distinct: sigma_free_x = {float(loaded_trainable.sigma_free_x):.4f}, kzz_noise = {float(loaded_trainable.kzz_noise):.4e}")

    print("\n=== TEST 4: GP Evaluations (psi, piola, kl_divergence, cov) ===")
    key = jr.PRNGKey(42)
    F_sample = jnp.eye(3)
    psi_val = model.psi(F_sample, key, params=loaded_trainable)
    assert jnp.isfinite(psi_val)
    piola_val = model.piola(F_sample, key, params=loaded_trainable)
    assert jnp.all(jnp.isfinite(piola_val))
    kl_val = model.kl_divergence(params=loaded_trainable)
    assert jnp.isfinite(kl_val)
    cov_val = model.psi_gp_cov(F_sample, params=loaded_trainable)
    assert jnp.isfinite(cov_val)
    p_dist = model.piola_dist(F_sample, params=loaded_trainable)
    assert jnp.all(jnp.isfinite(p_dist.mean)) and jnp.all(jnp.isfinite(p_dist.var))
    print(f"✓ GP functions OK (psi={float(psi_val):.4f}, kl={float(kl_val):.4f}, cov={float(cov_val):.4e})")

    print("\n=== TEST 5: Gradient calculation w.r.t. BOTH log_kzz_noise and log_sigma_free ===")
    n_nodes = 4
    n_steps = 2
    cells = jnp.array([[0, 1, 2], [1, 2, 3]])
    node_type = jnp.zeros((n_nodes, 5))
    node_type = node_type.at[0, 1].set(1) # fix x
    node_type = node_type.at[0, 2].set(1) # fix y
    f_neu_nodes = jnp.zeros((n_steps, n_nodes, 2))
    dNdX = jnp.ones((2, 3, 2)) * 0.1
    dA = jnp.ones(2) * 0.5
    f3x3 = jnp.tile(jnp.eye(3)[None, :, :], (n_steps, 2, 1, 1))

    def simple_loss(p, k):
        return total_stochastic_loss(
            p, model, f3x3, cells, n_nodes, f_neu_nodes, node_type, dNdX, dA,
            k, n_s=2, normalize_ell=0
        )

    (loss, aux), grads = jax.value_and_grad(simple_loss, has_aux=True)(p_trainable, key)
    assert jnp.isfinite(loss)
    assert grads.log_kzz_noise is not None and jnp.isfinite(grads.log_kzz_noise)
    assert grads.log_sigma_free_x is not None and jnp.isfinite(grads.log_sigma_free_x)
    assert grads.log_sigma_free_y is not None and jnp.isfinite(grads.log_sigma_free_y)
    print(f"✓ Loss = {float(loss):.4f}")
    print(f"✓ grad(log_kzz_noise) = {float(grads.log_kzz_noise):.6e}")
    print(f"✓ grad(log_sigma_free_x) = {float(grads.log_sigma_free_x):.6e}")
    print(f"✓ grad(log_sigma_free_y) = {float(grads.log_sigma_free_y):.6e}")

    # Optimizer step check
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(p_trainable)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_p = optax.apply_updates(p_trainable, updates)
    assert new_p.log_kzz_noise != p_trainable.log_kzz_noise
    assert new_p.log_sigma_free_x != p_trainable.log_sigma_free_x
    print(f"✓ Independent updates applied:")
    print(f"  old log_kzz_noise = {float(p_trainable.log_kzz_noise):.4f} -> new = {float(new_p.log_kzz_noise):.4f}")
    print(f"  old log_sigma_free_x = {float(p_trainable.log_sigma_free_x):.4f} -> new = {float(new_p.log_sigma_free_x):.4f}")

    print("\n================ ALL TESTS PASSED SUCCESSFULLY! ================")

if __name__ == "__main__":
    run_tests()

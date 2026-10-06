import jax
import jax.numpy as jnp

# free_x_loss shape: (n_steps, n_freedofs_x)
n_steps = 3
n_freedofs = 5
# Suppose node residuals vary across nodes: node 0 has 1.0, node 1 has 2.0, node 2 has 0.1, etc.
free_x_loss = jnp.array([
    [1.0, 2.0, 0.1, 0.5, 0.2],
    [1.2, 1.8, 0.1, 0.4, 0.3],
    [0.9, 2.1, 0.2, 0.6, 0.1]
])

def nodal_log_likelihood(log_sigma, free_loss, normalize_ell=0):
    sigma = jnp.maximum(jnp.exp(log_sigma), 1e-4) # shape: (n_freedofs,)
    # Sum of squared residuals for each node across all load steps
    node_sq_res = jnp.sum(free_loss**2, axis=0) # shape: (n_freedofs,)
    
    # Quadratic term per node: - 0.5 * sum_t (R_{t, i}^2 / sigma_i^2)
    quad_terms = - 0.5 * (node_sq_res / (sigma**2)) # shape: (n_freedofs,)
    
    # Normalizing log term per node: - 0.5 * n_steps * log(2 * pi * sigma_i^2)
    log_terms = - 0.5 * n_steps * jnp.log(2.0 * jnp.pi * (sigma**2)) # shape: (n_freedofs,)
    
    node_ll = quad_terms + log_terms # log-likelihood for each node
    
    if normalize_ell == 1:
        # Normalized by total DOFs (steps * nodes)
        return jnp.sum(node_ll) / (n_steps * n_freedofs)
    else:
        return jnp.sum(node_ll)

log_sigma_init = jnp.zeros(n_freedofs)
val, grads = jax.value_and_grad(nodal_log_likelihood)(log_sigma_init, free_x_loss)

print("node_sq_res:", jnp.sum(free_x_loss**2, axis=0))
print("Gradients per node:", grads)


import os
import shutil
import tempfile
import numpy as np
import jax
import jax.numpy as jnp
import optax
jax.config.update("jax_enable_x64", True)

from core.material_models import get_material
from core.datasetclass import DatasetFactory
from core.virtual_fields import build_kinematic_virtual_fields, compute_global_virtual_residuals
from core.loss_function import total_stochastic_loss
from core.model import SparseHyperelasticityGP
from core.features import IsotropicFeatureExtractor
from core.trainer import HyperelasticGPTrainer
from core.dataclass import GPRawParams

print("=== 1. Testing Virtual Fields Basis Generation & Orthonormality ===")
data_path = "dataset/preprocessed/syn_f/isihara_0.0001_0.01_8.0_0.95.npz"
dataset = DatasetFactory.create("dataset/precomputed_vfm", data_path=data_path)
prep_data = dataset.get_data()

mesh_pos = prep_data["mesh_pos"]
node_type = np.asarray(prep_data["node_type"])
cells = prep_data["cells"]
dA = prep_data["dA"]
dNdX = prep_data["dNdX"]
f_neu = prep_data["f_neu"]
f2x2 = prep_data["F"]

is_fix_x = (node_type[:, 1] == 1)
is_fix_y = (node_type[:, 2] == 1)

V_basis = build_kinematic_virtual_fields(mesh_pos, node_type, order=2)
print(f"V_basis shape: {V_basis.shape}, dtype: {V_basis.dtype}")
assert V_basis.shape[0] == 12, f"Expected 12 virtual fields for order=2, got {V_basis.shape[0]}"

# Check kinematic admissibility
max_fix_x = float(np.max(np.abs(V_basis[:, is_fix_x, 0])))
max_fix_y = float(np.max(np.abs(V_basis[:, is_fix_y, 1])))
print(f"Max Dirichlet violation: x={max_fix_x:.2e}, y={max_fix_y:.2e}")
assert max_fix_x < 1e-12, "Virtual fields violate Dirichlet X boundary condition!"
assert max_fix_y < 1e-12, "Virtual fields violate Dirichlet Y boundary condition!"

# Check orthonormality
V_flat = V_basis.reshape(12, -1)
gram = V_flat @ V_flat.T
gram_err = float(np.max(np.abs(gram - np.eye(12))))
print(f"Gram matrix max deviation from identity: {gram_err:.2e}")
assert gram_err < 1e-12, "Virtual fields are not orthonormal!"
print("✅ Virtual fields basis verified.")

print("\n=== 2. Testing GP Model & Loss Compilation Across 3 VFM Modes ===")
from core.utils import fto3x3
train_steps = [1, 4, 7]
f3x3 = jax.vmap(jax.vmap(fto3x3))(f2x2[train_steps])
f_neu_nodes = f_neu[train_steps]
load_noise_std_steps = prep_data["load_noise_std_steps"][train_steps]

# Dummy GP initialization
k1 = jax.random.PRNGKey(42)
n_ip = 5
fps_idx = np.linspace(0, f3x3.shape[0]*f3x3.shape[1]-1, n_ip, dtype=int)
I_z = np.zeros((n_ip, 3))
I_z[:, 0] = 3.0
I_z[:, 1] = 3.0
I_z[:, 2] = 1.0
I_z = jnp.array(I_z)

raw_params = GPRawParams(
    raw_dev_ls=jax.random.normal(k1, (2,)),
    raw_dev_sig=jax.random.normal(k1, ()),
    raw_dev_z=jnp.zeros((n_ip, 2)),
    raw_dev_u_mean=jnp.zeros((n_ip,)),
    raw_dev_u_var=jnp.ones((n_ip,)),
    raw_vol_ls=jax.random.normal(k1, (1,)),
    raw_vol_sig=jax.random.normal(k1, ()),
    raw_vol_z=jnp.zeros((n_ip, 1)),
    raw_vol_u_mean=jnp.zeros((n_ip,)),
    raw_vol_u_var=jnp.ones((n_ip,)),
    log_sigma_free_x=jnp.log(jnp.array(1.0)),
    log_sigma_free_y=jnp.log(jnp.array(1.0)),
    log_sigma_fix_x=jnp.log(jnp.array([1.0, 1.0, 1.0])),
    log_sigma_fix_y=jnp.log(jnp.array([1.0, 1.0, 1.0])),
    log_sigma_global=jnp.log(jnp.array(1.0))
)

extractor = IsotropicFeatureExtractor()
model = SparseHyperelasticityGP(
    raw_params=raw_params,
    I_z=I_z,
    min_dev=jnp.array([3.0, 3.0]),
    min_vol=jnp.array([1.0]),
    max_dev=jnp.array([10.0, 10.0]),
    max_vol=jnp.array([2.0]),
    sampling_mode="pathwise",
    beta=50.0,
    L=200,
    feature_extractor=extractor
)

# Test load_params includes sigma_global
phys_params = model.load_params(raw_params)
assert phys_params.sigma_global is not None, "phys_params.sigma_global should not be None!"
print(f"phys_params.sigma_global = {float(phys_params.sigma_global):.4f}")

modes = ["linear_triangle", "global_vf", "mix"]
for mode in modes:
    print(f"\nTesting mode: '{mode}'...")
    vb = V_basis if mode in ["global_vf", "mix"] else None
    
    def loss_closure(p, k):
        return total_stochastic_loss(
            p, model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type,
            dNdX, dA, k, n_s=2, normalize_ell=0,
            vfm_mode=mode, V_basis=vb
        )
    
    val, aux = loss_closure(raw_params, k1)
    print(f"  Loss value: {float(val):.4f} | phy: {float(aux[6]):.6f} | phy2: {float(aux[7]):.6f}")
    assert jnp.isfinite(val), f"Loss is not finite in mode {mode}!"
    assert jnp.isfinite(aux[6]), f"phy is not finite in mode {mode}!"
    
    # Test gradients
    val_grad_fn = jax.jit(jax.value_and_grad(loss_closure, has_aux=True))
    (loss_val, aux_val), grads = val_grad_fn(raw_params, k1)
    print(f"  JIT loss value: {float(loss_val):.4f}")
    assert jnp.isfinite(loss_val), f"JIT Loss is not finite in mode {mode}!"
    # Check that grads has non-zero finite values
    grad_norm = float(jnp.linalg.norm(grads.raw_dev_u_mean))
    print(f"  Gradient norm of dev_u_mean: {grad_norm:.6f}")
    assert jnp.isfinite(grad_norm) and grad_norm > 0, "Gradient norm should be finite and non-zero!"
    print(f"  Mode '{mode}' passed JIT value_and_grad.")

print("\n=== 3. Testing 5-step HyperelasticGPTrainer Run in 'global_vf' Mode ===")
tmpdir = tempfile.mkdtemp()
try:
    opt = optax.adam(learning_rate=0.01)
    opt_state = opt.init(raw_params)
    true_mat = get_material("isihara", dev_params=[0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], vol_params=[1.5, 0.0, 0.0])
    
    def loss_closure_global(p, k):
        return total_stochastic_loss(
            p, model, f3x3, cells, cells.max() + 1, f_neu_nodes, node_type,
            dNdX, dA, k, n_s=2, normalize_ell=0,
            vfm_mode="global_vf", V_basis=V_basis
        )
        
    trainer = HyperelasticGPTrainer(
        model=model,
        initial_params=raw_params,
        loss_fn=loss_closure_global,
        opt_state=opt_state,
        optimizer=opt,
        save_path=tmpdir,
        true_mat_model=true_mat,
        I_z=I_z,
        I_all=I_z,
        min_dev=jnp.array([3.0, 3.0]),
        min_vol=jnp.array([1.0]),
        max_dev=jnp.array([10.0, 10.0]),
        max_vol=jnp.array([2.0]),
        seed=42,
        vfm_mode="global_vf"
    )
    
    best_p = trainer.train(n_iterations=5, main_key=k1, log_info_str="test run", block_size=5)
    print("Trainer finished 5 iterations successfully!")
    
    # Verify metadata.json
    import json
    with open(os.path.join(tmpdir, "metadata.json")) as f:
        meta = json.load(f)
    print(f"metadata.json content: {meta}")
    assert meta["vfm_mode"] == "global_vf"
    
    # Verify best_params.npy can be reloaded
    saved_dict = np.load(os.path.join(tmpdir, "best_params.npy"), allow_pickle=True).item()
    reloaded = GPRawParams(**saved_dict)
    assert reloaded.log_sigma_global is not None
    print("Reloaded best_params successfully with log_sigma_global.")
    
finally:
    shutil.rmtree(tmpdir)

print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")


# Repository Rules: Unsupervised GP for Hyperelasticity Discovery (`ugp_hyper_discovery`)

When interacting with this repository, AI agents and contributors must strictly adhere to the following architectural conventions, coding paradigms, and physics formulations.

---

## 1. Core JAX & Precision Standards
- **Functional Purity & State:** All JAX computations must be strictly functional. Avoid stateful classes for GP evaluations and physics calculations. Explicitly manage and pass `jax.random.PRNGKey` instances.
- **64-Bit Precision (Mandatory):** Because hyperelastic stress/strain relationships and covariance Gram matrices are sensitive to precision degradation, always ensure 64-bit precision is enabled at the entry of every script or computation:
  ```python
  import jax
  jax.config.update("jax_enable_x64", True)
  ```
- **Array Shape & Type Annotation:** Always document array dimensions in docstrings or annotations (e.g., `Float[Array, "batch nodes dim"]`, `F: (..., 3, 3)`).

---

## 2. Constitutive Laws & Automatic Differentiation
- **Material Models (`core/material_models.py`):**
  - All ground-truth or analytical constitutive models must inherit from `BaseMaterialModel`.
  - Child classes MUST ONLY implement the strain energy density scalar function $\psi(\mathbf{F})$ via `psi(F)`.
  - **Never hardcode analytical stress derivatives.** First Piola-Kirchhoff stress calculations ($\mathbf{P}(\mathbf{F}) = \frac{\partial \psi}{\partial \mathbf{F}}$) must be performed using the base class automatic differentiation routines (`P(F)`, which wraps `jax.grad`, `jax.vmap`, and `jax.jit`).
  - Register all new constitutive models using the `@register_material("model_name")` decorator.

---

## 3. Dataset Integration (`core/datasetclass.py`)
- **Abstract Inheritance:** All dataset ingest implementations must inherit from `HyperelasticDataset` and implement `get_data()` along with standard sequencing methods (`__len__`, `__getitem__`).
- **Factory Registration:** Any newly created dataset class must be plugged into `DatasetFactory.create()` under a unique descriptive string identifier.
- **Deformation Gradients:** When constructing deformation gradients ($\mathbf{F}$) from node coordinates and displacements, utilize vectorized element-wise operations:
  ```python
  f = jax.vmap(lambda ce, de: deformation_gradient_element(ce, de))(coords_elems, disp_elems)
  ```
  Ensure all 2D plane stress/strain representations are promoted to full `(3, 3)` arrays with $\mathbf{F}_{33} = 1.0$ when passing to materials or invariant functions.

---

## 4. Feature Extraction & Invariants (`core/features.py`)
- All input transformations into GP kernels must utilize subclass implementations of `FeatureExtractor(ABC)`.
- Currently, isotropic models utilize `IsotropicFeatureExtractor`, which separates invariants ($I_1, I_2, J$) into deviatoric (`dev`) and volumetric (`vol`) features.
- Ensure all kernel evaluations in `core/kernel.py` and GP predictions in `core/model.py` (`SparseHyperelasticityGP`) adhere to this `(dev, vol)` separation.

---

## 5. JAX $\leftrightarrow$ PyTorch Interop & Distillation (`distillation/`)
This repository combines JAX (for fast GPU Gaussian Process inference & hyperelastic autograd) with PyTorch (for uncertainty-aware symbolic distillation and model discovery).
- **GPU Memory Safety:** Never run simultaneous active GPU allocations where JAX arrays and PyTorch tensors directly convert on-device without memory buffers. Doing so causes CUDA device collisions or GPU OOM errors.
- **The NumPy Intermediary Bridge:** When transferring GP predictive posteriors (mean and uncertainty variance) to PyTorch discovery scripts (e.g., `distill_uqmodeldisc.py`, `distill_parameters_wasserstein.py`):
  1. Evaluate JAX models and convert all resulting arrays to standard host NumPy arrays using `np.array(jax_arr)`.
  2. Either save intermediates to structured `.npy` / `.npz` files (matching the design in `export_gp_to_pytorch.py`) or decouple script executions between extraction and PyTorch distillation.

---

## 6. Execution Environment (Docker)
- **Always Run Python in Docker:** All Python scripts must be executed within a Docker container rather than using the local Python environment. There is an existing container named `ugp_disc_hyperelastic`. Always use `docker exec` to run commands inside this existing container first (e.g., `docker exec ugp_disc_hyperelastic python <script>`).

import jax
import jax.numpy as jnp
from jax import random, vmap, grad, jit
from typing import Optional, Tuple, Callable, Any

# Enforce mandatory 64-bit precision standard for hyperelastic computations
jax.config.update("jax_enable_x64", True)

from .kernel import rbf
from .dataclass import EnergyDist, StressDist, GPParams, GPWeights
from .features import IsotropicFeatureExtractor, FeatureExtractor


class SparseHyperelasticityGP:
    """
    Sparse Gaussian Process model for hyperelasticity.
    Uses pathwise sampling and Matheron's rule to condition on inducing points.
    Strictly assumes a zero-mean prior for the strain energy density components.
    All evaluation methods support optional explicit parameter and weight passing
    to maintain functional purity during JAX transformations.
    """
    def __init__(self, raw_params: Any, I_z: jnp.ndarray, min_dev: jnp.ndarray, min_vol: jnp.ndarray,
                 max_dev: jnp.ndarray, max_vol: jnp.ndarray, sampling_mode: str = "pws", 
                 beta: float = 1.0, L: int = 200, feature_extractor: Optional[FeatureExtractor] = None,
                 min_aniso: Optional[jnp.ndarray] = None, max_aniso: Optional[jnp.ndarray] = None, aniso_z: Optional[jnp.ndarray] = None):
        self.feature_extractor = feature_extractor if feature_extractor is not None else IsotropicFeatureExtractor()
        # 1. Inducing points split
        self.dev_z = jnp.asarray(I_z[:, :2], dtype=jnp.float64)
        self.vol_z = jnp.asarray(I_z[:, 2:], dtype=jnp.float64)
        self.min_dev = jnp.asarray(min_dev, dtype=jnp.float64)
        self.max_dev = jnp.asarray(max_dev, dtype=jnp.float64)
        self.min_vol = jnp.asarray(min_vol, dtype=jnp.float64)
        self.max_vol = jnp.asarray(max_vol, dtype=jnp.float64)
        
        self.is_anisotropic = (aniso_z is not None)
        if self.is_anisotropic:
            self.aniso_z = jnp.asarray(aniso_z, dtype=jnp.float64)
            self.min_aniso = jnp.asarray(min_aniso, dtype=jnp.float64)
            self.max_aniso = jnp.asarray(max_aniso, dtype=jnp.float64)

        self.sampling_mode = sampling_mode
        self.L = L  # Number of Random Fourier Features for pathwise sampling
        self.beta = beta
        
        # 2. Setup Parameters and Weights
        self.params: GPParams = self.load_params(raw_params)
        self.gpweight: GPWeights = self.precompute_weights(raw_params)

    def _resolve_state(self, params: Optional[GPParams], weights: Optional[GPWeights]) -> Tuple[GPParams, GPWeights]:
        """Resolves functional parameter passing, falling back to instance state if omitted."""
        p = params if params is not None else self.params
        if weights is not None:
            w = weights
        elif params is not None and params is not self.params:
            w = self.precompute_weights_from_loaded(p)
        else:
            w = self.gpweight
        return p, w

    # ---------------------------------------------------------
    # 1. Parameter Management
    # ---------------------------------------------------------
    def load_params(self, p: Any) -> GPParams:
        """Applies physical constraints (e.g., positivity via softplus/exp) to raw parameters."""
        def to_f64(x):
            return jnp.asarray(x, dtype=jnp.float64)

        dev_mu = to_f64(jax.nn.softplus(p.raw_dev_u_mean))
        vol_mu = to_f64(jax.nn.softplus(p.raw_vol_u_mean))
        dev_var = to_f64(jax.nn.softplus(p.raw_dev_u_var))
        vol_var = to_f64(jax.nn.softplus(p.raw_vol_u_var))

        # Force anchor points (First inducing point at zero energy)
        dev_z = to_f64(jax.nn.softplus(p.raw_dev_z)) + to_f64(jnp.array([3.0, 3.0]))
        vol_z = to_f64(jax.nn.softplus(p.raw_vol_z))

        dev_z = dev_z.at[0].set(to_f64(jnp.array([3.0, 3.0])))
        vol_z = vol_z.at[0].set(to_f64(jnp.array([1.0])))

        dev_u_mean = dev_mu.at[0].set(0.0)
        dev_u_var  = dev_var.at[0].set(1e-8)
        vol_u_mean = vol_mu.at[0].set(0.0)
        vol_u_var  = vol_var.at[0].set(1e-8)
        
        kwargs = {}
        if self.is_anisotropic:
            aniso_mu = to_f64(jax.nn.softplus(p.raw_aniso_u_mean))
            aniso_var = to_f64(jax.nn.softplus(p.raw_aniso_u_var))
            aniso_z = to_f64(jax.nn.softplus(p.raw_aniso_z))
            # anchor point at zero energy
            aniso_z = aniso_z.at[0].set(to_f64(jnp.array([0.0])))
            aniso_u_mean = aniso_mu.at[0].set(0.0)
            aniso_u_var = aniso_var.at[0].set(1e-8)
            
            kwargs = dict(
                aniso_ls=to_f64(self.max_aniso * 2 * jax.nn.sigmoid(p.raw_aniso_ls)),
                aniso_sig=to_f64(jnp.exp(p.raw_aniso_sig)),
                aniso_u_mean=aniso_u_mean,
                aniso_u_var=aniso_u_var,
                aniso_z=aniso_z,
                aniso_kappa=to_f64(jax.nn.softplus(p.raw_aniso_kappa))
            )

        return GPParams(
            dev_ls=to_f64(self.max_dev.mean() * 2 * jax.nn.sigmoid(p.raw_dev_ls)),
            dev_sig=to_f64(jnp.exp(p.raw_dev_sig)),
            dev_u_mean=dev_u_mean,
            dev_u_var=dev_u_var,
            dev_z=dev_z,

            vol_ls=to_f64(self.max_vol * 2 * jax.nn.sigmoid(p.raw_vol_ls)),
            vol_sig=to_f64(jnp.exp(p.raw_vol_sig)),
            vol_u_mean=vol_u_mean,
            vol_u_var=vol_u_var,
            vol_z=vol_z,
            vol_kappa=to_f64(jax.nn.softplus(p.raw_vol_kappa)),

            sigma_free_x=to_f64(jnp.exp(p.log_sigma_free_x)),
            sigma_free_y=to_f64(jnp.exp(p.log_sigma_free_y)),
            sigma_fix_x=to_f64(jnp.exp(p.log_sigma_fix_x)),
            sigma_fix_y=to_f64(jnp.exp(p.log_sigma_fix_y)),
            
            **kwargs
        )

    # ---------------------------------------------------------
    # 2. Core GP Mathematics & Weight Precomputation
    # ---------------------------------------------------------
    def _compute_component_weights(self, z: jnp.ndarray, u_mean: jnp.ndarray, u_var: jnp.ndarray, 
                                   ls: jnp.ndarray, sig: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
        """Helper to precompute reusable covariance matrices and vectors for GP."""
        Kzz = rbf(z, z, sig, ls) + 1e-6 * jnp.eye(z.shape[0], dtype=jnp.float64)
        K_inv = jnp.linalg.solve(Kzz, jnp.eye(z.shape[0], dtype=jnp.float64))
        
        # We strictly assume a zero-mean prior, so v_diff is just u_mean - 0
        v_diff = u_mean
        
        M_mat = K_inv @ (Kzz - jnp.diag(u_var)) @ K_inv.T
        trace_term = jnp.trace(K_inv @ jnp.diag(u_var))
        mahalanobis_term = v_diff.T @ K_inv @ v_diff
        log_term = jnp.log(jnp.linalg.det(Kzz)) - jnp.log(jnp.linalg.det(jnp.diag(u_var)))
        
        return Kzz, K_inv, v_diff, trace_term, mahalanobis_term, M_mat, log_term

    def precompute_weights_from_loaded(self, p: GPParams) -> GPWeights:
        """Precomputes weights directly from loaded GPParams."""
        d_res = self._compute_component_weights(p.dev_z, p.dev_u_mean, p.dev_u_var, p.dev_ls, p.dev_sig)
        v_res = self._compute_component_weights(p.vol_z, p.vol_u_mean, p.vol_u_var, p.vol_ls, p.vol_sig)
        
        kwargs = {}
        if self.is_anisotropic:
            a_res = self._compute_component_weights(p.aniso_z, p.aniso_u_mean, p.aniso_u_var, p.aniso_ls, p.aniso_sig)
            kwargs = dict(
                aniso_Kzz=a_res[0], aniso_Kzz_inv=a_res[1], aniso_v=a_res[2], aniso_trace_term=a_res[3], 
                aniso_mahalanobis_term=a_res[4], aniso_M_mat=a_res[5], aniso_logterm=a_res[6]
            )

        return GPWeights(
            dev_Kzz=d_res[0], dev_Kzz_inv=d_res[1], dev_v=d_res[2], dev_trace_term=d_res[3], 
            dev_mahalanobis_term=d_res[4], dev_M_mat=d_res[5], dev_logterm=d_res[6],
            vol_Kzz=v_res[0], vol_Kzz_inv=v_res[1], vol_v=v_res[2], vol_trace_term=v_res[3], 
            vol_mahalanobis_term=v_res[4], vol_M_mat=v_res[5], vol_logterm=v_res[6],
            **kwargs
        )

    def precompute_weights(self, params: Any) -> GPWeights:
        """Precomputes weights for both deviatoric and volumetric components from raw parameters."""
        p = self.load_params(params)
        return self.precompute_weights_from_loaded(p)

    # ---------------------------------------------------------
    # 3. Pathwise Sampling (Physics-Informed)
    # ---------------------------------------------------------
    def _sample_path_components(self, key: jnp.ndarray, p: GPParams, w: GPWeights):
        """
        Generates independent pathwise sample functions for deviatoric and volumetric components.
        Splits PRNGKey into 12 statistically independent streams to prevent Fourier feature 
        correlation with variational inducing values.
        """
        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12 = random.split(key, 12)
        
        # 1. Random Fourier Features for Prior Paths (enforcing 64-bit precision)
        w_dev_prior = random.normal(k1, (self.L,), dtype=jnp.float64)
        w_vol_prior = random.normal(k2, (self.L,), dtype=jnp.float64)

        W_dev = random.normal(k3, (2, self.L), dtype=jnp.float64) 
        b_dev = random.uniform(k4, (self.L,), dtype=jnp.float64) * 2 * jnp.pi
        W_vol = random.normal(k5, (1, self.L), dtype=jnp.float64) 
        b_vol = random.uniform(k6, (self.L,), dtype=jnp.float64) * 2 * jnp.pi

        def f_prior_dev(d):
            phi = jnp.sqrt(2.0 * p.dev_sig**2 / self.L) * jnp.cos(jnp.dot(d, W_dev / p.dev_ls[:, None]) + b_dev)
            return jnp.dot(phi, w_dev_prior)

        def f_prior_vol(v):
            phi = jnp.sqrt(2.0 * p.vol_sig**2 / self.L) * jnp.cos(jnp.dot(v, W_vol / p.vol_ls[:, None]) + b_vol)
            return jnp.dot(phi, w_vol_prior)

        # 2. Sample Inducing Values u ~ q(u) using independent PRNG keys
        u_dev = jax.random.multivariate_normal(k7, p.dev_u_mean, jnp.diag(p.dev_u_var), dtype=jnp.float64)
        u_vol = jax.random.multivariate_normal(k8, p.vol_u_mean, jnp.diag(p.vol_u_var), dtype=jnp.float64)

        # 3. Correction Vectors (Matheron's Rule)
        v_dev_corr = jnp.linalg.solve(w.dev_Kzz, u_dev - vmap(f_prior_dev)(p.dev_z))
        v_vol_corr = jnp.linalg.solve(w.vol_Kzz, u_vol - vmap(f_prior_vol)(p.vol_z))

        def path_dev(dev_feats):
            k_dz = rbf(dev_feats, p.dev_z, p.dev_sig, p.dev_ls)
            return f_prior_dev(dev_feats) + jnp.dot(k_dz, v_dev_corr)

        def path_vol(vol_feats):
            k_vz = rbf(vol_feats, p.vol_z, p.vol_sig, p.vol_ls)
            return f_prior_vol(vol_feats) + jnp.dot(k_vz, v_vol_corr)

        if self.is_anisotropic:
            w_aniso_prior = random.normal(k9, (self.L,), dtype=jnp.float64)
            W_aniso = random.normal(k10, (1, self.L), dtype=jnp.float64) 
            b_aniso = random.uniform(k11, (self.L,), dtype=jnp.float64) * 2 * jnp.pi

            def f_prior_aniso(v):
                phi = jnp.sqrt(2.0 * p.aniso_sig**2 / self.L) * jnp.cos(jnp.dot(v, W_aniso / p.aniso_ls[:, None]) + b_aniso)
                return jnp.dot(phi, w_aniso_prior)

            u_aniso = jax.random.multivariate_normal(k12, p.aniso_u_mean, jnp.diag(p.aniso_u_var), dtype=jnp.float64)
            v_aniso_corr = jnp.linalg.solve(w.aniso_Kzz, u_aniso - vmap(f_prior_aniso)(p.aniso_z))

            def path_aniso(aniso_feats):
                k_az = rbf(aniso_feats, p.aniso_z, p.aniso_sig, p.aniso_ls)
                return f_prior_aniso(aniso_feats) + jnp.dot(k_az, v_aniso_corr)
            
            return path_dev, path_vol, path_aniso

        return path_dev, path_vol

    def get_path_psi_fn(self, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> Callable[[jnp.ndarray], jnp.ndarray]:
        """
        Returns a differentiable scalar function psi(F) for one realization.
        This uses Matheron's rule to condition random prior features on the inducing points.
        """
        p, w = self._resolve_state(params, weights)
        paths = self._sample_path_components(key, p, w)

        def path_psi(f: jnp.ndarray) -> jnp.ndarray:
            feats = self.feature_extractor.extract(f)
            psi_dev = paths[0](feats[0])
            psi_vol = paths[1](feats[1])
            total_psi = psi_dev + psi_vol
            if self.is_anisotropic:
                total_psi += paths[2](feats[2])
            return total_psi.squeeze()

        return path_psi

    def get_path_components_psi_fn(self, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> Callable[[jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        """Returns a scalar function that outputs (psi_dev, psi_vol, psi_aniso) separately."""
        p, w = self._resolve_state(params, weights)
        paths = self._sample_path_components(key, p, w)
        path_dev, path_vol = paths[0], paths[1]
        path_aniso = paths[2] if self.is_anisotropic else None

        def path_components_psi(f: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            feats = self.feature_extractor.extract(f)
            dev, vol = feats[0], feats[1]
            psi_dev = path_dev(dev)
            psi_vol = path_vol(vol)
            if self.is_anisotropic:
                psi_aniso = path_aniso(feats[2])
            else:
                psi_aniso = jnp.zeros_like(psi_dev)
            return psi_dev.squeeze(), psi_vol.squeeze(), psi_aniso.squeeze()

        return path_components_psi

    # ---------------------------------------------------------
    # 4. API Endpoints for Loss / Evaluation
    # ---------------------------------------------------------
    def psi(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Calculates Energy across a mesh for a single realization."""
        if self.sampling_mode == "mds":
            return self.psi_mds(f_mesh, key, params=params, weights=weights)
        elif self.sampling_mode == "pws":
            return self.psi_pws(f_mesh, key, params=params, weights=weights)
        raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

    def piola(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Calculates Piola Stress field across a mesh for a single realization."""
        if self.sampling_mode == "mds":
            return self.piola_mds(f_mesh, key, params=params, weights=weights)
        elif self.sampling_mode == "pws":
            return self.piola_pws(f_mesh, key, params=params, weights=weights)
        raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

    def psi_pws(self, f: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        path_psi = self.get_path_psi_fn(key, params=params, weights=weights)
        if f.ndim == 3:
            return jax.vmap(path_psi)(f)
        return path_psi(f)

    def piola_pws(self, f: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        path_psi = self.get_path_psi_fn(key, params=params, weights=weights)
        piola_fn = grad(path_psi)
        if f.ndim == 3:
            return jax.vmap(piola_fn)(f)
        return piola_fn(f)

    def kl_divergence(self, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes the KL divergence for ELBO training."""
        p, w = self._resolve_state(params, weights)
        def component_kl(ma, log_t, tr, M):
            return 0.5 * (log_t - M + tr + ma)
        
        dev_kl = component_kl(w.dev_mahalanobis_term, w.dev_logterm, 
                              w.dev_trace_term, p.dev_z.shape[0])
        vol_kl = component_kl(w.vol_mahalanobis_term, w.vol_logterm, 
                              w.vol_trace_term, p.vol_z.shape[0])
        total_kl = dev_kl + vol_kl
        if self.is_anisotropic:
            aniso_kl = component_kl(w.aniso_mahalanobis_term, w.aniso_logterm, 
                                    w.aniso_trace_term, p.aniso_z.shape[0])
            total_kl += aniso_kl
            
        return total_kl * self.beta

    # ---------------------------------------------------------
    # 5. Analytical GP Moments (Mean & Covariance for MDS)
    # ---------------------------------------------------------
    def dev_gp_mean(self, d: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        k_dz = rbf(d, p.dev_z, p.dev_sig, p.dev_ls)
        gp_term = k_dz @ w.dev_Kzz_inv @ w.dev_v
        return gp_term  # Zero mean prior
    
    def vol_gp_mean(self, v: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        k_vz = rbf(v, p.vol_z, p.vol_sig, p.vol_ls)
        gp_term = k_vz @ w.vol_Kzz_inv @ w.vol_v
        return gp_term

    def aniso_gp_mean(self, a: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        k_az = rbf(a, p.aniso_z, p.aniso_sig, p.aniso_ls)
        gp_term = k_az @ w.aniso_Kzz_inv @ w.aniso_v
        return gp_term

    def psi_gp_mean(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        is_single = (f.ndim == 2)
        if is_single:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev, vol = feats[0], feats[1]
        gp_mean = self.dev_gp_mean(dev, params=p, weights=w) + self.vol_gp_mean(vol, params=p, weights=w)
        if self.is_anisotropic:
            aniso = feats[2]
            gp_mean += self.aniso_gp_mean(aniso, params=p, weights=w)
        res = gp_mean.squeeze()
        return res if not is_single else jnp.reshape(res, ())

    def psi_gp_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes marginal energy variance using O(N) memory without assembling N x N Gram matrices."""
        p, w = self._resolve_state(params, weights)
        is_single = (f.ndim == 2)
        if is_single:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev, vol = feats[0], feats[1]
        
        k_dz = rbf(dev, p.dev_z, p.dev_sig, p.dev_ls)
        var_dev = jnp.maximum(p.dev_sig**2 - jnp.sum((k_dz @ w.dev_M_mat) * k_dz, axis=-1), 1e-8)

        k_vz = rbf(vol, p.vol_z, p.vol_sig, p.vol_ls)
        var_vol = jnp.maximum(p.vol_sig**2 - jnp.sum((k_vz @ w.vol_M_mat) * k_vz, axis=-1), 1e-8)
        res = var_dev + var_vol
        if self.is_anisotropic:
            aniso = feats[2]
            k_az = rbf(aniso, p.aniso_z, p.aniso_sig, p.aniso_ls)
            var_aniso = jnp.maximum(p.aniso_sig**2 - jnp.sum((k_az @ w.aniso_M_mat) * k_az, axis=-1), 1e-8)
            res += var_aniso
            
        return res if not is_single else res[0]

    def psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Returns the full N x N dense joint covariance matrix for psi."""
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev, vol = feats[0], feats[1]
        k_dz = rbf(dev, p.dev_z, p.dev_sig, p.dev_ls)
        k_dd = rbf(dev, dev, p.dev_sig, p.dev_ls)
        cov_mat_dev = k_dd - k_dz @ w.dev_M_mat @ k_dz.T

        k_vz = rbf(vol, p.vol_z, p.vol_sig, p.vol_ls)
        k_vv = rbf(vol, vol, p.vol_sig, p.vol_ls)
        cov_mat_vol = k_vv - k_vz @ w.vol_M_mat @ k_vz.T
        
        cov_full = cov_mat_dev + cov_mat_vol
        if self.is_anisotropic:
            aniso = feats[2]
            k_az = rbf(aniso, p.aniso_z, p.aniso_sig, p.aniso_ls)
            k_aa = rbf(aniso, aniso, p.aniso_sig, p.aniso_ls)
            cov_mat_aniso = k_aa - k_az @ w.aniso_M_mat @ k_az.T
            cov_full += cov_mat_aniso
            
        cov_full = 0.5 * (cov_full + cov_full.T) # Guarantee symmetry
        return cov_full

    def dev_psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev = feats[0]
        k_dz = rbf(dev, p.dev_z, p.dev_sig, p.dev_ls)
        k_dd = rbf(dev, dev, p.dev_sig, p.dev_ls)
        cov_mat_dev = k_dd - k_dz @ w.dev_M_mat @ k_dz.T
        return 0.5 * (cov_mat_dev + cov_mat_dev.T)

    def vol_psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        vol = feats[1]
        k_vz = rbf(vol, p.vol_z, p.vol_sig, p.vol_ls)
        k_vv = rbf(vol, vol, p.vol_sig, p.vol_ls)
        cov_mat_vol = k_vv - k_vz @ w.vol_M_mat @ k_vz.T
        return 0.5 * (cov_mat_vol + cov_mat_vol.T)

    def piola_gp_cov_pair(self, f1: jnp.ndarray, f2: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes double-differentiation cross-covariance between two deformation gradient tensors."""
        p, w = self._resolve_state(params, weights)

        def psi_cov_single(fa, fb):
            if self.is_anisotropic:
                dev1, vol1, aniso1 = self.feature_extractor.extract(fa)
                dev2, vol2, aniso2 = self.feature_extractor.extract(fb)
                
                k_d1z = rbf(dev1[None, :], p.dev_z, p.dev_sig, p.dev_ls)
                k_dz2 = rbf(p.dev_z, dev2[None, :], p.dev_sig, p.dev_ls)
                k_d1d2 = rbf(dev1[None, :], dev2[None, :], p.dev_sig, p.dev_ls)
                cov_dev = k_d1d2 - k_d1z @ w.dev_M_mat @ k_dz2
                
                k_v1z = rbf(vol1[None, :], p.vol_z, p.vol_sig, p.vol_ls)
                k_vz2 = rbf(p.vol_z, vol2[None, :], p.vol_sig, p.vol_ls)
                k_v1v2 = rbf(vol1[None, :], vol2[None, :], p.vol_sig, p.vol_ls)
                cov_vol = k_v1v2 - k_v1z @ w.vol_M_mat @ k_vz2

                k_a1z = rbf(aniso1[None, :], p.aniso_z, p.aniso_sig, p.aniso_ls)
                k_az2 = rbf(p.aniso_z, aniso2[None, :], p.aniso_sig, p.aniso_ls)
                k_a1a2 = rbf(aniso1[None, :], aniso2[None, :], p.aniso_sig, p.aniso_ls)
                cov_aniso = k_a1a2 - k_a1z @ w.aniso_M_mat @ k_az2

                return (cov_dev + cov_vol + cov_aniso).squeeze()
            else:
                dev1, vol1 = self.feature_extractor.extract(fa)
                dev2, vol2 = self.feature_extractor.extract(fb)
                
                k_d1z = rbf(dev1[None, :], p.dev_z, p.dev_sig, p.dev_ls)
                k_dz2 = rbf(p.dev_z, dev2[None, :], p.dev_sig, p.dev_ls)
                k_d1d2 = rbf(dev1[None, :], dev2[None, :], p.dev_sig, p.dev_ls)
                cov_dev = k_d1d2 - k_d1z @ w.dev_M_mat @ k_dz2
                
                k_v1z = rbf(vol1[None, :], p.vol_z, p.vol_sig, p.vol_ls)
                k_vz2 = rbf(p.vol_z, vol2[None, :], p.vol_sig, p.vol_ls)
                k_v1v2 = rbf(vol1[None, :], vol2[None, :], p.vol_sig, p.vol_ls)
                cov_vol = k_v1v2 - k_v1z @ w.vol_M_mat @ k_vz2
                
                return (cov_dev + cov_vol).squeeze()

        hessian_cov = jax.jacfwd(jax.jacrev(psi_cov_single, argnums=0), argnums=1)
        return hessian_cov(f1, f2)

    def piola_gp_var(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """
        Computes the variance of the Piola Stress components using 
        double differentiation of the predictive covariance at f.
        """
        return self.piola_gp_cov_pair(f, f, params=params, weights=weights)

    def psi_det(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if self.is_anisotropic:
            dev, vol, aniso = self.feature_extractor.extract(f)
            return (self.dev_gp_mean(dev[None, :], params=p, weights=w).reshape() + 
                    self.vol_gp_mean(vol[None, :], params=p, weights=w).reshape() + 
                    self.aniso_gp_mean(aniso[None, :], params=p, weights=w).reshape())
        else:
            dev, vol = self.feature_extractor.extract(f)
            return self.dev_gp_mean(dev[None, :], params=p, weights=w).reshape() + self.vol_gp_mean(vol[None, :], params=p, weights=w).reshape()
    
    def piola_det(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        def single_psi_det(f_single):
            if self.is_anisotropic:
                dev, vol, aniso = self.feature_extractor.extract(f_single)
                return (self.dev_gp_mean(dev[None, :], params=p, weights=w).reshape() + 
                        self.vol_gp_mean(vol[None, :], params=p, weights=w).reshape() + 
                        self.aniso_gp_mean(aniso[None, :], params=p, weights=w).reshape())
            else:
                dev, vol = self.feature_extractor.extract(f_single)
                return self.dev_gp_mean(dev[None, :], params=p, weights=w).reshape() + self.vol_gp_mean(vol[None, :], params=p, weights=w).reshape()
        piola_det_fn = jax.grad(single_psi_det)
        return piola_det_fn(f)

    def psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        p, w = self._resolve_state(params, weights)
        f_mesh = jnp.asarray(f_mesh, dtype=jnp.float64)
        posterior_mean = self.psi_gp_mean(f_mesh, params=p, weights=w)
        posterior_var = self.psi_gp_cov(f_mesh, params=p, weights=w)
        return EnergyDist(posterior_mean, posterior_var)

    def dev_psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f_mesh)
        dev = feats[0]
        mean = self.dev_gp_mean(dev, params=p, weights=w)
        k_dz = rbf(dev, p.dev_z, p.dev_sig, p.dev_ls)
        var_dev = jnp.maximum(p.dev_sig**2 - jnp.sum((k_dz @ w.dev_M_mat) * k_dz, axis=-1), 1e-8)
        if is_single:
            return EnergyDist(mean.reshape(), var_dev[0])
        return EnergyDist(mean.squeeze(), var_dev)
        
    def vol_psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f_mesh)
        vol = feats[1]
        mean = self.vol_gp_mean(vol, params=p, weights=w)
        k_vz = rbf(vol, p.vol_z, p.vol_sig, p.vol_ls)
        var_vol = jnp.maximum(p.vol_sig**2 - jnp.sum((k_vz @ w.vol_M_mat) * k_vz, axis=-1), 1e-8)
        if is_single:
            return EnergyDist(mean.reshape(), var_vol[0])
        return EnergyDist(mean.squeeze(), var_vol)

    def aniso_psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        if not self.is_anisotropic:
            return EnergyDist(jnp.zeros(f_mesh.shape[0]), jnp.zeros(f_mesh.shape[0]))
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f_mesh)
        aniso = feats[2]
        mean = self.aniso_gp_mean(aniso, params=p, weights=w)
        k_az = rbf(aniso, p.aniso_z, p.aniso_sig, p.aniso_ls)
        var_aniso = jnp.maximum(p.aniso_sig**2 - jnp.sum((k_az @ w.aniso_M_mat) * k_az, axis=-1), 1e-8)
        if is_single:
            return EnergyDist(mean.reshape(), var_aniso[0])
        return EnergyDist(mean.squeeze(), var_aniso)

    def piola_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> StressDist:
        """
        Calculates the Mean Piola Stress and variances across a mesh of deformation gradients.
        Supports both (N, 2, 2) and promoted (N, 3, 3) arrays.
        """
        p, w = self._resolve_state(params, weights)
        f_mesh = jnp.asarray(f_mesh, dtype=jnp.float64)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        
        def single_psi_mean(f):
            if self.is_anisotropic:
                dev, vol, aniso = self.feature_extractor.extract(f)
                return (self.dev_gp_mean(dev[None, :], params=p, weights=w) + 
                        self.vol_gp_mean(vol[None, :], params=p, weights=w) + 
                        self.aniso_gp_mean(aniso[None, :], params=p, weights=w)).reshape()
            else:
                dev, vol = self.feature_extractor.extract(f)
                return (self.dev_gp_mean(dev[None, :], params=p, weights=w) + 
                        self.vol_gp_mean(vol[None, :], params=p, weights=w)).reshape()

        piola_mean_fn = jax.vmap(jax.grad(single_psi_mean))
        piola_means = piola_mean_fn(f_mesh)

        def single_piola_var(f):
            return jnp.einsum('ijij->ij', self.piola_gp_var(f, params=p, weights=w))

        piola_vars_fn = jax.vmap(single_piola_var)
        piola_vars = piola_vars_fn(f_mesh)
        if is_single:
            return StressDist(piola_means[0], piola_vars[0])
        return StressDist(piola_means, piola_vars)
    
    def psi_mds(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        dist = self.psi_dist(f_mesh, params=p, weights=w)
        # Convert marginal variance vector into diagonal covariance for multivariate drawing
        psi = jax.random.multivariate_normal(key, dist.mean, jnp.diag(dist.var), dtype=jnp.float64)
        return psi if not is_single else psi[0]

    def piola_mds(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """
        Samples the ENTIRE correlated Piola stress field across a mesh.
        Dynamically handles both (N, 2, 2) and promoted (N, 3, 3) deformation gradients.
        """
        p, w = self._resolve_state(params, weights)
        f_mesh = jnp.asarray(f_mesh, dtype=jnp.float64)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        N = f_mesh.shape[0]
        d1, d2 = f_mesh.shape[1], f_mesh.shape[2]
        d_flat = N * d1 * d2

        dist_mean = self.piola_dist(f_mesh, params=p, weights=w).mean.reshape(-1)
        
        var_fn = lambda f1, f2: self.piola_gp_cov_pair(f1, f2, params=p, weights=w)
        K_full_tensor = jax.vmap(jax.vmap(var_fn, in_axes=(None, 0)), in_axes=(0, None))(f_mesh, f_mesh)
        
        # Permute (N1, N2, row1, col1, row2, col2) -> (N1, row1, col1, N2, row2, col2)
        K_joint = K_full_tensor.transpose(0, 2, 3, 1, 4, 5).reshape(d_flat, d_flat)
        K_joint += 1e-6 * jnp.eye(d_flat, dtype=jnp.float64)  # Numerical stability jitter
        
        sample_flat = jax.random.multivariate_normal(key, dist_mean, K_joint, dtype=jnp.float64)
        res = sample_flat.reshape(N, d1, d2)
        return res if not is_single else res[0]

import jax
import jax.numpy as jnp
from jax import random, vmap, grad, jit
from typing import Optional, Tuple, Callable, Any

# Enforce mandatory 64-bit precision standard for hyperelastic computations
jax.config.update("jax_enable_x64", True)

from .kernel import rbf
from .dataclass import EnergyDist, StressDist, GPParams, GPWeights
from .features import IsotropicFeatureExtractor, FeatureExtractor
from .gp_component import (
    load_component_params,
    compute_component_weights,
    sample_component_path,
    predict_component_mean,
    predict_component_var,
    predict_component_joint_cov,
    compute_component_kl,
)


class SparseHyperelasticityGP:
    """
    Sparse Gaussian Process model for hyperelasticity.
    Uses pathwise sampling (RFF + Matheron's rule) or Cholesky joint sampling.
    Strictly assumes a zero-mean prior for the strain energy density components.
    All evaluation methods support optional explicit parameter and weight passing
    to maintain functional purity during JAX transformations.
    """
    def __init__(self, raw_params: Any, I_z: jnp.ndarray, min_dev: jnp.ndarray, min_vol: jnp.ndarray,
                 max_dev: jnp.ndarray, max_vol: jnp.ndarray, sampling_mode: str = "pathwise", 
                 beta: float = 1.0, L: int = 200, feature_extractor: Optional[FeatureExtractor] = None,
                 min_aniso: Optional[jnp.ndarray] = None, max_aniso: Optional[jnp.ndarray] = None, aniso_z: Optional[jnp.ndarray] = None,
                 covariance_mode: str = "diag", normalize_ell: int = 0,
                 u_var_anchor: float = 1e-12, kzz_jitter: float = 1e-8, **kwargs):
        self.feature_extractor = feature_extractor if feature_extractor is not None else IsotropicFeatureExtractor()
        # 1. Inducing points split
        self.dev_z = jnp.asarray(I_z[:, :2], dtype=jnp.float64)
        self.vol_z = jnp.asarray(I_z[:, 2:3], dtype=jnp.float64)
        self.min_dev = jnp.asarray(min_dev, dtype=jnp.float64)
        self.max_dev = jnp.asarray(max_dev, dtype=jnp.float64)
        self.min_vol = jnp.asarray(min_vol, dtype=jnp.float64)
        self.max_vol = jnp.asarray(max_vol, dtype=jnp.float64)
        
        self.is_anisotropic = (aniso_z is not None)
        if self.is_anisotropic:
            self.aniso_z = jnp.asarray(aniso_z, dtype=jnp.float64)
            self.min_aniso = jnp.asarray(min_aniso, dtype=jnp.float64)
            self.max_aniso = jnp.asarray(max_aniso, dtype=jnp.float64)

        # Standardize sampling mode: pathwise (formerly pws) vs cholesky (formerly mds)
        if sampling_mode in ("pathwise", "pws"):
            self.sampling_mode = "pathwise"
        elif sampling_mode in ("cholesky", "mds"):
            self.sampling_mode = "cholesky"
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}. Expected 'pathwise' or 'cholesky'.")

        self.L = L  # Number of Random Fourier Features for pathwise sampling
        self.beta = beta
        self.covariance_mode = covariance_mode
        self.normalize_ell = int(normalize_ell)
        self.u_var_anchor = float(u_var_anchor)
        self.kzz_jitter = float(kzz_jitter)
        
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
        to_f64 = lambda x: jnp.asarray(x, dtype=jnp.float64)

        dev_p = load_component_params(
            raw_ls=p.raw_dev_ls, raw_sig=p.raw_dev_sig,
            raw_u_mean=p.raw_dev_u_mean, raw_u_var=p.raw_dev_u_var,
            raw_z=p.raw_dev_z, anchor=jnp.array([3.0, 3.0]),
            covariance_mode=self.covariance_mode, max_val=self.max_dev,
            u_var_anchor=self.u_var_anchor, is_dev=True
        )

        vol_p = load_component_params(
            raw_ls=p.raw_vol_ls, raw_sig=p.raw_vol_sig,
            raw_u_mean=p.raw_vol_u_mean, raw_u_var=p.raw_vol_u_var,
            raw_z=p.raw_vol_z, anchor=jnp.array([1.0]),
            covariance_mode=self.covariance_mode, max_val=self.max_vol,
            u_var_anchor=self.u_var_anchor, is_dev=False
        )

        kwargs = {}
        if self.is_anisotropic:
            dim_aniso = p.raw_aniso_z.shape[-1]
            aniso_p = load_component_params(
                raw_ls=p.raw_aniso_ls, raw_sig=p.raw_aniso_sig,
                raw_u_mean=p.raw_aniso_u_mean, raw_u_var=p.raw_aniso_u_var,
                raw_z=p.raw_aniso_z, anchor=jnp.ones(dim_aniso),
                covariance_mode=self.covariance_mode, max_val=self.max_aniso,
                u_var_anchor=self.u_var_anchor, is_dev=False
            )
            kwargs = dict(
                aniso_ls=aniso_p.ls,
                aniso_sig=aniso_p.sig,
                aniso_u_mean=aniso_p.u_mean,
                aniso_u_var=aniso_p.u_var,
                aniso_z=aniso_p.z
            )
            if getattr(p, "raw_aniso_theta_mean", None) is not None:
                kwargs["aniso_theta_mean"] = to_f64(jnp.pi * (jax.nn.sigmoid(p.raw_aniso_theta_mean) - 0.5))
            if getattr(p, "raw_aniso_theta_var", None) is not None:
                kwargs["aniso_theta_var"] = to_f64(jax.nn.softplus(p.raw_aniso_theta_var) + 1e-6)

        return GPParams(
            dev_ls=dev_p.ls,
            dev_sig=dev_p.sig,
            dev_u_mean=dev_p.u_mean,
            dev_u_var=dev_p.u_var,
            dev_z=dev_p.z,

            vol_ls=vol_p.ls,
            vol_sig=vol_p.sig,
            vol_u_mean=vol_p.u_mean,
            vol_u_var=vol_p.u_var,
            vol_z=vol_p.z,

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
        w = compute_component_weights(z, u_mean, u_var, ls, sig, 
                                      covariance_mode=self.covariance_mode, kzz_jitter=self.kzz_jitter)
        return w.Kzz, w.Kzz_inv, w.v, w.trace_term, w.mahalanobis_term, w.M_mat, w.logterm

    def precompute_weights_from_loaded(self, p: GPParams) -> GPWeights:
        """Precomputes weights directly from loaded GPParams."""
        d_w = compute_component_weights(p.dev_z, p.dev_u_mean, p.dev_u_var, p.dev_ls, p.dev_sig,
                                        covariance_mode=self.covariance_mode, kzz_jitter=self.kzz_jitter)
        v_w = compute_component_weights(p.vol_z, p.vol_u_mean, p.vol_u_var, p.vol_ls, p.vol_sig,
                                        covariance_mode=self.covariance_mode, kzz_jitter=self.kzz_jitter)
        
        kwargs = {}
        if self.is_anisotropic:
            a_w = compute_component_weights(p.aniso_z, p.aniso_u_mean, p.aniso_u_var, p.aniso_ls, p.aniso_sig,
                                            covariance_mode=self.covariance_mode, kzz_jitter=self.kzz_jitter)
            kwargs = dict(
                aniso_Kzz=a_w.Kzz, aniso_Kzz_inv=a_w.Kzz_inv, aniso_v=a_w.v, aniso_trace_term=a_w.trace_term, 
                aniso_mahalanobis_term=a_w.mahalanobis_term, aniso_M_mat=a_w.M_mat, aniso_logterm=a_w.logterm
            )

        return GPWeights(
            dev_Kzz=d_w.Kzz, dev_Kzz_inv=d_w.Kzz_inv, dev_v=d_w.v, dev_trace_term=d_w.trace_term, 
            dev_mahalanobis_term=d_w.mahalanobis_term, dev_M_mat=d_w.M_mat, dev_logterm=d_w.logterm,
            vol_Kzz=v_w.Kzz, vol_Kzz_inv=v_w.Kzz_inv, vol_v=v_w.v, vol_trace_term=v_w.trace_term, 
            vol_mahalanobis_term=v_w.mahalanobis_term, vol_M_mat=v_w.M_mat, vol_logterm=v_w.logterm,
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
        keys = random.split(key, 12)
        
        path_dev = sample_component_path(
            k_prior_w=keys[0], k_W=keys[2], k_b=keys[3], k_u=keys[6],
            z=p.dev_z, u_mean=p.dev_u_mean, u_var=p.dev_u_var, ls=p.dev_ls, sig=p.dev_sig,
            Kzz=w.dev_Kzz, L=self.L, covariance_mode=self.covariance_mode
        )
        path_vol = sample_component_path(
            k_prior_w=keys[1], k_W=keys[4], k_b=keys[5], k_u=keys[7],
            z=p.vol_z, u_mean=p.vol_u_mean, u_var=p.vol_u_var, ls=p.vol_ls, sig=p.vol_sig,
            Kzz=w.vol_Kzz, L=self.L, covariance_mode=self.covariance_mode
        )

        if self.is_anisotropic:
            path_aniso = sample_component_path(
                k_prior_w=keys[8], k_W=keys[9], k_b=keys[10], k_u=keys[11],
                z=p.aniso_z, u_mean=p.aniso_u_mean, u_var=p.aniso_u_var, ls=p.aniso_ls, sig=p.aniso_sig,
                Kzz=w.aniso_Kzz, L=self.L, covariance_mode=self.covariance_mode
            )
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
        if self.sampling_mode in ("cholesky", "mds"):
            return self.psi_cholesky(f_mesh, key, params=params, weights=weights)
        elif self.sampling_mode in ("pathwise", "pws"):
            return self.psi_pathwise(f_mesh, key, params=params, weights=weights)
        raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

    def piola(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Calculates Piola Stress field across a mesh for a single realization."""
        if self.sampling_mode in ("cholesky", "mds"):
            return self.piola_cholesky(f_mesh, key, params=params, weights=weights)
        elif self.sampling_mode in ("pathwise", "pws"):
            return self.piola_pathwise(f_mesh, key, params=params, weights=weights)
        raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")

    def psi_pws(self, f: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Pathwise energy prediction (alias: psi_pathwise)."""
        path_psi = self.get_path_psi_fn(key, params=params, weights=weights)
        if f.ndim == 3:
            return jax.vmap(path_psi)(f)
        return path_psi(f)

    psi_pathwise = psi_pws

    def piola_pws(self, f: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Pathwise Piola stress prediction (alias: piola_pathwise)."""
        path_psi = self.get_path_psi_fn(key, params=params, weights=weights)
        piola_fn = grad(path_psi)
        if f.ndim == 3:
            return jax.vmap(piola_fn)(f)
        return piola_fn(f)

    piola_pathwise = piola_pws

    def kl_divergence(self, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes the KL divergence for ELBO training."""
        p, w = self._resolve_state(params, weights)
        
        dev_kl = compute_component_kl(w.dev_mahalanobis_term, w.dev_logterm, w.dev_trace_term, p.dev_z.shape[0])
        vol_kl = compute_component_kl(w.vol_mahalanobis_term, w.vol_logterm, w.vol_trace_term, p.vol_z.shape[0])
        total_kl = dev_kl + vol_kl

        if self.is_anisotropic:
            aniso_kl = compute_component_kl(w.aniso_mahalanobis_term, w.aniso_logterm, w.aniso_trace_term, p.aniso_z.shape[0])
            total_kl += aniso_kl
            
        return total_kl * self.beta

    # ---------------------------------------------------------
    # 5. Analytical GP Moments (Mean & Covariance for Cholesky / Analytics)
    # ---------------------------------------------------------
    def dev_gp_mean(self, d: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        return predict_component_mean(d, p.dev_z, p.dev_sig, p.dev_ls, w.dev_Kzz_inv, w.dev_v)
    
    def vol_gp_mean(self, v: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        return predict_component_mean(v, p.vol_z, p.vol_sig, p.vol_ls, w.vol_Kzz_inv, w.vol_v)

    def aniso_gp_mean(self, a: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        return predict_component_mean(a, p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_Kzz_inv, w.aniso_v)

    def psi_gp_mean(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        is_single = (f.ndim == 2)
        if is_single:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev, vol = feats[0], feats[1]
        gp_mean = self.dev_gp_mean(dev, params=p, weights=w) + self.vol_gp_mean(vol, params=p, weights=w)
        if self.is_anisotropic:
            gp_mean += self.aniso_gp_mean(feats[2], params=p, weights=w)
        res = gp_mean.reshape(f.shape[0])
        return res if not is_single else res[0]

    def psi_gp_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes marginal energy variance using O(N) memory without assembling N x N Gram matrices."""
        p, w = self._resolve_state(params, weights)
        is_single = (f.ndim == 2)
        if is_single:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        
        var_dev = predict_component_var(feats[0], p.dev_z, p.dev_sig, p.dev_ls, w.dev_M_mat)
        var_vol = predict_component_var(feats[1], p.vol_z, p.vol_sig, p.vol_ls, w.vol_M_mat)
        res = var_dev + var_vol
        if self.is_anisotropic:
            res += predict_component_var(feats[2], p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_M_mat)
        res = res.reshape(f.shape[0])
        return res if not is_single else res[0]

    def psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Returns the full N x N dense joint covariance matrix for psi."""
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        dev, vol = feats[0], feats[1]
        
        cov_full = (predict_component_joint_cov(dev, dev, p.dev_z, p.dev_sig, p.dev_ls, w.dev_M_mat) +
                    predict_component_joint_cov(vol, vol, p.vol_z, p.vol_sig, p.vol_ls, w.vol_M_mat))
        if self.is_anisotropic:
            cov_full += predict_component_joint_cov(feats[2], feats[2], p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_M_mat)
            
        return cov_full

    def dev_psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        return predict_component_joint_cov(feats[0], feats[0], p.dev_z, p.dev_sig, p.dev_ls, w.dev_M_mat)

    def vol_psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        return predict_component_joint_cov(feats[1], feats[1], p.vol_z, p.vol_sig, p.vol_ls, w.vol_M_mat)

    def aniso_psi_joint_cov(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        p, w = self._resolve_state(params, weights)
        if f.ndim == 2:
            f = f[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f)
        return predict_component_joint_cov(feats[2], feats[2], p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_M_mat)

    def piola_gp_cov_pair(self, f1: jnp.ndarray, f2: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes double-differentiation cross-covariance between two deformation gradient tensors."""
        p, w = self._resolve_state(params, weights)

        def psi_cov_single(fa, fb):
            feats_a = self.feature_extractor.extract(fa)
            feats_b = self.feature_extractor.extract(fb)
            
            cov_dev = predict_component_joint_cov(feats_a[0][None, :], feats_b[0][None, :], p.dev_z, p.dev_sig, p.dev_ls, w.dev_M_mat)
            cov_vol = predict_component_joint_cov(feats_a[1][None, :], feats_b[1][None, :], p.vol_z, p.vol_sig, p.vol_ls, w.vol_M_mat)
            total = cov_dev + cov_vol
            if self.is_anisotropic:
                cov_aniso = predict_component_joint_cov(feats_a[2][None, :], feats_b[2][None, :], p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_M_mat)
                total += cov_aniso
            return total.squeeze()

        hessian_cov = jax.jacfwd(jax.jacrev(psi_cov_single, argnums=0), argnums=1)
        return hessian_cov(f1, f2)

    def piola_gp_var(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Computes the variance of the Piola Stress components using double differentiation."""
        return self.piola_gp_cov_pair(f, f, params=params, weights=weights)

    def psi_det(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Deterministic strain energy density (GP mean)."""
        return self.psi_gp_mean(f, params=params, weights=weights)
    
    def piola_det(self, f: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """Deterministic Piola stress via automatic differentiation of deterministic energy."""
        p, w = self._resolve_state(params, weights)
        def single_psi_det(f_single):
            return self.psi_gp_mean(f_single, params=p, weights=w)
        return jax.grad(single_psi_det)(f)

    def psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        """Evaluates predictive energy distribution (mean and marginal variance)."""
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
        mean = self.dev_gp_mean(dev, params=p, weights=w).reshape(f_mesh.shape[0])
        var_dev = predict_component_var(dev, p.dev_z, p.dev_sig, p.dev_ls, w.dev_M_mat).reshape(f_mesh.shape[0])
        if is_single:
            return EnergyDist(mean[0], var_dev[0])
        return EnergyDist(mean, var_dev)
        
    def vol_psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f_mesh)
        vol = feats[1]
        mean = self.vol_gp_mean(vol, params=p, weights=w).reshape(f_mesh.shape[0])
        var_vol = predict_component_var(vol, p.vol_z, p.vol_sig, p.vol_ls, w.vol_M_mat).reshape(f_mesh.shape[0])
        if is_single:
            return EnergyDist(mean[0], var_vol[0])
        return EnergyDist(mean, var_vol)

    def aniso_psi_dist(self, f_mesh: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> EnergyDist:
        if not self.is_anisotropic:
            zeros = jnp.zeros(f_mesh.shape[0] if f_mesh.ndim > 2 else ())
            return EnergyDist(zeros, zeros)
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        feats = jax.vmap(self.feature_extractor.extract)(f_mesh)
        aniso = feats[2]
        mean = self.aniso_gp_mean(aniso, params=p, weights=w).reshape(f_mesh.shape[0])
        var_aniso = predict_component_var(aniso, p.aniso_z, p.aniso_sig, p.aniso_ls, w.aniso_M_mat).reshape(f_mesh.shape[0])
        if is_single:
            return EnergyDist(mean[0], var_aniso[0])
        return EnergyDist(mean, var_aniso)

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
            return self.psi_gp_mean(f, params=p, weights=w)

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
        """Cholesky / joint normal energy sampling across mesh (alias: psi_cholesky)."""
        p, w = self._resolve_state(params, weights)
        is_single = (f_mesh.ndim == 2)
        if is_single:
            f_mesh = f_mesh[None, ...]
        dist = self.psi_dist(f_mesh, params=p, weights=w)
        # Convert marginal variance vector into diagonal covariance for multivariate drawing
        psi = jax.random.multivariate_normal(key, dist.mean, jnp.diag(dist.var), dtype=jnp.float64)
        return psi if not is_single else psi[0]

    psi_cholesky = psi_mds

    def piola_mds(self, f_mesh: jnp.ndarray, key: jnp.ndarray, params: Optional[GPParams] = None, weights: Optional[GPWeights] = None) -> jnp.ndarray:
        """
        Samples the ENTIRE correlated Piola stress field across a mesh via Cholesky / joint Gaussian.
        Alias: piola_cholesky.
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

    piola_cholesky = piola_mds

"""
Reduced-rank GP strain-energy model (HSGP extraction method).

    psi(F) = beta_mu h_mu(I1_bar) + beta_kappa h_kappa(J) + phi_dev(I1_bar, I2_bar)^T w_dev + phi_vol(J)^T w_vol
           = Phi(F)^T theta,      theta = [beta (2), w_dev (K_dev), w_vol (K_vol)],

with h_mu = (I1_bar - 3)/2, h_kappa = (J - 1)^2/2 (linear-elastic prior mean) and phi the HSGP bases (core/hsgp_basis).
The energy is linear in theta, so the posterior N(theta_mean, F F^T) gives every derived quantity in closed form:
energy and stress means/covariances, and exact pathwise samples theta = theta_mean + F z.

Optional amplitude envelope (large deformations, energies over orders of magnitude): each GP basis is multiplied by
a fixed positive envelope of its own inputs, keeping the dev/vol split,
    s_dev(I1_bar) = 1 + mu_e h_mu / e_bar,   s_vol(J) = 1 + kappa_e h_kappa / e_bar,
(mu_e, kappa_e from the linear-elastic initialisation, e_bar the data energy scale), so the GP models a relative
deviation with an uncertainty band that grows with the energy. The model stays linear in theta (s phi_k is just another
basis), and s = 1 with zero slope at F = I leaves the reference constraints unchanged.

Optional input warping (large deformations with a dense core and a long tail, e.g. strain concentrations): the GP
bases take x_dev = log(I_bar - 2) (component-wise, I_bar >= 3) and x_vol = log J, so equal basis resolution covers equal
relative strain. The reference maps to x = 0 and dpsi/dlog J = J dpsi/dJ, so the reference constraints keep their form;
the linear-elastic basis and the envelopes stay functions of the raw invariants.

The reference conditions psi_dev(I) = 0, psi_vol(J=1) = 0, dpsi_vol/dJ(1) = 0 hold exactly for every sample
(the GP prior is conditioned on them, see hsgp_inference.constrained_transform); dev stress at F = I vanishes by
itself because dI1_bar/dF = dI2_bar/dF = 0 there.

Duck-type compatible with the SparseHyperelasticityGP methods used by evaluation, plotting and the distillation export.
"""
import json
from typing import Callable, Dict, Optional

import numpy as np
import jax
import jax.numpy as jnp

from .dataclass import EnergyDist, StressDist
from .features import IsotropicFeatureExtractor
from .hsgp_basis import Box, eigenfunctions

jax.config.update("jax_enable_x64", True)

DEV_REFERENCE = np.array([3.0, 3.0])
VOL_REFERENCE = np.array([1.0])


class HSGPHyperelasticity:
    is_anisotropic = False

    def __init__(self, dev_box: Box, vol_box: Box, theta_mean, theta_factor, hyper: Dict[str, np.ndarray] = None,
                 noise: Dict[str, float] = None, info: Dict = None, envelope: Optional[Dict[str, float]] = None,
                 warp: bool = False):
        self.dev_box, self.vol_box = dev_box, vol_box
        self.warp = bool(warp)
        self.envelope = None if envelope is None else {k: float(v) for k, v in envelope.items()}   # mu, kappa, scale
        self.feature_extractor = IsotropicFeatureExtractor()
        self.n_lin = 2
        self.k_dev, self.k_vol = dev_box.num_basis, vol_box.num_basis
        self.sl_dev = slice(self.n_lin, self.n_lin + self.k_dev)
        self.sl_vol = slice(self.n_lin + self.k_dev, self.n_lin + self.k_dev + self.k_vol)
        self.num_params = self.n_lin + self.k_dev + self.k_vol
        # component index sets: dev = mu term + dev GP, vol = kappa term + vol GP
        self.idx_dev = np.concatenate([[0], np.arange(self.n_lin, self.n_lin + self.k_dev)])
        self.idx_vol = np.concatenate([[1], np.arange(self.n_lin + self.k_dev, self.num_params)])
        self.set_posterior(theta_mean, theta_factor)
        self.hyper = {k: np.asarray(v) for k, v in (hyper or {}).items()}
        self.noise = dict(noise or {})
        self.info = dict(info or {})

    def set_posterior(self, theta_mean, theta_factor):
        self.theta_mean = jnp.asarray(theta_mean, dtype=jnp.float64)
        self.theta_factor = jnp.asarray(theta_factor, dtype=jnp.float64)   # Cov(theta) = factor @ factor.T

    # ---------------- basis ----------------
    def envelopes(self, dev: jnp.ndarray, vol: jnp.ndarray):
        """(s_dev, s_vol) of shape (...,): ones without an envelope."""
        if self.envelope is None:
            return jnp.ones(dev.shape[:-1]), jnp.ones(vol.shape[:-1])
        e = self.envelope
        return (1.0 + max(e["mu"], 0.0) * 0.5 * (dev[..., 0] - 3.0) / e["scale"],
                1.0 + max(e["kappa"], 0.0) * 0.5 * (vol[..., 0] - 1.0) ** 2 / e["scale"])

    def gp_inputs(self, dev: jnp.ndarray, vol: jnp.ndarray):
        """Raw features -> GP inputs (identity, or the log warp)."""
        if not self.warp:
            return dev, vol
        return jnp.log(jnp.maximum(dev - 2.0, 1e-12)), jnp.log(jnp.maximum(vol, 1e-12))

    def dev_basis(self, dev: jnp.ndarray) -> jnp.ndarray:
        s_dev, _ = self.envelopes(dev, jnp.ones(dev.shape[:-1] + (1,)))
        x_dev, _ = self.gp_inputs(dev, jnp.ones(dev.shape[:-1] + (1,)))
        return s_dev[..., None] * eigenfunctions(self.dev_box, x_dev)

    def vol_basis(self, vol: jnp.ndarray) -> jnp.ndarray:
        _, s_vol = self.envelopes(jnp.full(vol.shape[:-1] + (2,), 3.0), vol)
        _, x_vol = self.gp_inputs(jnp.full(vol.shape[:-1] + (2,), 3.0), vol)
        return s_vol[..., None] * eigenfunctions(self.vol_box, x_vol)

    def basis_from_features(self, dev: jnp.ndarray, vol: jnp.ndarray) -> jnp.ndarray:
        """(..., 2), (..., 1) -> (..., P) values of [h_mu, h_kappa, s_dev phi_dev, s_vol phi_vol]."""
        h = jnp.stack([0.5 * (dev[..., 0] - 3.0), 0.5 * (vol[..., 0] - 1.0) ** 2], axis=-1)
        return jnp.concatenate([h, self.dev_basis(dev), self.vol_basis(vol)], axis=-1)

    def basis(self, f: jnp.ndarray) -> jnp.ndarray:
        """Single F (3, 3) -> (P,)."""
        dev, vol = self.feature_extractor.extract(f)
        return self.basis_from_features(dev, vol)

    def basis_batch(self, f: jnp.ndarray) -> jnp.ndarray:
        f = jnp.asarray(f, dtype=jnp.float64)
        return jax.vmap(self.basis)(f.reshape(-1, 3, 3)).reshape(*f.shape[:-2], self.num_params)

    def _rows(self, f, idx=None):
        phi = self.basis_batch(f)
        if idx is None:
            return phi, self.theta_mean, self.theta_factor
        return phi[..., idx], self.theta_mean[idx], self.theta_factor[idx]

    # ---------------- energy ----------------
    def psi_det(self, f: jnp.ndarray, params=None, weights=None) -> jnp.ndarray:
        """Posterior-mean energy at one F (3, 3); differentiable (stress, tangent)."""
        return self.basis(f) @ self.theta_mean

    def psi_gp_mean(self, f, params=None, weights=None):
        return self.basis_batch(f) @ self.theta_mean

    def _dist(self, f, idx=None):
        f = jnp.asarray(f, dtype=jnp.float64)
        single = f.ndim == 2
        phi, m, F = self._rows(f[None] if single else f, idx)
        mean, var = phi @ m, jnp.sum((phi @ F) ** 2, axis=-1)
        return EnergyDist(mean[0], var[0]) if single else EnergyDist(mean, var)

    def psi_dist(self, f, params=None, weights=None) -> EnergyDist:
        return self._dist(f)

    def dev_psi_dist(self, f, params=None, weights=None) -> EnergyDist:
        return self._dist(f, self.idx_dev)

    def vol_psi_dist(self, f, params=None, weights=None) -> EnergyDist:
        return self._dist(f, self.idx_vol)

    def _joint_cov(self, f, idx=None):
        phi, _, F = self._rows(jnp.asarray(f, dtype=jnp.float64).reshape(-1, 3, 3), idx)
        G = phi @ F
        return G @ G.T

    def psi_joint_cov(self, f, params=None, weights=None):
        return self._joint_cov(f)

    def dev_psi_joint_cov(self, f, params=None, weights=None):
        return self._joint_cov(f, self.idx_dev)

    def vol_psi_joint_cov(self, f, params=None, weights=None):
        return self._joint_cov(f, self.idx_vol)

    def joint_factor(self, f, component: str = "total"):
        """Exact low-rank factor G (N, Q) of the energy covariance G G^T at the states f (export / sampling)."""
        idx = {"total": None, "dev": self.idx_dev, "vol": self.idx_vol}[component]
        phi, _, F = self._rows(jnp.asarray(f, dtype=jnp.float64).reshape(-1, 3, 3), idx)
        return phi @ F

    def dev_gp_mean(self, dev_feats, params=None, weights=None):
        dev = jnp.asarray(dev_feats).reshape(-1, 2)
        h = 0.5 * (dev[:, 0] - 3.0)
        return h * self.theta_mean[0] + self.dev_basis(dev) @ self.theta_mean[self.sl_dev]

    def vol_gp_mean(self, vol_feats, params=None, weights=None):
        vol = jnp.asarray(vol_feats).reshape(-1, 1)
        h = 0.5 * (vol[:, 0] - 1.0) ** 2
        return h * self.theta_mean[1] + self.vol_basis(vol) @ self.theta_mean[self.sl_vol]

    # ---------------- stress ----------------
    def piola_det(self, f, params=None, weights=None):
        return jax.grad(self.psi_det)(jnp.asarray(f, dtype=jnp.float64))

    def piola_basis(self, f):
        """d Phi / dF at one F: (P, 3, 3)."""
        return jax.jacfwd(self.basis)(jnp.asarray(f, dtype=jnp.float64))

    def piola_dist(self, f_mesh, params=None, weights=None) -> StressDist:
        """Mean and marginal variance of each Piola component, (N, 3, 3) each."""
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        single = f.ndim == 2
        dphi = jax.vmap(self.piola_basis)(f.reshape(-1, 3, 3))                    # (N, P, 3, 3)
        mean = jnp.einsum("npij,p->nij", dphi, self.theta_mean)
        var = jnp.sum(jnp.einsum("npij,pq->nijq", dphi, self.theta_factor) ** 2, axis=-1)
        return StressDist(mean[0], var[0]) if single else StressDist(mean, var)

    # ---------------- pathwise samples (exact) ----------------
    def sample_theta(self, key):
        return self.theta_mean + self.theta_factor @ jax.random.normal(key, (self.theta_factor.shape[1],), dtype=jnp.float64)

    # Path samples share one jitted evaluator taking theta as an argument: a fresh closure per sample would be
    # compiled (and cached) anew for every draw, which grows memory without bound in sampling loops.
    def _evaluators(self):
        if getattr(self, "_jit_eval", None) is None:
            idx_dev, idx_vol = np.asarray(self.idx_dev), np.asarray(self.idx_vol)   # constants: no tracers in the cache

            def total(theta, f):
                return self.basis(f) @ theta

            def comps(theta, f):
                phi = self.basis(f)
                return phi[idx_dev] @ theta[idx_dev], phi[idx_vol] @ theta[idx_vol], jnp.zeros(())
            self._jit_eval = (jax.jit(total), jax.jit(comps))
        return self._jit_eval

    def get_path_psi_fn(self, key, params=None, weights=None) -> Callable:
        theta = self.sample_theta(key)
        total = self._evaluators()[0]
        return lambda f: total(theta, f)

    def get_path_components_psi_fn(self, key, params=None, weights=None) -> Callable:
        theta = self.sample_theta(key)
        comps = self._evaluators()[1]
        return lambda f: comps(theta, f)

    def get_path_dev_vol_psi_fn(self, key, params=None, weights=None) -> Callable:
        comps = self.get_path_components_psi_fn(key)
        return lambda f: comps(f)[:2]

    def psi(self, f_mesh, key, params=None, weights=None):
        """One posterior sample of the energy at F (3, 3) or a batch (..., 3, 3)."""
        path = self.get_path_psi_fn(key)
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        return path(f) if f.ndim == 2 else jax.vmap(path)(f.reshape(-1, 3, 3)).reshape(f.shape[:-2])

    def piola(self, f_mesh, key, params=None, weights=None):
        """One posterior sample of the first Piola stress at F (3, 3) or a batch (..., 3, 3)."""
        path = jax.grad(self.get_path_psi_fn(key))
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        return path(f) if f.ndim == 2 else jax.vmap(path)(f.reshape(-1, 3, 3)).reshape(f.shape)

    psi_pws = psi        # pathwise-sampling names used by the plotting code
    piola_pws = piola

    # ---------------- linear-elastic coefficients ----------------
    def linear_elastic_summary(self) -> Dict[str, float]:
        m = np.asarray(self.theta_mean[:2])
        F = np.asarray(self.theta_factor[:2])
        C = F @ F.T
        sd = np.sqrt(np.diag(C))
        return dict(mu=float(m[0]), kappa=float(m[1]), mu_std=float(sd[0]), kappa_std=float(sd[1]),
                    corr_mu_kappa=float(C[0, 1] / (sd[0] * sd[1])))

    def small_strain_moduli(self, n_samples: int = 400, key=None) -> Dict[str, float]:
        """
        Effective small-strain moduli of the whole energy (parametric mean + GP): mu = 2 (dpsi/dI1_bar + dpsi/dI2_bar)
        and kappa = d2psi/dJ2 at F = I, posterior mean and std. Unlike the linear-elastic coefficients, these do not
        depend on how the energy is split between the parametric basis and the GP.
        """
        dev0, vol0 = jnp.array([3.0, 3.0]), jnp.array([1.0])

        def moduli(theta):
            gd = jax.grad(lambda x: self.basis_from_features(x, vol0)[self.idx_dev] @ theta[self.idx_dev])(dev0)
            kv = jax.hessian(lambda j: self.basis_from_features(dev0, j)[self.idx_vol] @ theta[self.idx_vol])(vol0)[0, 0]
            return jnp.array([2.0 * (gd[0] + gd[1]), kv])

        keys = jax.random.split(jax.random.PRNGKey(0) if key is None else key, n_samples)
        S = np.asarray(jax.vmap(lambda k: moduli(self.sample_theta(k)))(keys))
        m = np.asarray(moduli(self.theta_mean))
        return dict(mu_eff=float(m[0]), kappa_eff=float(m[1]), mu_eff_std=float(S[:, 0].std()),
                    kappa_eff_std=float(S[:, 1].std()), corr_mu_kappa_eff=float(np.corrcoef(S.T)[0, 1]))

    # ---------------- persistence ----------------
    def save(self, path: str):
        np.savez(path, dev_centre=self.dev_box.centre, dev_half_width=self.dev_box.half_width, dev_indices=self.dev_box.indices,
                 vol_centre=self.vol_box.centre, vol_half_width=self.vol_box.half_width, vol_indices=self.vol_box.indices,
                 theta_mean=np.asarray(self.theta_mean), theta_factor=np.asarray(self.theta_factor),
                 hyper=json.dumps({k: np.asarray(v).tolist() for k, v in self.hyper.items()}),
                 noise=json.dumps(self.noise), info=json.dumps(self.info, default=float),
                 envelope=json.dumps(self.envelope), warp=self.warp)

    @classmethod
    def load(cls, path: str) -> "HSGPHyperelasticity":
        d = np.load(path, allow_pickle=False)
        dev_box = Box(d["dev_centre"], d["dev_half_width"], d["dev_indices"])
        vol_box = Box(d["vol_centre"], d["vol_half_width"], d["vol_indices"])
        return cls(dev_box, vol_box, d["theta_mean"], d["theta_factor"],
                   hyper={k: np.asarray(v) for k, v in json.loads(str(d["hyper"])).items()},
                   noise=json.loads(str(d["noise"])), info=json.loads(str(d["info"])),
                   envelope=json.loads(str(d["envelope"])) if "envelope" in d.files else None,
                   warp=bool(d["warp"]) if "warp" in d.files else False)

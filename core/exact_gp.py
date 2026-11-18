"""
Exact Gaussian-process strain-energy model (no basis approximation), for small problems (1-3 load steps).

The energy is a GP over the same inputs as the HSGP model (core/hsgp_model.py),
    z = [I1_bar, I2_bar (deviatoric) | J (volumetric, compressible only) | I4_bar, I6_bar (fibres, optional)],
    psi(z) = g_dev(z_dev) + g_vol(z_vol) + g_aniso(z_aniso),   g_c ~ GP(0, sigma_c^2 exp(-|r / l_c|^2 / 2)),
with zero prior mean, like the HSGP (its linear-elastic coefficients are kept at zero).

Observations. At the measured deformation, each internal nodal force is a LINEAR functional of the energy:
    f_(a,i) = sum_e dA_e sum_d dpsi/dz_d (z_e) [dz_d/dF (F_e) dN_a]_i ,
so with "atoms" = the gradient components dpsi/dz_d at the element states (and values / gradients at the reference
state), every observation row is y_r = A_r . atoms with a sparse, known matrix A. The free-DOF rows (target 0,
sigma = s |R_t|), the reaction rows (target R_t, load-cell sigma) and the reference conditions (psi(I) = 0, dpsi/dJ(1)
= 0, dpsi/dI4 = dpsi/dI6 = 0 at I4 = I6 = 1, noise-free) give a Gaussian likelihood, and the GP posterior is exact:
    C = A K A^T + Sigma,   alpha = A^T C^-1 y,   U = A^T L^-T (L L^T = C),
    mean of a functional q:  k_q . alpha,      covariance:  k_qq - (k_q U)(k_q U)^T,
where K and k_q are covariances between atoms (kernel derivatives up to second order, in closed form for the
squared-exponential kernel). Hyperparameters (amplitudes, ARD lengthscales) by maximum evidence; s by cross-validation
on held-out reactions, as for the HSGP (or given).

Cost: K is (n_atoms)^2 with n_atoms = steps x elements x D (+ reference atoms); intended for 1-3 load steps.

Pathwise samples (plots, sef_stress export) use Matheron's rule with random-Fourier-feature prior draws; that is the
only approximation (the posterior means, variances and joint covariances used by the sef/sef_split export are exact).

Same evaluation interface as HSGPHyperelasticity (psi_dist, dev/vol/aniso_psi_dist, *_joint_cov, *_gp_mean,
piola_dist, piola, get_path_*_fn, small_strain_moduli, reaction_prediction, save/load).
"""
import json
import time
from typing import Optional

import numpy as np
import jax
import jax.numpy as jnp
import optax

from .dataclass import EnergyDist, StressDist
from .features import AnisotropicFeatureExtractor, make_feature_extractor
from .hsgp_model import ANISO_REFERENCE, DEV_REFERENCE, VOL_REFERENCE
from .mechanics import dof_sets

jax.config.update("jax_enable_x64", True)

VALUE = -1          # atom kind: value; kinds >= 0 = derivative along z_d
N_RFF = 1000        # random Fourier features per component for pathwise samples
CHUNK = 256         # query chunk size (memory)


# ----------------------------------------------------------------------------- layout and kernel
class Layout:
    """Active input dimensions of z and the kernel components over them."""

    def __init__(self, incompressible_plane_stress: bool, anisotropic: bool):
        self.dev = [0, 1]
        d = 2
        self.vol = [] if incompressible_plane_stress else [d]
        d += len(self.vol)
        self.aniso = [d, d + 1] if anisotropic else []
        d += len(self.aniso)
        self.D = d
        self.comps = [("dev", self.dev)] + ([("vol", self.vol)] if self.vol else []) \
            + ([("aniso", self.aniso)] if self.aniso else [])
        ref = list(DEV_REFERENCE) + (list(VOL_REFERENCE) if self.vol else []) + (list(ANISO_REFERENCE) if self.aniso else [])
        self.z_ref = np.asarray(ref, dtype=np.float64)

    def names(self):
        return [c for c, _ in self.comps]


def atom_cov(Z1, k1, Z2, k2, hyper, layout: Layout, comps=None):
    """Covariance between atoms (Z1 (n1, D), kinds k1 (n1,)) and (Z2, k2): value (-1) or d/dz_d (d >= 0) of psi."""
    Z1, Z2 = jnp.asarray(Z1), jnp.asarray(Z2)
    k1, k2 = jnp.asarray(k1), jnp.asarray(k2)
    v1, v2 = (k1 == VALUE).astype(Z1.dtype), (k2 == VALUE).astype(Z1.dtype)
    out = jnp.zeros((Z1.shape[0], Z2.shape[0]), dtype=Z1.dtype)
    for name, dims in layout.comps:
        if comps is not None and name not in comps:
            continue
        sig, ls = hyper[f"{name}_sig"], jnp.reshape(jnp.asarray(hyper[f"{name}_ls"]), (-1,))
        da = jnp.asarray(dims)
        dz = Z1[:, None, da] - Z2[None, :, da]                         # (n1, n2, dc)
        diff = dz / ls ** 2                                            # (z - z') / l^2
        k = sig ** 2 * jnp.exp(-0.5 * jnp.sum((dz / ls) ** 2, axis=-1))
        a1 = (k1[:, None] == da[None, :]).astype(Z1.dtype)             # one-hot derivative direction (n1, dc)
        a2 = (k2[:, None] == da[None, :]).astype(Z1.dtype)
        g1 = jnp.einsum("ijc,ic->ij", diff, a1)                        # (z_c - z'_c) / l_c^2 at atom 1's direction
        g2 = jnp.einsum("ijd,jd->ij", diff, a2)
        is1, is2 = a1.sum(1), a2.sum(1)                                # 1 if the atom differentiates inside this comp
        same = a1 @ (a2 / ls[None, :] ** 2).T                          # delta_cd / l_d^2
        term = (v1[:, None] * v2[None, :] + v1[:, None] * is2[None, :] * g2 - is1[:, None] * v2[None, :] * g1
                + is1[:, None] * is2[None, :] * (same - g1 * g2))
        out = out + k * term
    return out


def prior_var_value(hyper, layout, comps=None):
    return sum(hyper[f"{n}_sig"] ** 2 for n, _ in layout.comps if comps is None or n in comps)


# ----------------------------------------------------------------------------- observation operator per step
def step_operator(extractor, layout, F, mesh, R, sR, global_equilibrium, S=None, sS=None):
    """Atoms (Z, kinds), dense rows A (n_rows, n_atoms), targets y, fixed noise var, free-row scale |R|^2 (rows whose
    noise is s^2 |R|^2 get the scale, others 0). Reaction rows: loaded edges (R, sR) and the supports with a measured
    reaction (S, sS: (2,), NaN = not measured), as core.hsgp_inference.pseudo_observation."""
    F = jnp.asarray(F, dtype=jnp.float64)

    def z_of(f3):
        feats = extractor.extract(f3)
        parts = [feats[0]] + ([feats[1]] if layout.vol else []) + ([feats[2][:2]] if layout.aniso else [])
        return jnp.concatenate(parts)

    Z = np.asarray(jax.vmap(z_of)(F))                                                  # (C, D)
    Jz = np.asarray(jax.vmap(lambda f3: jax.jacfwd(lambda f2: z_of(f3.at[:2, :2].set(f2)))(f3[:2, :2]))(F))   # (C, D, 2, 2)
    dNdX, dA, cells = np.asarray(mesh["dNdX"]), np.asarray(mesh["dA"]), np.asarray(mesh["cells"])
    C, D = Z.shape
    W = np.einsum("cdij,caj->caid", Jz, dNdX) * dA[:, None, None, None]                # (C, 3, 2, D)
    G = np.zeros((mesh["n_nodes"], 2, C, D))
    for a in range(3):
        np.add.at(G, (cells[:, a], slice(None), np.arange(C)), np.transpose(W[:, a], (0, 1, 2)))
    G = G.reshape(2 * mesh["n_nodes"], C * D)                                          # DOF force <- atoms (e, d)
    r2 = float(max(np.sum(np.asarray(R) ** 2), 1e-30))
    rows, y, noise, fscale = [], [], [], []
    free = np.asarray(mesh["free"])
    rows.append(G[free])
    y += [0.0] * len(free)
    noise += [0.0] * len(free)
    fscale += [r2] * len(free)
    n_react = 0
    sets = [(mesh["rx"], float(np.asarray(R)[0]), float(np.asarray(sR)[0])),
            (mesh["ry"], float(np.asarray(R)[1]), float(np.asarray(sR)[1]))]
    if S is not None:
        sets += [(mesh["fx"], float(np.asarray(S)[0]), max(float(np.asarray(sS)[0]), 1e-3)),
                 (mesh["fy"], float(np.asarray(S)[1]), max(float(np.asarray(sS)[1]), 1e-3))]
    for dofs, value, s in sets:
        if len(dofs) == 0 or not np.isfinite(value) or not s <= 1e6:   # no measurement for this boundary: no row
            continue
        rows.append(G[np.asarray(dofs)].sum(0)[None])
        y.append(value)
        noise.append(s ** 2)
        fscale.append(0.0)
        n_react += 1
    if global_equilibrium > 0:
        for par in (0, 1):
            rows.append(G[free[free % 2 == par]].sum(0)[None])
            y.append(0.0)
            noise.append(float(global_equilibrium) ** 2 * r2)
            fscale.append(0.0)
    kinds = np.tile(np.arange(D), C)
    Zat = np.repeat(Z, D, axis=0)
    return dict(Z=Zat, kinds=kinds, A=np.concatenate(rows), y=np.asarray(y), noise=np.asarray(noise),
                fscale=np.asarray(fscale), n_free=len(free), n_react=n_react)


def reaction_rows_operator(extractor, layout, F, mesh):
    """Atoms and rows (R_x, R_y) of the loaded edges at one state (for reaction prediction)."""
    st = step_operator(extractor, layout, F, mesh, np.ones(2), np.ones(2), 0.0)
    nf = st["n_free"]
    return st["Z"], st["kinds"], st["A"][nf:nf + 2]


# ----------------------------------------------------------------------------- model
class ExactGPHyperelasticity:
    model_type = "exact_gp"

    def __init__(self, Z, kinds, A, y, noise, L, hyper, incompressible_plane_stress=False, a0=None, a1=None,
                 info=None, noise_info=None):
        self.incompressible_plane_stress = bool(incompressible_plane_stress)
        self.is_anisotropic = a0 is not None
        if self.is_anisotropic:
            self.a0 = jnp.asarray(a0, dtype=jnp.float64)
            self.a1 = None if a1 is None else jnp.asarray(a1, dtype=jnp.float64)
            self.feature_extractor = AnisotropicFeatureExtractor(self.a0, a1=self.a1,
                                                                 incompressible_plane_stress=self.incompressible_plane_stress)
        else:
            self.a0 = self.a1 = None
            self.feature_extractor = make_feature_extractor(self.incompressible_plane_stress)
        self.layout = Layout(self.incompressible_plane_stress, self.is_anisotropic)
        self.Z, self.kinds = jnp.asarray(Z, dtype=jnp.float64), jnp.asarray(kinds, dtype=jnp.int32)
        self.A, self.y, self.obs_noise = jnp.asarray(A), jnp.asarray(y), jnp.asarray(noise)
        self.L = jnp.asarray(L)
        self.hyper = {k: np.asarray(v) for k, v in hyper.items()}
        self._h = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in hyper.items()}
        self.info = dict(info or {})
        self.noise = dict(noise_info or {})
        w = jax.scipy.linalg.cho_solve((self.L, True), self.y)
        self.alpha = self.A.T @ w                                                        # (n_atoms,)
        self.U = self.A.T @ jax.scipy.linalg.solve_triangular(self.L, jnp.eye(self.L.shape[0]), lower=True).T  # A^T L^-T
        self.dev_z = self.vol_z = None

    # ---------------- inputs ----------------
    def z_of(self, f):
        feats = self.feature_extractor.extract(jnp.asarray(f, dtype=jnp.float64))
        parts = [feats[0]] + ([feats[1]] if self.layout.vol else []) + ([feats[2][:2]] if self.layout.aniso else [])
        return jnp.concatenate(parts)

    def _z_batch(self, f):
        return jax.vmap(self.z_of)(jnp.asarray(f, dtype=jnp.float64).reshape(-1, 3, 3))

    def _z_from_parts(self, dev=None, vol=None, aniso=None, n=None):
        """z with the given parts, the reference elsewhere."""
        n = n if n is not None else next(x.shape[0] for x in (dev, vol, aniso) if x is not None)
        z = jnp.tile(jnp.asarray(self.layout.z_ref), (n, 1))
        if dev is not None:
            z = z.at[:, :2].set(jnp.asarray(dev).reshape(n, 2))
        if vol is not None and self.layout.vol:
            z = z.at[:, self.layout.vol[0]].set(jnp.asarray(vol).reshape(n))
        if aniso is not None and self.layout.aniso:
            z = z.at[:, self.layout.aniso].set(jnp.asarray(aniso).reshape(n, -1)[:, :2])
        return z

    # ---------------- core: functionals ----------------
    def _kq(self, Zq, kq, comps=None):
        return atom_cov(Zq, kq, self.Z, self.kinds, self._h, self.layout, comps)

    def _value_mean_var(self, Zq, comps=None):
        out_m, out_v = [], []
        for i in range(0, Zq.shape[0], CHUNK):
            z = Zq[i:i + CHUNK]
            kq = self._kq(z, jnp.full(z.shape[0], VALUE), comps)
            out_m.append(kq @ self.alpha)
            out_v.append(prior_var_value(self._h, self.layout, comps) - jnp.sum((kq @ self.U) ** 2, axis=1))
        return jnp.concatenate(out_m), jnp.maximum(jnp.concatenate(out_v), 0.0)

    def _value_joint(self, Zq, comps=None):
        kq = self._kq(Zq, jnp.full(Zq.shape[0], VALUE), comps)
        kk = atom_cov(Zq, jnp.full(Zq.shape[0], VALUE), Zq, jnp.full(Zq.shape[0], VALUE), self._h, self.layout, comps)
        G = kq @ self.U
        return kq @ self.alpha, kk - G @ G.T

    # ---------------- energy ----------------
    def psi_det(self, f, params=None, weights=None):
        z = self.z_of(f)
        return (self._kq(z[None], jnp.array([VALUE])) @ self.alpha)[0]

    def psi_gp_mean(self, f, params=None, weights=None):
        return self._value_mean_var(self._z_batch(f))[0]

    def _dist(self, f, comps=None):
        f = jnp.asarray(f, dtype=jnp.float64)
        single = f.ndim == 2
        m, v = self._value_mean_var(self._z_batch(f), comps)
        return EnergyDist(m[0], v[0]) if single else EnergyDist(m.reshape(f.shape[:-2]), v.reshape(f.shape[:-2]))

    def psi_dist(self, f, params=None, weights=None):
        return self._dist(f)

    def dev_psi_dist(self, f, params=None, weights=None):
        return self._dist(f, ["dev"])

    def vol_psi_dist(self, f, params=None, weights=None):
        if not self.layout.vol:
            f = jnp.asarray(f)
            z = jnp.zeros(()) if f.ndim == 2 else jnp.zeros(f.shape[:-2])
            return EnergyDist(z, z)
        return self._dist(f, ["vol"])

    def aniso_psi_dist(self, f, params=None, weights=None):
        if not self.is_anisotropic:
            f = jnp.asarray(f)
            z = jnp.zeros(()) if f.ndim == 2 else jnp.zeros(f.shape[:-2])
            return EnergyDist(z, z)
        return self._dist(f, ["aniso"])

    def _joint(self, f, comps):
        if comps is not None and not any(c in self.layout.names() for c in comps):
            n = jnp.asarray(f).reshape(-1, 3, 3).shape[0]
            return jnp.zeros((n, n))
        return self._value_joint(self._z_batch(f), comps)[1]

    def psi_joint_cov(self, f, params=None, weights=None):
        return self._joint(f, None)

    def dev_psi_joint_cov(self, f, params=None, weights=None):
        return self._joint(f, ["dev"])

    def vol_psi_joint_cov(self, f, params=None, weights=None):
        return self._joint(f, ["vol"])

    def aniso_psi_joint_cov(self, f, params=None, weights=None):
        return self._joint(f, ["aniso"])

    def joint_factor(self, f, component: str = "total"):
        comps = None if component == "total" else [component]
        K = np.asarray(self._joint(f, comps))
        w, V = np.linalg.eigh(0.5 * (K + K.T))
        return jnp.asarray(V * np.sqrt(np.clip(w, 0.0, None))[None, :])

    def dev_gp_mean(self, dev_feats, params=None, weights=None):
        return self._value_mean_var(self._z_from_parts(dev=jnp.asarray(dev_feats).reshape(-1, 2)), ["dev"])[0]

    def vol_gp_mean(self, vol_feats, params=None, weights=None):
        v = jnp.asarray(vol_feats).reshape(-1)
        if not self.layout.vol:
            return jnp.zeros(v.shape[0])
        return self._value_mean_var(self._z_from_parts(vol=v), ["vol"])[0]

    def aniso_gp_mean(self, aniso_feats, params=None, weights=None):
        a = jnp.asarray(aniso_feats)
        a = a.reshape(-1, a.shape[-1])
        if not self.is_anisotropic:
            return jnp.zeros(a.shape[0])
        return self._value_mean_var(self._z_from_parts(aniso=a), ["aniso"])[0]

    # ---------------- stress ----------------
    def piola_det(self, f, params=None, weights=None):
        return jax.grad(self.psi_det)(jnp.asarray(f, dtype=jnp.float64))

    def piola_dist(self, f_mesh, params=None, weights=None) -> StressDist:
        """Mean and marginal variance of each first Piola component (exact): P = sum_d dpsi/dz_d dz_d/dF."""
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        single = f.ndim == 2
        F = f.reshape(-1, 3, 3)
        D = self.layout.D
        inv_l2 = np.zeros(D)
        sig2 = np.zeros(D)
        for n, dims in self.layout.comps:
            inv_l2[dims] = 1.0 / np.reshape(self.hyper[f"{n}_ls"], (-1,)) ** 2
            sig2[dims] = float(self.hyper[f"{n}_sig"]) ** 2
        prior_gg = jnp.asarray(sig2 * inv_l2)                                       # diag d2k/dz dz' at r = 0
        means, vars_ = [], []
        for i in range(0, F.shape[0], CHUNK // max(D, 1)):
            Fc = F[i:i + CHUNK // max(D, 1)]
            n = Fc.shape[0]
            Zc = jax.vmap(self.z_of)(Fc)                                              # (n, D)
            J = jax.vmap(jax.jacfwd(self.z_of))(Fc)                                   # (n, D, 3, 3)
            Zq = jnp.repeat(Zc, D, axis=0)
            kq = self._kq(Zq, jnp.tile(jnp.arange(D), n))                             # (n D, atoms)
            gm = (kq @ self.alpha).reshape(n, D)
            GU = (kq @ self.U).reshape(n, D, -1)
            post_gg = jnp.einsum("ndq,neq->nde", GU, GU)                               # (n, D, D)
            cov_gg = jnp.eye(D)[None] * prior_gg[None, None, :] - post_gg
            means.append(jnp.einsum("nd,ndij->nij", gm, J))
            vars_.append(jnp.einsum("ndij,nde,neij->nij", J, cov_gg, J))
        mean, var = jnp.concatenate(means), jnp.maximum(jnp.concatenate(vars_), 0.0)
        return StressDist(mean[0], var[0]) if single else StressDist(mean.reshape(f.shape), var.reshape(f.shape))

    # ---------------- reactions ----------------
    def reaction_prediction(self, f3x3_steps, cells, node_type, dNdX, dA):
        """Reactions (R_x, R_y) at the measured states of each step: mean (T, 2), epistemic covariance (T, 2, 2)."""
        nt, c = np.asarray(node_type), np.asarray(cells)
        mesh = dict(cells=c, n_nodes=nt.shape[0], dNdX=np.asarray(dNdX), dA=np.asarray(dA),
                    **dict(dof_sets(nt), fx=np.array([], int), fy=np.array([], int)))   # loaded-edge reactions only
        means, covs = [], []
        for F in np.asarray(f3x3_steps):
            Zq, kq_, Aq = reaction_rows_operator(self.feature_extractor, self.layout, F, mesh)
            Kq = atom_cov(Zq, kq_, self.Z, self.kinds, self._h, self.layout)           # (n_q, atoms)
            Kqq = atom_cov(Zq, kq_, Zq, kq_, self._h, self.layout)
            Aq = jnp.asarray(Aq)
            AK = Aq @ Kq
            G = AK @ self.U
            means.append(np.asarray(AK @ self.alpha))
            covs.append(np.asarray(Aq @ Kqq @ Aq.T - G @ G.T))
        return np.array(means), np.array(covs)

    # ---------------- moduli ----------------
    def small_strain_moduli(self, n_samples: int = 200, key=None):
        """mu = 2 (dpsi/dI1_bar + dpsi/dI2_bar) at F = I (exact mean and std); kappa = d2psi/dJ2 (mean exact via the
        posterior mean, std from pathwise samples) or 0 if incompressible."""
        zr = jnp.asarray(self.layout.z_ref)[None]
        Zq, kq = jnp.concatenate([zr, zr]), jnp.array([0, 1])
        K = self._kq(Zq, kq)
        w = jnp.array([2.0, 2.0])
        mu = float(w @ (K @ self.alpha))
        prior = atom_cov(Zq, kq, Zq, kq, self._h, self.layout)
        G = K @ self.U
        var = float(w @ (prior - G @ G.T) @ w)
        out = dict(mu_eff=mu, mu_eff_std=float(np.sqrt(max(var, 0.0))), kappa_eff=0.0, kappa_eff_std=0.0,
                   corr_mu_kappa_eff=float("nan"))
        if self.layout.vol:
            j = self.layout.vol[0]

            def vol_mean(Jv):
                z = jnp.asarray(self.layout.z_ref).at[j].set(Jv)
                return (self._kq(z[None], jnp.array([VALUE]), ["vol"]) @ self.alpha)[0]
            out["kappa_eff"] = float(jax.grad(jax.grad(vol_mean))(1.0))
        return out

    def linear_elastic_summary(self):
        m = self.small_strain_moduli()
        return dict(mu=m["mu_eff"] / 2, kappa=m["kappa_eff"], mu_std=m["mu_eff_std"] / 2, kappa_std=0.0, corr_mu_kappa=float("nan"))

    # ---------------- pathwise samples (Matheron's rule, RFF prior draws) ----------------
    def _path(self, key):
        """(total(f), comps(f)) of one posterior function draw."""
        keys = jax.random.split(key, 2 * len(self.layout.comps) + 1)
        feats = {}
        for i, (name, dims) in enumerate(self.layout.comps):
            ls = jnp.reshape(self._h[f"{name}_ls"], (-1,))
            om = jax.random.normal(keys[2 * i], (N_RFF, len(dims))) / ls[None, :]
            kb, kw = jax.random.split(keys[2 * i + 1])
            b = jax.random.uniform(kb, (N_RFF,), maxval=2 * jnp.pi)
            w = jax.random.normal(kw, (N_RFF,)) * self._h[f"{name}_sig"] * jnp.sqrt(2.0 / N_RFF)
            feats[name] = (jnp.asarray(dims), om, b, w)

        def prior_rows(Z, kinds, comps=None):
            out = jnp.zeros(Z.shape[0])
            for name, (dims, om, b, w) in feats.items():
                if comps is not None and name not in comps:
                    continue
                arg = Z[:, dims] @ om.T + b                                       # (n, M)
                val = jnp.cos(arg) @ w
                a = (kinds[:, None] == dims[None, :]).astype(Z.dtype)             # (n, dc)
                grad = -(jnp.sin(arg) * w[None, :]) @ om                            # (n, dc)
                out = out + jnp.where(kinds == VALUE, val, jnp.sum(grad * a, axis=1))
            return out

        def prior_atoms(Z, kinds, comps=None):
            """Prior draw at many atoms, in chunks (the (n, M) feature arrays dominate the memory, also when the
            caller vectorises over draws)."""
            n = Z.shape[0]
            if n <= CHUNK:
                return prior_rows(Z, kinds, comps)
            pad = (-n) % CHUNK
            Zp = jnp.concatenate([Z, jnp.zeros((pad, Z.shape[1]), Z.dtype)]).reshape(-1, CHUNK, Z.shape[1])
            kp = jnp.concatenate([kinds, jnp.full(pad, VALUE, kinds.dtype)]).reshape(-1, CHUNK)
            return jax.lax.map(lambda zk: prior_rows(zk[0], zk[1], comps), (Zp, kp)).reshape(-1)[:n]
        eps = jax.random.normal(keys[-1], self.y.shape) * jnp.sqrt(self.obs_noise)
        f_at = prior_atoms(self.Z, self.kinds)
        v = jax.scipy.linalg.cho_solve((self.L, True), self.y - self.A @ f_at - eps)
        corr = self.A.T @ v                                                         # (n_atoms,)
        names = [n for n, _ in self.layout.comps]

        def comp_value(f, name):
            z = self.z_of(f)[None]
            pri = prior_atoms(z, jnp.array([VALUE]), [name])[0]
            return pri + (self._kq(z, jnp.array([VALUE]), [name]) @ corr)[0]

        def total(f):
            return sum(comp_value(f, n) for n in names)

        def comps(f):
            dev = comp_value(f, "dev")
            vol = comp_value(f, "vol") if "vol" in names else jnp.zeros(())
            an = comp_value(f, "aniso") if "aniso" in names else jnp.zeros(())
            return dev, vol, an
        return total, comps

    def get_path_psi_fn(self, key, params=None, weights=None):
        return self._path(key)[0]

    def get_path_components_psi_fn(self, key, params=None, weights=None):
        return self._path(key)[1]

    def get_path_dev_vol_psi_fn(self, key, params=None, weights=None):
        comps = self._path(key)[1]
        return lambda f: comps(f)[:2]

    def psi(self, f_mesh, key, params=None, weights=None):
        path = self.get_path_psi_fn(key)
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        return path(f) if f.ndim == 2 else jax.lax.map(path, f.reshape(-1, 3, 3)).reshape(f.shape[:-2])

    def piola(self, f_mesh, key, params=None, weights=None):
        path = jax.grad(self.get_path_psi_fn(key))
        f = jnp.asarray(f_mesh, dtype=jnp.float64)
        return path(f) if f.ndim == 2 else jax.lax.map(path, f.reshape(-1, 3, 3)).reshape(f.shape)

    psi_pws = psi
    piola_pws = piola

    # ---------------- curves (reference-energy plots) ----------------
    def dev_curve_dist(self, feats_fn, lam):
        """Energy and dW/dlambda (mean, std) along a curve of deviatoric features feats_fn(lambda) -> (I1_bar, I2_bar)."""
        lam = jnp.asarray(lam, dtype=jnp.float64)
        dev = jax.vmap(feats_fn)(lam)
        ddev = jax.vmap(jax.jacfwd(feats_fn))(lam)                                   # (n, 2)
        Z = self._z_from_parts(dev=dev)
        W, vW = self._value_mean_var(Z)
        n = Z.shape[0]
        Zq = jnp.repeat(Z, 2, axis=0)
        kq = self._kq(Zq, jnp.tile(jnp.arange(2), n))
        gm = (kq @ self.alpha).reshape(n, 2)
        GU = (kq @ self.U).reshape(n, 2, -1)
        prior = np.zeros(2)
        prior[:] = float(self.hyper["dev_sig"]) ** 2 / np.reshape(self.hyper["dev_ls"], (-1,)) ** 2
        cov = jnp.eye(2)[None] * jnp.asarray(prior)[None, None] - jnp.einsum("ndq,neq->nde", GU, GU)
        dW = jnp.sum(gm * ddev, axis=1)
        vdW = jnp.einsum("nd,nde,ne->n", ddev, cov, ddev)
        return np.asarray(W), np.sqrt(np.maximum(np.asarray(vW), 0)), np.asarray(dW), np.sqrt(np.maximum(np.asarray(vdW), 0))

    # ---------------- persistence ----------------
    def save(self, path: str):
        np.savez(path, model_type="exact_gp", Z=np.asarray(self.Z), kinds=np.asarray(self.kinds), A=np.asarray(self.A),
                 y=np.asarray(self.y), obs_noise=np.asarray(self.obs_noise), L=np.asarray(self.L),
                 hyper=json.dumps({k: np.asarray(v).tolist() for k, v in self.hyper.items()}),
                 noise=json.dumps(self.noise), info=json.dumps(self.info, default=float),
                 incompressible_plane_stress=self.incompressible_plane_stress,
                 a0=np.asarray(self.a0) if self.is_anisotropic else np.zeros(0),
                 a1=np.asarray(self.a1) if (self.is_anisotropic and self.a1 is not None) else np.zeros(0))

    @classmethod
    def load(cls, path: str) -> "ExactGPHyperelasticity":
        d = np.load(path, allow_pickle=False)
        return cls(d["Z"], d["kinds"], d["A"], d["y"], d["obs_noise"], d["L"],
                   hyper={k: np.asarray(v) for k, v in json.loads(str(d["hyper"])).items()},
                   incompressible_plane_stress=bool(d["incompressible_plane_stress"]),
                   a0=d["a0"] if d["a0"].size else None, a1=d["a1"] if d["a1"].size else None,
                   info=json.loads(str(d["info"])), noise_info=json.loads(str(d["noise"])))


# ----------------------------------------------------------------------------- fit
def fit_exact_gp(f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA, *, energy_scale: float,
                 amplitude_factor: float = 10.0, lengthscale_factor: float = 1.5, sigma_cap: float = 10.0,
                 hyper_restarts: bool = True, incompressible_plane_stress: bool = False,
                 sigma_f_rel: Optional[float] = None, cv_folds: int = 2, cv_grid=None, cv_tol: float = 0.0,
                 cv_data: Optional[dict] = None, fibers=None, step_labels=None, hyper_steps: int = 200,
                 global_equilibrium: float = 0.0, support_R=None, support_sigma=None, verbose: bool = True,
                 log_fn=print, **unused):
    """Same data arguments and noise / s options as core.hsgp_inference.fit_hsgp (single or several experiments).
    The HSGP basis options (num_basis_*, box_factor, priors) do not apply and are ignored.
    Returns (ExactGPHyperelasticity, history)."""
    t0 = time.time()
    multi = isinstance(cells, (list, tuple))
    if not multi:
        f3x3_steps, R_obs, sigma_R, cells, node_type, dNdX, dA = ([x] for x in (f3x3_steps, R_obs, sigma_R, cells,
                                                                               node_type, dNdX, dA))
        step_labels = None if step_labels is None else [step_labels]
        support_R, support_sigma = [support_R], [support_sigma]
    if support_R is None:
        support_R = support_sigma = [None] * len(f3x3_steps)
    aniso_on = fibers is not None
    if aniso_on:
        extractor = AnisotropicFeatureExtractor(jnp.asarray(fibers[0], dtype=jnp.float64),
                                                a1=None if fibers[1] is None else jnp.asarray(fibers[1], dtype=jnp.float64),
                                                incompressible_plane_stress=incompressible_plane_stress)
    else:
        extractor = make_feature_extractor(incompressible_plane_stress)
    layout = Layout(incompressible_plane_stress, aniso_on)
    meshes = []
    for c, nt, dn, da in zip(cells, node_type, dNdX, dA):
        nt, c = np.asarray(nt), np.asarray(c)
        meshes.append(dict(cells=c, n_nodes=nt.shape[0], dNdX=np.asarray(dn), dA=np.asarray(da), **dof_sets(nt)))
    steps, labels = [], []
    for e, Fe in enumerate(f3x3_steps):
        Fe = np.asarray(Fe)
        lab_e = list(step_labels[e]) if step_labels is not None and step_labels[e] is not None else list(range(Fe.shape[0]))
        for t in range(Fe.shape[0]):
            R_t = np.asarray(R_obs[e])[t]
            sR_t = np.maximum(np.asarray(sigma_R[e])[t], 1e-3)
            S_t = None if support_R[e] is None else np.asarray(support_R[e])[t]
            sS_t = None if support_sigma[e] is None else np.asarray(support_sigma[e])[t]
            steps.append(step_operator(extractor, layout, Fe[t], meshes[e], R_t, sR_t, global_equilibrium, S_t, sS_t))
            labels.append((lab_e[t] if isinstance(lab_e[t], str) else f"{e}:{int(lab_e[t])}") if multi else int(lab_e[t]))
    T = len(steps)
    if T > 3:
        log_fn(f"[ExactGP] warning: {T} load steps; the exact GP is meant for 1-3 steps (cost ~ (steps x elements)^2)")

    # reference rows: psi(z_ref) = 0 [, dpsi/dJ = 0] [, dpsi/dI4 = dpsi/dI6 = 0]
    ref_kinds = [VALUE] + list(layout.vol) + list(layout.aniso)
    ref = dict(Z=np.tile(layout.z_ref, (len(ref_kinds), 1)), kinds=np.asarray(ref_kinds))
    ref_noise = (1e-6 * energy_scale) ** 2

    def assemble(idx):
        """Stacked atoms, block-diagonal rows, targets, fixed noise and free-row scales of the steps idx (+ ref)."""
        Zs = [steps[t]["Z"] for t in idx] + [ref["Z"]]
        ks = [steps[t]["kinds"] for t in idx] + [ref["kinds"]]
        n_at = sum(z.shape[0] for z in Zs)
        n_rows = sum(steps[t]["A"].shape[0] for t in idx) + len(ref_kinds)
        A = np.zeros((n_rows, n_at))
        r = c = 0
        for t in idx:
            a = steps[t]["A"]
            A[r:r + a.shape[0], c:c + a.shape[1]] = a
            r, c = r + a.shape[0], c + a.shape[1]
        A[r:, c:] = np.eye(len(ref_kinds))
        y = np.concatenate([steps[t]["y"] for t in idx] + [np.zeros(len(ref_kinds))])
        noise = np.concatenate([steps[t]["noise"] for t in idx] + [np.full(len(ref_kinds), ref_noise)])
        fs = np.concatenate([steps[t]["fscale"] for t in idx] + [np.zeros(len(ref_kinds))])
        return dict(Z=jnp.asarray(np.concatenate(Zs)), kinds=jnp.asarray(np.concatenate(ks)), A=jnp.asarray(A),
                    y=jnp.asarray(y), noise=jnp.asarray(noise), fs=jnp.asarray(fs))

    all_idx = list(range(T))
    full = assemble(all_idx)
    n_at, n_obs = full["Z"].shape[0], full["A"].shape[0]
    log_fn(f"[ExactGP] {T} load steps | {n_at} atoms (gradients at the element states + reference) | {n_obs} observation "
           f"rows | inputs {layout.names()} ({layout.D} dims); assembled in {time.time() - t0:.0f}s")

    # hyperparameters: log amplitudes (capped), lengthscales in [0.02, 50] x data span (sigmoid)
    Zall = np.asarray(full["Z"])
    span = np.maximum(Zall.max(0) - Zall.min(0), 1e-6)
    sig_centre = float(np.log(amplitude_factor * energy_scale))
    cap = sig_centre + np.log(sigma_cap) if sigma_cap and sigma_cap > 0 else np.inf
    lo_l, hi_l = np.log(0.02 * span), np.log(50.0 * span)

    def to_raw(log_l, lo, hi):
        s = np.clip((log_l - lo) / (hi - lo), 0.02, 0.98)
        return np.log(s / (1 - s))

    def unpack(eta):
        h = {}
        for name, dims in layout.comps:
            lo, hi = jnp.asarray(lo_l[dims]), jnp.asarray(hi_l[dims])
            h[f"{name}_sig"] = jnp.exp(jnp.minimum(eta[f"log_sig_{name}"], cap))
            h[f"{name}_ls"] = jnp.exp(lo + (hi - lo) * jax.nn.sigmoid(eta[f"raw_ls_{name}"]))
        return h

    def eta_start(factor):
        return {**{f"log_sig_{n}": jnp.asarray(sig_centre) for n, _ in layout.comps},
                **{f"raw_ls_{n}": jnp.asarray(to_raw(np.log(lengthscale_factor * factor * span[d]), lo_l[d], hi_l[d]))
                   for n, d in layout.comps}}
    starts = [eta_start(1.0)] + ([eta_start(0.5), eta_start(2.0)] if hyper_restarts else [])

    def chol_C(eta, P, s):
        h = unpack(eta)
        K = atom_cov(P["Z"], P["kinds"], P["Z"], P["kinds"], h, layout)
        C = P["A"] @ K @ P["A"].T + jnp.diag(P["noise"] + s ** 2 * P["fs"])
        C = 0.5 * (C + C.T) + 1e-10 * jnp.mean(jnp.diag(C)) * jnp.eye(C.shape[0])
        return jnp.linalg.cholesky(C)

    def neg_log_ev(eta, P, s):
        L = chol_C(eta, P, s)
        a = jax.scipy.linalg.solve_triangular(L, P["y"], lower=True)
        return 0.5 * a @ a + jnp.sum(jnp.log(jnp.diag(L))) + 0.5 * P["y"].shape[0] * jnp.log(2 * jnp.pi)

    def optimise(P, s):
        fun = jax.jit(lambda e: neg_log_ev(e, P, s))
        opt = optax.lbfgs()
        vg = optax.value_and_grad_from_state(fun)

        @jax.jit
        def run(e0):
            def body(_, carry):
                e, st = carry
                val, g = vg(e, state=st)
                u, st = opt.update(g, st, e, value=val, grad=g, value_fn=fun)
                return optax.apply_updates(e, u), st
            return jax.lax.fori_loop(0, hyper_steps, body, (e0, opt.init(e0)))[0]
        best = None
        for e0 in starts:
            e = run(e0)
            v = float(fun(e))
            if np.isfinite(v) and (best is None or v < best[0]):
                best = (v, e)
        return best

    def predict_reactions(P, eta, s, step_ops):
        """Held-out reaction rows of other steps: (mean, cov) per step given the fit on P."""
        h = unpack(eta)
        L = chol_C(eta, P, s)
        w = jax.scipy.linalg.cho_solve((L, True), P["y"])
        alpha = P["A"].T @ w
        out = []
        for st in step_ops:
            nf, nr = st["n_free"], st["n_react"]
            if nr == 0:
                continue
            Aq, yq, nq = jnp.asarray(st["A"][nf:nf + nr]), st["y"][nf:nf + nr], st["noise"][nf:nf + nr]   # reaction rows
            Kq = atom_cov(jnp.asarray(st["Z"]), jnp.asarray(st["kinds"]), P["Z"], P["kinds"], h, layout)
            Kqq = atom_cov(jnp.asarray(st["Z"]), jnp.asarray(st["kinds"]), jnp.asarray(st["Z"]), jnp.asarray(st["kinds"]), h, layout)
            AK = Aq @ Kq
            G = jax.scipy.linalg.solve_triangular(L, P["A"] @ AK.T, lower=True)
            m = AK @ alpha
            C = Aq @ Kqq @ Aq.T - G.T @ G + jnp.diag(jnp.asarray(nq))
            d = jnp.asarray(yq) - m
            out.append(float(-0.5 * (d @ jnp.linalg.solve(C, d) + jnp.linalg.slogdet(C)[1] + len(yq) * np.log(2 * np.pi))))
        return sum(out)

    cv = None
    if sigma_f_rel is None:
        grid = sorted(float(g) for g in (cv_grid if cv_grid is not None else np.logspace(-4, 0, 9)))
        if cv_data is not None:
            v_ops = [step_operator(extractor, layout, np.asarray(cv_data["f3x3"])[v], meshes[0],
                                   np.asarray(cv_data["R"])[v], np.maximum(np.asarray(cv_data["sigma_R"])[v], 1e-3),
                                   global_equilibrium,
                                   None if cv_data.get("support_R") is None else np.asarray(cv_data["support_R"])[v],
                                   None if cv_data.get("support_sigma") is None else np.asarray(cv_data["support_sigma"])[v])
                     for v in range(np.asarray(cv_data["f3x3"]).shape[0])]
            folds = []
        else:
            if T < 2:
                raise ValueError("exact GP with one load step: give hsgp_force_noise_rel > 0, or hsgp_cv_holdout: val")
            K = min(max(2, int(cv_folds)), T)
            folds = [all_idx[k::K] for k in range(K)]

        def score(s):
            if cv_data is not None:
                b = optimise(full, s)
                return -np.inf if b is None else predict_reactions(full, b[1], s, v_ops)
            tot = 0.0
            for hold in folds:
                P = assemble([t for t in all_idx if t not in hold])
                b = optimise(P, s)
                if b is None:
                    return -np.inf
                tot += predict_reactions(P, b[1], s, [steps[t] for t in hold])
            return tot
        scores = {}
        for g in grid:
            scores[g] = score(g)
            log_fn(f"[ExactGP] cross-validation: s = {g:.3g}: held-out reaction score {scores[g]:.2f}")
        s_best = max(scores, key=scores.get)
        sel = min(g for g, v in scores.items() if v >= scores[s_best] - cv_tol) if cv_tol > 0 else s_best
        sigma_f_rel = sel
        cv = dict(method="cross-validation", holdout="validation" if cv_data is not None else "folds", s=sel,
                  s_best_score=s_best, tol=float(cv_tol), folds=[[labels[t] for t in f] for f in folds],
                  validation_steps=[int(x) for x in cv_data.get("labels", [])] if cv_data is not None else None,
                  curve={f"{k:.6g}": v for k, v in sorted(scores.items())})
        log_fn(f"[ExactGP] cross-validation: selected s = {sel:.3g}")

    s = float(sigma_f_rel)
    best = optimise(full, s)
    if best is None:
        raise RuntimeError("exact GP: no finite evidence")
    h = {k: np.asarray(v) for k, v in unpack(best[1]).items()}
    L = np.asarray(chol_C(best[1], full, s))
    log_ev = -best[0]
    hyper = dict(h, s=np.asarray(s))
    if not layout.vol:
        hyper.update(vol_sig=np.asarray(0.0), vol_ls=np.asarray([1.0]))
    rec = dict(iteration=0, log_evidence=log_ev, s=s, time=time.time() - t0,
               **{f"{n}_sig": float(h[f"{n}_sig"]) for n, _ in layout.comps},
               **{f"{n}_ls": np.reshape(h[f"{n}_ls"], (-1,)).tolist() for n, _ in layout.comps})
    if verbose:
        log_fn(f"[ExactGP] posterior: logZ {log_ev:.2f} | sigma_free = {s:.3g} x |R_t| | "
               + " | ".join(f"{n}: sig {float(h[f'{n}_sig']):.3g} l {np.round(np.reshape(h[f'{n}_ls'], (-1,)), 3)}"
                            for n, _ in layout.comps) + f" ({rec['time']:.0f}s)")
    noise_full = np.asarray(full["noise"] + s ** 2 * full["fs"])
    info = dict(likelihood="residual_exact", model="exact_gp", sigma_f_rel=s, force_noise_cv=cv, converged=True,
                global_equilibrium=float(global_equilibrium), reaction_rows=int(steps[0]["n_react"]),
                log_evidence=log_ev, valid_load_steps=labels, n_atoms=int(n_at), n_obs=int(n_obs),
                lengthscale_bounds=dict(lo=np.exp(lo_l).tolist(), hi=np.exp(hi_l).tolist()),
                amplitude_prior_centre=float(np.exp(sig_centre)), sigma_cap=sigma_cap, fit_time=time.time() - t0,
                anisotropic=aniso_on)
    model = ExactGPHyperelasticity(np.asarray(full["Z"]), np.asarray(full["kinds"]), np.asarray(full["A"]),
                                   np.asarray(full["y"]), noise_full, L, hyper,
                                   incompressible_plane_stress=incompressible_plane_stress,
                                   a0=fibers[0] if aniso_on else None, a1=fibers[1] if aniso_on else None,
                                   info=info, noise_info=dict(sigma_f_rel=s))
    return model, [rec]

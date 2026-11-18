"""
plots/training.py: extraction visualizations of the GP teacher (HSGP / exact GP): standard-mode energy and stress
validation, interpolation / extrapolation regimes, invariant-space modes and training R^2.
"""

import os
import json
import yaml
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from plots.theme import (
    apply_style,
    save_figure,
    MODE_NAMES,
    MODE_COLORS,
    CURVE_STYLES,
)


def standard_mode_F(learned_gp, gamma):
    """
    Deformation gradients (6, len(gamma), 3, 3) of the six standard modes (MODE_NAMES order: uniaxial tension,
    equibiaxial tension, pure shear, uniaxial compression, equibiaxial compression, simple shear) for gamma in [0, 1].
    Plane strain / compressible: in-plane modes with F33 = 1. Incompressible plane stress (learned_gp.
    incompressible_plane_stress): F33 = 1 / det F_2D, so the in-plane F is chosen to give each mode's 3D stretches
    (with the plane-strain definitions, uniaxial tension/compression would equal pure shear).
    """
    gamma = jnp.asarray(gamma)
    n = gamma.shape[0]

    def set_F(f11, f22, f33, f12=0.0):
        arr = jnp.zeros((n, 3, 3))
        return arr.at[:, 0, 0].set(f11).at[:, 1, 1].set(f22).at[:, 2, 2].set(f33).at[:, 0, 1].set(f12)

    lam = 1 + gamma
    if getattr(learned_gp, "incompressible_plane_stress", False):
        modes = [set_F(lam, lam ** -0.5, lam ** -0.5),          # uniaxial tension
                 set_F(lam, lam, lam ** -2),                    # equibiaxial tension
                 set_F(lam, 1 / lam, 1.0),                      # pure shear
                 set_F(1 / lam, lam ** 0.5, lam ** 0.5),        # uniaxial compression
                 set_F(1 / lam, 1 / lam, lam ** 2),             # equibiaxial compression
                 set_F(1.0, 1.0, 1.0, f12=gamma)]               # simple shear
    else:
        modes = [set_F(lam, 1.0, 1.0), set_F(lam, lam, 1.0), set_F(lam, 1 / lam, 1.0),
                 set_F(1 / lam, 1.0, 1.0), set_F(1 / lam, 1 / lam, 1.0), set_F(1.0, 1.0, 1.0, f12=gamma)]
    return jnp.stack(modes)


def compute_interpolation_masks(learned_gp, F_all, I_obs=None, dev_tol: float = 1e-3, vol_tol: float = 1e-3):
    """
    Interpolation/extrapolation masks of standard deformation modes with respect to the TRAINING DATA support.

    psi = psi_dev(I1_bar, I2_bar) + psi_vol(J) [+ psi_aniso], so each GP component interpolates when its own
    input lies in its data support: the convex hull of the observed (I1_bar, I2_bar), the observed J range and,
    if anisotropic, the observed range of each anisotropic invariant.

    F_all: (n_modes, n_points, 3, 3). I_obs: observed features (..., d) as saved in I_obs_all.npy
    (dev in [:2], vol in [2:3], aniso in [3:]); if None, learned_gp.dev_z / vol_z (the observed invariants that
    core.teacher.load_hsgp_teacher attaches).
    Returns dict of boolean arrays (n_modes, n_points): 'dev', 'vol', 'aniso', 'tot'.
    """
    if I_obs is None:
        parts = [np.asarray(learned_gp.dev_z), np.asarray(learned_gp.vol_z)]
        if getattr(learned_gp, "is_anisotropic", False):
            parts.append(np.asarray(learned_gp.aniso_z))
        I_obs = np.concatenate(parts, axis=-1)
    I_obs = np.asarray(I_obs).reshape(-1, np.asarray(I_obs).shape[-1])
    dev_obs, vol_obs, aniso_obs = I_obs[:, :2], I_obs[:, 2:3], I_obs[:, 3:]
    # data support: concave hulls (DataRegion) including the reference state F = I (zero energy and stress)
    region_dev = deviatoric_region(dev_obs)
    region_an = fibre_region(aniso_obs[:, :2]) if aniso_obs.shape[1] >= 2 else None
    vol_obs = np.vstack([vol_obs, [[1.0]]])
    vol_lo, vol_hi = vol_obs.min() - vol_tol, vol_obs.max() + vol_tol

    masks = {"dev": [], "vol": [], "aniso": [], "tot": []}
    for mode in range(F_all.shape[0]):
        feats = jax.vmap(learned_gp.feature_extractor.extract)(F_all[mode])
        dev_m, vol_m = np.asarray(feats[0]), np.asarray(feats[1])
        in_dev = region_dev.contains(dev_m)
        in_vol = (vol_m[:, 0] >= vol_lo) & (vol_m[:, 0] <= vol_hi)
        in_aniso = np.ones_like(in_dev)
        if len(feats) > 2 and region_an is not None:
            in_aniso = region_an.contains(np.asarray(feats[2])[:, :2])
        masks["dev"].append(in_dev)
        masks["vol"].append(in_vol)
        masks["aniso"].append(in_aniso)
        masks["tot"].append(in_dev & in_vol & in_aniso)
    return {k: np.stack(v) for k, v in masks.items()}


def shade_regimes(ax, gamma, inside, label: bool = False):
    """Shades interpolation (green) / extrapolation (red) segments of one mode; a mode may cross the data
    support several times, and every crossing is marked with a dotted line."""
    gamma = np.asarray(gamma)
    inside = np.asarray(inside, dtype=bool)
    cross = np.where(np.diff(inside.astype(int)) != 0)[0]
    # boundary halfway between the last point on one side and the first on the other
    bounds = [gamma[0]] + [0.5 * (gamma[i] + gamma[i + 1]) for i in cross] + [gamma[-1]]
    states = [inside[0]] + [inside[i + 1] for i in cross]
    labelled = set()
    if label:  # proxies: both regimes appear in the legend even if this mode never leaves one of them
        for name, color in (("Interpolation", "green"), ("Extrapolation", "red")):
            ax.fill_between([], [], color=color, alpha=0.10, label=name)
            labelled.add(name)
    for g0, g1, ins in zip(bounds[:-1], bounds[1:], states):
        name = "Interpolation" if ins else "Extrapolation"
        ax.axvspan(g0, g1, color="green" if ins else "red", alpha=0.10, zorder=1,
                   label=name if (label and name not in labelled) else "")
        labelled.add(name)
    for b in bounds[1:-1]:
        ax.axvline(x=b, color="darkred", linestyle=":", lw=1.5, alpha=0.8, zorder=4)


def plot_combined_validation(learned_gp, true_model, save_path, step, I_obs=None):
    """Plots clamped strain energy density and stress predictions against ground truth across 6 deformation modes."""
    apply_style()
    num_points = 50
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points)

    F_all = standard_mode_F(learned_gp, gamma)

    mode_names = MODE_NAMES

    psi_vmap = jax.vmap(jax.vmap(learned_gp.psi, in_axes=(0, None)), in_axes=(0, None))
    piola_vmap = jax.vmap(jax.vmap(learned_gp.piola, in_axes=(0, None)), in_axes=(0, None))

    if true_model is not None:
        psi_true = jax.vmap(true_model.psi)(F_all)
        P_true = jax.vmap(jax.vmap(true_model.P))(F_all)
    else:
        psi_true = None
        P_true = None

    psi_dist_mean = [learned_gp.psi_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    psi_dist_var = [learned_gp.psi_dist(F_all[mode]).var for mode in range(len(mode_names))]

    P_dist_mean = [learned_gp.piola_dist(F_all[mode]).mean for mode in range(len(mode_names))]
    P_dist_var = [learned_gp.piola_dist(F_all[mode]).var for mode in range(len(mode_names))]

    keys = jax.random.split(jax.random.PRNGKey(step), num_samples)
    psi_samples = jax.vmap(psi_vmap, in_axes=(None, 0))(F_all, keys)
    psi_dets = [jax.vmap(learned_gp.psi_det)(F_all[mode]) for mode in range(len(mode_names))]

    P_samples = jax.vmap(piola_vmap, in_axes=(None, 0))(F_all, keys)
    P_dets = [jax.vmap(learned_gp.piola_det)(F_all[mode]) for mode in range(len(mode_names))]
    regime = compute_interpolation_masks(learned_gp, F_all, I_obs)

    fig, axes = plt.subplots(6, 2, figsize=(12, 24))
    fig.suptitle(f"Material Discovery Validation - Step {step}", fontsize=18, y=1.01)

    for i, name in enumerate(mode_names):
        if name == "Pure Shear":
            idx_comp = (1, 1)
        elif name == "Simple Shear":
            idx_comp = (0, 1)
        else:
            idx_comp = (0, 0)

        # Column 0: Energy
        ax_psi = axes[i, 0]
        if psi_true is not None:
            ax_psi.plot(gamma, psi_true[i], **CURVE_STYLES["gt"])
            y_min, y_max = jnp.min(psi_true[i]), jnp.max(psi_true[i])
        else:
            y_min = jnp.min(psi_dist_mean[i] - 1.96 * jnp.sqrt(jnp.maximum(1e-12, psi_dist_var[i])))
            y_max = jnp.max(psi_dist_mean[i] + 1.96 * jnp.sqrt(jnp.maximum(1e-12, psi_dist_var[i])))

        ax_psi.plot(gamma, psi_dets[i], color="red", lw=0.8, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_samples[:, i, :].T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_psi.plot(gamma, psi_dist_mean[i], **CURVE_STYLES["gp"])
        psi_std_i = jnp.sqrt(jnp.maximum(1e-12, psi_dist_var[i]))
        ax_psi.fill_between(gamma, psi_dist_mean[i] - 1.96 * psi_std_i,
                            psi_dist_mean[i] + 1.96 * psi_std_i, **CURVE_STYLES["gp_ci"])

        pad = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
        ax_psi.set_ylim(y_min - pad, y_max + pad)
        ax_psi.set_xlim(0, gamma.max())

        # Column 1: Stress
        ax_p = axes[i, 1]
        p_mean_comp = P_dist_mean[i][:, idx_comp[0], idx_comp[1]]
        p_std_comp = jnp.sqrt(jnp.maximum(1e-12, P_dist_var[i][:, idx_comp[0], idx_comp[1]]))
        p_samples_comp = P_samples[:, i, :, idx_comp[0], idx_comp[1]]
        p_det_comp = P_dets[i][:, idx_comp[0], idx_comp[1]]

        if P_true is not None:
            p_true_comp = P_true[i, :, idx_comp[0], idx_comp[1]]
            ax_p.plot(gamma, p_true_comp, **CURVE_STYLES["gt"])
            y_min_p, y_max_p = jnp.min(p_true_comp), jnp.max(p_true_comp)
        else:
            y_min_p = jnp.min(p_mean_comp - 1.96 * p_std_comp)
            y_max_p = jnp.max(p_mean_comp + 1.96 * p_std_comp)

        ax_p.plot(gamma, p_det_comp, color="red", lw=0.8, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_samples_comp.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
        ax_p.plot(gamma, p_mean_comp, **CURVE_STYLES["gp"])
        ax_p.fill_between(gamma, p_mean_comp - 1.96 * p_std_comp,
                          p_mean_comp + 1.96 * p_std_comp, **CURVE_STYLES["gp_ci"])

        pad_p = (y_max_p - y_min_p) * 0.1 if y_max_p != y_min_p else 1.0
        ax_p.set_ylim(y_min_p - pad_p, y_max_p + pad_p)
        ax_p.set_xlim(0, gamma.max())

        ax_psi.set_title(f"{name}: Energy")
        ax_p.set_title(f"{name}: Stress")
        for ax in [ax_psi, ax_p]:
            shade_regimes(ax, gamma, regime["tot"][i], label=(i == 0 and ax == ax_psi))
            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_file = os.path.join(save_path, f"clamped_validation_{step}.pdf")
    save_figure(fig, save_file)
    plt.close(fig)
    print(f"Clamped limits plot saved to: {save_file}")


class DataRegion:
    """
    Region of a 2D input space covered by the training states: a concave hull (alpha shape) instead of the convex
    hull, which over-states sparse data (e.g. the empty area above a narrow band of states).
      - the points (with an optional reference point, e.g. F = I) are thinned on a grid of `frac` x their range per
        axis, the thinned points triangulated (Delaunay), and the triangles with circumradius <= k x (95% quantile of
        the thinned nearest-neighbour spacing) kept;
      - a point is inside when it lies in a kept triangle or close to a training state (states on a curved edge of
        the band fall just outside the straight triangle edges): within one grid cell (its diagonal), and with a
        reference point within 5% of the point's distance from it (near F = I the invariants are small, so an
        absolute allowance would let every mode count as inside there); points closer to the reference than its 10th
        nearest training state are inside (the GP is pinned there).
    near_transform / near_scale: optional map (n, 2) -> (n, 2) and per-axis scale in which "close to a training
    state" is measured in addition (distance <= 1 after dividing by the scale), e.g. the deviatoric mode-separating
    coordinates (log10(I1_bar - 3), (I2_bar - 3) / (I1_bar - 3)): similar strain level and mode at any strain size.
    Falls back to the convex hull when the points are degenerate.
    """
    def __init__(self, points, reference=None, k: float = 4.0, frac: float = 0.01, near_transform=None, near_scale=None):
        from scipy.spatial import Delaunay, cKDTree, QhullError
        P = np.asarray(points, dtype=float).reshape(-1, 2)
        self.reference = None if reference is None else np.asarray(reference, dtype=float).reshape(2)
        if reference is not None:
            P = np.vstack([P, np.asarray(reference, dtype=float).reshape(1, 2)])
        self.points = P
        h = frac * np.maximum(P.max(0) - P.min(0), 1e-12)
        _, idx = np.unique(np.floor((P - P.min(0)) / h).astype(np.int64), axis=0, return_index=True)
        T = P[np.sort(idx)]
        self.delta = float(np.linalg.norm(h))
        self.tree = cKDTree(P)
        self.near_transform, self.near_scale = near_transform, None
        if near_transform is not None:
            self.near_scale = np.asarray(near_scale, dtype=float)
            data_t = near_transform(P if self.reference is None else P[np.linalg.norm(P - self.reference, axis=1) > 1e-12])
            self.tree_t = cKDTree(data_t / self.near_scale)
        self.r_ref = 0.0
        if self.reference is not None:
            d_ref = np.sort(np.linalg.norm(P - self.reference, axis=1))
            d_ref = d_ref[d_ref > 1e-12]
            self.r_ref = float(d_ref[min(9, len(d_ref) - 1)]) if len(d_ref) else 0.0
        try:
            self.tri = Delaunay(T)
        except QhullError:
            self.tri = None
            return
        A, B, C = (T[self.tri.simplices[:, i]] for i in range(3))
        a, b, c = (np.linalg.norm(B - C, axis=1), np.linalg.norm(A - C, axis=1), np.linalg.norm(A - B, axis=1))
        area = 0.5 * np.abs((B - A)[:, 0] * (C - A)[:, 1] - (B - A)[:, 1] * (C - A)[:, 0])
        R = a * b * c / np.maximum(4.0 * area, 1e-300)
        d_nn = cKDTree(T).query(T, k=2)[0][:, 1] if len(T) > 2 else np.ones(1)
        self.keep = R <= k * float(np.quantile(d_nn, 0.95))
        if not self.keep.any():
            self.keep[:] = True
        self.vertices = T
        tris = self.tri.simplices[self.keep]
        E = np.sort(np.vstack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
        u, cnt = np.unique(E, axis=0, return_counts=True)
        self.boundary_edges = u[cnt == 1]
        self.triangles = tris

    def contains(self, Q, tol: float = 0.0):
        Q = np.asarray(Q, dtype=float).reshape(-1, 2)
        allow = np.full(len(Q), self.delta)
        if self.near_transform is not None:
            # close in the transformed coordinates (similar mode, relative strain size) AND in absolute terms
            # (within one grid cell: a relative measure alone is too generous at large strain)
            near = ((self.tree_t.query(self.near_transform(Q) / self.near_scale)[0] <= 1.0)
                    & (self.tree.query(Q)[0] <= self.delta + tol))
            if self.reference is not None:
                near |= np.linalg.norm(Q - self.reference, axis=1) <= self.r_ref
        elif self.reference is not None:
            d_ref = np.linalg.norm(Q - self.reference, axis=1)
            allow = np.minimum(allow, 0.05 * d_ref)
            near = (self.tree.query(Q)[0] <= allow + tol) | (d_ref <= self.r_ref)
        else:
            near = self.tree.query(Q)[0] <= allow + tol
        if self.tri is None:
            return near
        s = self.tri.find_simplex(Q)
        return near | ((s >= 0) & self.keep[np.maximum(s, 0)])

    def boundary_polylines(self, n: int = 24):
        """Each boundary edge densified to n points (for drawing in transformed coordinates)."""
        if self.tri is None:
            return []
        V = self.vertices
        return [np.linspace(V[i], V[j], n) for i, j in self.boundary_edges]


def mode_coordinates(dev):
    """(I1_bar, I2_bar) -> (log10(I1_bar - 3), (I2_bar - 3) / (I1_bar - 3)): strain size and mode type."""
    dev = np.asarray(dev, dtype=float).reshape(-1, 2)
    a = np.maximum(dev[:, 0] - 3.0, 1e-12)
    return np.column_stack([np.log10(a), (dev[:, 1] - 3.0) / a])


def deviatoric_region(dev_obs):
    """Data region of the deviatoric GP inputs: concave hull with F = I; closeness in mode coordinates
    (within 0.25 decades of strain size and 0.05 in the I2/I1 ratio of a training state)."""
    return DataRegion(dev_obs, reference=[3.0, 3.0], near_transform=mode_coordinates, near_scale=[0.25, 0.05])


def fibre_coordinates(an):
    """(I4_bar, I6_bar) -> (log10 of the distance from (1, 1), direction angle): strain size and fibre-mode type."""
    an = np.asarray(an, dtype=float).reshape(-1, 2) - 1.0
    return np.column_stack([np.log10(np.maximum(np.linalg.norm(an, axis=1), 1e-12)), np.arctan2(an[:, 1], an[:, 0])])


def fibre_region(an_obs):
    """Data region of the fibre GP inputs (I4_bar, I6_bar): concave hull with F = I; closeness within 0.25 decades
    of distance from (1, 1) and 0.1 rad of direction (and one grid cell absolute)."""
    return DataRegion(an_obs, reference=[1.0, 1.0], near_transform=fibre_coordinates, near_scale=[0.25, 0.1])


def plot_invariant_space_modes(learned_gp, save_path, I_obs, gamma_max: float = 1.0, n_gamma: int = 400,
                               picked=None, picked_label="picked states", filename="invariant_space_modes.pdf"):
    """
    Deviatoric GP input space (I1_bar - 3, I2_bar - 3): the training invariants, their convex hull (the interpolation
    region of the energy plots) and the paths of the six standard modes for gamma in [0, gamma_max], with a cross and
    the gamma value where each path leaves the hull. (a) full range in mode-separating coordinates: I1_bar - 3 (log)
    against the ratio (I2_bar - 3) / (I1_bar - 3) (< 1: uniaxial-type stretch, > 1: biaxial-type; all paths share the
    diagonal in plain (I1, I2)); (b) linear zoom on the data in (I1_bar - 3, I2_bar - 3). For a
    compressible model, (c) J along each mode against the observed J range. For an anisotropic GP (I_obs columns
    3, 4 = I4_bar, I6_bar), a last panel shows the fibre invariants I4_bar against I6_bar with their hull, and a mode
    counts as leaving the data when it leaves EITHER the (I1_bar, I2_bar) hull or the (I4_bar, I6_bar) hull.
    picked: optional (N, 3) GP inputs (I1_bar, I2_bar, J) of selected states (e.g. the distillation export points),
    drawn on top of the data in (a), (b) and as a J histogram in (c).
    """
    from plots.theme import MODE_SHORT_NAMES
    apply_style()
    I_obs = np.asarray(I_obs).reshape(-1, np.asarray(I_obs).shape[-1])
    dev_obs, vol_obs = I_obs[:, :2], I_obs[:, 2:3]
    # data region of the deviatoric inputs: concave hull with the reference state F = I (zero energy and stress)
    region = deviatoric_region(dev_obs)
    gamma = jnp.linspace(0.0, gamma_max, n_gamma)
    F_all = standard_mode_F(learned_gp, gamma)
    incompressible = getattr(learned_gp, "incompressible_plane_stress", False)
    aniso = bool(getattr(learned_gp, "is_anisotropic", False)) and I_obs.shape[1] >= 5
    if aniso:                               # fibre invariants (I4_bar, I6_bar) of the training states and their region
        an_obs = I_obs[:, 3:5]
        region_an = fibre_region(an_obs)
    n_panels = (2 if incompressible else 3) + (1 if aniso else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.2 * n_panels, 4.6))
    rng = np.random.default_rng(0)
    sub = dev_obs[rng.choice(len(dev_obs), min(len(dev_obs), 6000), replace=False)]
    eps = 1e-8
    summary = {}
    def to_a(x1, x2):                    # panel (a): (I1 - 3, (I2 - 3) / (I1 - 3))
        a = np.maximum(np.asarray(x1) - 3, eps)
        return a, (np.asarray(x2) - 3) / a

    def to_b(x1, x2):                    # panel (b): (I1 - 3, I2 - 3)
        return np.asarray(x1) - 3, np.asarray(x2) - 3
    from matplotlib.collections import LineCollection
    import matplotlib.tri as mtri_
    polys = region.boundary_polylines()
    for ax, tf in ((axes[0], to_a), (axes[1], to_b)):
        ax.scatter(*tf(sub[:, 0], sub[:, 1]), s=5, c="#37474f", alpha=0.6, lw=0, zorder=3, label="training invariants")
        if tf is to_b and region.tri is not None:
            V = region.vertices
            ax.tripcolor(mtri_.Triangulation(V[:, 0] - 3, V[:, 1] - 3, region.triangles), np.ones(len(V)),
                         cmap="Greens", vmin=0, vmax=8, alpha=0.6, lw=0, zorder=0)
        ax.add_collection(LineCollection([np.column_stack(tf(pl[:, 0], pl[:, 1])) for pl in polys], colors="#2e7d32",
                                         linewidths=1.0, zorder=1))
        ax.plot([], [], color="#2e7d32", lw=1.0, label="interpolation region (data)")
        if picked is not None:
            pk = np.asarray(picked).reshape(-1, np.asarray(picked).shape[-1])
            ax.scatter(*tf(pk[:, 0], pk[:, 1]), s=22, marker="x", c="#111111", lw=1.0, zorder=6,
                       label=f"{picked_label} (N = {len(pk)})")
    if aniso:
        ax_an = axes[-1]
        sub_an = an_obs[rng.choice(len(an_obs), min(len(an_obs), 6000), replace=False)]
        ax_an.scatter(sub_an[:, 0], sub_an[:, 1], s=5, c="#37474f", alpha=0.6, lw=0, zorder=3, label="training invariants")
        if region_an.tri is not None:
            Va = region_an.vertices
            ax_an.tripcolor(mtri_.Triangulation(Va[:, 0], Va[:, 1], region_an.triangles), np.ones(len(Va)),
                            cmap="Greens", vmin=0, vmax=8, alpha=0.6, lw=0, zorder=0)
        ax_an.add_collection(LineCollection(region_an.boundary_polylines(), colors="#2e7d32", linewidths=1.0, zorder=1))
        ax_an.plot([], [], color="#2e7d32", lw=1.0, label="interpolation region (data)")
        if picked is not None and np.asarray(picked).reshape(-1, np.asarray(picked).shape[-1]).shape[1] >= 5:
            pk_an = np.asarray(picked).reshape(-1, np.asarray(picked).shape[-1])[:, 3:5]
            ax_an.scatter(pk_an[:, 0], pk_an[:, 1], s=22, marker="x", c="#111111", lw=1.0, zorder=6,
                          label=f"{picked_label} (N = {len(pk_an)})")
        an_paths = []
    for m in range(6):
        feats = jax.vmap(learned_gp.feature_extractor.extract)(F_all[m])
        dev_m, vol_m = np.asarray(feats[0]), np.asarray(feats[1])
        inside = region.contains(dev_m)
        if aniso:                           # outside the data if either input pair leaves its hull
            an_m = np.asarray(feats[2])[:, :2]
            inside_an = region_an.contains(an_m)
            inside = inside & inside_an
            an_paths.append(an_m)
        # first point after the reference state F = I that is outside the data (F = I itself can lie just outside the
        # hull, e.g. when every training state stretches a fibre; a path that starts outside then leaves at once)
        out_k = np.where(~inside[1:])[0]
        exits = [int(out_k[0]) + 1] if len(out_k) else []
        g_exit = float(gamma[exits[0]]) if exits else None
        summary[MODE_NAMES[m]] = g_exit
        col = MODE_COLORS[m]
        lab = f"{MODE_SHORT_NAMES[m]}" + (f" (exits at $\\gamma$ = {g_exit:.2f})" if g_exit is not None else " (inside up to $\\gamma$ = %g)" % gamma_max)
        for ax, tf in ((axes[0], to_a), (axes[1], to_b)):
            x, y = tf(dev_m[1:, 0], dev_m[1:, 1])
            ax.plot(x, y, color=col, lw=1.8, label=lab if ax is axes[0] else None)
            for k in exits:
                px, py = tf(dev_m[k:k + 1, 0], dev_m[k:k + 1, 1])
                ax.plot(px, py, "x", color=col, ms=9, mew=2.2)
                ax.annotate(f"{float(gamma[k]):.2f}", (float(px[0]), float(py[0])), textcoords="offset points",
                            xytext=(5, 4), fontsize=8, color=col)
        if aniso:
            ax_an.plot(an_m[:, 0], an_m[:, 1], color=col, lw=1.8)
            for k in exits:
                ax_an.plot(an_m[k, 0], an_m[k, 1], "x", color=col, ms=9, mew=2.2)
                ax_an.annotate(f"{float(gamma[k]):.2f}", (float(an_m[k, 0]), float(an_m[k, 1])),
                               textcoords="offset points", xytext=(5, 4), fontsize=8, color=col)
        if not incompressible:
            ax = axes[2]
            ax.plot(np.asarray(gamma), vol_m[:, 0], color=col, lw=1.8, label=MODE_SHORT_NAMES[m])
            lo_j, hi_j = vol_obs.min() - 1e-3, vol_obs.max() + 1e-3
            out = np.where((vol_m[:, 0] < lo_j) | (vol_m[:, 0] > hi_j))[0]
            if len(out):
                ax.plot(float(gamma[out[0]]), vol_m[out[0], 0], "x", color=col, ms=9, mew=2.2)
    axes[0].set_xscale("log")
    pos = (dev_obs[:, 0] - 3)[(dev_obs[:, 0] - 3) > 0]
    axes[0].set_xlim(left=max(1e-6, 0.3 * float(np.quantile(pos, 0.001))) if len(pos) else 1e-6)
    axes[0].set_ylim(0.0, 2.5)
    axes[0].set_title("(a) Mode-separating coordinates, full range")
    ext = (dev_obs - 3).max(0)
    axes[1].set_xlim(-0.05 * ext[0], 1.5 * ext[0]); axes[1].set_ylim(-0.05 * ext[1], 1.5 * ext[1])
    axes[1].set_title("(b) Zoom on the data")
    axes[0].set_xlabel(r"$\bar I_1 - 3$"); axes[0].set_ylabel(r"$(\bar I_2 - 3)\,/\,(\bar I_1 - 3)$")
    axes[1].set_xlabel(r"$\bar I_1 - 3$"); axes[1].set_ylabel(r"$\bar I_2 - 3$")
    for ax in axes[:2]:
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, loc="upper left", frameon=False)
    axes[1].legend(fontsize=7.5, loc="lower right", frameon=False)
    if not incompressible:
        ax = axes[2]
        ax.axhspan(vol_obs.min(), vol_obs.max(), color="#4caf50", alpha=0.12, label="observed J range")
        if picked is not None and pk.shape[1] > 2:   # J of the picked states as a rug on the right edge
            ax.plot(np.full(len(pk), gamma_max), pk[:, 2], "_", color="#111111", ms=10, mew=1.2, clip_on=False,
                    label=f"{picked_label} J")
        ax.set_xlabel(r"$\gamma$"); ax.set_ylabel("J (volumetric GP input)"); ax.grid(alpha=0.3)
        ax.set_title("(c) Volumetric input along the modes"); ax.legend(fontsize=7.5, frameon=False)
    if aniso:
        lo_d, hi_d = an_obs.min(0), an_obs.max(0)
        span = np.maximum(hi_d - lo_d, 1e-3)
        lo_v = np.minimum(lo_d - 0.15 * span, np.min([p.min(0) for p in an_paths], axis=0))
        lo_v = np.maximum(lo_v, lo_d - 0.6 * span)       # paths can run far; keep the data readable
        hi_v = hi_d + 0.4 * span
        lim_lo, lim_hi = float(min(lo_v)), float(max(hi_v))
        ax_an.plot([lim_lo, lim_hi], [lim_lo, lim_hi], ls="--", color="#555555", lw=0.9, label=r"$\bar I_4 = \bar I_6$")
        ax_an.axvline(1.0, color="#999999", lw=0.7, ls=":")
        ax_an.axhline(1.0, color="#999999", lw=0.7, ls=":")
        ax_an.set_xlim(lim_lo, lim_hi)
        ax_an.set_ylim(lim_lo, lim_hi)
        ax_an.set_aspect("equal", adjustable="box")
        ax_an.set_xlabel(r"$\bar I_4$ (fibre family 1)")
        ax_an.set_ylabel(r"$\bar I_6$ (fibre family 2)")
        ax_an.set_title(f"({'abcd'[n_panels - 1]}) Fibre invariants (< 1: fibre compressed)")
        ax_an.grid(alpha=0.3)
        ax_an.legend(fontsize=7.5, loc="upper left", frameon=False)
    fig.suptitle("Invariant space: data, interpolation region and standard deformation modes", fontsize=12)
    fig.tight_layout()
    save_figure(fig, os.path.join(save_path, filename))
    plt.close(fig)
    return summary


def plot_energy_decomposition_validation(learned_gp, true_model, save_path, I_obs=None):
    """Plots energy decomposition (Deviatoric, Volumetric, Anisotropic, Total) across 6 deformation modes."""
    apply_style()
    print("Generating Energy Decomposition Validation Plot...")
    num_points = 50
    num_samples = 32
    gamma = jnp.linspace(0.0, 1.0, num_points)

    F_all = standard_mode_F(learned_gp, gamma)

    mode_names = MODE_NAMES

    if true_model is not None:
        psi_true_tot = jax.vmap(jax.vmap(true_model.psi))(F_all)
        psi_true_dev = jax.vmap(jax.vmap(true_model.psi_dev))(F_all)
        psi_true_vol = jax.vmap(jax.vmap(true_model.psi_vol))(F_all)
        if hasattr(true_model, 'psi_aniso'):
            psi_true_aniso = jax.vmap(jax.vmap(true_model.psi_aniso))(F_all)
        else:
            psi_true_aniso = jnp.zeros_like(psi_true_tot)
    else:
        psi_true_tot = None
        psi_true_dev = None
        psi_true_vol = None
        psi_true_aniso = None

    dist_tot = jax.vmap(learned_gp.psi_dist)(F_all)
    psi_mean_tot, psi_std_tot = dist_tot.mean, jnp.sqrt(dist_tot.var)

    dist_dev = jax.vmap(learned_gp.dev_psi_dist)(F_all)
    psi_mean_dev, psi_std_dev = dist_dev.mean, jnp.sqrt(dist_dev.var)

    if hasattr(learned_gp, 'is_anisotropic') and learned_gp.is_anisotropic:
        dist_aniso = jax.vmap(learned_gp.aniso_psi_dist)(F_all)
        psi_mean_aniso, psi_std_aniso = dist_aniso.mean, jnp.sqrt(dist_aniso.var)
    else:
        psi_mean_aniso = jnp.zeros_like(psi_mean_tot)
        psi_std_aniso = jnp.zeros_like(psi_std_tot)

    dist_vol = jax.vmap(learned_gp.vol_psi_dist)(F_all)
    psi_mean_vol, psi_std_vol = dist_vol.mean, jnp.sqrt(dist_vol.var)

    keys = jax.random.split(jax.random.PRNGKey(42), num_samples)

    psi_samples_dev, psi_samples_vol, psi_samples_aniso_list, psi_samples_tot = [], [], [], []
    for k in keys:
        fn = learned_gp.get_path_components_psi_fn(k)
        dev_s, vol_s, aniso_s = jax.vmap(jax.vmap(fn))(F_all)
        psi_samples_dev.append(dev_s)
        psi_samples_vol.append(vol_s)
        psi_samples_aniso_list.append(aniso_s)
        psi_samples_tot.append(dev_s + vol_s + aniso_s)

    psi_samples_dev = jnp.stack(psi_samples_dev, axis=0)
    psi_samples_vol = jnp.stack(psi_samples_vol, axis=0)
    psi_samples_aniso = jnp.stack(psi_samples_aniso_list, axis=0)
    psi_samples_tot = jnp.stack(psi_samples_tot, axis=0)
    regime = compute_interpolation_masks(learned_gp, F_all, I_obs)

    def calc_metrics(true, mean, std):
        rmse = jnp.sqrt(jnp.mean((true - mean)**2))
        lower = mean - 1.96 * std
        upper = mean + 1.96 * std
        coverage = jnp.mean((true >= lower) & (true <= upper)) * 100
        return rmse, coverage

    fig, axes = plt.subplots(6, 4, figsize=(24, 24))
    fig.suptitle("Energy Decomposition Validation", fontsize=20, y=1.01)

    for i, name in enumerate(mode_names):
        configs = [
            (0, "Deviatoric", psi_true_dev[i] if psi_true_dev is not None else None, psi_mean_dev[i], psi_std_dev[i], psi_samples_dev[:, i, :], regime["dev"][i]),
            (1, "Volumetric", psi_true_vol[i] if psi_true_vol is not None else None, psi_mean_vol[i], psi_std_vol[i], psi_samples_vol[:, i, :], regime["vol"][i]),
            (2, "Anisotropic", psi_true_aniso[i] if psi_true_aniso is not None else None, psi_mean_aniso[i], psi_std_aniso[i], psi_samples_aniso[:, i, :], regime["aniso"][i]),
            (3, "Total Energy", psi_true_tot[i] if psi_true_tot is not None else None, psi_mean_tot[i], psi_std_tot[i], psi_samples_tot[:, i, :], regime["tot"][i])
        ]
        for col, col_name, true_val, mean_val, std_val, samples, inside in configs:
            ax = axes[i, col]
            if true_val is not None:
                ax.plot(gamma, true_val, 'k--', lw=1.8, label="True", zorder=5)
            ax.plot(gamma, samples.T, color="lightblue", lw=0.8, alpha=0.3, zorder=1)
            ax.plot(gamma, mean_val, color="#1f77b4", lw=2, label="GP Mean", zorder=3)
            ax.fill_between(gamma, mean_val - 1.96 * std_val, mean_val + 1.96 * std_val, color="#1f77b4", alpha=0.2, zorder=2)

            shade_regimes(ax, gamma, inside, label=(i == 0 and col == 3))

            if true_val is not None:
                rmse, coverage = calc_metrics(true_val, mean_val, std_val)
                ax.set_title(f"{name}: {col_name}\nRMSE: {rmse:.4f} | Cov: {coverage:.1f}%")
                y_min, y_max = jnp.min(true_val), jnp.max(true_val)
            else:
                ax.set_title(f"{name}: {col_name}\n(GP Discovery)")
                y_min = jnp.min(mean_val - 1.96 * std_val)
                y_max = jnp.max(mean_val + 1.96 * std_val)

            ax.set_xlabel(r"$\gamma$")
            ax.grid(True, alpha=0.25)

            pad = (y_max - y_min) * 0.1 if y_max != y_min else 1.0
            ax.set_ylim(y_min - pad, y_max + pad)
            ax.set_xlim(0, float(gamma.max()))
            if i == 0 and col == 3:
                ax.legend(loc="upper left", framealpha=0.9)

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "energy_decomposition.pdf"))
    plt.close(fig)


class ExtractionR2Metrics(tuple):
    """
    3-tuple (r2, rmse, coverage) ensuring backwards compatibility for callers
    expecting a 3-tuple return value, while exposing .train_metrics, .val_metrics,
    .test_metrics, and .by_component for detailed evaluation.
    """
    def __new__(cls, r2, rmse, coverage, train_metrics=None, val_metrics=None, test_metrics=None, by_component=None):
        return super().__new__(cls, (r2, rmse, coverage))

    def __init__(self, r2, rmse, coverage, train_metrics=None, val_metrics=None, test_metrics=None, by_component=None):
        self.r2 = r2
        self.rmse = rmse
        self.coverage = coverage
        self.train_metrics = train_metrics or {}
        self.val_metrics = val_metrics or {}
        self.test_metrics = test_metrics or {}
        self.by_component = by_component or {}


def _format_step_indices(steps):
    """Utility to format a collection of load step indices into compact strings (e.g. '1, 7, 13, 19' or '2-6, 8-12, 14-18')."""
    if not steps:
        return "None"
    steps = sorted(list(set(steps)))
    ranges = []
    start = steps[0]
    prev = steps[0]
    for s in steps[1:]:
        if s == prev + 1:
            prev = s
        else:
            if start == prev:
                ranges.append(f"{start}")
            elif prev == start + 1:
                ranges.append(f"{start}, {prev}")
            else:
                ranges.append(f"{start}-{prev}")
            start = s
            prev = s
    if start == prev:
        ranges.append(f"{start}")
    elif prev == start + 1:
        ranges.append(f"{start}, {prev}")
    else:
        ranges.append(f"{start}-{prev}")
    return ", ".join(ranges)


def _resolve_load_steps(num_steps, save_path=None, train_steps=None, val_steps=None, test_steps=None):
    """Resolves and validates train and validation step indices from explicit lists or config files (test is deprecated/omitted)."""
    if (val_steps is None or train_steps is None) and save_path:
        for cfg_file in ["recipe_config.yaml", "config.yaml", "config.json"]:
            cfg_path = os.path.join(save_path, cfg_file)
            if not os.path.exists(cfg_path):
                parent_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(save_path))), cfg_file)
                if os.path.exists(parent_cfg):
                    cfg_path = parent_cfg
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        if cfg_path.endswith(".json"):
                            cfg = json.load(f)
                        else:
                            cfg = yaml.safe_load(f)
                        if cfg:
                            if val_steps is None and "val_load_steps_indices" in cfg:
                                val_steps = cfg["val_load_steps_indices"]
                            if train_steps is None and "train_load_steps_indices" in cfg:
                                train_steps = cfg["train_load_steps_indices"]
                        if val_steps is not None and train_steps is not None:
                            break
                except Exception:
                    pass

    # If test_steps is None, check config file if available
    if test_steps is None and save_path:
        for cfg_file in ["recipe_config.yaml", "config.yaml", "config.json"]:
            cfg_path = os.path.join(save_path, cfg_file)
            if not os.path.exists(cfg_path):
                parent_cfg = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(save_path))), cfg_file)
                if os.path.exists(parent_cfg):
                    cfg_path = parent_cfg
            if os.path.exists(cfg_path):
                try:
                    with open(cfg_path, "r") as f:
                        if cfg_path.endswith(".json"):
                            cfg = json.load(f)
                        else:
                            cfg = yaml.safe_load(f)
                        if cfg and "test_load_steps_indices" in cfg and cfg["test_load_steps_indices"] is not None:
                            test_steps = cfg["test_load_steps_indices"]
                            break
                except Exception:
                    pass

    # Normalize train_steps
    if train_steps is not None and len(train_steps) > 0:
        if isinstance(train_steps, (int, float)):
            train_steps_list = [int(train_steps)]
        else:
            train_steps_list = [int(s) for s in train_steps if int(s) < num_steps]
        train_set = set(train_steps_list)
    else:
        train_set = set(range(num_steps))

    # Normalize val_steps
    if val_steps is not None and len(val_steps) > 0:
        if isinstance(val_steps, (int, float)):
            val_steps_list = [int(val_steps)]
        else:
            val_steps_list = [int(s) for s in val_steps if int(s) < num_steps]
        # Val steps are those in val_steps_list not in train_set
        val_set = set(val_steps_list) - train_set
    else:
        val_set = set()

    # If train_steps was not explicitly specified, train_set is all steps minus val_set
    if train_steps is None or len(train_steps) == 0:
        train_set = train_set - val_set

    # Test load steps is completely optional: only evaluate if explicitly defined and non-empty
    if test_steps is not None and len(test_steps) > 0:
        if isinstance(test_steps, (int, float)):
            test_steps_list = [int(test_steps)]
        else:
            test_steps_list = [int(s) for s in test_steps if int(s) < num_steps]
        # Test steps cannot overlap with train_set or val_set
        test_set = set(test_steps_list) - train_set - val_set
    else:
        test_set = set()

    train_steps_clean = [s for s in range(num_steps) if s in train_set]
    val_steps_clean = [s for s in range(num_steps) if s in val_set]
    test_steps_clean = [s for s in range(num_steps) if s in test_set]

    train_steps_str = _format_step_indices(train_steps_clean)
    val_steps_str = _format_step_indices(val_steps_clean)
    test_steps_str = _format_step_indices(test_steps_clean)

    return (
        train_set, val_set, test_set,
        train_steps_clean, val_steps_clean, test_steps_clean,
        train_steps_str, val_steps_str, test_steps_str
    )


def plot_training_r2(learned_gp, true_model, F_train_full, save_path, train_steps=None, val_steps=None, test_steps=None, n_pws_samples: int = 256):
    """
    Plots energy parity across training, validation (calibration), and test (extrapolation) steps.
    Row 0 displays Energy Parity (Psi).
    Row 1 displays True Energy Distribution (Histograms) across train, validation, and test datasets.
    Uses pathwise sampling (default 256 draws) to compute predictive uncertainties.
    """
    if true_model is None:
        print("[INFO] Ground truth material model is None (experimental data). Skipping true energy parity plots.")
        empty_metrics = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
        return ExtractionR2Metrics(
            0.0, 0.0, 0.0,
            train_metrics=empty_metrics,
            val_metrics=empty_metrics,
            test_metrics=empty_metrics,
            by_component={}
        )

    apply_style()
    print("Generating Training, Validation & Test Data Energy Parity & Distribution Plot...")
    num_steps = F_train_full.shape[0]

    (
        train_set, val_set, test_set,
        train_steps_clean, val_steps_clean, test_steps_clean,
        train_steps_str, val_steps_str, test_steps_str
    ) = _resolve_load_steps(num_steps, save_path, train_steps, val_steps, test_steps)

    has_aniso = (hasattr(learned_gp, 'is_anisotropic') and learned_gp.is_anisotropic) and hasattr(true_model, 'psi_aniso')

    if hasattr(true_model, 'psi_dev') and hasattr(true_model, 'psi_vol'):
        psi_dev_fn = lambda f: jax.vmap(true_model.psi_dev)(f)
        psi_vol_fn = lambda f: jax.vmap(true_model.psi_vol)(f)
    elif hasattr(true_model, 'dev_params') and hasattr(true_model, 'vol_params'):
        from core.material_models import get_material
        model_name = getattr(true_model, 'name', 'gmr')
        t_dev = get_material(model_name, dev_params=list(true_model.dev_params), vol_params=[0] * len(true_model.vol_params), jit_P=False)
        t_vol = get_material(model_name, dev_params=[0] * len(true_model.dev_params), vol_params=list(true_model.vol_params), jit_P=False)
        psi_dev_fn = lambda f: jax.vmap(t_dev.psi)(f)
        psi_vol_fn = lambda f: jax.vmap(t_vol.psi)(f)
    else:
        psi_dev_fn = None
        psi_vol_fn = None

    components = []
    if psi_dev_fn is not None and psi_vol_fn is not None:
        components.append(("Deviatoric", r"Deviatoric Energy ($\Psi_{\mathrm{dev}}$)", psi_dev_fn, lambda f: learned_gp.dev_psi_dist(f)))
        components.append(("Volumetric", r"Volumetric Energy ($\Psi_{\mathrm{vol}}$)", psi_vol_fn, lambda f: learned_gp.vol_psi_dist(f)))
    if has_aniso:
        components.append(("Anisotropic", r"Anisotropic Energy ($\Psi_{\mathrm{aniso}}$)", lambda f: jax.vmap(true_model.psi_aniso)(f), lambda f: learned_gp.aniso_psi_dist(f)))
    components.append(("Total Energy", r"Total Energy ($\Psi_{\mathrm{total}}$)", lambda f: jax.vmap(true_model.psi)(f), lambda f: learned_gp.psi_dist(f)))

    n_panels = len(components)
    fig, axes = plt.subplots(2, n_panels, figsize=(6 * n_panels, 12), squeeze=False)

    # Reference state F = I (3, 3)
    I_ref = jnp.eye(3)[None, ...]
    I_ref_single = jnp.eye(3)

    # Pre-evaluate energy pathwise sample realizations if enabled
    step_energy_samples = {}
    use_pws = (n_pws_samples is not None and n_pws_samples > 0 and hasattr(learned_gp, 'get_path_components_psi_fn'))
    if use_pws:
        print(f"Pre-evaluating energy pathwise realizations (n_samples = {n_pws_samples})...")
        from functools import partial
        batch_draws = min(32, n_pws_samples)

        @partial(jax.jit, static_argnums=(2,))
        def _eval_batch_energy(F_s, key, n_draws):
            keys = jax.random.split(key, n_draws)
            def single_draw(k):
                fn = learned_gp.get_path_components_psi_fn(k)
                dev_F, vol_F, aniso_F = jax.vmap(fn)(F_s)
                dev_I, vol_I, aniso_I = fn(I_ref_single)
                tot_F = dev_F + vol_F + (aniso_F if has_aniso else 0.0)
                tot_I = dev_I + vol_I + (aniso_I if has_aniso else 0.0)
                return (
                    dev_F, dev_F - dev_I,
                    vol_F, vol_F - vol_I,
                    aniso_F if has_aniso else jnp.zeros_like(dev_F),
                    (aniso_F - aniso_I) if has_aniso else jnp.zeros_like(dev_F),
                    tot_F, tot_F - tot_I
                )
            return jax.vmap(single_draw)(keys)

        for s in range(num_steps):
            if s in train_set or s in val_set or s in test_set:
                step_key = jax.random.PRNGKey(42 + s)
                accum_tuples = []
                for b_start in range(0, n_pws_samples, batch_draws):
                    b_size = min(batch_draws, n_pws_samples - b_start)
                    step_key, subk = jax.random.split(step_key)
                    batch_res = _eval_batch_energy(F_train_full[s], subk, b_size)
                    accum_tuples.append([np.array(x) for x in batch_res])
                
                # Concatenate across sample dimension (axis=0)
                step_energy_samples[s] = tuple(
                    np.concatenate([accum_tuples[b_idx][i] for b_idx in range(len(accum_tuples))], axis=0)
                    for i in range(len(accum_tuples[0]))
                )

    comp_metrics = {}
    ret_train = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
    ret_val = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
    ret_test = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}
    ret_overall = {"r2": 0.0, "rmse": 0.0, "ec": 0.0}

    def compute_metrics(y_t_list, y_m_list, y_s_list):
        if len(y_t_list) == 0:
            return np.nan, np.nan, np.nan
        y_t = jnp.concatenate(y_t_list)
        y_m = jnp.concatenate(y_m_list)
        y_s = jnp.concatenate(y_s_list)

        ss_tot = jnp.sum((y_t - jnp.mean(y_t)) ** 2)
        ss_res = jnp.sum((y_t - y_m) ** 2)
        r2_val_c = 1.0 - ss_res / (ss_tot + 1e-12)
        rmse_val_c = jnp.sqrt(jnp.mean((y_t - y_m) ** 2))
        lower = y_m - 1.96 * y_s
        upper = y_m + 1.96 * y_s
        cov_val_c = jnp.mean((y_t >= lower) & (y_t <= upper)) * 100.0
        return float(r2_val_c), float(rmse_val_c), float(cov_val_c)

    comp_idx_map = {
        "Deviatoric": (0, 1),
        "Volumetric": (2, 3),
        "Anisotropic": (4, 5),
        "Total Energy": (6, 7),
    }

    for col_idx, (comp_name, comp_label, true_fn, gp_dist_fn) in enumerate(components):
        # Evaluation at reference state I
        psi_I_true = float(jnp.squeeze(true_fn(I_ref)))
        psi_I_gp = float(jnp.squeeze(gp_dist_fn(I_ref).mean))

        train_true, train_mean, train_std = [], [], []
        val_true, val_mean, val_std = [], [], []
        test_true, test_mean, test_std = [], [], []

        train_true_rec, train_mean_rec, train_std_rec = [], [], []
        val_true_rec, val_mean_rec, val_std_rec = [], [], []
        test_true_rec, test_mean_rec, test_std_rec = [], [], []

        raw_idx, rec_idx = comp_idx_map.get(comp_name, (6, 7))

        for step in range(num_steps):
            if step not in train_set and step not in val_set and step not in test_set:
                continue

            F_step = F_train_full[step]
            true_psi = true_fn(F_step)
            dist = gp_dist_fn(F_step)
            mean_psi = dist.mean

            if use_pws and step in step_energy_samples:
                std_psi = jnp.std(step_energy_samples[step][raw_idx], axis=0)
                std_psi_rec = jnp.std(step_energy_samples[step][rec_idx], axis=0)
            else:
                std_psi = jnp.sqrt(dist.var)
                std_psi_rec = std_psi

            true_rec = true_psi - psi_I_true
            mean_rec = mean_psi - psi_I_gp

            if step in train_set:
                train_true.append(true_psi)
                train_mean.append(mean_psi)
                train_std.append(std_psi)
                train_true_rec.append(true_rec)
                train_mean_rec.append(mean_rec)
                train_std_rec.append(std_psi_rec)
            elif step in val_set:
                val_true.append(true_psi)
                val_mean.append(mean_psi)
                val_std.append(std_psi)
                val_true_rec.append(true_rec)
                val_mean_rec.append(mean_rec)
                val_std_rec.append(std_psi_rec)
            elif step in test_set:
                test_true.append(true_psi)
                test_mean.append(mean_psi)
                test_std.append(std_psi)
                test_true_rec.append(true_rec)
                test_mean_rec.append(mean_rec)
                test_std_rec.append(std_psi_rec)

        # Row 0: Energy Parity Plot
        ax_parity = axes[0, col_idx]
        tr_t, tr_m, tr_s = train_true, train_mean, train_std
        v_t, v_m, v_s = val_true, val_mean, val_std
        te_t, te_m, te_s = test_true, test_mean, test_std
        title_str = f"Energy Parity: {comp_name}"
        xlabel_str = f"True {comp_label}"
        ylabel_str = f"Predicted GP Mean {comp_label}"

        if len(te_t) > 0:
            ax_parity.errorbar(jnp.concatenate(te_t), jnp.concatenate(te_m),
                               yerr=1.96 * jnp.concatenate(te_s), fmt='o',
                               color='#d62728', ecolor='#e45756', alpha=0.40,
                               markersize=3.5, elinewidth=0.85,
                               label=f"Test Steps ({test_steps_str})")
        if len(v_t) > 0:
            ax_parity.errorbar(jnp.concatenate(v_t), jnp.concatenate(v_m),
                               yerr=1.96 * jnp.concatenate(v_s), fmt='o',
                               color='#1f77b4', ecolor='#4ba3e3', alpha=0.35,
                               markersize=3.2, elinewidth=0.8,
                               label=f"Val/Calib Steps ({val_steps_str})")
        if len(tr_t) > 0:
            ax_parity.errorbar(jnp.concatenate(tr_t), jnp.concatenate(tr_m),
                               yerr=1.96 * jnp.concatenate(tr_s), fmt='o',
                               color='gray', ecolor='#b0b0b0', alpha=0.25,
                               markersize=3.0, elinewidth=0.8,
                               label=f"Train Steps ({train_steps_str})")

        r2_tr, rmse_tr, cov_tr = compute_metrics(tr_t, tr_m, tr_s)
        r2_v, rmse_v, cov_v = compute_metrics(v_t, v_m, v_s)
        r2_te, rmse_te, cov_te = compute_metrics(te_t, te_m, te_s)
        r2_tot, rmse_tot, cov_tot = compute_metrics(
            tr_t + v_t + te_t,
            tr_m + v_m + te_m,
            tr_s + v_s + te_s
        )

        all_pts_t = tr_t + v_t + te_t
        all_pts_m = tr_m + v_m + te_m
        if len(all_pts_t) > 0:
            cat_true = jnp.concatenate(all_pts_t)
            cat_mean = jnp.concatenate(all_pts_m)
            min_val = min(float(cat_true.min()), float(cat_mean.min()))
            max_val = max(float(cat_true.max()), float(cat_mean.max()))
            margin = max((max_val - min_val) * 0.05, 1e-4)
            ax_parity.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', lw=1.5, label="Parity")

        box_lines = []
        if len(tr_t) > 0:
            box_lines.append(f"Train ({train_steps_str}):")
            box_lines.append(f"  $R^2$: {r2_tr:.4f}")
            box_lines.append(f"  RMSE: {rmse_tr:.4f}")
            box_lines.append(f"  EC: {cov_tr:.1f}%")

        if len(v_t) > 0:
            if len(box_lines) > 0:
                box_lines.append("")
            box_lines.append(f"Val ({val_steps_str}):")
            box_lines.append(f"  $R^2$: {r2_v:.4f}")
            box_lines.append(f"  RMSE: {rmse_v:.4f}")
            box_lines.append(f"  EC: {cov_v:.1f}%")

        if len(te_t) > 0:
            if len(box_lines) > 0:
                box_lines.append("")
            box_lines.append(f"Test ({test_steps_str}):")
            box_lines.append(f"  $R^2$: {r2_te:.4f}")
            box_lines.append(f"  RMSE: {rmse_te:.4f}")
            box_lines.append(f"  EC: {cov_te:.1f}%")

        box_text = "\n".join(box_lines)
        ax_parity.text(0.05, 0.95, box_text, transform=ax_parity.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'), fontsize=9.0)

        ax_parity.set_title(title_str, fontsize=13, fontweight='bold')
        ax_parity.set_xlabel(xlabel_str, fontsize=11)
        ax_parity.set_ylabel(ylabel_str, fontsize=11)
        ax_parity.grid(True, alpha=0.25)
        ax_parity.set_aspect('equal', adjustable='datalim')
        ax_parity.legend(loc='lower right', fontsize=8.5, framealpha=0.85)

        comp_metrics[comp_name] = {
            "train": {"r2": r2_tr, "rmse": rmse_tr, "ec": cov_tr},
            "val": {"r2": r2_v, "rmse": rmse_v, "ec": cov_v},
            "test": {"r2": r2_te, "rmse": rmse_te, "ec": cov_te},
            "overall": {"r2": r2_tot, "rmse": rmse_tot, "ec": cov_tot}
        }
        if comp_name == "Total Energy":
            ret_train = {"r2": r2_tr, "rmse": rmse_tr, "ec": cov_tr, "steps": train_steps_clean}
            ret_val = {"r2": r2_v, "rmse": rmse_v, "ec": cov_v, "steps": val_steps_clean}
            ret_test = {"r2": r2_te, "rmse": rmse_te, "ec": cov_te, "steps": test_steps_clean}
            ret_overall = {"r2": r2_tot, "rmse": rmse_tot, "ec": cov_tot, "steps": list(range(num_steps))}

        # Row 1: True Energy Distribution (Histogram)
        ax_hist = axes[1, col_idx]
        hist_bins = 30
        
        # Determine common bin range across datasets for this component
        vals_to_bin = []
        if len(tr_t) > 0:
            vals_to_bin.append(np.array(jnp.concatenate(tr_t)))
        if len(v_t) > 0:
            vals_to_bin.append(np.array(jnp.concatenate(v_t)))
        if len(te_t) > 0:
            vals_to_bin.append(np.array(jnp.concatenate(te_t)))

        if len(vals_to_bin) > 0:
            all_cat = np.concatenate(vals_to_bin)
            bin_min, bin_max = float(np.min(all_cat)), float(np.max(all_cat))
            if bin_max == bin_min:
                bin_edges = np.linspace(bin_min - 0.1, bin_max + 0.1, hist_bins + 1)
            else:
                bin_edges = np.linspace(bin_min, bin_max, hist_bins + 1)
        else:
            bin_edges = hist_bins

        if len(tr_t) > 0:
            ax_hist.hist(np.array(jnp.concatenate(tr_t)), bins=bin_edges, density=True,
                         alpha=0.45, color='gray', edgecolor='dimgray',
                         label=f"Train ({train_steps_str})")
        if len(v_t) > 0:
            ax_hist.hist(np.array(jnp.concatenate(v_t)), bins=bin_edges, density=True,
                         alpha=0.45, color='#1f77b4', edgecolor='#17598c',
                         label=f"Val/Calib ({val_steps_str})")
        if len(te_t) > 0:
            ax_hist.hist(np.array(jnp.concatenate(te_t)), bins=bin_edges, density=True,
                         alpha=0.45, color='#d62728', edgecolor='#a81f20',
                         label=f"Test ({test_steps_str})")

        ax_hist.set_title(f"True Energy Distribution: {comp_name}", fontsize=13, fontweight='bold')
        ax_hist.set_xlabel(f"True {comp_label}", fontsize=11)
        ax_hist.set_ylabel("Probability Density", fontsize=11)
        ax_hist.grid(True, alpha=0.25)
        ax_hist.legend(loc='upper right', fontsize=8.5, framealpha=0.85)

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "training_r2_energy.pdf"))
    save_figure(fig, os.path.join(save_path, "training_r2_energy.png"))
    plt.close(fig)

    primary_r2 = ret_train["r2"] if not np.isnan(ret_train["r2"]) else ret_overall["r2"]
    primary_rmse = ret_train["rmse"] if not np.isnan(ret_train["rmse"]) else ret_overall["rmse"]
    primary_cov = ret_train["ec"] if not np.isnan(ret_train["ec"]) else ret_overall["ec"]

    return ExtractionR2Metrics(
        primary_r2, primary_rmse, primary_cov,
        train_metrics=ret_train, val_metrics=ret_val, test_metrics=ret_test, by_component=comp_metrics
    )


def plot_training_r2_piola(
    learned_gp,
    true_model,
    F_train_full,
    save_path,
    train_steps=None,
    val_steps=None,
    test_steps=None,
    n_pws_samples: int = 256
):
    """
    Plots First Piola-Kirchhoff stress parity across training, validation, and test steps
    for all tensor components (P11, P22, P33, P12, P21) and Frobenius norm ||P||_F.
    Uses fast pathwise sampling (default 256 draws) to compute predictive uncertainties.
    """
    if true_model is None:
        print("[INFO] Ground truth material model is None (experimental data). Skipping true Piola stress parity plots.")
        return {}

    apply_style()
    print("Generating First Piola-Kirchhoff Stress Parity Plot...")
    num_steps = F_train_full.shape[0]

    (
        train_set, val_set, test_set,
        train_steps_clean, val_steps_clean, test_steps_clean,
        train_steps_str, val_steps_str, test_steps_str
    ) = _resolve_load_steps(num_steps, save_path, train_steps, val_steps, test_steps)

    stress_components = [
        ("P11", r"$P_{11}$ ($P_{xx}$)", (0, 0)),
        ("P22", r"$P_{22}$ ($P_{yy}$)", (1, 1)),
        ("P33", r"$P_{33}$ ($P_{zz}$)", (2, 2)),
        ("P12", r"$P_{12}$ ($P_{xy}$)", (0, 1)),
        ("P21", r"$P_{21}$ ($P_{yx}$)", (1, 0)),
        ("Norm", r"$\|\mathbf{P}\|_{\mathrm{F}}$ (Frobenius Norm)", None),
    ]

    # Pre-evaluate true stress, GP mean, and pathwise sample realizations per step
    step_results = {}
    print(f"Evaluating Piola stress across {num_steps} steps (pathwise draws = {n_pws_samples})...")
    @jax.jit
    def _eval_step_piola(F_s, key):
        keys = jax.random.split(key, n_pws_samples)
        # draws in batches of 8: a full vmap over the draws holds every draw's basis Jacobians at once
        return jax.lax.map(lambda k: learned_gp.piola_pws(F_s, k), keys, batch_size=8)

    for step in range(num_steps):
        if step not in train_set and step not in val_set and step not in test_set:
            continue
        F_step = F_train_full[step]
        P_tr = np.asarray(jax.vmap(true_model.P)(F_step))
        P_gp = np.asarray(jax.vmap(learned_gp.piola_det)(F_step))
        P_samps = np.asarray(_eval_step_piola(F_step, jax.random.PRNGKey(42 + step)))
        step_results[step] = (P_tr, P_gp, P_samps)

    fig, axes = plt.subplots(2, 3, figsize=(18, 12), squeeze=False)
    metrics_summary = {}

    def compute_metrics(y_t_list, y_m_list, y_s_list):
        if len(y_t_list) == 0:
            return np.nan, np.nan, np.nan
        y_t = jnp.concatenate(y_t_list)
        y_m = jnp.concatenate(y_m_list)
        y_s = jnp.concatenate(y_s_list)

        ss_tot = jnp.sum((y_t - jnp.mean(y_t)) ** 2)
        ss_res = jnp.sum((y_t - y_m) ** 2)
        if float(ss_tot) < 1e-6:
            r2_val_c = np.nan
        else:
            r2_val_c = 1.0 - ss_res / (ss_tot + 1e-12)
        rmse_val_c = jnp.sqrt(jnp.mean((y_t - y_m) ** 2))
        lower = y_m - 1.96 * y_s
        upper = y_m + 1.96 * y_s
        cov_val_c = jnp.mean((y_t >= lower) & (y_t <= upper)) * 100.0
        return float(r2_val_c), float(rmse_val_c), float(cov_val_c)

    def _fmt_r2(val):
        return "N/A (Const 0)" if (np.isnan(val) or val < -100.0) else f"{val:.4f}"

    for c_idx, (comp_id, comp_label, idx) in enumerate(stress_components):
        row_idx = c_idx // 3
        col_idx = c_idx % 3
        ax = axes[row_idx, col_idx]

        train_true, train_mean, train_std = [], [], []
        val_true, val_mean, val_std = [], [], []
        test_true, test_mean, test_std = [], [], []

        for step in range(num_steps):
            if step not in step_results:
                continue
            P_tr, P_gp, P_samps = step_results[step]
            if idx is not None:
                yt = P_tr[:, idx[0], idx[1]]
                ym = P_gp[:, idx[0], idx[1]]
                ys = jnp.std(P_samps[:, :, idx[0], idx[1]], axis=0)
            else:
                yt = jnp.linalg.norm(P_tr, axis=(-2, -1))
                ym = jnp.linalg.norm(P_gp, axis=(-2, -1))
                norms = jnp.linalg.norm(P_samps, axis=(-2, -1))
                ys = jnp.std(norms, axis=0)

            if step in test_set:
                test_true.append(yt)
                test_mean.append(ym)
                test_std.append(ys)
            elif step in val_set:
                val_true.append(yt)
                val_mean.append(ym)
                val_std.append(ys)
            else:
                train_true.append(yt)
                train_mean.append(ym)
                train_std.append(ys)

        if len(test_true) > 0:
            ax.errorbar(jnp.concatenate(test_true), jnp.concatenate(test_mean),
                        yerr=1.96 * jnp.concatenate(test_std), fmt='o',
                        color='#d62728', ecolor='#e45756', alpha=0.40,
                        markersize=3.5, elinewidth=0.85,
                        label=f"Test Steps ({test_steps_str})")
        if len(val_true) > 0:
            ax.errorbar(jnp.concatenate(val_true), jnp.concatenate(val_mean),
                        yerr=1.96 * jnp.concatenate(val_std), fmt='o',
                        color='#1f77b4', ecolor='#4ba3e3', alpha=0.35,
                        markersize=3.2, elinewidth=0.8,
                        label=f"Val/Calib Steps ({val_steps_str})")
        if len(train_true) > 0:
            ax.errorbar(jnp.concatenate(train_true), jnp.concatenate(train_mean),
                        yerr=1.96 * jnp.concatenate(train_std), fmt='o',
                        color='gray', ecolor='#b0b0b0', alpha=0.25,
                        markersize=3.0, elinewidth=0.8,
                        label=f"Train Steps ({train_steps_str})")

        r2_tr, rmse_tr, cov_tr = compute_metrics(train_true, train_mean, train_std)
        r2_v, rmse_v, cov_v = compute_metrics(val_true, val_mean, val_std)
        r2_te, rmse_te, cov_te = compute_metrics(test_true, test_mean, test_std)
        r2_tot, rmse_tot, cov_tot = compute_metrics(
            train_true + val_true + test_true,
            train_mean + val_mean + test_mean,
            train_std + val_std + test_std
        )

        all_pts_t = train_true + val_true + test_true
        all_pts_m = train_mean + val_mean + test_mean
        if len(all_pts_t) > 0:
            cat_true = jnp.concatenate(all_pts_t)
            cat_mean = jnp.concatenate(all_pts_m)
            min_val = min(float(cat_true.min()), float(cat_mean.min()))
            max_val = max(float(cat_true.max()), float(cat_mean.max()))
            margin = max((max_val - min_val) * 0.1, 0.05)
            ax.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'k--', lw=1.5, label="Parity")
            ax.set_xlim(min_val - margin, max_val + margin)
            ax.set_ylim(min_val - margin, max_val + margin)

        box_lines = []
        if len(train_true) > 0:
            box_lines.append(f"Train ({train_steps_str}):")
            box_lines.append(f"  $R^2$: {_fmt_r2(r2_tr)}")
            box_lines.append(f"  RMSE: {rmse_tr:.4f}")
            box_lines.append(f"  EC: {cov_tr:.1f}%")

        if len(val_true) > 0:
            if len(box_lines) > 0:
                box_lines.append("")
            box_lines.append(f"Val ({val_steps_str}):")
            box_lines.append(f"  $R^2$: {_fmt_r2(r2_v)}")
            box_lines.append(f"  RMSE: {rmse_v:.4f}")
            box_lines.append(f"  EC: {cov_v:.1f}%")

        if len(test_true) > 0:
            if len(box_lines) > 0:
                box_lines.append("")
            box_lines.append(f"Test ({test_steps_str}):")
            box_lines.append(f"  $R^2$: {_fmt_r2(r2_te)}")
            box_lines.append(f"  RMSE: {rmse_te:.4f}")
            box_lines.append(f"  EC: {cov_te:.1f}%")

        box_text = "\n".join(box_lines)
        ax.text(0.05, 0.95, box_text, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='#cccccc'), fontsize=9.0)

        ax.set_title(f"First Piola Stress Parity: {comp_label}", fontsize=13, fontweight='bold')
        ax.set_xlabel(f"True Stress {comp_label}", fontsize=11)
        ax.set_ylabel(f"Predicted GP Mean {comp_label}", fontsize=11)
        ax.grid(True, alpha=0.25)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(loc='lower right', fontsize=8.5, framealpha=0.85)

        metrics_summary[comp_id] = {
            "train": {"r2": r2_tr, "rmse": rmse_tr, "ec": cov_tr},
            "val": {"r2": r2_v, "rmse": rmse_v, "ec": cov_v},
            "test": {"r2": r2_te, "rmse": rmse_te, "ec": cov_te},
            "overall": {"r2": r2_tot, "rmse": rmse_tot, "ec": cov_tot}
        }

    plt.tight_layout()
    save_figure(fig, os.path.join(save_path, "training_r2_piola_stress.pdf"))
    save_figure(fig, os.path.join(save_path, "training_r2_piola_stress.png"))
    plt.close(fig)

    metrics_file = os.path.join(save_path, "piola_stress_metrics.json")
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics_summary, f, indent=2)
    except Exception as e:
        print(f"Warning: could not write {metrics_file}: {e}")

    return metrics_summary



import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from scipy.spatial import ConvexHull
import jax
from jax import config
config.update("jax_enable_x64", True)
import jax.numpy as jnp

from plots.theme import apply_style
from core.model import SparseHyperelasticityGP
from core.dataclass import GPRawParams
from core.material_models import get_material
from core.features import IsotropicFeatureExtractor
from core.utils import generate_standard_deformation_modes as generate_standard_modes

apply_style()

def main():
    exp_dir = "/home/mmdiscovery/shared/results/20260925T090659_gentthomas_1e-05_0.01_1.0_0.5_8_1.0_isotropic_block"
    seed15_dir = os.path.join(exp_dir, "15")
    saved_model_dir = os.path.join(seed15_dir, "extracted")
    distilled_dir_s15 = os.path.join(seed15_dir, "distilled")
    distilled_dir_s6 = os.path.join(exp_dir, "6", "distilled")
    out_dir = os.path.join(exp_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load UGP Model
    best_params_dict = np.load(os.path.join(saved_model_dir, "best_params.npy"), allow_pickle=True).item()
    gp_params = GPRawParams(**best_params_dict)
    I_z = jnp.load(os.path.join(saved_model_dir, "I_z.npy"))
    dev_z = I_z[:, :2]
    vol_z = I_z[:, 2:3] if I_z.shape[1] > 3 else I_z[:, 2:]
    min_dev, max_dev = jnp.min(dev_z, axis=0), jnp.max(dev_z, axis=0)
    min_vol, max_vol = jnp.min(vol_z, axis=0), jnp.max(vol_z, axis=0)

    learned_gp = SparseHyperelasticityGP(
        gp_params, I_z, min_dev, min_vol, max_dev, max_vol,
        beta=1.0, feature_extractor=IsotropicFeatureExtractor(),
        covariance_mode="diag",
        constraint_lengthscale=1
    )

    # 2. Ground Truth Model
    true_model = get_material(
        "gentthomas",
        dev_params=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        vol_params=[1.5, 0.0, 0.0],
        jit_P=False
    )

    # 3. Generate standard modes up to gamma=2.5
    max_gamma = 2.5
    num_points = 200
    F_all, gamma = generate_standard_modes(
        num_points=num_points, max_gamma=max_gamma,
        stress_mode="plane_stress", material_model=true_model
    )
    mode_names = ["UT", "ET", "PS", "UC", "EC", "SS"]
    mode_full_names = {
        "UT": "UT", "ET": "EBT", "PS": "PS",
        "UC": "UC", "EC": "EBC", "SS": "SS"
    }

    # 4. Invariant Hull for Extrapolation Shading
    I_obs_all = np.load(os.path.join(saved_model_dir, "I_obs_all.npy")).reshape(-1, 3)
    dev_obs = I_obs_all[:, :2]
    vol_obs = I_obs_all[:, 2:3]
    limit_min_vol, limit_max_vol = np.min(vol_obs, axis=0), np.max(vol_obs, axis=0)
    hull = ConvexHull(dev_obs)
    hull_eqs = hull.equations
    extractor = IsotropicFeatureExtractor()

    mode_limits_dev = {}
    mode_limits_vol = {}
    for i, name in enumerate(mode_names):
        feats_ext = jax.vmap(extractor.extract)(F_all[i])
        dev_I, vol_J = np.array(feats_ext[0]), np.array(feats_ext[1])
        inside_vol = (vol_J[:, 0] >= limit_min_vol[0] - 0.001) & (vol_J[:, 0] <= limit_max_vol[0] + 0.001)
        inside_dev = np.all(dev_I @ hull_eqs[:, :-1].T + hull_eqs[:, -1] <= 0.001, axis=1)

        crossings_dev = np.where(np.diff(inside_dev.astype(int)) != 0)[0]
        limits_dev = [float(gamma[idx]) for idx in crossings_dev]
        mode_limits_dev[name] = limits_dev

        crossings_vol = np.where(np.diff(inside_vol.astype(int)) != 0)[0]
        limits_vol = [float(gamma[idx]) for idx in crossings_vol]
        mode_limits_vol[name] = limits_vol

    # 5. Evaluate Ground Truth & GP
    psi_true = [jax.vmap(true_model.psi)(F_all[m]) for m in range(len(mode_names))]
    psi_gp_mean = [learned_gp.psi_dist(F_all[m]).mean for m in range(len(mode_names))]
    psi_gp_var = [learned_gp.psi_dist(F_all[m]).var for m in range(len(mode_names))]

    # 6. Distilled Models:
    # A) Seed 15 Distilled (Discovered Log Term E + C10)
    dev_s15 = np.load(os.path.join(distilled_dir_s15, "dev_flow_samples.npy"))[:64]
    vol_s15 = np.load(os.path.join(distilled_dir_s15, "vol_flow_samples.npy"))[:64]

    # B) Seed 6 Distilled (Typical Polynomial Fit: C01 + C10)
    dev_s6 = np.load(os.path.join(distilled_dir_s6, "dev_flow_samples.npy"))[:64]
    vol_s6 = np.load(os.path.join(distilled_dir_s6, "vol_flow_samples.npy"))[:64]

    def eval_distilled_samples(dev_samples, vol_samples, F_chunk):
        def single_eval(td, tv):
            d_p = list(td) + [0.0] * max(0, 10 - len(td))
            v_p = list(tv) + [0.0] * max(0, 3 - len(tv))
            m_dev = get_material("gmr", dev_params=d_p[:10], vol_params=[0.0]*3, jit_P=False)
            m_vol = get_material("gmr", dev_params=[0.0]*10, vol_params=v_p[:3], jit_P=False)
            return jax.vmap(m_dev.psi)(F_chunk) + jax.vmap(m_vol.psi)(F_chunk)
        return jax.vmap(single_eval)(dev_samples, vol_samples)

    psi_dist_s15 = [eval_distilled_samples(dev_s15, vol_s15, F_all[m]) for m in range(len(mode_names))]
    psi_dist_s6 = [eval_distilled_samples(dev_s6, vol_s6, F_all[m]) for m in range(len(mode_names))]

    # 7. Render Plot (2 rows x 3 cols, sized for 50% A4 width ~4.6 in or slightly wider ~5.2 in for 2.5 gamma)
    fig_w = 5.2
    fig_h = 3.9
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.30, top=0.93, bottom=0.18, left=0.10, right=0.98)

    color_true = "k"
    color_gp = "gray"
    color_s15 = "#009E73"  # green (Seed 15 with Log Term)
    color_s6 = "#D55E00"   # vermilion/orange (Seed 6 with C01 polynomial)
    extrap_alpha = 0.08

    for i, name in enumerate(mode_names):
        row, col = divmod(i, 3)
        ax = fig.add_subplot(gs[row, col])
        ax.set_box_aspect(1)

        # Ground Truth
        ax.plot(gamma, psi_true[i], color=color_true, lw=1.2, ls="--", dashes=(3, 2), label="Ground Truth", zorder=6)

        # GP Baseline
        gp_mean = psi_gp_mean[i]
        gp_std = jnp.sqrt(jnp.maximum(psi_gp_var[i], 1e-12))
        ax.fill_between(gamma, gp_mean - 1.96 * gp_std, gp_mean + 1.96 * gp_std, color=color_gp, alpha=0.22, label="GP 95% CI", zorder=1)
        ax.plot(gamma, gp_mean, color=color_gp, lw=1.1, ls=":", label="GP Mean", zorder=3)

        # Distilled Seed 15 (Log term discovered)
        s15_mean = psi_dist_s15[i].mean(axis=0)
        s15_low = np.percentile(psi_dist_s15[i], 2.5, axis=0)
        s15_high = np.percentile(psi_dist_s15[i], 97.5, axis=0)
        ax.fill_between(gamma, s15_low, s15_high, color=color_s15, alpha=0.18, zorder=2)
        ax.plot(gamma, s15_mean, color=color_s15, lw=1.3, ls="-", label=r"Distilled (Log Term $E$)", zorder=5)

        # Distilled Seed 6 (Collinear polynomial C01)
        s6_mean = psi_dist_s6[i].mean(axis=0)
        s6_low = np.percentile(psi_dist_s6[i], 2.5, axis=0)
        s6_high = np.percentile(psi_dist_s6[i], 97.5, axis=0)
        ax.fill_between(gamma, s6_low, s6_high, color=color_s6, alpha=0.15, zorder=2)
        ax.plot(gamma, s6_mean, color=color_s6, lw=1.2, ls="-.", label=r"Distilled ($C_{01}$ Poly)", zorder=4)

        # Shading for Extrapolation
        crossings = mode_limits_dev[name] + mode_limits_vol[name]
        if crossings:
            min_c = min(crossings)
            ax.axvspan(min_c, max_gamma, color="#E69F00", alpha=extrap_alpha, zorder=0)
            ax.axvline(min_c, color="#E69F00", linestyle=":", linewidth=0.8, zorder=0)
        else:
            # Check if out
            ax.axvspan(0.0, max_gamma, color="#E69F00", alpha=extrap_alpha, zorder=0)

        # Mark original training limit gamma=1.0 with subtle dashed vertical line
        ax.axvline(1.0, color="#666666", linestyle="--", linewidth=0.7, alpha=0.6, zorder=1)

        ax.set_title(f"({i+1}) {mode_full_names[name]}", fontsize=9.2, fontweight="bold", pad=4)
        ax.set_xlim(0.0, max_gamma)
        if row == 1:
            ax.set_xlabel(r"$\gamma$", fontsize=9.5)
        else:
            ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel(r"$\Psi$", fontsize=9.5)

        ax.tick_params(axis="both", which="major", labelsize=8.0)

    # 1-row legend across the bottom
    legend_handles = [
        mlines.Line2D([], [], color=color_s15, linestyle="-", linewidth=1.3, label=r"Dist. Seed 15 (Log $E$)"),
        mlines.Line2D([], [], color=color_s6, linestyle="-.", linewidth=1.2, label=r"Dist. Seed 6 ($C_{01}$)"),
        mlines.Line2D([], [], color=color_true, linestyle="--", linewidth=1.1, dashes=(3, 2), label="Ground Truth"),
        mlines.Line2D([], [], color=color_gp, linestyle=":", linewidth=1.1, label="GP Mean"),
        mpatches.Patch(color=color_gp, alpha=0.22, label="GP 95% CI"),
        mpatches.Patch(color="#E69F00", alpha=extrap_alpha, label="Extrap. Region"),
    ]
    fig.legend(
        handles=legend_handles, loc="lower center", ncol=6,
        bbox_to_anchor=(0.54, 0.02), fontsize=6.2, frameon=False,
        handlelength=1.1, handletextpad=0.25, columnspacing=0.55
    )

    out_pdf = os.path.join(out_dir, "split_energy_gentthomas_extended_gamma25.pdf")
    out_png = os.path.join(out_dir, "split_energy_gentthomas_extended_gamma25.png")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ Generated extended gamma plot: {out_pdf} and {out_png}")

    # Copy to artifact directory
    artifact_dir = "/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03"
    try:
        import shutil
        shutil.copy2(out_pdf, os.path.join(artifact_dir, "split_energy_gentthomas_extended_gamma25.pdf"))
        shutil.copy2(out_png, os.path.join(artifact_dir, "split_energy_gentthomas_extended_gamma25.png"))
    except Exception:
        pass

if __name__ == "__main__":
    main()

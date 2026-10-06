"""
generate_domain_diagrams.py

Generates publication-quality schematic diagrams of both training (Block)
and validation/generalization (Holes) domains with comprehensive dimensioning,
finite-element mesh overlay, and rigorous mechanical boundary condition specifications.

Notation standards:
- \partial \Omega for all boundaries (\partial \Omega_{\mathrm{top}}, \partial \Omega_{\mathrm{bottom}}, etc.)
- \mu for the biaxial asymmetry loading factor (\mu \in [0.5, 0.9])
- "u_x free" / "u_y free" / "u_x, u_y free" for unconstrained displacement DOFs
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.tri as tri

# Ensure plots/ directory is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from plots.theme import apply_style
    apply_style("paper")
except ImportError:
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams["figure.dpi"] = 300
    plt.rcParams["savefig.dpi"] = 300


# ==============================================================================
# Helper Drawing Utilities for Engineering Schematics
# ==============================================================================

def draw_dim_h(ax, x0, x1, y, text, offset_text=0.045, text_color="#0f172a", arrow_color="#334155"):
    """Draws a horizontal dimension line with dual arrowheads and centered text."""
    ax.annotate(
        "", xy=(x0, y), xytext=(x1, y),
        arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.5, shrinkA=0, shrinkB=0),
        zorder=5
    )
    # Ticks at endpoints
    tick_h = 0.022
    ax.plot([x0, x0], [y - tick_h, y + tick_h], color=arrow_color, lw=1.3, zorder=5)
    ax.plot([x1, x1], [y - tick_h, y + tick_h], color=arrow_color, lw=1.3, zorder=5)
    # Text
    xm = (x0 + x1) / 2.0
    ax.text(
        xm, y + offset_text, text, ha="center", va="center", fontsize=14.0,
        color=text_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.96),
        zorder=6
    )


def draw_dim_v(ax, y0, y1, x, text, offset_text=0.050, text_color="#0f172a", arrow_color="#334155"):
    """Draws a vertical dimension line with dual arrowheads and centered text."""
    ax.annotate(
        "", xy=(x, y0), xytext=(x, y1),
        arrowprops=dict(arrowstyle="<->", color=arrow_color, lw=1.5, shrinkA=0, shrinkB=0),
        zorder=5
    )
    # Ticks at endpoints
    tick_w = 0.022
    ax.plot([x - tick_w, x + tick_w], [y0, y0], color=arrow_color, lw=1.3, zorder=5)
    ax.plot([x - tick_w, x + tick_w], [y1, y1], color=arrow_color, lw=1.3, zorder=5)
    # Text
    ym = (y0 + y1) / 2.0
    ax.text(
        x + offset_text, ym, text, ha="center", va="center", fontsize=14.0,
        rotation=90, color=text_color, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.96),
        zorder=6
    )


def draw_clamped_support_bottom(ax, y=0.0, x_min=0.0, x_max=1.0, n_hatches=32):
    """Draws fixed/clamped ground hatching along bottom boundary (ux = 0, uy = 0)."""
    ax.plot([x_min, x_max], [y, y], color="#0f172a", lw=2.4, zorder=5)
    xs = np.linspace(x_min, x_max, n_hatches)
    h_len = 0.038
    for xp in xs:
        ax.plot([xp, xp - h_len * 0.7], [y, y - h_len], color="#475569", lw=1.1, zorder=5)


def draw_traction_arrows_top(ax, y=1.0, x_min=0.0, x_max=1.0, n_arrows=9, arrow_len=0.09, color="#c2410c", label=None):
    """Draws tensile pull arrows along the top boundary."""
    xs = np.linspace(x_min + 0.05, x_max - 0.05, n_arrows)
    for i, xp in enumerate(xs):
        ax.annotate(
            "", xy=(xp, y + arrow_len), xytext=(xp, y),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.2,
                mutation_scale=17, shrinkA=0, shrinkB=0
            ),
            zorder=6
        )
    # Collective tie bar
    ax.plot([x_min, x_max], [y + arrow_len, y + arrow_len], color=color, lw=1.3, ls="--", alpha=0.7, zorder=5)
    if label:
        ax.text(
            (x_min + x_max) / 2.0, y + arrow_len + 0.040, label,
            ha="center", va="bottom", fontsize=15.0, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#fffaf0", edgecolor=color, lw=1.3, alpha=0.96),
            zorder=7
        )


def draw_traction_arrows_right(ax, x=1.0, y_min=0.0, y_max=1.0, n_arrows=9, arrow_len=0.09, color="#1d4ed8", label=None):
    """Draws tensile pull arrows along the right boundary."""
    ys = np.linspace(y_min + 0.05, y_max - 0.05, n_arrows)
    for i, yp in enumerate(ys):
        ax.annotate(
            "", xy=(x + arrow_len, yp), xytext=(x, yp),
            arrowprops=dict(
                arrowstyle="-|>", color=color, lw=2.2,
                mutation_scale=17, shrinkA=0, shrinkB=0
            ),
            zorder=6
        )
    ax.plot([x + arrow_len, x + arrow_len], [y_min, y_max], color=color, lw=1.3, ls="--", alpha=0.7, zorder=5)
    if label:
        ax.text(
            x + arrow_len + 0.042, (y_min + y_max) / 2.0, label,
            ha="left", va="center", fontsize=15.0, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#f0f7ff", edgecolor=color, lw=1.3, alpha=0.96),
            zorder=7
        )


# ==============================================================================
# Plot 1: Block Domain (Training Geometry)
# ==============================================================================

def plot_block_domain(ax, mesh_npz_path="mesh/block_mesh.npz", panel_label=r"$\mathbf{(a)}$"):
    """Renders the detailed Block geometry with dimensions, FE mesh, and BC specifications."""
    # 1. Load actual FE Mesh
    if os.path.exists(mesh_npz_path):
        data = np.load(mesh_npz_path)
        node_coords = data["node_coords"][:, :2]
        cells = data["cells"]
        triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)
        ax.triplot(triang, color="#64748b", lw=0.45, alpha=0.55, zorder=2)
        ax.tripcolor(triang, np.ones(len(node_coords)), cmap="Blues", alpha=0.05, zorder=1)
    else:
        poly_pts = [[0.1, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]]
        theta = np.linspace(np.pi / 2, 0, 30)
        arc = np.column_stack([0.1 * np.cos(theta), 0.1 * np.sin(theta)])
        poly_pts = np.vstack([arc, [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.1]]])
        ax.fill(poly_pts[:, 0], poly_pts[:, 1], color="#f8fafc", zorder=1)

    # 2. Draw Domain Outline
    # Corner circular cutout at (0, 0) with R = 0.10
    theta_arc = np.linspace(0, np.pi / 2.0, 100)
    arc_x = 0.10 * np.cos(theta_arc)
    arc_y = 0.10 * np.sin(theta_arc)
    ax.plot(arc_x, arc_y, color="#0f172a", lw=2.4, zorder=4)

    # Straight boundary edges
    ax.plot([0.10, 1.0], [0.0, 0.0], color="#0f172a", lw=2.4, zorder=4)   # Bottom
    ax.plot([1.0, 1.0], [0.0, 1.0], color="#0f172a", lw=2.4, zorder=4)   # Right
    ax.plot([0.0, 1.0], [1.0, 1.0], color="#0f172a", lw=2.4, zorder=4)   # Top
    ax.plot([0.0, 0.0], [0.10, 1.0], color="#0f172a", lw=2.4, zorder=4)  # Left

    # 3. Boundary Conditions
    # Top edge: Prescribed vertical displacement / pull
    draw_traction_arrows_top(
        ax, y=1.0, x_min=0.0, x_max=1.0, n_arrows=9, arrow_len=0.09,
        color="#c2410c",
        label=r"$\partial \Omega_{\mathrm{top}}: u_y = \bar{u}_{\mathrm{top}}$"
    )
    
    # Right edge: Prescribed horizontal displacement (asymmetric biaxial with \mu)
    draw_traction_arrows_right(
        ax, x=1.0, y_min=0.0, y_max=1.0, n_arrows=9, arrow_len=0.09,
        color="#1d4ed8",
        label=r"$\partial \Omega_{\mathrm{right}}: u_x = \mu\,\bar{u}_{\mathrm{top}}$"
    )

    # Radius Dimension for Notch: R = 0.10
    ax.annotate(
        "", xy=(0.10 * np.cos(np.radians(65)), 0.10 * np.sin(np.radians(65))), xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.8),
        zorder=6
    )
    ax.text(
        0.045, 0.095, r"$R = 0.10$", fontsize=15.0, color="#b91c1c", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=7
    )

    # 4. Mechanical BC Callouts (Positioned cleanly along symmetry boundaries)
    ax.text(
        -0.05, 0.55, r"$\partial \Omega_{\mathrm{left}}:\; u_x = 0$",
        fontsize=15.0, ha="right", va="center", color="#1e293b", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2, alpha=0.96),
        zorder=7
    )
    ax.text(
        0.55, -0.065, r"$\partial \Omega_{\mathrm{bottom}}:\; u_y = 0$",
        fontsize=15.0, ha="center", va="top", color="#1e293b", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2, alpha=0.96),
        zorder=7
    )

    # 5. Dimensions (Well clear of labels)
    # Overall horizontal width L_X = 1.00
    draw_dim_h(ax, x0=0.0, x1=1.00, y=-0.22, text=r"$L_X = 1.00$", offset_text=-0.055)
    ax.plot([0.0, 0.0], [0.0, -0.25], color="#94a3b8", lw=1.0, ls=":")
    ax.plot([1.0, 1.0], [0.0, -0.25], color="#94a3b8", lw=1.0, ls=":")

    # Overall vertical height L_Y = 1.00
    draw_dim_v(ax, y0=0.0, y1=1.00, x=-0.38, text=r"$L_Y = 1.00$", offset_text=-0.060)
    ax.plot([0.0, -0.41], [0.0, 0.0], color="#94a3b8", lw=1.0, ls=":")
    ax.plot([0.0, -0.41], [1.0, 1.0], color="#94a3b8", lw=1.0, ls=":")

    # 6. Coordinate Origin & Extended Axes
    # X-axis along Y=0 (extending past domain envelope)
    ax.annotate(
        "", xy=(1.22, 0.0), xytext=(-0.10, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0),
        zorder=8
    )
    ax.text(1.27, 0.0, r"$X$", fontsize=18.0, fontweight="bold", ha="left", va="center", color="#0f172a", zorder=9)

    # Y-axis along X=0 (extending past domain envelope)
    ax.annotate(
        "", xy=(0.0, 1.22), xytext=(0.0, -0.10),
        arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0),
        zorder=8
    )
    ax.text(0.0, 1.27, r"$Y$", fontsize=18.0, fontweight="bold", ha="center", va="bottom", color="#0f172a", zorder=9)

    ax.plot(0.0, 0.0, "ko", ms=5.0, zorder=9)
    ax.text(-0.025, -0.025, r"$(0,0)$", fontsize=13.0, fontweight="semibold", ha="right", va="top", color="#475569", zorder=9)

    # Panel tag (replaces large title)
    if panel_label:
        ax.text(-0.48, 1.26, panel_label, fontsize=19.0, fontweight="bold", color="#0f172a", zorder=10)

    ax.set_xlim(-0.54, 1.55)
    ax.set_ylim(-0.35, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")


# ==============================================================================
# Plot 2: Holes Domain (Validation / Generalization Geometry)
# ==============================================================================

def plot_holes_domain(ax, mesh_npz_path="mesh/holes_mesh.npz", panel_label=r"$\mathbf{(b)}$"):
    """Renders the detailed Holes geometry with dimensions, FE mesh, and BC specifications."""
    # 1. Load actual FE Mesh
    if os.path.exists(mesh_npz_path):
        data = np.load(mesh_npz_path)
        node_coords = data["node_coords"][:, :2]
        cells = data["cells"]
        triang = tri.Triangulation(node_coords[:, 0], node_coords[:, 1], cells)
        ax.triplot(triang, color="#64748b", lw=0.45, alpha=0.55, zorder=2)
        ax.tripcolor(triang, np.ones(len(node_coords)), cmap="Blues", alpha=0.05, zorder=1)
    else:
        rect = patches.Rectangle((0, 0), 1, 1, facecolor="#f8fafc", zorder=1)
        ax.add_patch(rect)

    # 2. Draw Domain Outer Boundaries
    ax.plot([0.0, 1.0], [0.0, 0.0], color="#0f172a", lw=2.4, zorder=4)  # Bottom (Clamped)
    ax.plot([1.0, 1.0], [0.0, 1.0], color="#0f172a", lw=2.4, zorder=4)  # Right
    ax.plot([0.0, 1.0], [1.0, 1.0], color="#0f172a", lw=2.4, zorder=4)  # Top (Tension)
    ax.plot([0.0, 0.0], [0.0, 1.0], color="#0f172a", lw=2.4, zorder=4)  # Left

    # 3. Draw Elliptical Cutouts
    # Hole 1 (Upper Right): Center (0.70, 0.70), a1 = 0.20, b1 = 0.18
    h1_cx, h1_cy = 0.70, 0.70
    h1_a, h1_b = 0.20, 0.18
    ell1 = patches.Ellipse(
        (h1_cx, h1_cy), 2 * h1_a, 2 * h1_b, angle=0.0,
        facecolor="white", edgecolor="#0f172a", lw=2.2, zorder=3
    )
    ax.add_patch(ell1)

    # Hole 2 (Lower Left): Center (0.25, 0.30), a2 = 0.04, b2 = 0.18
    h2_cx, h2_cy = 0.25, 0.30
    h2_a, h2_b = 0.04, 0.18
    ell2 = patches.Ellipse(
        (h2_cx, h2_cy), 2 * h2_a, 2 * h2_b, angle=0.0,
        facecolor="white", edgecolor="#0f172a", lw=2.2, zorder=3
    )
    ax.add_patch(ell2)

    # Center crosshairs for holes
    ch_len = 0.025
    ax.plot([h1_cx - ch_len, h1_cx + ch_len], [h1_cy, h1_cy], color="#64748b", lw=1.0, zorder=4)
    ax.plot([h1_cx, h1_cx], [h1_cy - ch_len, h1_cy + ch_len], color="#64748b", lw=1.0, zorder=4)
    ax.plot([h2_cx - ch_len, h2_cx + ch_len], [h2_cy, h2_cy], color="#64748b", lw=1.0, zorder=4)
    ax.plot([h2_cx, h2_cx], [h2_cy - ch_len, h2_cy + ch_len], color="#64748b", lw=1.0, zorder=4)

    # 4. Boundary Conditions
    # Bottom: Clamped Support (ux = 0, uy = 0)
    draw_clamped_support_bottom(ax, y=0.0, x_min=0.0, x_max=1.0, n_hatches=32)

    # Top: Prescribed Uniaxial Tension with clamped transverse displacement (uy = u_top, ux = 0)
    draw_traction_arrows_top(
        ax, y=1.0, x_min=0.0, x_max=1.0, n_arrows=9, arrow_len=0.085,
        color="#c2410c",
        label=r"$\partial \Omega_{\mathrm{top}}: u_y = \bar{u}_{\mathrm{top}}, \quad u_x = 0$"
    )

    # Bottom Clamped label (placed right below the hatching)
    ax.text(
        0.50, -0.065, r"$\partial \Omega_{\mathrm{bottom}}:\; u_x = 0,\; u_y = 0$",
        fontsize=15.0, ha="center", va="top", color="#0f172a", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#f1f5f9", edgecolor="#94a3b8", lw=1.2, alpha=0.96),
        zorder=7
    )

    # Center projection lines to domain edges
    ax.plot([0.0, h1_cx], [h1_cy, h1_cy], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([h1_cx, h1_cx], [0.0, h1_cy], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([0.0, h2_cx], [h2_cy, h2_cy], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([h2_cx, h2_cx], [0.0, h2_cy], color="#94a3b8", lw=0.9, ls=":")

    # --- Hole 1 (Upper Right) ---
    # Horizontal semi-axis a_1 inside Hole 1 (label below arrow)
    ax.annotate(
        "", xy=(h1_cx + h1_a, h1_cy), xytext=(h1_cx, h1_cy),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=14),
        zorder=5
    )
    ax.text(
        h1_cx + h1_a * 0.5, h1_cy - 0.024, r"$a_1 = 0.20$",
        fontsize=13.0, color="#b91c1c", fontweight="bold", ha="center", va="top",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=6
    )

    # Vertical semi-axis b_1 inside Hole 1 (label to the left of arrow)
    ax.annotate(
        "", xy=(h1_cx, h1_cy + h1_b), xytext=(h1_cx, h1_cy),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=14),
        zorder=5
    )
    ax.text(
        h1_cx - 0.022, h1_cy + h1_b * 0.5, r"$b_1 = 0.18$",
        fontsize=13.0, color="#b91c1c", fontweight="bold", ha="right", va="center",
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=6
    )

    # Name tag inside Hole 1 (lower-left quadrant of the hole)
    ax.text(
        h1_cx - 0.10, h1_cy - 0.09, r"$\mathbf{Hole\;1}$",
        fontsize=14.5, color="#0f172a", fontweight="bold", ha="center", va="center",
        zorder=6
    )

    # --- Hole 2 (Lower Left) ---
    # Vertical semi-axis b_2 arrow inside Hole 2
    ax.annotate(
        "", xy=(h2_cx, h2_cy + h2_b), xytext=(h2_cx, h2_cy),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=12),
        zorder=5
    )
    # Horizontal semi-axis a_2 arrow inside Hole 2
    ax.annotate(
        "", xy=(h2_cx + h2_a, h2_cy), xytext=(h2_cx, h2_cy),
        arrowprops=dict(arrowstyle="->", color="#b91c1c", lw=1.8, shrinkA=0, shrinkB=0, mutation_scale=10),
        zorder=5
    )

    # Semi-axes dimension callout for Hole 2 (pointing to upper-left perimeter)
    h2_rim_x = h2_cx - 0.04 * 0.866  # cos(150 deg)
    h2_rim_y = h2_cy + 0.18 * 0.500  # sin(150 deg)
    ax.annotate(
        r"$\mathbf{Hole\;2}$" + "\n" + r"$a_2 = 0.04$" + "\n" + r"$b_2 = 0.18$",
        xy=(h2_rim_x, h2_rim_y),
        xytext=(0.04, 0.62),
        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.6, connectionstyle="arc3,rad=-0.10"),
        fontsize=13.0, ha="left", va="center", fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="#fffdf0", edgecolor="#cbd5e1", lw=1.2, alpha=0.96),
        zorder=7
    )

    # 6. Center coordinate dimensions on axes (Clean staggered hierarchy)
    # Horizontal dimensions
    draw_dim_h(ax, x0=0.0, x1=h2_cx, y=-0.16, text=r"$X_2 = 0.25$", offset_text=-0.050, text_color="#475569")
    ax.plot([0.0, 0.0], [0.0, -0.19], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([h2_cx, h2_cx], [0.0, -0.19], color="#94a3b8", lw=0.9, ls=":")

    draw_dim_h(ax, x0=0.0, x1=h1_cx, y=-0.28, text=r"$X_1 = 0.70$", offset_text=-0.050, text_color="#475569")
    ax.plot([0.0, 0.0], [0.0, -0.31], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([h1_cx, h1_cx], [0.0, -0.31], color="#94a3b8", lw=0.9, ls=":")

    draw_dim_h(ax, x0=0.0, x1=1.00, y=-0.40, text=r"$L_X = 1.00$", offset_text=-0.055)
    ax.plot([0.0, 0.0], [0.0, -0.43], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([1.0, 1.0], [0.0, -0.43], color="#94a3b8", lw=0.9, ls=":")

    # Vertical dimensions (Shifted left to avoid any overlap)
    draw_dim_v(ax, y0=0.0, y1=h2_cy, x=-0.14, text=r"$Y_2 = 0.30$", offset_text=-0.050, text_color="#475569")
    ax.plot([0.0, -0.17], [0.0, 0.0], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([0.0, -0.17], [h2_cy, h2_cy], color="#94a3b8", lw=0.9, ls=":")

    draw_dim_v(ax, y0=0.0, y1=h1_cy, x=-0.26, text=r"$Y_1 = 0.70$", offset_text=-0.050, text_color="#475569")
    ax.plot([0.0, -0.29], [0.0, 0.0], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([0.0, -0.29], [h1_cy, h1_cy], color="#94a3b8", lw=0.9, ls=":")

    draw_dim_v(ax, y0=0.0, y1=1.00, x=-0.38, text=r"$L_Y = 1.00$", offset_text=-0.055)
    ax.plot([0.0, -0.41], [0.0, 0.0], color="#94a3b8", lw=0.9, ls=":")
    ax.plot([0.0, -0.41], [1.0, 1.0], color="#94a3b8", lw=0.9, ls=":")

    # Coordinate Origin & Extended Axes
    # X-axis along Y=0 (extending past domain envelope)
    ax.annotate(
        "", xy=(1.22, 0.0), xytext=(-0.10, 0.0),
        arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0),
        zorder=8
    )
    ax.text(1.27, 0.0, r"$X$", fontsize=18.0, fontweight="bold", ha="left", va="center", color="#0f172a", zorder=9)

    # Y-axis along X=0 (extending past domain envelope)
    ax.annotate(
        "", xy=(0.0, 1.22), xytext=(0.0, -0.10),
        arrowprops=dict(arrowstyle="-|>", color="#0f172a", lw=2.0, mutation_scale=15, shrinkA=0, shrinkB=0),
        zorder=8
    )
    ax.text(0.0, 1.27, r"$Y$", fontsize=18.0, fontweight="bold", ha="center", va="bottom", color="#0f172a", zorder=9)

    ax.plot(0.0, 0.0, "ko", ms=5.0, zorder=9)
    ax.text(-0.025, -0.025, r"$(0,0)$", fontsize=13.0, fontweight="semibold", ha="right", va="top", color="#475569", zorder=9)

    # Panel tag (replaces large title)
    if panel_label:
        ax.text(-0.48, 1.26, panel_label, fontsize=19.0, fontweight="bold", color="#0f172a", zorder=10)

    ax.set_xlim(-0.56, 1.45)
    ax.set_ylim(-0.55, 1.34)
    ax.set_aspect("equal")
    ax.axis("off")


# ==============================================================================
# Main Generation Function
# ==============================================================================

def generate_all_diagrams(output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    mesh_dir = "mesh"
    block_npz = os.path.join(mesh_dir, "block_mesh.npz")
    holes_npz = os.path.join(mesh_dir, "holes_mesh.npz")

    # 1. Combined Two-Panel Paper Figure
    print("Generating combined paper figure (side-by-side)...")
    fig, axes = plt.subplots(1, 2, figsize=(16.5, 8.0), dpi=300)
    plot_block_domain(axes[0], block_npz, panel_label=r"$\mathbf{(a)}$")
    plot_holes_domain(axes[1], holes_npz, panel_label=r"$\mathbf{(b)}$")
    axes[0].set_ylim(-0.55, 1.34)  # Align baseline with panel (b) for seamless side-by-side comparison
    
    plt.tight_layout(pad=1.5)
    combined_pdf = os.path.join(output_dir, "domain_diagrams_paper.pdf")
    combined_png = os.path.join(output_dir, "domain_diagrams_paper.png")
    fig.savefig(combined_pdf, bbox_inches="tight")
    fig.savefig(combined_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {combined_pdf}")
    print(f"Saved: {combined_png}")

    # 2. Standalone Block Domain Figure
    print("Generating standalone Block diagram...")
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.2), dpi=300)
    plot_block_domain(ax, block_npz, panel_label=None)
    plt.tight_layout(pad=0.8)
    block_pdf = os.path.join(output_dir, "domain_diagram_block.pdf")
    block_png = os.path.join(output_dir, "domain_diagram_block.png")
    fig.savefig(block_pdf, bbox_inches="tight")
    fig.savefig(block_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {block_pdf}")

    # 3. Standalone Holes Domain Figure
    print("Generating standalone Holes diagram...")
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 7.5), dpi=300)
    plot_holes_domain(ax, holes_npz, panel_label=None)
    plt.tight_layout(pad=0.8)
    holes_pdf = os.path.join(output_dir, "domain_diagram_holes.pdf")
    holes_png = os.path.join(output_dir, "domain_diagram_holes.png")
    fig.savefig(holes_pdf, bbox_inches="tight")
    fig.savefig(holes_png, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {holes_pdf}")

    # 4. Copy to mesh/ and artifacts directory
    artifact_dir = "/root/.gemini/antigravity/brain/da90ac59-033b-4891-87d4-47e237c15c03"
    import shutil
    for fname in ["domain_diagrams_paper.png", "domain_diagram_block.png", "domain_diagram_holes.png",
                  "domain_diagrams_paper.pdf", "domain_diagram_block.pdf", "domain_diagram_holes.pdf"]:
        src = os.path.join(output_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join("mesh", fname))
            if os.path.exists(artifact_dir):
                shutil.copy(src, os.path.join(artifact_dir, fname))
    print("All diagrams successfully generated and distributed!")


if __name__ == "__main__":
    generate_all_diagrams()

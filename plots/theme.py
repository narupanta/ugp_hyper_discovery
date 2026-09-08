"""
theme.py: Unified Styling and Theme Configuration for Hyperelasticity Discovery Plots

Provides standard fonts, figure aesthetics, standardized color palettes for 
deformation modes and material components, and uniform save utilities.
"""

import os
import logging
from typing import Optional, Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib as mpl

# Silence benign fontTools warnings regarding font metadata timestamp epoch
logging.getLogger("fontTools").setLevel(logging.ERROR)

# ==============================================================================
# 1. Standard Color Palettes & Labels
# ==============================================================================

# Standard 6 Deformation Modes
MODE_NAMES: List[str] = [
    "Uniaxial Tension",
    "Equibiaxial Tension",
    "Pure Shear",
    "Uniaxial Compression",
    "Equibiaxial Compression",
    "Simple Shear"
]

MODE_SHORT_NAMES: List[str] = ["UT", "EBT", "PS", "UC", "EBC", "SS"]

# Mode Colors: consistent across all 6-mode plots
MODE_COLORS: Dict[Any, str] = {
    0: "#1f77b4",      # UT: Blue
    1: "#ff7f0e",      # EBT: Orange
    2: "#2ca02c",      # PS: Green
    3: "#d62728",      # UC: Red
    4: "#9467bd",      # EBC: Purple
    5: "#8c564b",      # SS: Brown
    "UT": "#1f77b4",
    "EBT": "#ff7f0e",
    "PS": "#2ca02c",
    "UC": "#d62728",
    "EBC": "#9467bd",
    "SS": "#8c564b",
    "Uniaxial Tension": "#1f77b4",
    "Equibiaxial Tension": "#ff7f0e",
    "Pure Shear": "#2ca02c",
    "Uniaxial Compression": "#d62728",
    "Equibiaxial Compression": "#9467bd",
    "Simple Shear": "#8c564b",
}

MODE_LINESTYLES: Dict[str, Any] = {
    "Uniaxial Tension": "-",
    "Equibiaxial Tension": "-.",
    "Pure Shear": (0, (3, 1, 1, 1)),
    "Uniaxial Compression": "--",
    "Equibiaxial Compression": ":",
    "Simple Shear": (0, (5, 2)),
    "UT": "-",
    "EBT": "-.",
    "PS": (0, (3, 1, 1, 1)),
    "UC": "--",
    "EBC": ":",
    "SS": (0, (5, 2)),
}

# Standard Component Colors
COMPONENT_COLORS: Dict[str, str] = {
    "dev": "#1f77b4",    # Deviatoric: Steel Blue
    "vol": "#2ca02c",    # Volumetric: Forest Green
    "aniso": "#e377c2",  # Anisotropic: Orchid / Pink
    "total": "#333333",  # Total: Charcoal
}

# Standard Curve Styles (Ground Truth, GP Posterior, Distilled)
CURVE_STYLES: Dict[str, Dict[str, Any]] = {
    "gt": {
        "color": "black",
        "linestyle": "-",
        "linewidth": 2.2,
        "label": "Ground Truth",
        "zorder": 4
    },
    "gp": {
        "color": "#1f77b4",
        "linestyle": "--",
        "linewidth": 2.0,
        "label": "GP Posterior",
        "zorder": 3
    },
    "gp_ci": {
        "color": "#1f77b4",
        "alpha": 0.22,
        "label": r"GP $\pm 2\sigma$",
        "zorder": 2
    },
    "distilled": {
        "color": "#d95f02",
        "linestyle": "-.",
        "linewidth": 2.0,
        "label": "Distilled Model",
        "zorder": 3
    },
    "distilled_ci": {
        "color": "#d95f02",
        "alpha": 0.22,
        "label": r"Distilled $\pm 2\sigma$",
        "zorder": 2
    }
}

MODE_LABELS_P: List[str] = [
    r"$P_{11}$",
    r"$P_{11}$",
    r"$P_{22}$",
    r"$P_{11}$",
    r"$P_{11}$",
    r"$P_{12}$"
]


# ==============================================================================
# 2. Theme & Matplotlib rcParams Configuration
# ==============================================================================

def apply_style(mode: str = "paper") -> None:
    """
    Applies unified matplotlib styling across the repository.
    Ensures publication-ready serif typography, Computer Modern math notation,
    consistent font sizes, and high-resolution rendering without requiring system LaTeX.
    """
    params = {
        # Typography & Math Rendering
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "text.usetex": False,

        # Font Sizes
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 15,

        # Figure & Export Resolution
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,

        # Grid & Axes Aesthetics
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "grid.color": "#b0b0b0",
        "axes.axisbelow": True,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,

        # Legend Aesthetics
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": "#cccccc",
        "legend.fancybox": True,

        # Lines
        "lines.linewidth": 1.8,
        "lines.markersize": 5,
    }

    if mode == "presentation":
        params.update({
            "font.size": 14,
            "axes.titlesize": 16,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 13,
            "lines.linewidth": 2.5,
        })

    plt.rcParams.update(params)


# ==============================================================================
# 3. Figure & Axes Helper Utilities
# ==============================================================================

def save_figure(
    fig: plt.Figure,
    path: str,
    dpi: int = 300,
    close: bool = True,
    make_png: bool = False
) -> str:
    """
    Saves figure to specified path with tight bounding box, creating parent directories
    if they don't exist. Optionally produces a matching .png file alongside .pdf.
    """
    parent_dir = os.path.dirname(os.path.abspath(path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    
    if make_png and path.endswith(".pdf"):
        png_path = path[:-4] + ".png"
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")

    if close:
        plt.close(fig)

    return path


def setup_axes(
    ax: plt.Axes,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    grid: bool = True
) -> None:
    """Convenience helper to set standard title, labels, and grid on an axes."""
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if grid:
        ax.grid(True, alpha=0.25, linestyle="--")

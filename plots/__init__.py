"""
plots: Centralized plotting and visualization framework for hyperelasticity discovery.
"""
import core.figure_policy  # noqa: F401  (PDF-only figure output, see core/figure_policy.py)


from .theme import (
    apply_style,
    save_figure,
    setup_axes,
    MODE_COLORS,
    MODE_LINESTYLES,
    MODE_NAMES,
    MODE_LABELS_P,
    COMPONENT_COLORS,
    CURVE_STYLES,
)

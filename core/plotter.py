"""
core/plotter.py: Backward Compatibility Bridge

All active plotting routines are centralized and maintained in `plots.training`.
This module re-exports them to preserve full backwards compatibility.
"""

from plots.training import (
    plot_loss_analysis,
    plot_vfm_loss_analysis,
    plot_parameters_hist,
    plot_inducing_points,
    plot_combined_validation,
    plot_stress_validation,
    plot_energy_decomposition_validation,
    plot_training_r2,
    ExtractionR2Metrics,
    _compute_regime_transitions,
    _format_step_indices,
)

__all__ = [
    "plot_loss_analysis",
    "plot_vfm_loss_analysis",
    "plot_parameters_hist",
    "plot_inducing_points",
    "plot_combined_validation",
    "plot_stress_validation",
    "plot_energy_decomposition_validation",
    "plot_training_r2",
    "ExtractionR2Metrics",
    "_compute_regime_transitions",
    "_format_step_indices",
]
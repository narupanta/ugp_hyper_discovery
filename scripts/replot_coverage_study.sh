#!/bin/bash
# Post-distillation plots for the distillations of a coverage study, as in the normal pipeline (scripts/run_pipeline.sh):
# distilled-vs-GP validation, split summary, effective moduli, energy R^2 and sensitivity plots (no block validation, no FEM).
# Also called by validation/distillation_coverage_study.py after each seed/init pair. Runs every <study>/seed<s>/init<m> that has flow samples.
#
# Usage: bash scripts/replot_coverage_study.sh results/<experiment>/coverage_study_<states> [--seeds "1 2"] [--inits "0 1"]
set -e
cd "$(dirname "$0")/.."
STUDY="${1%/}"; shift || true
EXP=$(dirname "$STUDY")
SEEDS=""; INITS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --inits) INITS="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -d "$STUDY" ] && [ -f "$EXP/config.yaml" ] || { echo "Not a coverage study folder: $STUDY"; exit 1; }
CFG="$EXP/config.yaml"
cfg() { python3 -c "import yaml; d=yaml.safe_load(open('$CFG')); v=d.get('$1', '$2'); print(' '.join(map(str, v)) if isinstance(v, list) else v)"; }
DIST_MODEL=$(cfg distilled_material_model gmr)
DIST_TARGET=$(cfg distill_target sef_split)
FEM_VAL_STEPS=$(cfg val_load_steps_indices "")
export XLA_PYTHON_CLIENT_PREALLOCATE=false

for RUN in "$STUDY"/seed*/init*; do
    S=$(basename "$(dirname "$RUN")"); S=${S#seed}; M=$(basename "$RUN"); M=${M#init}
    [ -n "$SEEDS" ] && [[ " $SEEDS " != *" $S "* ]] && continue
    [ -n "$INITS" ] && [[ " $INITS " != *" $M "* ]] && continue
    [ -f "$RUN/dev_flow_samples.npy" ] || { echo "skip $RUN (no flow samples yet)"; continue; }
    EXTRACT_DIR="$EXP/$S/extracted"
    echo "=== seed $S init $M: $RUN"
    if [ ! -f "$RUN/vol_flow_samples.npy" ]; then   # dev-only distillation: zero vol samples keep the [dev, vol] layout
        python3 -c "import numpy as np; d=np.load('$RUN/dev_flow_samples.npy'); np.save('$RUN/vol_flow_samples.npy', np.zeros((d.shape[0], 3)))"
    fi
    COMMON=(--distilled_dir "$RUN" --saved_model_dir "$EXTRACT_DIR" --material_model "$DIST_MODEL" --distill_target "$DIST_TARGET")
    python3 plots/plot_distilled_validation.py "${COMMON[@]}" || echo "  plot_distilled_validation failed"
    python3 plots/plot_split_summary.py "${COMMON[@]}" || echo "  plot_split_summary failed"
    python3 plots/plot_effective_moduli.py --distilled_dir "$RUN" --saved_model_dir "$EXTRACT_DIR" --material_model "$DIST_MODEL" || echo "  plot_effective_moduli failed"
    python3 plots/plot_distilled_r2_energy.py "${COMMON[@]}" --val_load_steps $FEM_VAL_STEPS || echo "  plot_distilled_r2_energy failed"
    for COMP in dev vol aniso; do
        if [ -d "$RUN/output/${COMP}_sensitivities" ] || [ -d "$RUN/${COMP}_sensitivities" ]; then
            python3 plots/plot_invariant_sensitivity.py --distilled_dir "$RUN" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
            python3 plots/plot_invariant_sensitivity_3d_pairs.py --distilled_dir "$RUN" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
            python3 plots/plot_deformation_sensitivity.py --distilled_dir "$RUN" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
        fi
    done
    python3 plots/plot_combined_invariant_sensitivity.py --distilled_dir "$RUN" 2>/dev/null || true
    python3 plots/plot_all_invariant_sensitivity.py --distilled_dir "$RUN" 2>/dev/null || true
done
echo "done"

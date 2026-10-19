#!/bin/bash
# Regenerates the post-extraction plots and metrics of an HSGP experiment (no fitting, distillation or FEM solves):
#   distillation plots (split summary, distilled validation, distilled energy metrics), the pre-FEM free-node check,
#   the 2x2 FEM validation figures (standard, emulated experiment, holes) and the cross-seed summary table.
#
# Usage: scripts/regen_hsgp_plots.sh <experiment dir> [--seeds "1 2 3"] [--use_fem_bundle]
#   --use_fem_bundle: take the training kinematics from <seed>/fem_validation/block/fem_distilled_samples.npz instead
#                     of the clean dataset (exact when the clean dataset is not available on this machine).
set -u
export XLA_PYTHON_CLIENT_PREALLOCATE=false

EXP=""; SEEDS=""; BUNDLE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --seeds) SEEDS="$2"; shift 2 ;;
        --use_fem_bundle) BUNDLE="--use_fem_bundle"; shift ;;
        *) EXP="$1"; shift ;;
    esac
done
[ -d "$EXP" ] || { echo "Usage: $0 <experiment dir> [--seeds '1 2'] [--use_fem_bundle]"; exit 1; }
EXP="$(cd "$EXP" && pwd)"
[ -n "$SEEDS" ] || SEEDS=$(ls "$EXP" | grep -E '^[0-9]+$' | sort -n | tr '\n' ' ')
cfg() { python3 -c "import yaml; d=yaml.safe_load(open('$EXP/config.yaml')); v=d.get('$1', '$2'); print(*v) if isinstance(v, (list, tuple)) else print(v)"; }
DIST_MODEL=$(cfg distilled_material_model gmr)
DIST_TARGET=$(cfg distill_target sef_split)
VAL_STEPS=$(cfg val_load_steps_indices "")
GEOM_TRAIN=$(cfg geometry_train block)

run() {   # run a step, report failures without stopping the loop
    local name="$1"; shift
    if "$@" > "$LOG_DIR/${name}.log" 2>&1; then echo -n "$name ok | "; else echo -n "$name FAILED (see $LOG_DIR/${name}.log) | "; fi
}

for S in $SEEDS; do
    SD="$EXP/$S"
    [ -f "$SD/distilled/dev_flow_samples.npy" ] || { echo "seed $S: no distillation, skipped"; continue; }
    LOG_DIR="$SD/regen_logs"; mkdir -p "$LOG_DIR"
    echo -n "seed $S: "
    COMMON=(--distilled_dir "$SD/distilled" --saved_model_dir "$SD/extracted" --material_model "$DIST_MODEL" --distill_target "$DIST_TARGET")
    run split_summary python3 plots/plot_split_summary.py "${COMMON[@]}"
    run distilled_validation python3 plots/plot_distilled_validation.py "${COMMON[@]}"
    if [ -n "$BUNDLE" ] && [ ! -f "$SD/fem_validation/block/fem_distilled_samples.npz" ]; then
        echo -n "energy/free-node skipped (no FEM bundle) | "
    else
        run energy_metrics python3 plots/plot_distilled_r2_energy.py "${COMMON[@]}" --val_load_steps $VAL_STEPS $BUNDLE
        run free_node python3 plots/plot_free_node_residual_eiv.py --seed_dir "$SD" $BUNDLE
    fi
    if [ -d "$SD/fem_validation" ]; then
        run fem_2x2 python3 plots/plot_fem_validation_2x2.py --seed_dir "$SD"
        run fem_2x2_exp python3 plots/plot_fem_validation_2x2.py --seed_dir "$SD" --geometries "$GEOM_TRAIN" --experimental
        mkdir -p "$EXP/plots"
        for G in block holes block_exp; do
            f="$SD/fem_validation/${G%_exp}/fem_validation_2x2_$G.pdf"
            [ -f "$f" ] && cp "$f" "$EXP/plots/fem_validation_2x2_${G}_seed${S}.pdf"
        done
    fi
    echo
done

python3 plots/generate_experiment_summary.py --experiment_dir "$EXP" > /dev/null 2>&1 \
    && echo "summary: $EXP/summary_across_seeds.md" || echo "summary FAILED"

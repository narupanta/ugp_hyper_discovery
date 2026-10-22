#!/bin/bash
# ==============================================================================
# One entry point for the three studies. Choose the setup with MODE (below or --mode), adjust its settings block,
# and run it here or on the cluster (scripts/run_study_slurm.sh passes everything through to this script).
#
#   MODE=threshold   noise-threshold sweep: grid of displacement x load noise, HSGP extraction per grid point
#                    (validation/noise_threshold_sweep.py)
#   MODE=coverage    coverage of the distilled material parameters over noise seeds x normalizing-flow
#                    initialisations (validation/distillation_coverage_study.py); runs the extractions first if needed
#   MODE=normal      the standard pipeline for one recipe: HSGP extraction [+ distillation, FEM validation, plots]
#                    (scripts/run_hsgp_pipeline.sh)
#
# Usage:
#   bash scripts/run_study.sh                                   # uses MODE and the settings below
#   bash scripts/run_study.sh --mode normal --recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --seeds "1 2"
#   bash scripts/run_study.sh --mode threshold --grid_index 3   # one grid point (Slurm arrays set it automatically)
#   bash scripts/run_study.sh --mode threshold --collect_only   # summary of a sweep
#   bash scripts/run_study.sh --mode coverage --experiment_dir results/<exp> --evaluate_only
# Any setting below can be given as --<name in lower case> <value> (e.g. --noise_levels "1e-5 1e-4").
# Results: all three write to results/ (threshold: SWEEP_DIR; coverage: <experiment>/coverage_study; normal: a new
# results/<timestamp>_hsgp_... folder, or the given experiment directory).
# ==============================================================================
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

MODE="normal"                     # threshold | coverage | normal

# --- 1. threshold: noise-threshold sweep ---------------------------------------
THR_RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"
NOISE_LEVELS="1e-5 4e-5 7e-5 1e-4"                # displacement noise std (relative to the specimen size)
LOAD_NOISE_LEVELS="0.02 0.05 0.1"                 # load noise (relative); reactions in the likelihood match it
THR_SEEDS="1 2 3"
FORCE_NOISE_RULE="cv"                             # cv | evidence | scaled | fixed | derived | from_reaction
CV_TOL="2"                                        # cv: smallest s within this many nats of the best ("" = best)
R2_MIN="0.99"                                     # a grid point "works" if mean val energy R^2 >= R2_MIN ...
EC_MIN="90"                                       # ... and mean val energy coverage >= EC_MIN (%)
SWEEP_DIR="results/noise_sweep_isihara_cv"
THR_HSGP_ARGS=""                                  # extra '--hsgp_<option> <value>' for every grid point

# --- 2. coverage: distilled-parameter coverage ----------------------------------
COV_RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"
COV_EXPERIMENT_DIR=""                             # existing experiment; empty: create one from COV_RECIPE
COV_N_NOISE_SEEDS="10"
COV_N_INITS="5"
COV_N_ITERATIONS=""                               # Wasserstein iterations ("" = recipe value)
COV_CHECKPOINTS=""                                # e.g. "2000 4000 6000 8000 10000" (pilot)
COV_JOBS="2"                                      # distillations in parallel
COV_LEVEL="0.95"

# --- 3. normal: standard pipeline ---------------------------------------------------
RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"   # or an existing experiment directory
SEEDS="1"
DOWNSTREAM="1"                                    # 1: also distillation, FEM validation and validation plots
NORMAL_HSGP_ARGS=""

# --- command-line overrides -----------------------------------------------------
PASS=()                                           # flags passed on unchanged (e.g. --collect_only, --evaluate_only)
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --collect_only|--evaluate_only|--list_grid|--extract|--force|--no_sensitivity) PASS+=("$1"); shift ;;
        --grid_index) PASS+=("$1" "$2"); shift 2 ;;
        --*) VAR=$(echo "${1#--}" | tr '[:lower:]' '[:upper:]'); printf -v "$VAR" '%s' "$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
cd "$(dirname "$0")/.."
echo "=== study: $MODE ($(date))"

case "$MODE" in
threshold)
    ARGS=(--base_recipe "$THR_RECIPE" --force_noise_rule "$FORCE_NOISE_RULE" --noise_levels $NOISE_LEVELS
          --load_noise_levels $LOAD_NOISE_LEVELS --seeds $THR_SEEDS --r2_min "$R2_MIN" --ec_min "$EC_MIN"
          --sweep_dir "$SWEEP_DIR")
    [ -n "$CV_TOL" ] && ARGS+=(--cv_tol "$CV_TOL")
    [ -n "$THR_HSGP_ARGS" ] && ARGS+=(--hsgp_args "$THR_HSGP_ARGS")
    # inside a Slurm array: one grid point per task (unless only the summary is requested)
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ " ${PASS[*]} " != *" --collect_only "* ]] \
            && [[ " ${PASS[*]} " != *" --grid_index "* ]]; then
        PASS+=(--grid_index "$SLURM_ARRAY_TASK_ID")
    fi
    python3 validation/noise_threshold_sweep.py "${ARGS[@]}" "${PASS[@]}"
    ;;
coverage)
    if [ -z "$COV_EXPERIMENT_DIR" ]; then
        SEEDS_LIST=$(seq -s ' ' 1 "$COV_N_NOISE_SEEDS")
        echo "No experiment directory given: running the extractions of $COV_RECIPE for noise seeds $SEEDS_LIST"
        LOG=$(mktemp)
        bash scripts/run_hsgp_pipeline.sh "$COV_RECIPE" --seeds "$SEEDS_LIST" 2>&1 | tee "$LOG"
        COV_EXPERIMENT_DIR=$(grep -oE "Created experiment directory: [^ ]+" "$LOG" | head -1 | awk '{print $4}')
        rm -f "$LOG"
        [ -n "$COV_EXPERIMENT_DIR" ] || { echo "Could not determine the experiment directory"; exit 1; }
    fi
    ARGS=(--experiment_dir "$COV_EXPERIMENT_DIR" --n_noise_seeds "$COV_N_NOISE_SEEDS" --n_inits "$COV_N_INITS"
          --jobs "$COV_JOBS" --level "$COV_LEVEL" --extract)
    [ -n "$COV_N_ITERATIONS" ] && ARGS+=(--n_iterations "$COV_N_ITERATIONS")
    [ -n "$COV_CHECKPOINTS" ] && ARGS+=(--checkpoints $COV_CHECKPOINTS)
    python3 validation/distillation_coverage_study.py "${ARGS[@]}" "${PASS[@]}"
    ;;
normal)
    ARGS=("$RECIPE" --seeds "$SEEDS")
    [ "$DOWNSTREAM" = "1" ] && ARGS+=(--downstream)
    bash scripts/run_hsgp_pipeline.sh "${ARGS[@]}" $NORMAL_HSGP_ARGS
    ;;
*)
    echo "MODE must be threshold, coverage or normal (got '$MODE')"; exit 1 ;;
esac
echo "=== study $MODE finished ($(date))"

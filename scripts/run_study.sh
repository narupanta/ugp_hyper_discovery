#!/bin/bash
# ==============================================================================
# One entry point for the three studies. Choose the setup with MODE (below or --mode), adjust its settings block,
# and run it here or on the cluster (scripts/run_study_slurm.sh passes everything through to this script).
#
#   MODE=threshold   noise-threshold sweep: grid of displacement x load noise, HSGP extraction per grid point
#                    (validation/noise_threshold_sweep.py)
#   MODE=coverage    coverage of the distilled material parameters over noise seeds x normalizing-flow
#                    initialisations (validation/distillation_coverage_study.py); runs the extractions first if needed
#
# Typical workflow: 1) threshold  2) pick a grid point (e.g. d4e-05_l2e-02, see SWEEP_DIR/sweep_summary.md)
#                   3) coverage with COV_FROM_SWEEP=<that tag>: continues in that grid point's experiment folder,
#                      adds the missing extractions up to COV_N_NOISE_SEEDS seeds (same config), then distils
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
# The grid is read from THR_RECIPE (sweep_disp_noise, sweep_load_noise, sweep_seeds, sweep_force_noise_rule);
# anything set here (or on the command line) overrides the recipe. Leave empty to use the recipe.
NOISE_LEVELS=""                                   # displacement noise std, e.g. "1e-5 4e-5 7e-5 1e-4"
LOAD_NOISE_LEVELS=""                              # load noise (relative), e.g. "0.02 0.05 0.1"; reactions match it
THR_SEEDS=""                                      # e.g. "1 2 3"
FORCE_NOISE_RULE=""                               # cv | evidence | scaled | fixed | derived | from_reaction
CV_TOL="2"                                        # cv: smallest s within this many nats of the best ("" = best)
R2_MIN="0.99"                                     # a grid point "works" if mean val energy R^2 >= R2_MIN ...
EC_MIN="90"                                       # ... and mean val energy coverage >= EC_MIN (%)
SWEEP_DIR="results/noise_sweep_isihara_cv"
THR_HSGP_ARGS=""                                  # extra '--hsgp_<option> <value>' for every grid point

# --- 2. coverage: distilled-parameter coverage ----------------------------------
COV_FROM_SWEEP=""                                 # grid point of the threshold sweep (tag, e.g. d4e-05_l2e-02):
                                                  # its experiment folder in SWEEP_DIR is continued
COV_EXPERIMENT_DIR=""                             # or an existing experiment directory
COV_RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"   # used when neither of the two above is given
COV_DISP_NOISE=""                                 # with COV_RECIPE: override disp_noise ("" = recipe value)
COV_LOAD_NOISE=""                                 # with COV_RECIPE: override load_noise (reactions follow it)
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
    ARGS=(--base_recipe "$THR_RECIPE" --r2_min "$R2_MIN" --ec_min "$EC_MIN" --sweep_dir "$SWEEP_DIR")
    [ -n "$NOISE_LEVELS" ] && ARGS+=(--noise_levels $NOISE_LEVELS)
    [ -n "$LOAD_NOISE_LEVELS" ] && ARGS+=(--load_noise_levels $LOAD_NOISE_LEVELS)
    [ -n "$THR_SEEDS" ] && ARGS+=(--seeds $THR_SEEDS)
    [ -n "$FORCE_NOISE_RULE" ] && ARGS+=(--force_noise_rule "$FORCE_NOISE_RULE")
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
    if [ -n "$COV_FROM_SWEEP" ]; then
        STATE="$SWEEP_DIR/state/$COV_FROM_SWEEP.json"
        [ -f "$STATE" ] || { echo "No grid point $COV_FROM_SWEEP in $SWEEP_DIR/state (see --list_grid)"; exit 1; }
        COV_EXPERIMENT_DIR=$(python3 -c "import json; print(json.load(open('$STATE'))['experiment_dir'])")
        echo "Coverage study continues the threshold grid point $COV_FROM_SWEEP: $COV_EXPERIMENT_DIR"
    fi
    if [ -z "$COV_EXPERIMENT_DIR" ]; then
        SEEDS_LIST=$(seq -s ' ' 1 "$COV_N_NOISE_SEEDS")
        REC="$COV_RECIPE"
        if [ -n "$COV_DISP_NOISE$COV_LOAD_NOISE" ]; then      # recipe copy with the chosen noise levels
            mkdir -p results/coverage_recipes
            REC="results/coverage_recipes/$(basename "${COV_RECIPE%.yaml}")_d${COV_DISP_NOISE:-rec}_l${COV_LOAD_NOISE:-rec}.yaml"
            python3 - "$COV_RECIPE" "$REC" "$COV_DISP_NOISE" "$COV_LOAD_NOISE" <<'PY'
import sys, yaml
src, dst, d, l = sys.argv[1:5]
r = yaml.safe_load(open(src))
if d:
    r["disp_noise"] = float(d)
if l:
    r["load_noise"] = float(l); r["reaction_noise_rel"] = float(l)
yaml.safe_dump(r, open(dst, "w"), sort_keys=False)
PY
        fi
        echo "No experiment directory given: running the extractions of $REC for noise seeds $SEEDS_LIST"
        LOG=$(mktemp)
        bash scripts/run_hsgp_pipeline.sh "$REC" --seeds "$SEEDS_LIST" 2>&1 | tee "$LOG"
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

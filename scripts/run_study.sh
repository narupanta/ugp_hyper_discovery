#!/bin/bash
# ==============================================================================
# One entry point for the studies. Choose the setup with MODE (below or --mode), adjust its settings block, and run
# it here or on the cluster (scripts/run_study_slurm.sh passes everything through to this script).
#
#   MODE=normal      the standard pipeline for one recipe: HSGP extraction [+ distillation, FEM validation, plots]
#                    (scripts/run_hsgp_pipeline.sh)
#   MODE=coverage    coverage of the distilled material parameters over noise seeds x normalizing-flow
#                    initialisations (validation/distillation_coverage_study.py); runs the extractions first if needed
#
# Usage:
#   bash scripts/run_study.sh                                   # uses MODE and the settings below
#   bash scripts/run_study.sh --mode normal --recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --seeds "1 2"
#   bash scripts/run_study.sh --mode coverage --cov_experiment_dir results/<exp>
#   bash scripts/run_study.sh --mode coverage --cov_disp_noise 1e-5 --cov_load_noise 0.02
#   bash scripts/run_study.sh --mode coverage --cov_experiment_dir results/<exp> --evaluate_only
# Any setting below can be given as --<name in lower case> <value>.
# Results: normal: a new results/<timestamp>_hsgp_... folder (or the given experiment directory);
#          coverage: <experiment>/coverage_study.
# ==============================================================================
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

MODE="normal"                     # normal | coverage
RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"   # e.g. nh2_ / gentthomas_ / ttc_hsgp_residual_cv

# --- normal: standard pipeline (RECIPE, or an existing experiment directory) ---------
SEEDS="1"
DOWNSTREAM="1"                                    # 1: also distillation, FEM validation and validation plots
NORMAL_HSGP_ARGS=""                               # extra '--hsgp_<option> <value>' for the extraction

# --- coverage: distilled-parameter coverage ----------------------------------
COV_EXPERIMENT_DIR=""                             # an existing experiment directory (continued: missing seeds added)
COV_RECIPE=""                                     # "" = RECIPE; used when no experiment directory is given
COV_DISP_NOISE=""                                 # with COV_RECIPE: override disp_noise ("" = recipe value)
COV_LOAD_NOISE=""                                 # with COV_RECIPE: override load_noise (reactions follow it)
COV_N_NOISE_SEEDS="10"
COV_N_INITS="5"
COV_N_ITERATIONS=""                               # Wasserstein iterations ("" = recipe value)
COV_CHECKPOINTS=""                                # e.g. "2000 4000 6000 8000 10000" (pilot)
COV_JOBS="2"
COV_LEVEL="0.95"

# --- command-line overrides -----------------------------------------------------
PASS=()                                           # flags passed on unchanged (e.g. --evaluate_only)
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode) MODE="$2"; shift 2 ;;
        --evaluate_only|--extract|--force|--no_sensitivity) PASS+=("$1"); shift ;;
        --*) VAR=$(echo "${1#--}" | tr '[:lower:]' '[:upper:]'); printf -v "$VAR" '%s' "$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
cd "$(dirname "$0")/.."
COV_RECIPE="${COV_RECIPE:-$RECIPE}"
echo "=== study: $MODE ($(date)) | recipe: $([ "$MODE" = coverage ] && echo "$COV_RECIPE" || echo "$RECIPE")"

case "$MODE" in
coverage)
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
    echo "MODE must be normal or coverage (got '$MODE')"; exit 1 ;;
esac
echo "=== study $MODE finished ($(date))"

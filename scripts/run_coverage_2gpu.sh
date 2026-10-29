#!/bin/bash
# ==============================================================================
# Coverage study on several GPU slots (default 2), resumable:
#   1. HSGP extraction of the missing noise seeds 1..N_SEEDS, the seeds shared round-robin over the slots
#      (one seed after another per slot);
#   2. distillation, one initialisation at a time: for init m, slot g runs the seeds s with (s - 1) % N_SLOTS == g
#      (2 slots: odd seeds on the first, even seeds on the second); init m + 1 starts when every seed of init m is done;
#   3. the coverage summary over all seeds and inits.
# Every step skips what already exists (extraction: hsgp_posterior.npz + extraction_metrics.json; distillation: the
# samples of the seed/init pair), so a stopped job is continued by resubmitting the same command.
#
# Usage (normally through scripts/run_coverage_2gpu_slurm.sh):
#   bash scripts/run_coverage_2gpu.sh --recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --n_seeds 10 --n_inits 5
#   bash scripts/run_coverage_2gpu.sh --experiment_dir results/<exp> --n_seeds 10 --n_inits 5 \
#        --sample_mode dataset_f_random --num_points 128
# Options:
#   --experiment_dir DIR   existing experiment (continued); or --recipe FILE to create a new one
#   --n_seeds N            noise seeds 1..N (default 10); or --seeds "1 2 5" for an explicit list
#   --n_inits M            initialisations 0..M-1 (default 5)
#   --gpus "0 1"           GPU id of each slot (default: "0 1"); a repeated id puts several slots on one GPU, e.g.
#                          --gpus "0 0 0" = 3 seeds at a time on one GPU (each slot gets a share of its memory)
#   --hsgp_<opt> <value>   passed to the extraction (e.g. --hsgp_cv_folds 3)
#   anything else          passed to validation/distillation_coverage_study.py (e.g. --sample_mode, --num_points,
#                          --n_iterations, --checkpoints ..., --no_plots, --no_sensitivity)
# Logs: <experiment>/coverage_logs/
# ==============================================================================
set -e
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1
cd "$(dirname "$0")/.."

EXP=""; RECIPE=""; N_SEEDS=10; SEEDS=""; N_INITS=5; GPUS="0 1"
HSGP_ARGS=(); STUDY_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --experiment_dir) EXP="$2"; shift 2 ;;
        --recipe) RECIPE="$2"; shift 2 ;;
        --n_seeds) N_SEEDS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --n_inits) N_INITS="$2"; shift 2 ;;
        --gpus) GPUS="$2"; shift 2 ;;
        --hsgp_*) HSGP_ARGS+=("$1" "$2"); shift 2 ;;
        --checkpoints)                      # list-valued: everything up to the next option
            STUDY_ARGS+=("$1"); shift
            while [[ $# -gt 0 && "$1" != --* ]]; do STUDY_ARGS+=("$1"); shift; done ;;
        --no_plots|--no_sensitivity|--force) STUDY_ARGS+=("$1"); shift ;;
        --*) STUDY_ARGS+=("$1" "$2"); shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done
[ -n "$SEEDS" ] || SEEDS=$(seq -s ' ' 1 "$N_SEEDS")
read -r -a SLOTS <<< "$GPUS"
NS=${#SLOTS[@]}
# JAX memory cap per process: a GPU shared by k slots gives each 0.9 / k (0.45 when a GPU has one slot)
mem_frac() { local k=0; for x in "${SLOTS[@]}"; do [ "$x" = "${SLOTS[$1]}" ] && k=$((k + 1)); done
             python3 -c "print(round(min(0.45, 0.9 / $k), 3))"; }
tag() { echo "slot$1-gpu${SLOTS[$1]}"; }

# --- experiment directory (a recipe creates a new one, named as in scripts/run_hsgp_pipeline.sh)
if [ -z "$EXP" ]; then
    [ -f "$RECIPE" ] || { echo "Give --experiment_dir or an existing --recipe"; exit 1; }
    NAME=$(python3 -c "
import yaml; d = yaml.safe_load(open('$RECIPE'))
print('_'.join(str(x) for x in [d.get('material_model_name', 'model'), d.get('disp_noise'), d.get('load_noise'),
      d.get('target_load_true_top'), d.get('asym_factor'), d.get('geometry_train', d.get('geometry', 'block'))]))")
    EXP="results/$(date +%Y%m%dT%H%M%S)_hsgp_${NAME}"
    mkdir -p "$EXP"
    cp "$RECIPE" "$EXP/config.yaml"
    echo "Created experiment directory: $EXP (recipe $RECIPE)"
fi
EXP="$(cd "$EXP" && pwd)"
LOGS="$EXP/coverage_logs"
mkdir -p "$LOGS"
echo "=== coverage study on ${NS} slot(s) (GPU per slot: ${GPUS}) | $EXP | seeds: $SEEDS | inits: $N_INITS | $(date)"

extracted() { [ -f "$EXP/$1/extracted/hsgp_posterior.npz" ] && [ -f "$EXP/$1/extracted/extraction_metrics.json" ]; }

# --- 1. extractions of the missing seeds, round-robin over the slots
MISSING=(); for s in $SEEDS; do extracted "$s" || MISSING+=("$s"); done
if [ ${#MISSING[@]} -gt 0 ]; then
    echo "--- extraction of seeds: ${MISSING[*]}"
    PIDS=()
    for g in $(seq 0 $((NS - 1))); do
        MINE=(); for i in "${!MISSING[@]}"; do [ $((i % NS)) -eq "$g" ] && MINE+=("${MISSING[$i]}"); done
        [ ${#MINE[@]} -gt 0 ] || continue
        (
            # the first slot solves a missing clean dataset; the others wait for its first dataset spec (no race)
            if [ "$g" -gt 0 ]; then
                for _ in $(seq 1 720); do [ -f "$EXP/${MISSING[0]}/dataset_train.spec" ] && break; sleep 5; done
            fi
            for s in "${MINE[@]}"; do
                echo "[$(tag "$g")] seed $s extraction ($(date))"
                CUDA_VISIBLE_DEVICES=${SLOTS[$g]} XLA_PYTHON_CLIENT_MEM_FRACTION=$(mem_frac "$g") \
                    bash scripts/run_hsgp_pipeline.sh "$EXP" --seeds "$s" "${HSGP_ARGS[@]}" \
                    > "$LOGS/extract_seed${s}.log" 2>&1 || echo "[$(tag "$g")] seed $s extraction FAILED (see $LOGS/extract_seed${s}.log)"
            done
        ) &
        PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait "$p" || true; done
fi
READY=(); for s in $SEEDS; do if extracted "$s"; then READY+=("$s"); else echo "seed $s has no extraction: left out"; fi; done
[ ${#READY[@]} -gt 0 ] || { echo "No extracted seeds"; exit 1; }

# --- 2. distillations, init by init; slot g takes the seeds with (s - 1) % NS == g
for m in $(seq 0 $((N_INITS - 1))); do
    echo "--- distillation init $m ($(date))"
    PIDS=()
    for g in $(seq 0 $((NS - 1))); do
        MINE=(); for s in "${READY[@]}"; do [ $(((s - 1) % NS)) -eq "$g" ] && MINE+=("$s"); done
        [ ${#MINE[@]} -gt 0 ] || continue
        echo "[$(tag "$g")] init $m seeds ${MINE[*]}"
        CUDA_VISIBLE_DEVICES=${SLOTS[$g]} XLA_PYTHON_CLIENT_MEM_FRACTION=$(mem_frac "$g") \
            python3 validation/distillation_coverage_study.py --experiment_dir "$EXP" \
            --noise_seeds "${MINE[@]}" --inits "$m" --no_evaluate "${STUDY_ARGS[@]}" \
            > "$LOGS/distill_init${m}_$(tag "$g").log" 2>&1 &
        PIDS+=($!)
    done
    FAIL=0
    for p in "${PIDS[@]}"; do wait "$p" || FAIL=1; done
    [ "$FAIL" -eq 0 ] || echo "init $m: a slot failed (see $LOGS/distill_init${m}_slot*.log); continuing"
done

# --- 3. summary over all seeds and inits
echo "--- coverage summary ($(date))"
python3 validation/distillation_coverage_study.py --experiment_dir "$EXP" --noise_seeds "${READY[@]}" \
    --n_inits "$N_INITS" --evaluate_only "${STUDY_ARGS[@]}" 2>&1 | tee "$LOGS/summary.log"
echo "=== done ($(date)): $EXP"

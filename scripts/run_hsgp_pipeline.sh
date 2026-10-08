#!/bin/bash
# HSGP extraction pipeline: dataset spec (clean FEM data solved once, reused) -> HSGP fit -> plots/metrics, per seed.
#
# Usage:
#   scripts/run_hsgp_pipeline.sh <recipe.yaml | recipe name (configs/recipes_v2 first) | experiment dir> [--seeds "6 7 8"] [--hsgp_warp 0 ...]
#   Any --hsgp_* option is passed on to extraction/train_hsgp.py and overrides the recipe (see its docstring).
#   --downstream: afterwards run distillation, FEM forward validation and validation plots with scripts/run_pipeline.sh
#                 on the same experiment directory (it accepts HSGP extractions; data generation only adds the missing
#                 validation geometry, the clean training data is reused).
#   --downstream-only: skip the extraction and run only those stages (for an existing HSGP experiment directory).
#
# Layout (same as run_pipeline.sh): results/<timestamp>_hsgp_<model>_<dnoise>_<lnoise>_<top>_<asym>_<geometry>/<seed>/extracted
# Each seed runs the fit and the plots as two processes so that peak memory stays low.
set -e

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
export PYTHONUNBUFFERED=1

INPUT_TARGET=""
SEEDS_CLI=""
HSGP_ARGS=()
SKIP_PLOTS=false
DOWNSTREAM=false
DOWNSTREAM_ONLY=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --downstream) DOWNSTREAM=true; shift ;;
        --downstream-only) DOWNSTREAM=true; DOWNSTREAM_ONLY=true; shift ;;
        --seeds|--seed) SEEDS_CLI="$2"; shift 2 ;;
        --skip-plots) SKIP_PLOTS=true; shift ;;
        --hsgp_*) HSGP_ARGS+=("$1" "$2"); shift 2 ;;
        -*) echo "Warning: unknown option $1"; shift ;;
        *) if [ -z "$INPUT_TARGET" ]; then INPUT_TARGET="$1"; fi; shift ;;
    esac
done
if [ -z "$INPUT_TARGET" ]; then
    echo "Usage: $0 <recipe.yaml | recipe name | experiment dir> [--seeds '6 7 8'] [--hsgp_<option> <value> ...]"
    exit 1
fi

# --- experiment directory and config (an existing directory is reused; a recipe creates a new one)
if [ -d "$INPUT_TARGET" ] || [ -d "results/$INPUT_TARGET" ]; then
    [ -d "$INPUT_TARGET" ] && EXPERIMENT_DIR="$(cd "$INPUT_TARGET" && pwd)" || EXPERIMENT_DIR="$(cd "results/$INPUT_TARGET" && pwd)"
    CONFIG_YAML="$EXPERIMENT_DIR/config.yaml"
    [ -f "$CONFIG_YAML" ] || { echo "No config.yaml in $EXPERIMENT_DIR"; exit 1; }
    echo "Using existing experiment directory: $EXPERIMENT_DIR"
else
    if [ -f "$INPUT_TARGET" ]; then RECIPE_FILE="$INPUT_TARGET"
    elif [ -f "${INPUT_TARGET}.yaml" ]; then RECIPE_FILE="${INPUT_TARGET}.yaml"
    elif [ -f "configs/recipes_v2/${INPUT_TARGET}.yaml" ]; then RECIPE_FILE="configs/recipes_v2/${INPUT_TARGET}.yaml"
    elif [ -f "configs/recipes/${INPUT_TARGET}.yaml" ]; then RECIPE_FILE="configs/recipes/${INPUT_TARGET}.yaml"
    else echo "Recipe or experiment '$INPUT_TARGET' not found"; exit 1; fi
    NAME=$(python3 -c "
import yaml; d = yaml.safe_load(open('$RECIPE_FILE'))
print('_'.join(str(x) for x in [d.get('material_model_name', 'model'), d.get('disp_noise'), d.get('load_noise'),
      d.get('target_load_true_top'), d.get('asym_factor'), d.get('geometry_train', d.get('geometry', 'block'))]))")
    EXPERIMENT_DIR="$(pwd)/results/$(date +%Y%m%dT%H%M%S)_hsgp_${NAME}"
    mkdir -p "$EXPERIMENT_DIR"
    cp "$RECIPE_FILE" "$EXPERIMENT_DIR/config.yaml"
    CONFIG_YAML="$EXPERIMENT_DIR/config.yaml"
    echo "Created experiment directory: $EXPERIMENT_DIR (recipe $RECIPE_FILE)"
fi

cfg() { python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); v=d.get('$1', '$2'); print(*v) if isinstance(v, (list, tuple)) else print(v)"; }
MODEL=$(cfg material_model_name isihara)
D_NOISE=$(cfg disp_noise 0.0001)
L_NOISE=$(cfg load_noise 0.01)
TOP_LOAD=$(cfg target_load_true_top 1.0)
ASYM=$(cfg asym_factor 1.0)
STEPS=$(cfg n_loadsteps 20)
MESH_SIZE=$(cfg mesh_size 0.08)
CONTROL_MODE=$(cfg control_mode displacement)
STRESS_MODE=$(cfg stress_mode plane_strain)
GEOMETRY_TRAIN=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('geometry_train', d.get('geometry', 'block')))")
CUSTOM_DATASET_PATH=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('dataset_path', '') or '')")
MAT_EXTRA_ARGS=$(python3 -c "
import yaml; d = yaml.safe_load(open('$CONFIG_YAML')); m = d.get('material_params', {}) or {}
out = []
for k in ('angles', 'dev_params', 'vol_params', 'aniso_params'):
    v = m.get(k, d.get(k) if k == 'angles' else None)
    if v is not None:
        out += ['--' + k] + [str(x) for x in (v if isinstance(v, list) else [v])]
print(' '.join(out))")
if [ -n "$SEEDS_CLI" ]; then SEEDS_LIST="${SEEDS_CLI//,/ }"; else SEEDS_LIST=$(cfg seeds 42); fi

echo "========================================================================"
echo "HSGP experiment: $EXPERIMENT_DIR"
echo "Seeds:           $SEEDS_LIST"
echo "Overrides:       ${HSGP_ARGS[*]:-none (recipe hsgp_* keys / defaults)}"
echo "========================================================================"

for SEED in $SEEDS_LIST; do
    [ "$DOWNSTREAM_ONLY" = true ] && break
    echo ""
    echo "### SEED = $SEED"
    SEED_DIR="$EXPERIMENT_DIR/$SEED"
    EXTRACT_DIR="$SEED_DIR/extracted"
    mkdir -p "$EXTRACT_DIR"

    # dataset spec: the clean FEM dataset is solved only if it does not exist yet; the seed's noise is added at load time
    if [ -n "$CUSTOM_DATASET_PATH" ]; then
        TRAIN_DATASET_PATH="$CUSTOM_DATASET_PATH"
    else
        python3 dataset/synthetic/force_control/syn_force_control.py \
            --recipe "$CONFIG_YAML" --model "$MODEL" --disp_noise "$D_NOISE" --load_noise "$L_NOISE" \
            --target_top "$TOP_LOAD" --asym "$ASYM" --n_steps "$STEPS" --geometry "$GEOMETRY_TRAIN" \
            --mesh_size "$MESH_SIZE" --control_mode "$CONTROL_MODE" --stress_mode "$STRESS_MODE" --clamp_top_x 0 \
            --seed "$SEED" --spec_out "$SEED_DIR/dataset_train.spec" $MAT_EXTRA_ARGS
        TRAIN_DATASET_PATH=$(cat "$SEED_DIR/dataset_train.spec")
    fi
    echo "Dataset: $TRAIN_DATASET_PATH"

    COMMON=(--recipe "$CONFIG_YAML" --dataset_path "$TRAIN_DATASET_PATH" --seed "$SEED" --batch_dir "$EXTRACT_DIR" "${HSGP_ARGS[@]}")
    echo "--- HSGP fit (seed $SEED)"
    python3 extraction/train_hsgp.py "${COMMON[@]}" --phase fit
    if [ "$SKIP_PLOTS" = false ]; then
        echo "--- HSGP plots and metrics (seed $SEED)"
        python3 extraction/train_hsgp.py "${COMMON[@]}" --phase plot
    fi
    echo "Seed $SEED done -> $EXTRACT_DIR"
done
echo "All seeds finished: $EXPERIMENT_DIR"

if [ "$DOWNSTREAM" = true ]; then
    echo ""
    echo "### Downstream: distillation, FEM forward validation, validation plots (scripts/run_pipeline.sh)"
    bash scripts/run_pipeline.sh "$EXPERIMENT_DIR" --seeds "$SEEDS_LIST" --do-gen --do-distill --do-fem --do-val
fi

#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gp-pipeline
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere

set -e

# Prevent JAX from pre-allocating all GPU memory to avoid OOM
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=3
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# ==============================================================================
# 1. Argument Parsing & Flag Defaults
# ==============================================================================
INPUT_TARGET=""
SEEDS_CLI=""
DO_GEN=false
DO_EXT=false
DO_DISTILL=false
DO_FEM=false
DO_VAL=false
HAS_DO_FLAGS=false

SKIP_GEN=false
SKIP_EXT=false
SKIP_DISTILL=false
SKIP_FEM=false
SKIP_VAL=false

VAL_WORKERS_OVERRIDE=""
VAL_SAMPLES_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --do-gen)
            DO_GEN=true
            HAS_DO_FLAGS=true
            shift
            ;;
        --do-ext)
            DO_EXT=true
            HAS_DO_FLAGS=true
            shift
            ;;
        --do-distill)
            DO_DISTILL=true
            HAS_DO_FLAGS=true
            shift
            ;;
        --do-fem)
            DO_FEM=true
            HAS_DO_FLAGS=true
            shift
            ;;
        --do-val)
            DO_VAL=true
            HAS_DO_FLAGS=true
            shift
            ;;
        --skip-gen|--no-gen)
            SKIP_GEN=true
            shift
            ;;
        --skip-ext|--no-ext)
            SKIP_EXT=true
            shift
            ;;
        --skip-distill|--no-distill)
            SKIP_DISTILL=true
            shift
            ;;
        --skip-fem|--no-fem)
            SKIP_FEM=true
            shift
            ;;
        --skip-val|--no-val)
            SKIP_VAL=true
            shift
            ;;
        --seeds|--seed)
            SEEDS_CLI="$2"
            shift 2
            ;;
        --workers|--val-workers)
            VAL_WORKERS_OVERRIDE="$2"
            shift 2
            ;;
        --samples|--val-samples)
            VAL_SAMPLES_OVERRIDE="$2"
            shift 2
            ;;
        -*)
            echo "Warning: Unknown option: $1"
            shift
            ;;
        *)
            if [ -z "$INPUT_TARGET" ]; then
                INPUT_TARGET="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$INPUT_TARGET" ]; then
    echo "❌ Error: Please provide an experiment folder, recipe yaml, or recipe name!"
    echo "Usage: $0 <experiment_dir_or_recipe> [--seeds '1 2'] [--do-gen] [--do-ext] [--do-distill] [--do-fem] [--do-val]"
    exit 1
fi

# Determine whether to execute each stage based on --do-* or --skip-* flags
# If any --do-* flag was specified, ONLY those stages run.
if [ "$HAS_DO_FLAGS" = true ]; then
    RUN_GEN=$DO_GEN
    RUN_EXT=$DO_EXT
    RUN_DISTILL=$DO_DISTILL
    RUN_FEM=$DO_FEM
    RUN_VAL=$DO_VAL
else
    # Otherwise run all stages unless explicitly skipped
    RUN_GEN=true
    RUN_EXT=true
    RUN_DISTILL=true
    RUN_FEM=true
    RUN_VAL=true
    if [ "$SKIP_GEN" = true ]; then RUN_GEN=false; fi
    if [ "$SKIP_EXT" = true ]; then RUN_EXT=false; fi
    if [ "$SKIP_DISTILL" = true ]; then RUN_DISTILL=false; fi
    if [ "$SKIP_FEM" = true ]; then RUN_FEM=false; fi
    if [ "$SKIP_VAL" = true ]; then RUN_VAL=false; fi
fi

# ==============================================================================
# 2. Experiment Directory & Configuration Resolution
# ==============================================================================
IS_EXISTING_EXP=false
EXPERIMENT_DIR=""

# Check if input is an existing directory
if [ -d "$INPUT_TARGET" ]; then
    EXPERIMENT_DIR="$(cd "$INPUT_TARGET" && pwd)"
    IS_EXISTING_EXP=true
elif [ -d "results/$INPUT_TARGET" ]; then
    EXPERIMENT_DIR="$(cd "results/$INPUT_TARGET" && pwd)"
    IS_EXISTING_EXP=true
fi

if [ "$IS_EXISTING_EXP" = true ]; then
    echo "📁 Using existing experiment directory: $EXPERIMENT_DIR"
    CONFIG_YAML="$EXPERIMENT_DIR/config.yaml"
    if [ ! -f "$CONFIG_YAML" ]; then
        if [ -f "$EXPERIMENT_DIR/recipe_config.yaml" ]; then
            CONFIG_YAML="$EXPERIMENT_DIR/recipe_config.yaml"
        else
            echo "❌ Error: No config.yaml found inside $EXPERIMENT_DIR!"
            exit 1
        fi
    fi
else
    # Resolve recipe file
    RECIPE_FILE=""
    if [ -f "$INPUT_TARGET" ]; then
        RECIPE_FILE="$INPUT_TARGET"
    elif [ -f "configs/recipes/${INPUT_TARGET}.yaml" ]; then
        RECIPE_FILE="configs/recipes/${INPUT_TARGET}.yaml"
    elif [ -f "configs/recipes/${INPUT_TARGET}_benchmark.yaml" ]; then
        RECIPE_FILE="configs/recipes/${INPUT_TARGET}_benchmark.yaml"
    else
        echo "❌ Error: Recipe or experiment '$INPUT_TARGET' not found!"
        exit 1
    fi

    echo "📋 Initializing new experiment from recipe: $RECIPE_FILE"
    
    # Read core naming parameters from recipe
    MODEL=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('material_model_name', 'nh2'))")
    D_NOISE=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('disp_noise', 0.0001))")
    L_NOISE=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('load_noise', 0.01))")
    TOP_LOAD=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('target_load_true_top', 1.5))")
    ASYM=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('asym_factor', 0.95))")
    N_IP=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('n_ip', 5))")
    BETA=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('beta', d.get('beta_list', [1.0])[0] if isinstance(d.get('beta_list'), (list, tuple)) else 1.0))" 2>/dev/null || echo "1.0")
    MODEL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('model_mode', 'isotropic'))")
    GEOM_TRAIN=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('geometry_train', d.get('geometry', 'block')))")

    HAS_GT_VAL=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('has_ground_truth', True))" 2>/dev/null || echo "True")

    CURRENT_TIME=$(date +"%Y%m%dT%H%M%S")
    if [ "$HAS_GT_VAL" == "False" ] || [ "$HAS_GT_VAL" == "false" ] || [ "$GEOM_TRAIN" == "ttc" ] || [ "$MODEL" == "experimental" ]; then
        EXP_FOLDER_NAME="${CURRENT_TIME}_${GEOM_TRAIN}_${N_IP}_${BETA}_${MODEL_MODE}"
    else
        EXP_FOLDER_NAME="${CURRENT_TIME}_${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${N_IP}_${BETA}_${MODEL_MODE}_${GEOM_TRAIN}"
    fi
    EXPERIMENT_DIR="$(pwd)/results/${EXP_FOLDER_NAME}"
    mkdir -p "$EXPERIMENT_DIR"
    cp "$RECIPE_FILE" "$EXPERIMENT_DIR/config.yaml"
    CONFIG_YAML="$EXPERIMENT_DIR/config.yaml"
    echo "📁 Created experiment directory: $EXPERIMENT_DIR"
fi

# Extract parameters from experiment config.yaml
get_cfg_default() {
    python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('$1', '$2'))" 2>/dev/null || echo "$2"
}

CUSTOM_DATASET_PATH=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('dataset_path', ''))" 2>/dev/null || echo "")
HAS_GT=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('has_ground_truth', True))" 2>/dev/null || echo "True")
if [ -n "$CUSTOM_DATASET_PATH" ] || [ "$HAS_GT" == "False" ] || [ "$HAS_GT" == "false" ]; then
    RUN_GEN=false
fi

MODEL=$(get_cfg_default "material_model_name" "nh2")
D_NOISE=$(get_cfg_default "disp_noise" "0.0001")
L_NOISE=$(get_cfg_default "load_noise" "0.01")
ASYM=$(get_cfg_default "asym_factor" "1.0")
TOP_LOAD=$(get_cfg_default "target_load_true_top" "1.0")
TOP_LOAD_HOLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('target_load_holes', d.get('target_load_true_top', '$TOP_LOAD')))" 2>/dev/null || echo "$TOP_LOAD")
STEPS=$(get_cfg_default "n_loadsteps" "1")
MESH_SIZE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('mesh_size', '0.08'))" 2>/dev/null || echo "0.08")

# Extraction params
MCI_SAMPLING=$(get_cfg_default "number_of_mci_sampling" "1")
N_IP=$(get_cfg_default "n_ip" "5")
BETAS_LIST=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); b=d.get('beta_list', d.get('betas', d.get('beta', 1.0))); print(*b) if isinstance(b, (list, tuple)) else print(b)" 2>/dev/null || echo "1.0")
BETA=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('beta', d.get('beta_list', [1.0])[0] if isinstance(d.get('beta_list'), (list, tuple)) else 1.0))" 2>/dev/null || echo "1.0")
NUM_RFF=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('num_rff', 800))" 2>/dev/null || echo "800")
FIXED_NOISE=$(get_cfg_default "is_fixed_reaction_force_noise" "1")
FIXED_IP=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('is_fixed_inducing_points', 1))" 2>/dev/null || echo "1")
EXT_ITERS=$(get_cfg_default "extraction_n_iterations" "100")
EXT_LR=$(get_cfg_default "extraction_learning_rate" "0.01")
EXT_FINAL_LR=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('extraction_final_learning_rate', d.get('extraction_learning_rate')))" 2>/dev/null || echo "$EXT_LR")
CAP_COMPRESSION=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('cap_compression', 1))" 2>/dev/null || echo "1")
TRAIN_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(*(d.get('train_load_steps_indices', [0])))" 2>/dev/null || echo "0")
MODEL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('model_mode', 'isotropic'))" 2>/dev/null || echo "isotropic")
COVARIANCE_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('covariance_mode', 'diag'))" 2>/dev/null || echo "diag")
NORMALIZE_ELL=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('normalize_ell', 0))" 2>/dev/null || echo "0")
U_VAR_ANCHOR=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('u_var_anchor', '1e-12'))" 2>/dev/null || echo "1e-12")
KZZ_JITTER=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('kzz_jitter', '1e-8'))" 2>/dev/null || echo "1e-8")
VFM_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('vfm_mode', 'linear_triangle'))" 2>/dev/null || echo "linear_triangle")
VF_ORDER=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('vf_order', 2))" 2>/dev/null || echo "2")
CONTROL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('control_mode', 'force'))" 2>/dev/null || echo "force")
STRESS_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('stress_mode', d.get('stress_modes', 'plane_strain')))" 2>/dev/null || echo "plane_strain")
CONSTRAINT_LENGTHSCALE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(int(d.get('constraint_lengthscale', 1)))" 2>/dev/null || echo "1")
CLAMP_TOP_X=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(1 if d.get('clamp_top_x', True) else 0)" 2>/dev/null || echo "1")

# Distillation params
DIST_MODEL=$(get_cfg_default "distilled_material_model" "gmr")
DIST_ITERS=$(get_cfg_default "distillation_n_iterations" "100")
DEV_VOL_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('dev_vol_distillation_n_iterations', d.get('distillation_n_iterations', 5000)))" 2>/dev/null || echo "$DIST_ITERS")
ANISO_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); val=d.get('aniso_distillation_n_iterations'); print(val if val is not None else '')" 2>/dev/null || echo "")
DIST_TARGET=$(get_cfg_default "distill_target" "sef_split")
SAMPLE_MODE=$(get_cfg_default "sample_mode" "standard")
NUM_POINTS=$(get_cfg_default "num_points" "100")
NUM_FUNC_SAMPLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('num_func_samples', 512))" 2>/dev/null || echo "512")
MAX_GAMMA=$(get_cfg_default "max_gamma" "1.0")
DO_SENSITIVITY=$(get_cfg_default "do_sensitivity" "False")
SOBOL_THRESHOLD=$(get_cfg_default "sobol_threshold" "0.0001")
SOBOL_FACTOR=$(get_cfg_default "sobol_samples_factor" "2")
SENSITIVITY_FLAG=""
if [ "$DO_SENSITIVITY" == "0" ] || [ "$DO_SENSITIVITY" == "False" ] || [ "$DO_SENSITIVITY" == "false" ]; then
    SENSITIVITY_FLAG="--no_sensitivity"
fi

DEV_PARAMS_DISABLED=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); p=d.get('dev_params_disabled', []); print(*p) if isinstance(p, (list, tuple)) else (print(p) if p is not None else print(''))" 2>/dev/null || echo "")
VOL_PARAMS_DISABLED=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); p=d.get('vol_params_disabled', []); print(*p) if isinstance(p, (list, tuple)) else (print(p) if p is not None else print(''))" 2>/dev/null || echo "")
ANISO_PARAMS_DISABLED=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); p=d.get('aniso_params_disabled', []); print(*p) if isinstance(p, (list, tuple)) else (print(p) if p is not None else print(''))" 2>/dev/null || echo "")

DEV_PARAMS_DISABLED_ARG=""
if [ -n "$DEV_PARAMS_DISABLED" ]; then
    DEV_PARAMS_DISABLED_ARG="--dev_params_disabled $DEV_PARAMS_DISABLED"
fi

VOL_PARAMS_DISABLED_ARG=""
if [ -n "$VOL_PARAMS_DISABLED" ]; then
    VOL_PARAMS_DISABLED_ARG="--vol_params_disabled $VOL_PARAMS_DISABLED"
fi

ANISO_PARAMS_DISABLED_ARG=""
if [ -n "$ANISO_PARAMS_DISABLED" ]; then
    ANISO_PARAMS_DISABLED_ARG="--aniso_params_disabled $ANISO_PARAMS_DISABLED"
fi

# Validation params
if [ -n "$VAL_SAMPLES_OVERRIDE" ]; then
    VAL_SAMPLES="$VAL_SAMPLES_OVERRIDE"
else
    VAL_SAMPLES=$(get_cfg_default "val_number_samples" "8")
fi
VAL_LOAD_STEPS_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(*(d.get('val_load_steps_indices', [9])))")
TEST_LOAD_STEPS_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(*(d.get('test_load_steps_indices', [])))" 2>/dev/null || echo "")
TEST_LOAD_STEPS_ARG=""
if [ -n "$TEST_LOAD_STEPS_INDICES" ]; then
    TEST_LOAD_STEPS_ARG="--test_load_steps_indices $TEST_LOAD_STEPS_INDICES"
fi
FEM_VAL_STEPS="${TEST_LOAD_STEPS_INDICES:-$VAL_LOAD_STEPS_INDICES}"
VAL_WORKERS=2
if [ -n "$VAL_WORKERS_OVERRIDE" ]; then
    VAL_WORKERS="$VAL_WORKERS_OVERRIDE"
else
    VAL_WORKERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('val_workers', 2))" 2>/dev/null || echo "2")
fi

GEOMETRY_TRAIN=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('geometry_train', d.get('geometry', 'block')))" 2>/dev/null || echo "block")
GEOMETRY_VAL=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('geometry_val', 'holes'))" 2>/dev/null || echo "holes")

# Material parameters
ANGLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); a=d.get('material_params', {}).get('angles', d.get('angles', None)); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
DEV_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); a=d.get('material_params', {}).get('dev_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
VOL_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); a=d.get('material_params', {}).get('vol_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
ANISO_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); a=d.get('material_params', {}).get('aniso_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)

MAT_EXTRA_ARGS=""
if [ -n "$ANGLES" ]; then MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --angles $ANGLES"; fi
if [ -n "$DEV_PARAMS" ]; then MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --dev_params $DEV_PARAMS"; fi
if [ -n "$VOL_PARAMS" ]; then MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --vol_params $VOL_PARAMS"; fi
if [ -n "$ANISO_PARAMS" ]; then MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --aniso_params $ANISO_PARAMS"; fi

# Resolve seeds list
if [ -n "$SEEDS_CLI" ]; then
    SEEDS_LIST="${SEEDS_CLI//,/ }"
else
    SEEDS_LIST=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); s=d.get('seeds', d.get('seed_list', d.get('seed', 42))); print(*s) if isinstance(s, list) else print(s)" 2>/dev/null || echo "42")
fi

echo "========================================================================"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Material Model:       $MODEL (Candidate: $DIST_MODEL)"
echo "Seeds to process:     $SEEDS_LIST"
echo "Val samples:          $VAL_SAMPLES (Workers: $VAL_WORKERS)"
echo "Stages active:        GEN=$RUN_GEN, EXT=$RUN_EXT, DISTILL=$RUN_DISTILL, FEM=$RUN_FEM, VAL=$RUN_VAL"
echo "========================================================================"

# ==============================================================================
# 3. Seed Execution Loop
# ==============================================================================
for SEED in $SEEDS_LIST; do
    echo ""
    echo "########################################################################"
    echo "### Processing SEED = $SEED in $EXPERIMENT_DIR"
    echo "########################################################################"

    SEED_DIR="$EXPERIMENT_DIR/$SEED"
    EXTRACT_DIR="$SEED_DIR/extracted"
    DISTILL_DIR="$SEED_DIR/distilled"
    VAL_DIR="$SEED_DIR/fem_validation"

    mkdir -p "$SEED_DIR"

    if [ -n "$CUSTOM_DATASET_PATH" ]; then
        TRAIN_DATASET_PATH="$CUSTOM_DATASET_PATH"
    else
        TRAIN_DATASET_PATH="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${GEOMETRY_TRAIN}_${SEED}.npz"
    fi

    # --- STEP 1: DATA GENERATION ---
    if [ "$RUN_GEN" = true ]; then
        echo "--- Step 1: Data Generation (Seed: $SEED) ---"
        python3 dataset/synthetic/force_control/syn_force_control.py \
            --recipe "$CONFIG_YAML" \
            --model "$MODEL" \
            --disp_noise "$D_NOISE" \
            --load_noise "$L_NOISE" \
            --target_top "$TOP_LOAD" \
            --asym "$ASYM" \
            --n_steps "$STEPS" \
            --geometry "$GEOMETRY_TRAIN" \
            --mesh_size "$MESH_SIZE" \
            --control_mode "$CONTROL_MODE" \
            --stress_mode "$STRESS_MODE" \
            --clamp_top_x 0 \
            --seed "$SEED" \
            $MAT_EXTRA_ARGS

        if [ "$GEOMETRY_VAL" != "$GEOMETRY_TRAIN" ]; then
            python3 dataset/synthetic/force_control/syn_force_control.py \
                --recipe "$CONFIG_YAML" \
                --model "$MODEL" \
                --disp_noise "$D_NOISE" \
                --load_noise "$L_NOISE" \
                --target_top "$TOP_LOAD_HOLES" \
                --asym "$ASYM" \
                --n_steps "$STEPS" \
                --geometry "$GEOMETRY_VAL" \
                --mesh_size "$MESH_SIZE" \
                --control_mode "$CONTROL_MODE" \
                --stress_mode "$STRESS_MODE" \
                --clamp_top_x "$CLAMP_TOP_X" \
                --seed "$SEED" \
                $MAT_EXTRA_ARGS
        fi
        echo "✅ Step 1 (Data Generation for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 1 (Data Generation) for Seed $SEED."
    fi

    # --- STEP 2: EXTRACTION ---
    if [ "$RUN_EXT" = true ]; then
        echo "--- Step 2: UGP Extraction (Seed: $SEED) ---"
        
        NUM_BETAS=$(echo "$BETAS_LIST" | wc -w)
        if [ "$NUM_BETAS" -gt 1 ]; then
            echo "🔬 Multi-Beta Training active ($NUM_BETAS candidates: $BETAS_LIST)"
            CANDIDATES_DIR="$SEED_DIR/extracted_candidates"
            mkdir -p "$CANDIDATES_DIR"

            for BETA_VAL in $BETAS_LIST; do
                CANDIDATE_OUT="$CANDIDATES_DIR/beta_${BETA_VAL}"
                echo "  ▶ Training candidate model with beta = $BETA_VAL -> $CANDIDATE_OUT"
                mkdir -p "$CANDIDATE_OUT"
                python3 extraction/train_unsupervised.py \
                    --recipe "$CONFIG_YAML" \
                    --dataset_path "$TRAIN_DATASET_PATH" \
                    --material_model_name "$MODEL" \
                    --number_of_mci_sampling "$MCI_SAMPLING" \
                    --train_load_steps_indices $TRAIN_INDICES \
                    --val_load_steps_indices $VAL_LOAD_STEPS_INDICES \
                    $TEST_LOAD_STEPS_ARG \
                    --n_ip "$N_IP" \
                    --beta "$BETA_VAL" \
                    --num_rff "$NUM_RFF" \
                    --is_fixed_reaction_force_noise "$FIXED_NOISE" \
                    --is_fixed_inducing_points "$FIXED_IP" \
                    --n_iterations "$EXT_ITERS" \
                    --geometry "$GEOMETRY_TRAIN" \
                    --disp_noise "$D_NOISE" \
                    --load_noise "$L_NOISE" \
                    --target_load_true_top "$TOP_LOAD" \
                    --asym_factor "$ASYM" \
                    --learning_rate "$EXT_LR" \
                    --final_learning_rate "$EXT_FINAL_LR" \
                    --cap_compression "$CAP_COMPRESSION" \
                    --model_mode "$MODEL_MODE" \
                    --covariance_mode "$COVARIANCE_MODE" \
                    --normalize_ell "$NORMALIZE_ELL" \
                    --u_var_anchor "$U_VAR_ANCHOR" \
                    --kzz_jitter "$KZZ_JITTER" \
                    --vfm_mode "$VFM_MODE" \
                    --vf_order "$VF_ORDER" \
                    --control_mode "$CONTROL_MODE" \
                    --stress_mode "$STRESS_MODE" \
                    --seed "$SEED" \
                    --batch_dir "$CANDIDATE_OUT" \
                    $MAT_EXTRA_ARGS
            done

            echo "📊 Evaluating candidate models and selecting best-calibrated beta..."
            python3 extraction/select_best_beta.py \
                --candidates_dir "$CANDIDATES_DIR" \
                --output_dir "$EXTRACT_DIR" \
                --target_ec 95.0
        else
            echo "  ▶ Single Beta Training (beta = $BETA) -> $EXTRACT_DIR"
            mkdir -p "$EXTRACT_DIR"
            python3 extraction/train_unsupervised.py \
                --recipe "$CONFIG_YAML" \
                --dataset_path "$TRAIN_DATASET_PATH" \
                --material_model_name "$MODEL" \
                --number_of_mci_sampling "$MCI_SAMPLING" \
                --train_load_steps_indices $TRAIN_INDICES \
                --val_load_steps_indices $VAL_LOAD_STEPS_INDICES \
                $TEST_LOAD_STEPS_ARG \
                --n_ip "$N_IP" \
                --beta "$BETA" \
                --num_rff "$NUM_RFF" \
                --is_fixed_reaction_force_noise "$FIXED_NOISE" \
                --is_fixed_inducing_points "$FIXED_IP" \
                --n_iterations "$EXT_ITERS" \
                --geometry "$GEOMETRY_TRAIN" \
                --disp_noise "$D_NOISE" \
                --load_noise "$L_NOISE" \
                --target_load_true_top "$TOP_LOAD" \
                --asym_factor "$ASYM" \
                --learning_rate "$EXT_LR" \
                --final_learning_rate "$EXT_FINAL_LR" \
                --cap_compression "$CAP_COMPRESSION" \
                --model_mode "$MODEL_MODE" \
                --covariance_mode "$COVARIANCE_MODE" \
                --normalize_ell "$NORMALIZE_ELL" \
                --u_var_anchor "$U_VAR_ANCHOR" \
                --kzz_jitter "$KZZ_JITTER" \
                --vfm_mode "$VFM_MODE" \
                --vf_order "$VF_ORDER" \
                --control_mode "$CONTROL_MODE" \
                --stress_mode "$STRESS_MODE" \
                --constraint_lengthscale "$CONSTRAINT_LENGTHSCALE" \
                --seed "$SEED" \
                --batch_dir "$EXTRACT_DIR" \
                $MAT_EXTRA_ARGS
        fi
        echo "✅ Step 2 (Extraction for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 2 (Extraction) for Seed $SEED."
    fi

    # --- STEP 3: DISTILLATION ---
    if [ "$RUN_DISTILL" = true ]; then
        echo "--- Step 3: Distillation (Seed: $SEED) ---"
        mkdir -p "$DISTILL_DIR"
        
        # Verify extracted model exists
        if [ ! -f "$EXTRACT_DIR/best_params.npy" ]; then
            echo "❌ Error: Cannot run distillation; extracted model not found at $EXTRACT_DIR"
            exit 1
        fi

        if [ "$SAMPLE_MODE" == "dataset_all" ]; then
            EXPORT_SUB="pytorch_export_dataset_all"
        elif [ "$SAMPLE_MODE" == "dataset_f" ]; then
            EXPORT_SUB="pytorch_export_dataset_f_n${NUM_POINTS}"
        elif [ "$SAMPLE_MODE" == "standard_interp" ]; then
            EXPORT_SUB="pytorch_export_standard_interp"
        elif [ "$SAMPLE_MODE" == "inducing_points" ]; then
            EXPORT_SUB="pytorch_export_inducing_points"
        else
            EXPORT_SUB="pytorch_export_standard_g${MAX_GAMMA}"
        fi
        if [ "$DIST_TARGET" == "sef_stress" ] || [ "$DIST_TARGET" == "sef_cauchy" ]; then
            EXPORT_SUB="${EXPORT_SUB}_${DIST_TARGET}"
        fi

        if [ ! -f "$EXTRACT_DIR/$EXPORT_SUB/mean_dev.npy" ] && [ ! -f "$EXTRACT_DIR/$EXPORT_SUB/mean_psi.npy" ]; then
            echo "Exporting GP posterior to PyTorch bridge ($EXPORT_SUB)..."
            python3 distillation/export_gp_to_pytorch.py \
                --saved_model_dir "$EXTRACT_DIR" \
                --dataset_path "$TRAIN_DATASET_PATH" \
                --sample_mode "$SAMPLE_MODE" \
                --num_points "$NUM_POINTS" \
                --max_gamma "$MAX_GAMMA" \
                --distill_target "$DIST_TARGET" \
                --export_subfolder "$EXPORT_SUB"
        fi

        echo "Distilling DEV component into $DISTILL_DIR..."
        python3 distillation/distill_uqmodeldisc.py \
            --saved_model_dir "$EXTRACT_DIR" \
            --material_model "$DIST_MODEL" \
            --n_iterations "$DEV_VOL_DIST_ITERS" \
            --distill_target "$DIST_TARGET" \
            --component "dev" \
            --override_out_dir "$DISTILL_DIR" \
            --sample_mode "$SAMPLE_MODE" \
            --num_points "$NUM_POINTS" \
            --num_func_samples "$NUM_FUNC_SAMPLES" \
            --max_gamma "$MAX_GAMMA" \
            --sobol_threshold "$SOBOL_THRESHOLD" \
            --sobol_samples_factor "$SOBOL_FACTOR" \
            --seed "$SEED" \
            $DEV_PARAMS_DISABLED_ARG \
            $SENSITIVITY_FLAG &
            
        echo "Distilling VOL component into $DISTILL_DIR..."
        python3 distillation/distill_uqmodeldisc.py \
            --saved_model_dir "$EXTRACT_DIR" \
            --material_model "$DIST_MODEL" \
            --n_iterations "$DEV_VOL_DIST_ITERS" \
            --distill_target "$DIST_TARGET" \
            --component "vol" \
            --override_out_dir "$DISTILL_DIR" \
            --sample_mode "$SAMPLE_MODE" \
            --num_points "$NUM_POINTS" \
            --num_func_samples "$NUM_FUNC_SAMPLES" \
            --max_gamma "$MAX_GAMMA" \
            --sobol_threshold "$SOBOL_THRESHOLD" \
            --sobol_samples_factor "$SOBOL_FACTOR" \
            --seed "$SEED" \
            $VOL_PARAMS_DISABLED_ARG \
            $SENSITIVITY_FLAG &

        if [ "$MODEL_MODE" == "anisotropic" ] || [ "$DIST_MODEL" == "gmr_aniso" ]; then
            if [ -z "$ANISO_DIST_ITERS" ]; then
                echo "❌ Error: 'aniso_distillation_n_iterations' is not specified in $CONFIG_YAML!"
                echo "When running anisotropic extraction/distillation (model_mode: anisotropic or distilled_material_model: gmr_aniso), you must explicitly define 'aniso_distillation_n_iterations'."
                exit 1
            fi
            echo "Distilling ANISO component into $DISTILL_DIR (n_iterations: $ANISO_DIST_ITERS)..."
            python3 distillation/distill_uqmodeldisc.py \
                --saved_model_dir "$EXTRACT_DIR" \
                --material_model "$DIST_MODEL" \
                --n_iterations "$ANISO_DIST_ITERS" \
                --distill_target "$DIST_TARGET" \
                --component "aniso" \
                --override_out_dir "$DISTILL_DIR" \
                --sample_mode "$SAMPLE_MODE" \
                --num_points "$NUM_POINTS" \
                --num_func_samples "$NUM_FUNC_SAMPLES" \
                --max_gamma "$MAX_GAMMA" \
                --sobol_threshold "$SOBOL_THRESHOLD" \
                --sobol_samples_factor "$SOBOL_FACTOR" \
                --seed "$SEED" \
                $ANISO_PARAMS_DISABLED_ARG \
                $SENSITIVITY_FLAG &
        fi
            
        wait

        echo "Generating distillation diagnostics & sensitivity plots..."
        python3 plots/plot_distilled_validation.py \
            --distilled_dir "$DISTILL_DIR" \
            --saved_model_dir "$EXTRACT_DIR" \
            --material_model "$DIST_MODEL" \
            --distill_target "$DIST_TARGET" || true

        python3 plots/plot_split_summary.py \
            --distilled_dir "$DISTILL_DIR" \
            --saved_model_dir "$EXTRACT_DIR" \
            --material_model "$DIST_MODEL" \
            --distill_target "$DIST_TARGET" || true

        python3 plots/plot_distilled_r2_energy.py \
            --distilled_dir "$DISTILL_DIR" \
            --saved_model_dir "$EXTRACT_DIR" \
            --material_model "$DIST_MODEL" \
            --distill_target "$DIST_TARGET" \
            --val_load_steps $FEM_VAL_STEPS || true

        for COMP in "dev" "vol" "aniso"; do
            if [ -d "$DISTILL_DIR/output/${COMP}_sensitivities" ] || [ -d "$DISTILL_DIR/${COMP}_sensitivities" ]; then
                python3 plots/plot_invariant_sensitivity.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                python3 plots/plot_invariant_sensitivity_3d_pairs.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                python3 plots/plot_deformation_sensitivity.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
            fi
        done
        python3 plots/plot_combined_invariant_sensitivity.py --distilled_dir "$DISTILL_DIR" 2>/dev/null || true
        python3 plots/plot_all_invariant_sensitivity.py --distilled_dir "$DISTILL_DIR" 2>/dev/null || true

        echo "✅ Step 3 (Distillation for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 3 (Distillation) for Seed $SEED."
    fi

    # --- STEP 4: FEM FORWARD SIMULATION ---
    if [ "$RUN_FEM" = true ]; then
        echo "--- Step 4: FEM Forward Simulation (Seed: $SEED) ---"
        mkdir -p "$VAL_DIR/block"
        mkdir -p "$VAL_DIR/holes"

        # Verify distilled samples exist
        if [ ! -f "$DISTILL_DIR/dev_flow_samples.npy" ]; then
            echo "❌ Error: Cannot run FEM forward simulation; distilled flow samples not found at $DISTILL_DIR"
            exit 1
        fi

        export OMP_NUM_THREADS=2
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.40

        if [ "$GEOMETRY_TRAIN" == "ttc" ] || [ "$GEOMETRY_VAL" == "ttc" ]; then
            mkdir -p "$VAL_DIR/ttc"
            VAL_DATASET_TTC="${TRAIN_DATASET_PATH}"

            echo "Running FEM workers for ttc geometry..."
            TTC_PIDS=()
            for ((w=0; w<VAL_WORKERS; w++)); do
                W_LOG="$VAL_DIR/ttc/worker_${w}.log"
                python3 validation/forward_fem_distilled_piola_sample.py \
                    --distilled_dir "$DISTILL_DIR" \
                    --material_model "$DIST_MODEL" \
                    --dataset_path "$VAL_DATASET_TTC" \
                    --n_sample "$VAL_SAMPLES" \
                    --output_dir "$VAL_DIR/ttc" \
                    --geometry "ttc" \
                    --target_load "$TOP_LOAD" \
                    --asym_factor "$ASYM" \
                    --control_mode "$CONTROL_MODE" \
                    --stress_mode "$STRESS_MODE" \
                    --clamp_top_x 0 \
                    --total_workers "$VAL_WORKERS" \
                    --worker_id "$w" > "$W_LOG" 2>&1 &
                TTC_PIDS+=($!)
            done

            for pid in "${TTC_PIDS[@]}"; do
                wait "$pid"
            done

            if [ "$VAL_WORKERS" -gt 1 ]; then
                echo "Merging TTC worker outputs..."
                python3 validation/merge_fem_workers.py --folder "$VAL_DIR/ttc"
            fi
        else
            mkdir -p "$VAL_DIR/block"
            mkdir -p "$VAL_DIR/holes"

            VAL_DATASET_BLOCK="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${GEOMETRY_TRAIN}_${SEED}.npz"
            VAL_DATASET_HOLES="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD_HOLES}_${ASYM}_${GEOMETRY_VAL}_${SEED}.npz"

            echo "Running FEM workers for $GEOMETRY_TRAIN geometry..."
            BLOCK_PIDS=()
            for ((w=0; w<VAL_WORKERS; w++)); do
                W_LOG="$VAL_DIR/block/worker_${w}.log"
                python3 validation/forward_fem_distilled_piola_sample.py \
                    --distilled_dir "$DISTILL_DIR" \
                    --material_model "$DIST_MODEL" \
                    --dataset_path "$VAL_DATASET_BLOCK" \
                    --n_sample "$VAL_SAMPLES" \
                    --output_dir "$VAL_DIR/block" \
                    --geometry "$GEOMETRY_TRAIN" \
                    --target_load "$TOP_LOAD" \
                    --asym_factor "$ASYM" \
                    --control_mode "$CONTROL_MODE" \
                    --stress_mode "$STRESS_MODE" \
                    --clamp_top_x 0 \
                    --total_workers "$VAL_WORKERS" \
                    --worker_id "$w" > "$W_LOG" 2>&1 &
                BLOCK_PIDS+=($!)
            done

            for pid in "${BLOCK_PIDS[@]}"; do
                wait "$pid"
            done

            if [ "$VAL_WORKERS" -gt 1 ]; then
                echo "Merging Block worker outputs..."
                python3 validation/merge_fem_workers.py --folder "$VAL_DIR/block"
            fi

            echo "Running FEM workers for $GEOMETRY_VAL geometry..."
            HOLES_PIDS=()
            for ((w=0; w<VAL_WORKERS; w++)); do
                W_LOG_HOLES="$VAL_DIR/holes/worker_${w}.log"
                python3 validation/forward_fem_distilled_piola_sample.py \
                    --distilled_dir "$DISTILL_DIR" \
                    --material_model "$DIST_MODEL" \
                    --dataset_path "$VAL_DATASET_HOLES" \
                    --n_sample "$VAL_SAMPLES" \
                    --output_dir "$VAL_DIR/holes" \
                    --geometry "$GEOMETRY_VAL" \
                    --target_load "$TOP_LOAD_HOLES" \
                    --asym_factor "$ASYM" \
                    --control_mode "$CONTROL_MODE" \
                    --stress_mode "$STRESS_MODE" \
                    --clamp_top_x "$CLAMP_TOP_X" \
                    --total_workers "$VAL_WORKERS" \
                    --worker_id "$w" > "$W_LOG_HOLES" 2>&1 &
                HOLES_PIDS+=($!)
            done

            for pid in "${HOLES_PIDS[@]}"; do
                wait "$pid"
            done

            if [ "$VAL_WORKERS" -gt 1 ]; then
                echo "Merging Holes worker outputs..."
                python3 validation/merge_fem_workers.py --folder "$VAL_DIR/holes"
            fi
        fi

        echo "✅ Step 4 (FEM Forward Simulation for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 4 (FEM Forward Simulation) for Seed $SEED."
    fi

    # --- STEP 5: VALIDATION METRICS & PLOTS ---
    if [ "$RUN_VAL" = true ]; then
        echo "--- Step 5: Validation Metrics & Plots (Seed: $SEED) ---"
        if [ ! -d "$DISTILL_DIR" ] && [ ! -d "$VAL_DIR" ]; then
            echo "⚠️ Notice: Neither $DISTILL_DIR nor $VAL_DIR found for Seed $SEED; skipping validation metrics & plots."
        else
            FEM_VAL_STEPS="${TEST_LOAD_STEPS_INDICES:-$VAL_LOAD_STEPS_INDICES}"

            # If worker files exist but consolidated fem_distilled_samples.npz doesn't exist, merge them
            for GEOM_DIR in "$VAL_DIR/block" "$VAL_DIR/holes" "$VAL_DIR/ttc"; do
                if [ -d "$GEOM_DIR" ] && [ ! -f "$GEOM_DIR/fem_distilled_samples.npz" ]; then
                    if ls "$GEOM_DIR"/fem_distilled_samples_worker*.npz 1> /dev/null 2>&1; then
                        echo "Merging worker outputs in $GEOM_DIR..."
                        python3 validation/merge_fem_workers.py --folder "$GEOM_DIR" || true
                    fi
                fi
            done

            if [ -d "$VAL_DIR/ttc" ] && [ -f "$VAL_DIR/ttc/fem_distilled_samples.npz" ]; then
                echo "Generating UQ displacement verification plots for TTC (evaluating steps: $FEM_VAL_STEPS)..."
                python3 plots/uq_verification_disp.py \
                    --model_path "$VAL_DIR/ttc" \
                    --validation_load_step_indices $FEM_VAL_STEPS \
                    --n_sample "$VAL_SAMPLES" || true
            fi

            if [ -d "$VAL_DIR/block" ] && [ -f "$VAL_DIR/block/fem_distilled_samples.npz" ]; then
                echo "Generating UQ displacement verification plots for Block (evaluating steps: $FEM_VAL_STEPS)..."
                python3 plots/uq_verification_disp.py \
                    --model_path "$VAL_DIR/block" \
                    --validation_load_step_indices $FEM_VAL_STEPS \
                    --n_sample "$VAL_SAMPLES" || true
            else
                echo "⚠️ Notice: $VAL_DIR/block/fem_distilled_samples.npz not found; skipping Block UQ verification plots."
            fi

            if [ -d "$VAL_DIR/holes" ] && [ -f "$VAL_DIR/holes/fem_distilled_samples.npz" ]; then
                echo "Generating UQ displacement verification plots for Holes (evaluating steps: $FEM_VAL_STEPS)..."
                python3 plots/uq_verification_disp.py \
                    --model_path "$VAL_DIR/holes" \
                    --validation_load_step_indices $FEM_VAL_STEPS \
                    --n_sample "$VAL_SAMPLES" || true
            else
                echo "⚠️ Notice: $VAL_DIR/holes/fem_distilled_samples.npz not found; skipping Holes UQ verification plots."
            fi

            if [ -d "$VAL_DIR" ]; then
                echo "Generating Reaction Force verification plots for Seed $SEED..."
                python3 plots/plot_reaction_force_distilled.py --model_path "$VAL_DIR" || true
                python3 plots/plot_free_node_residuals.py --model_path "$VAL_DIR" || true
            fi

            # Re-run distilled energy R2 plot and split summary if distilled outputs exist
            if [ -f "$DISTILL_DIR/dev_flow_samples.npy" ]; then
                if [ -f "$EXTRACT_DIR/gp_posterior_predictions.npz" ]; then
                    python3 plots/plot_distilled_r2_energy.py \
                        --distilled_dir "$DISTILL_DIR" \
                        --saved_model_dir "$EXTRACT_DIR" \
                        --material_model "$DIST_MODEL" \
                        --distill_target "$DIST_TARGET" \
                        --val_load_steps $FEM_VAL_STEPS || true
                fi

                python3 plots/plot_split_summary.py \
                    --distilled_dir "$DISTILL_DIR" \
                    --saved_model_dir "$EXTRACT_DIR" \
                    --material_model "$DIST_MODEL" \
                    --distill_target "$DIST_TARGET" || true
            fi

            if [ -d "$DISTILL_DIR" ]; then
                echo "Updating validation_metrics.json for Seed $SEED..."
                python3 validation/update_validation_metrics_json.py \
                    --distilled_dir "$DISTILL_DIR" \
                    --val_load_steps $FEM_VAL_STEPS

                # Copy validation_metrics.json into fem_validation and seed dir
                cp "$DISTILL_DIR/validation_metrics.json" "$VAL_DIR/validation_metrics.json" 2>/dev/null || true
                cp "$DISTILL_DIR/validation_metrics.json" "$SEED_DIR/validation_metrics.json" 2>/dev/null || true
            fi

            echo "✅ Step 5 (Validation Metrics & Plots for Seed $SEED) completed."
        fi
    else
        echo "⏭️ Skipping Step 5 (Validation Metrics & Plots) for Seed $SEED."
    fi

done

# ==============================================================================
# 4. Post-Execution Aggregation Across Seeds
# ==============================================================================
if [ "$RUN_VAL" = true ] || [ "$HAS_DO_FLAGS" = false ]; then
    echo ""
    echo "========================================================================"
    echo "=== Updating Experiment Config & Compiling Multi-Seed Summary ==="
    echo "========================================================================"

    # Update config.yaml with all available numeric seeds present in the folder
    python3 - <<EOF
import os
import yaml

exp_dir = "$EXPERIMENT_DIR"
cfg_file = "$CONFIG_YAML"

if os.path.exists(cfg_file):
    with open(cfg_file, 'r') as f:
        data = yaml.safe_load(f) or {}
    
    seeds_found = sorted([int(e) for e in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, e)) and e.isdigit()])
    if seeds_found:
        data['seeds'] = seeds_found
        with open(cfg_file, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
        print(f"Updated {cfg_file} with complete seeds list: {seeds_found}")
EOF

    # Run cross-seed summary generator
    python3 plots/generate_experiment_summary.py --experiment_dir "$EXPERIMENT_DIR" || true

    echo ""
    echo "🎉 Pipeline finished successfully for experiment: $EXPERIMENT_DIR"
    echo "📊 Cross-seed results available in: $EXPERIMENT_DIR/summary_across_seeds.md and $EXPERIMENT_DIR/plots/"
else
    echo ""
    echo "🎉 Pipeline finished successfully for requested stages: $EXPERIMENT_DIR"
fi

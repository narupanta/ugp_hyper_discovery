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
DO_VAL=false
HAS_DO_FLAGS=false

SKIP_GEN=false
SKIP_EXT=false
SKIP_DISTILL=false
SKIP_VAL=false

VAL_WORKERS_OVERRIDE=""

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
    echo "Usage: $0 <experiment_dir_or_recipe> [--seeds '1 2'] [--do-gen] [--do-ext] [--do-distill] [--do-val]"
    exit 1
fi

# Determine whether to execute each stage based on --do-* or --skip-* flags
# If any --do-* flag was specified, ONLY those stages run.
if [ "$HAS_DO_FLAGS" = true ]; then
    RUN_GEN=$DO_GEN
    RUN_EXT=$DO_EXT
    RUN_DISTILL=$DO_DISTILL
    RUN_VAL=$DO_VAL
else
    # Otherwise run all stages unless explicitly skipped
    RUN_GEN=true
    RUN_EXT=true
    RUN_DISTILL=true
    RUN_VAL=true
    if [ "$SKIP_GEN" = true ]; then RUN_GEN=false; fi
    if [ "$SKIP_EXT" = true ]; then RUN_EXT=false; fi
    if [ "$SKIP_DISTILL" = true ]; then RUN_DISTILL=false; fi
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
    BETA=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('beta', 100.0))")
    MODEL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('model_mode', 'isotropic'))")
    GEOM_TRAIN=$(python3 -c "import yaml; d=yaml.safe_load(open('$RECIPE_FILE')); print(d.get('geometry_train', d.get('geometry', 'block')))")

    CURRENT_TIME=$(date +"%Y%m%dT%H%M%S")
    EXP_FOLDER_NAME="${CURRENT_TIME}_${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${N_IP}_${BETA}_${MODEL_MODE}_${GEOM_TRAIN}"
    EXPERIMENT_DIR="$(pwd)/results/${EXP_FOLDER_NAME}"
    mkdir -p "$EXPERIMENT_DIR"
    cp "$RECIPE_FILE" "$EXPERIMENT_DIR/config.yaml"
    CONFIG_YAML="$EXPERIMENT_DIR/config.yaml"
    echo "📁 Created experiment directory: $EXPERIMENT_DIR"
fi

# Extract parameters from experiment config.yaml
get_cfg() {
    python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d$1)"
}

MODEL=$(get_cfg "['material_model_name']")
D_NOISE=$(get_cfg "['disp_noise']")
L_NOISE=$(get_cfg "['load_noise']")
ASYM=$(get_cfg "['asym_factor']")
TOP_LOAD=$(get_cfg "['target_load_true_top']")
TOP_LOAD_HOLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('target_load_holes', d.get('target_load_true_top', $TOP_LOAD)))" 2>/dev/null || echo "$TOP_LOAD")
STEPS=$(get_cfg "['n_loadsteps']")
MESH_SIZE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('mesh_size', '0.08'))" 2>/dev/null || echo "0.08")

# Extraction params
MCI_SAMPLING=$(get_cfg "['number_of_mci_sampling']")
N_IP=$(get_cfg "['n_ip']")
BETA=$(get_cfg "['beta']")
NUM_RFF=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('num_rff', 800))" 2>/dev/null || echo "800")
FIXED_NOISE=$(get_cfg "['is_fixed_reaction_force_noise']")
FIXED_IP=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('is_fixed_inducing_points', 1))" 2>/dev/null || echo "1")
EXT_ITERS=$(get_cfg "['extraction_n_iterations']")
EXT_LR=$(get_cfg "['extraction_learning_rate']")
EXT_FINAL_LR=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('extraction_final_learning_rate', d.get('extraction_learning_rate')))" 2>/dev/null || echo "$EXT_LR")
CAP_COMPRESSION=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('cap_compression', 1))" 2>/dev/null || echo "1")
TRAIN_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(*(d['train_load_steps_indices']))")
MODEL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('model_mode', 'isotropic'))" 2>/dev/null || echo "isotropic")
COVARIANCE_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('covariance_mode', 'diag'))" 2>/dev/null || echo "diag")
NORMALIZE_ELL=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('normalize_ell', 0))" 2>/dev/null || echo "0")
U_VAR_ANCHOR=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('u_var_anchor', '1e-12'))" 2>/dev/null || echo "1e-12")
KZZ_JITTER=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('kzz_jitter', '1e-8'))" 2>/dev/null || echo "1e-8")

# Distillation params
DIST_MODEL=$(get_cfg "['distilled_material_model']")
DIST_ITERS=$(get_cfg "['distillation_n_iterations']")
DEV_VOL_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('dev_vol_distillation_n_iterations', d.get('distillation_n_iterations', 5000)))" 2>/dev/null || echo "$DIST_ITERS")
ANISO_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('aniso_distillation_n_iterations', 10000))" 2>/dev/null || echo "10000")
DIST_TARGET=$(get_cfg "['distill_target']")
SAMPLE_MODE=$(get_cfg "['sample_mode']")
NUM_POINTS=$(get_cfg "['num_points']")
NUM_FUNC_SAMPLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(d.get('num_func_samples', 512))" 2>/dev/null || echo "512")
MAX_GAMMA=$(get_cfg "['max_gamma']")
DO_SENSITIVITY=$(get_cfg "['do_sensitivity']")
SOBOL_THRESHOLD=$(get_cfg "['sobol_threshold']")
SOBOL_FACTOR=$(get_cfg "['sobol_samples_factor']")
SENSITIVITY_FLAG=""
if [ "$DO_SENSITIVITY" == "0" ] || [ "$DO_SENSITIVITY" == "False" ] || [ "$DO_SENSITIVITY" == "false" ]; then
    SENSITIVITY_FLAG="--no_sensitivity"
fi

# Validation params
VAL_SAMPLES=$(get_cfg "['val_number_samples']")
VAL_LOAD_STEPS_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); print(*(d.get('val_load_steps_indices', [9])))")
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
    SEEDS_LIST="$SEEDS_CLI"
else
    SEEDS_LIST=$(python3 -c "import yaml; d=yaml.safe_load(open('$CONFIG_YAML')); s=d.get('seeds', d.get('seed_list', d.get('seed', 42))); print(*s) if isinstance(s, list) else print(s)" 2>/dev/null || echo "42")
fi

echo "========================================================================"
echo "Experiment Directory: $EXPERIMENT_DIR"
echo "Material Model:       $MODEL (Candidate: $DIST_MODEL)"
echo "Seeds to process:     $SEEDS_LIST"
echo "Stages active:        GEN=$RUN_GEN, EXT=$RUN_EXT, DISTILL=$RUN_DISTILL, VAL=$RUN_VAL"
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

    # --- STEP 1: DATA GENERATION ---
    if [ "$RUN_GEN" = true ]; then
        echo "--- Step 1: Data Generation (Seed: $SEED) ---"
        python3 dataset/synthetic/force_control/syn_force_control.py \
            --model "$MODEL" \
            --disp_noise "$D_NOISE" \
            --load_noise "$L_NOISE" \
            --target_top "$TOP_LOAD" \
            --asym "$ASYM" \
            --n_steps "$STEPS" \
            --geometry "$GEOMETRY_TRAIN" \
            --mesh_size "$MESH_SIZE" \
            --seed "$SEED" \
            $MAT_EXTRA_ARGS

        if [ "$GEOMETRY_VAL" != "$GEOMETRY_TRAIN" ]; then
            python3 dataset/synthetic/force_control/syn_force_control.py \
                --model "$MODEL" \
                --disp_noise "$D_NOISE" \
                --load_noise "$L_NOISE" \
                --target_top "$TOP_LOAD_HOLES" \
                --asym "$ASYM" \
                --n_steps "$STEPS" \
                --geometry "$GEOMETRY_VAL" \
                --mesh_size "$MESH_SIZE" \
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
        mkdir -p "$EXTRACT_DIR"
        TRAIN_DATASET_PATH="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${GEOMETRY_TRAIN}_${SEED}.npz"
        python3 extraction/train_unsupervised.py \
            --dataset_path "$TRAIN_DATASET_PATH" \
            --material_model_name "$MODEL" \
            --number_of_mci_sampling "$MCI_SAMPLING" \
            --train_load_steps_indices $TRAIN_INDICES \
            --val_load_steps_indices $VAL_LOAD_STEPS_INDICES \
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
            --seed "$SEED" \
            --batch_dir "$EXTRACT_DIR" \
            $MAT_EXTRA_ARGS
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
            $SENSITIVITY_FLAG &

        if [ "$MODEL_MODE" == "anisotropic" ] || [ "$DIST_MODEL" == "gmr_aniso" ]; then
            echo "Distilling ANISO component into $DISTILL_DIR..."
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
            --val_load_steps $VAL_LOAD_STEPS_INDICES || true

        for COMP in "dev" "vol" "aniso"; do
            if [ -d "$DISTILL_DIR/output/${COMP}_sensitivities" ] || [ -d "$DISTILL_DIR/${COMP}_sensitivities" ]; then
                python3 plots/plot_invariant_sensitivity.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                python3 plots/plot_invariant_sensitivity_3d_pairs.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                python3 plots/plot_deformation_sensitivity.py --distilled_dir "$DISTILL_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
            fi
        done
        python3 plots/plot_combined_invariant_sensitivity.py --distilled_dir "$DISTILL_DIR" 2>/dev/null || true

        echo "✅ Step 3 (Distillation for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 3 (Distillation) for Seed $SEED."
    fi

    # --- STEP 4: FEM VALIDATION ---
    if [ "$RUN_VAL" = true ]; then
        echo "--- Step 4: FEM Validation (Seed: $SEED) ---"
        mkdir -p "$VAL_DIR/block"
        mkdir -p "$VAL_DIR/holes"

        # Verify distilled samples exist
        if [ ! -f "$DISTILL_DIR/dev_flow_samples.npy" ]; then
            echo "❌ Error: Cannot run FEM validation; distilled flow samples not found at $DISTILL_DIR"
            exit 1
        fi

        VAL_DATASET_BLOCK="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${GEOMETRY_TRAIN}_${SEED}.npz"
        VAL_DATASET_HOLES="dataset/preprocessed/syn_f/${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD_HOLES}_${ASYM}_${GEOMETRY_VAL}_${SEED}.npz"

        VAL_PIDS=()
        # Launch workers for Block geometry
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
                --total_workers "$VAL_WORKERS" \
                --worker_id "$w" > "$W_LOG" 2>&1 &
            VAL_PIDS+=($!)
        done

        # Launch workers for Holes geometry
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
                --total_workers "$VAL_WORKERS" \
                --worker_id "$w" > "$W_LOG_HOLES" 2>&1 &
            VAL_PIDS+=($!)
        done

        for pid in "${VAL_PIDS[@]}"; do
            wait "$pid"
        done

        if [ "$VAL_WORKERS" -gt 1 ]; then
            echo "Merging worker outputs..."
            python3 validation/merge_fem_workers.py --folder "$VAL_DIR/block"
            python3 validation/merge_fem_workers.py --folder "$VAL_DIR/holes"
        fi

        echo "Generating UQ displacement verification plots..."
        python3 plots/uq_verification_disp.py \
            --model_path "$VAL_DIR/block" \
            --validation_load_step_indices $VAL_LOAD_STEPS_INDICES \
            --n_sample "$VAL_SAMPLES" || true

        python3 plots/uq_verification_disp.py \
            --model_path "$VAL_DIR/holes" \
            --validation_load_step_indices $VAL_LOAD_STEPS_INDICES \
            --n_sample "$VAL_SAMPLES" || true

        echo "Updating validation_metrics.json for Seed $SEED..."
        python3 validation/update_validation_metrics_json.py \
            --distilled_dir "$DISTILL_DIR" \
            --val_load_steps $VAL_LOAD_STEPS_INDICES

        # Copy validation_metrics.json into fem_validation and seed dir
        cp "$DISTILL_DIR/validation_metrics.json" "$VAL_DIR/validation_metrics.json" 2>/dev/null || true
        cp "$DISTILL_DIR/validation_metrics.json" "$SEED_DIR/validation_metrics.json" 2>/dev/null || true

        echo "✅ Step 4 (FEM Validation for Seed $SEED) completed."
    else
        echo "⏭️ Skipping Step 4 (FEM Validation) for Seed $SEED."
    fi

done

# ==============================================================================
# 4. Post-Execution Aggregation Across Seeds
# ==============================================================================
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
python3 plots/generate_experiment_summary.py --experiment_dir "$EXPERIMENT_DIR"

echo ""
echo "🎉 Pipeline finished successfully for experiment: $EXPERIMENT_DIR"
echo "📊 Cross-seed results available in: $EXPERIMENT_DIR/summary_across_seeds.md and $EXPERIMENT_DIR/plots/"

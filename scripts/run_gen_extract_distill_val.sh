#!/bin/bash
#SBATCH --partition=gpu_teaching
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=gen-ext-dist-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere

set -e

# Prevent JAX from pre-allocating all GPU memory to avoid OOM, especially during JAX-PyTorch interop in distillation
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Configuration File & Argument Parsing
YAML_INPUT="distill_pipeline_config.yaml"
SKIP_GEN=false
SKIP_EXT=false
SKIP_DISTILL=false
SKIP_VAL=false
EXTRACTION_DIR_OVERRIDE=""
DISTILLED_DIR_OVERRIDE=""
BATCH_DIR=""
SEED_OVERRIDE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-gen|--no-gen|--skip-generation)
            SKIP_GEN=true
            shift
            ;;
        --skip-ext|--no-ext|--skip-extraction)
            SKIP_EXT=true
            shift
            ;;
        --skip-distill|--no-distill|--skip-distillation)
            SKIP_DISTILL=true
            shift
            ;;
        --skip-val|--no-val|--skip-validation)
            SKIP_VAL=true
            shift
            ;;
        --extraction-dir|--extracted-dir)
            EXTRACTION_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --distilled-dir)
            DISTILLED_DIR_OVERRIDE="$2"
            shift 2
            ;;
        --seed)
            SEED_OVERRIDE="$2"
            shift 2
            ;;
        --batch-dir)
            BATCH_DIR="$2"
            shift 2
            ;;
        -*)
            echo "Warning: Unknown option: $1"
            shift
            ;;
        *)
            YAML_INPUT="$1"
            shift
            ;;
    esac
done

# Recipe shortcut resolution (e.g. 'isihara', 'gentthomas', 'nh2', 'nh4')
if [ -f "$YAML_INPUT" ]; then
    YAML_FILE="$YAML_INPUT"
elif [ -f "configs/recipes/${YAML_INPUT}.yaml" ]; then
    YAML_FILE="configs/recipes/${YAML_INPUT}.yaml"
elif [ -f "configs/recipes/${YAML_INPUT}_benchmark.yaml" ]; then
    YAML_FILE="configs/recipes/${YAML_INPUT}_benchmark.yaml"
else
    echo "❌ Error: Configuration file or recipe '$YAML_INPUT' not found!"
    echo "Available recipes in configs/recipes/:"
    ls -1 configs/recipes/*.yaml 2>/dev/null || echo "  (none)"
    exit 1
fi

YAML_SKIP_GEN=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('skip_generation', False) or d.get('run_generation', True) is False or str(d.get('run_generation', 1)) == '0' or str(d.get('skip_generation', 0)) == '1')" 2>/dev/null || echo "False")
if [ "$YAML_SKIP_GEN" == "True" ]; then
    SKIP_GEN=true
fi

YAML_SKIP_EXT=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('skip_extraction', False) or d.get('run_extraction', True) is False or str(d.get('run_extraction', 1)) == '0' or str(d.get('skip_extraction', 0)) == '1')" 2>/dev/null || echo "False")
if [ "$YAML_SKIP_EXT" == "True" ]; then
    SKIP_EXT=true
fi

YAML_SKIP_DISTILL=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('skip_distillation', False) or d.get('run_distillation', True) is False or str(d.get('run_distillation', 1)) == '0' or str(d.get('skip_distillation', 0)) == '1')" 2>/dev/null || echo "False")
if [ "$YAML_SKIP_DISTILL" == "True" ]; then
    SKIP_DISTILL=true
fi

YAML_SKIP=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('skip_validation', False) or d.get('run_validation', True) is False or str(d.get('run_validation', 1)) == '0' or str(d.get('skip_validation', 0)) == '1')" 2>/dev/null || echo "False")
if [ "$YAML_SKIP" == "True" ]; then
    SKIP_VAL=true
fi

echo "========================================================================"
echo "=== Loading configuration from $YAML_FILE ==="
echo "========================================================================"

# Helper function to extract scalar values from YAML
get_yaml() {
  python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d$1)"
}

# 1. Dataset Generation Config
MODEL=$(get_yaml "['material_model_name']")
D_NOISE=$(get_yaml "['disp_noise']")
L_NOISE=$(get_yaml "['load_noise']")
ASYM=$(get_yaml "['asym_factor']")
TOP_LOAD=$(get_yaml "['target_load_true_top']")
TOP_LOAD_HOLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('target_load_holes', d.get('target_load_true_top')))" 2>/dev/null || echo "$TOP_LOAD")
STEPS=$(get_yaml "['n_loadsteps']")
MESH_SIZE=$(get_yaml "['mesh_size']")
if [ "$MESH_SIZE" == "None" ] || [ "$MESH_SIZE" == "" ] || [ "$MESH_SIZE" == "null" ]; then
    MESH_SIZE="0.08" # default fallback
fi

# 2. UGP Extraction Config
MCI_SAMPLING=$(get_yaml "['number_of_mci_sampling']")
N_IP=$(get_yaml "['n_ip']")
BETA=$(get_yaml "['beta']")
NUM_RFF=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('num_rff', 200))" 2>/dev/null || echo "200")
FIXED_NOISE=$(get_yaml "['is_fixed_reaction_force_noise']")
FIXED_IP=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('is_fixed_inducing_points', 1))" 2>/dev/null || echo "1")
PRIOR_MEAN=$(get_yaml "['is_include_prior_mean']")
EXT_ITERS=$(get_yaml "['extraction_n_iterations']")
EXT_LR=$(get_yaml "['extraction_learning_rate']")
EXT_FINAL_LR=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('extraction_final_learning_rate', d.get('extraction_learning_rate')))" 2>/dev/null || echo "$EXT_LR")
CAP_COMPRESSION=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('cap_compression', 1))" 2>/dev/null || echo "1")
TRAIN_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(*(d['train_load_steps_indices']))")
MODEL_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('model_mode', 'isotropic'))" 2>/dev/null || echo "isotropic")
COVARIANCE_MODE=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('covariance_mode', 'diag'))" 2>/dev/null || echo "diag")
if [ -n "$SEED_OVERRIDE" ]; then
    SEEDS_LIST="$SEED_OVERRIDE"
else
    SEEDS_LIST=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); s=d.get('seeds', d.get('seed_list', d.get('seed', 42))); print(*s) if isinstance(s, list) else print(s)" 2>/dev/null || echo "42")
fi

# 3. Distillation Config
DIST_MODEL=$(get_yaml "['distilled_material_model']")
DIST_ITERS=$(get_yaml "['distillation_n_iterations']")
DEV_VOL_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('dev_vol_distillation_n_iterations', d.get('distillation_n_iterations', 5000)))" 2>/dev/null || echo "$DIST_ITERS")
ANISO_DIST_ITERS=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('aniso_distillation_n_iterations', 10000))" 2>/dev/null || echo "10000")
DIST_TARGET=$(get_yaml "['distill_target']")
SAMPLE_MODE=$(get_yaml "['sample_mode']")
NUM_POINTS=$(get_yaml "['num_points']")
NUM_FUNC_SAMPLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('num_func_samples', d.get('num_functional_samples', 512)))" 2>/dev/null || echo "512")
MAX_GAMMA=$(get_yaml "['max_gamma']")
DO_SENSITIVITY=$(get_yaml "['do_sensitivity']")
SOBOL_THRESHOLD=$(get_yaml "['sobol_threshold']")
SOBOL_FACTOR=$(get_yaml "['sobol_samples_factor']")

SENSITIVITY_FLAG=""
if [ "$DO_SENSITIVITY" == "0" ] || [ "$DO_SENSITIVITY" == "False" ] || [ "$DO_SENSITIVITY" == "false" ]; then
    SENSITIVITY_FLAG="--no_sensitivity"
fi

# 4. Validation Config
VAL_SAMPLES=$(get_yaml "['val_number_samples']")
VAL_LOAD_STEPS_INDICES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(*(d.get('val_load_steps_indices', [9])))")

# 5. Geometry
GEOMETRY_TRAIN=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('geometry_train', d.get('geometry', 'block')))" 2>/dev/null || echo "block")
GEOMETRY_VAL=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('geometry_val', 'holes'))" 2>/dev/null || echo "holes")

# Material Parameter Overrides
ANGLES=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); a=d.get('material_params', {}).get('angles', d.get('angles', None)); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
DEV_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); a=d.get('material_params', {}).get('dev_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
VOL_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); a=d.get('material_params', {}).get('vol_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)
ANISO_PARAMS=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); a=d.get('material_params', {}).get('aniso_params', None); print(*(a if isinstance(a, list) else [a])) if a is not None else None" 2>/dev/null || true)

MAT_EXTRA_ARGS=""
if [ -n "$ANGLES" ]; then
    MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --angles $ANGLES"
fi
if [ -n "$DEV_PARAMS" ]; then
    MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --dev_params $DEV_PARAMS"
fi
if [ -n "$VOL_PARAMS" ]; then
    MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --vol_params $VOL_PARAMS"
fi
if [ -n "$ANISO_PARAMS" ]; then
    MAT_EXTRA_ARGS="$MAT_EXTRA_ARGS --aniso_params $ANISO_PARAMS"
fi

CURRENT_TIME=$(date +"%Y%m%dT%H%M%S")

for SEED in $SEEDS_LIST; do
    echo ""
    echo "========================================================================"
    echo "=== Running Extraction & Distillation Pipeline for SEED = $SEED ==="
    echo "========================================================================"

    FOLDER_NAME="${CURRENT_TIME}_${MODEL}_${D_NOISE}_${L_NOISE}_${TOP_LOAD}_${ASYM}_${N_IP}_${BETA}_${FIXED_NOISE}_fip${FIXED_IP}_${MODEL_MODE}_${GEOMETRY_TRAIN}_${SEED}"

    if [ "$SKIP_GEN" == "true" ]; then
        echo "⏭️ Skipping Step 1 (Sequential Data Generation) as requested."
    else
        echo "--- Step 1: Sequential Data Generation ($MODEL) on Geometry: $GEOMETRY_TRAIN (Seed: $SEED) ---"
        
        if [ "$GEOMETRY_TRAIN" == "holes" ]; then
            GEN_SCRIPT="dataset/synthetic/force_control/syn_force_control_holes.py"
        else
            GEN_SCRIPT="dataset/synthetic/force_control/syn_force_control.py"
        fi

        python3 $GEN_SCRIPT \
            --model "$MODEL" \
            --disp_noise "$D_NOISE" \
            --load_noise "$L_NOISE" \
            --target_top "$TOP_LOAD" \
            --asym "$ASYM" \
            --n_steps "$STEPS" \
            --geometry "$GEOMETRY_TRAIN" \
            --mesh_size "$MESH_SIZE" \
            --seed "$SEED" \
            $MAT_EXTRA_ARGS || { echo "❌ Step 1 (Data Generation) failed for seed $SEED. Skipping to next seed."; continue; }
        echo "✅ Step 1 (Data Generation) completed."
    fi

    EXTRACT_SEED_DIR="extraction/extracted_models/${FOLDER_NAME}"
    DISTILL_SEED_DIR="distillation/distilled_models/${FOLDER_NAME}"

    if [ "$SKIP_EXT" == "true" ]; then
        echo "⏭️ Skipping Step 2 (UGP Extraction Training) as requested."
    else
        echo "--- Step 2: UGP Extraction Training ($MODEL, $EXT_ITERS iterations, Seed: $SEED) ---"
        mkdir -p "$EXTRACT_SEED_DIR"
        cp "$YAML_FILE" "$EXTRACT_SEED_DIR/recipe_config.yaml" 2>/dev/null || true

        python3 extraction/train_unsupervised.py \
            --material_model_name "$MODEL" \
            --number_of_mci_sampling "$MCI_SAMPLING" \
            --train_load_steps_indices $TRAIN_INDICES \
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
            --seed "$SEED" \
            --batch_dir "$EXTRACT_SEED_DIR" \
            $MAT_EXTRA_ARGS
        echo "✅ Step 2 (Extraction for Seed $SEED) completed."
    fi

    if [ "$SKIP_DISTILL" == "true" ] && [ "$SKIP_VAL" == "true" ]; then
        echo "🎉 Pipeline execution finished for Seed $SEED (Distillation & Validation skipped)."
        continue
    fi

    # Find extraction directory matching this seed or use override/fallback
    if [ -n "$EXTRACTION_DIR_OVERRIDE" ] && [ -d "$EXTRACTION_DIR_OVERRIDE" ]; then
        SAVED_DIR="$EXTRACTION_DIR_OVERRIDE"
    elif [ -d "$EXTRACT_SEED_DIR" ] && [ -f "${EXTRACT_SEED_DIR}/best_params.npy" ]; then
        SAVED_DIR="$EXTRACT_SEED_DIR"
    else
        SAVED_DIR=$(ls -td extraction/extracted_models/*${MODEL}_${D_NOISE}_${L_NOISE}*${GEOMETRY_TRAIN}*_${SEED} extraction/extracted_models/*${MODEL}_${D_NOISE}_${L_NOISE}*${GEOMETRY_TRAIN}* 2>/dev/null | head -1)
    fi

    if [ -z "$SAVED_DIR" ] || [ ! -d "$SAVED_DIR" ]; then
        if [ "$SKIP_EXT" == "true" ]; then
            echo "❌ Error: Step 2 was skipped and no extracted model found for $MODEL (Seed $SEED)!"
        else
            echo "❌ Error: Could not locate extracted model directory for $MODEL (Seed $SEED)!"
        fi
        exit 1
    fi
    echo "ℹ️ Using extracted model at: $SAVED_DIR"


    if [ "$SKIP_DISTILL" == "true" ]; then
        echo "⏭️ Skipping Step 3 (Distillation) as requested."
    else
        echo "--- Step 3: Distillation ($DIST_MODEL candidate expression, DEV/VOL: $DEV_VOL_DIST_ITERS, ANISO: $ANISO_DIST_ITERS iterations, Seed: $SEED) ---"
        if [ -n "$DISTILLED_DIR_OVERRIDE" ]; then
            SHARED_OUT_DIR="$DISTILLED_DIR_OVERRIDE"
        else
            SHARED_OUT_DIR="$DISTILL_SEED_DIR"
        fi
        mkdir -p "$SHARED_OUT_DIR"
        cp "$YAML_FILE" "$SHARED_OUT_DIR/recipe_config.yaml" 2>/dev/null || true

        if [ "$DIST_TARGET" == "sef_split" ]; then
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
            if [ ! -f "${SAVED_DIR}/${EXPORT_SUB}/mean_dev.npy" ]; then
                echo "Exporting GP to PyTorch format before parallel distillation..."
                python3 distillation/export_gp_to_pytorch.py \
                    --saved_model_dir "$SAVED_DIR" \
                    --sample_mode "$SAMPLE_MODE" \
                    --num_points "$NUM_POINTS" \
                    --max_gamma "$MAX_GAMMA" \
                    --distill_target "$DIST_TARGET" \
                    --export_subfolder "$EXPORT_SUB"
            fi

            echo "Distilling DEV component into $SHARED_OUT_DIR ($DEV_VOL_DIST_ITERS iters)..."
            python3 distillation/distill_uqmodeldisc.py \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --n_iterations "$DEV_VOL_DIST_ITERS" \
                --distill_target "$DIST_TARGET" \
                --component "dev" \
                --override_out_dir "$SHARED_OUT_DIR" \
                --sample_mode "$SAMPLE_MODE" \
                --num_points "$NUM_POINTS" \
                --num_func_samples "$NUM_FUNC_SAMPLES" \
                --max_gamma "$MAX_GAMMA" \
                --sobol_threshold "$SOBOL_THRESHOLD" \
                --sobol_samples_factor "$SOBOL_FACTOR" \
                --seed "$SEED" \
                $SENSITIVITY_FLAG &
                
            echo "Distilling VOL component into $SHARED_OUT_DIR ($DEV_VOL_DIST_ITERS iters)..."
            python3 distillation/distill_uqmodeldisc.py \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --n_iterations "$DEV_VOL_DIST_ITERS" \
                --distill_target "$DIST_TARGET" \
                --component "vol" \
                --override_out_dir "$SHARED_OUT_DIR" \
                --sample_mode "$SAMPLE_MODE" \
                --num_points "$NUM_POINTS" \
                --num_func_samples "$NUM_FUNC_SAMPLES" \
                --max_gamma "$MAX_GAMMA" \
                --sobol_threshold "$SOBOL_THRESHOLD" \
                --sobol_samples_factor "$SOBOL_FACTOR" \
                --seed "$SEED" \
                $SENSITIVITY_FLAG &

            if [ "$MODEL_MODE" == "anisotropic" ] || [ "$MODEL_MODE" == "aniso_unk_fiber" ] || [ "$MODEL_MODE" == "aniso_fixed_fiber" ] || [ "$DIST_MODEL" == "gmr_aniso" ]; then
                echo "Distilling ANISO component into $SHARED_OUT_DIR ($ANISO_DIST_ITERS iters)..."
                python3 distillation/distill_uqmodeldisc.py \
                    --saved_model_dir "$SAVED_DIR" \
                    --material_model "$DIST_MODEL" \
                    --n_iterations "$ANISO_DIST_ITERS" \
                    --distill_target "$DIST_TARGET" \
                    --component "aniso" \
                    --override_out_dir "$SHARED_OUT_DIR" \
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
            
            echo "Generating split validation plots..."
            python3 plots/plot_distilled_validation.py \
                --distilled_dir "$SHARED_OUT_DIR" \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --distill_target "$DIST_TARGET"
                
            python3 plots/plot_split_summary.py \
                --distilled_dir "$SHARED_OUT_DIR" \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --distill_target "$DIST_TARGET"
                
            python3 plots/plot_distilled_r2_energy.py \
                --distilled_dir "$SHARED_OUT_DIR" \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --distill_target "$DIST_TARGET"

            echo "Generating invariant & deformation sensitivity plots..."
            for COMP in "dev" "vol" "aniso"; do
                if [ -d "$SHARED_OUT_DIR/output/${COMP}_sensitivities" ] || [ -d "$SHARED_OUT_DIR/${COMP}_sensitivities" ]; then
                    python3 plots/plot_invariant_sensitivity.py --distilled_dir "$SHARED_OUT_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                    python3 plots/plot_invariant_sensitivity_3d_pairs.py --distilled_dir "$SHARED_OUT_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                    python3 plots/plot_deformation_sensitivity.py --distilled_dir "$SHARED_OUT_DIR" --component "$COMP" --distill_target "$DIST_TARGET" 2>/dev/null || true
                fi
            done
        else
            python3 distillation/distill_uqmodeldisc.py \
                --saved_model_dir "$SAVED_DIR" \
                --material_model "$DIST_MODEL" \
                --n_iterations "$DIST_ITERS" \
                --distill_target "$DIST_TARGET" \
                --override_out_dir "$SHARED_OUT_DIR" \
                --sample_mode "$SAMPLE_MODE" \
                --num_points "$NUM_POINTS" \
                --num_func_samples "$NUM_FUNC_SAMPLES" \
                --max_gamma "$MAX_GAMMA" \
                --sobol_threshold "$SOBOL_THRESHOLD" \
                --sobol_samples_factor "$SOBOL_FACTOR" \
                --seed "$SEED" \
                $SENSITIVITY_FLAG

            if [ -d "$SHARED_OUT_DIR/output/sensitivities" ] || [ -d "$SHARED_OUT_DIR/sensitivities" ]; then
                python3 plots/plot_invariant_sensitivity.py --distilled_dir "$SHARED_OUT_DIR" 2>/dev/null || true
                python3 plots/plot_invariant_sensitivity_3d_pairs.py --distilled_dir "$SHARED_OUT_DIR" 2>/dev/null || true
                python3 plots/plot_deformation_sensitivity.py --distilled_dir "$SHARED_OUT_DIR" 2>/dev/null || true
            fi
        fi
        echo "✅ Step 3 (Distillation for Seed $SEED) completed."
    fi


    if [ "$SKIP_VAL" == "true" ]; then
        echo "⏭️ Skipping Step 4 (Stochastic Forward Sampling Validation) for Seed $SEED."
        continue
    fi

    if [ -n "$DISTILLED_DIR_OVERRIDE" ] && [ -d "$DISTILLED_DIR_OVERRIDE" ]; then
        if [ -d "${DISTILLED_DIR_OVERRIDE}/${SEED}/distillation" ]; then
            DISTILLED_DIR="${DISTILLED_DIR_OVERRIDE}/${SEED}/distillation"
        elif [ -d "${DISTILLED_DIR_OVERRIDE}/distillation" ]; then
            DISTILLED_DIR="${DISTILLED_DIR_OVERRIDE}/distillation"
        else
            DISTILLED_DIR="$DISTILLED_DIR_OVERRIDE"
        fi
    else
        DISTILLED_DIR="$DISTILL_SEED_DIR"
        if [ ! -d "$DISTILLED_DIR" ] || [ ! -f "${DISTILLED_DIR}/dev_flow_samples.npy" ]; then
            for dir in $(ls -td results/${MODEL}/*${MODEL}*d${D_NOISE}*l${L_NOISE}*${GEOMETRY_TRAIN}*/${SEED}/distillation distillation/distilled_models/*_${MODEL}* 2>/dev/null); do
                if [ -f "${dir}/dev_flow_samples.npy" ]; then
                    DISTILLED_DIR="$dir"
                    break
                fi
            done
        fi
    fi
    echo "ℹ️ Using distilled model at: $DISTILLED_DIR"

    if [ "$SKIP_DISTILL" != "true" ]; then
        echo "Generating combined summary plot..."
        python3 plots/plot_combined_summary.py \
            --distilled_dir "$DISTILLED_DIR" \
            --sobol_threshold "$SOBOL_THRESHOLD"
            
        echo "Generating split summary plots (Energy / Params)..."
        python3 plots/plot_split_summary.py \
            --distilled_dir "$DISTILLED_DIR"
    fi

    echo "--- Step 4: Parallel FEM Validation on $GEOMETRY_TRAIN & $GEOMETRY_VAL (Seed: $SEED) ---"

    export OMP_NUM_THREADS=4 
    export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

    mkdir -p "${DISTILLED_DIR}/fem_validation"
    mkdir -p "${DISTILLED_DIR}/fem_validation_${GEOMETRY_VAL}"

    VAL_LOG="${DISTILLED_DIR}/validation_${MODEL}_seed${SEED}_${GEOMETRY_TRAIN}.log"
    VAL_LOG_HOLES="${DISTILLED_DIR}/validation_${MODEL}_seed${SEED}_${GEOMETRY_VAL}.log"

    python3 validation/forward_fem_distilled_piola_sample.py \
        --model_path "$MODEL_PATH" \
        --distilled_dir "$DISTILLED_DIR" \
        --material_model "$DIST_MODEL" \
        --n_sample "$VAL_SAMPLES" \
        --subfolder "fem_validation" \
        --geometry "$GEOMETRY_TRAIN" \
        --target_load "$TOP_LOAD" > "$VAL_LOG" 2>&1 &
    PID1=$!

    python3 validation/forward_fem_distilled_piola_sample.py \
        --model_path "$MODEL_PATH" \
        --distilled_dir "$DISTILLED_DIR" \
        --material_model "$DIST_MODEL" \
        --n_sample "$VAL_SAMPLES" \
        --subfolder "fem_validation_${GEOMETRY_VAL}" \
        --geometry "$GEOMETRY_VAL" \
        --target_load "$TOP_LOAD_HOLES" > "$VAL_LOG_HOLES" 2>&1 &
    PID2=$!

    echo "Processes started: Validation $GEOMETRY_TRAIN (PID: $PID1) and Validation $GEOMETRY_VAL (PID: $PID2)"
    wait $PID1 || true
    EXIT_CODE1=$?
    wait $PID2 || true
    EXIT_CODE2=$?

    if [ $EXIT_CODE1 -ne 0 ] || [ $EXIT_CODE2 -ne 0 ]; then
        echo "❌ Step 4 Validation failed for seed $SEED."
    else
        echo "✅ Validation on $GEOMETRY_TRAIN and $GEOMETRY_VAL finished."
    fi

    echo "Generating UQ verification displacement plots..."
    python3 plots/uq_verification_disp.py \
        --model_path "$DISTILLED_DIR" \
        --validation_load_step_indices $VAL_LOAD_STEPS_INDICES \
        --n_sample "$VAL_SAMPLES" \
        --subfolder "fem_validation"

    python3 plots/uq_verification_disp.py \
        --model_path "$DISTILLED_DIR" \
        --validation_load_step_indices $VAL_LOAD_STEPS_INDICES \
        --n_sample "$VAL_SAMPLES" \
        --subfolder "fem_validation_${GEOMETRY_VAL}"
done

echo ""
echo "🎉 Multi-Seed Sequential Pipeline Execution Completed Successfully across Seeds: $SEEDS_LIST!"

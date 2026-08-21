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
        --extraction-dir)
            EXTRACTION_DIR_OVERRIDE="$2"
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
    SEED="$SEED_OVERRIDE"
else
    SEED=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(d.get('seed', 42))" 2>/dev/null || echo "42")
fi


# 3. Distillation Config
DIST_MODEL=$(get_yaml "['distilled_material_model']")
DIST_ITERS=$(get_yaml "['distillation_n_iterations']")
DIST_TARGET=$(get_yaml "['distill_target']")
SAMPLE_MODE=$(get_yaml "['sample_mode']")
NUM_POINTS=$(get_yaml "['num_points']")
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
GEOMETRY=$(get_yaml "['geometry']")
if [ "$GEOMETRY" == "" ] || [ "$GEOMETRY" == "null" ]; then
    GEOMETRY="block"
fi


if [ "$SKIP_GEN" == "true" ]; then
    echo "⏭️ Skipping Step 1 (Sequential Data Generation) as requested."
else
    echo "--- Step 1: Sequential Data Generation ($MODEL) on Geometry: $GEOMETRY ---"
    
    if [ "$GEOMETRY" == "holes" ]; then
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
        --geometry "$GEOMETRY" \
        --mesh_size "$MESH_SIZE"
    echo "✅ Step 1 (Data Generation) completed."
fi


if [ "$SKIP_EXT" == "true" ]; then
    echo "⏭️ Skipping Step 2 (UGP Extraction Training) as requested."
else
    echo "--- Step 2: UGP Extraction Training ($MODEL, $EXT_ITERS iterations) ---"
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
        --geometry "$GEOMETRY" \
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
        --batch_dir "$BATCH_DIR"
    echo "✅ Step 2 (Extraction) completed."
fi

if [ "$SKIP_DISTILL" == "true" ] && [ "$SKIP_VAL" == "true" ]; then
    echo "🎉 Pipeline execution finished (Distillation & Validation skipped as requested)."
    exit 0
fi

# Find the most recently created extraction directory matching this recipe and geometry or use override
if [ -n "$EXTRACTION_DIR_OVERRIDE" ]; then
    SAVED_DIR="$EXTRACTION_DIR_OVERRIDE"
else
    SAVED_DIR=$(ls -td extraction/extracted_models/*${MODEL}_${D_NOISE}_${L_NOISE}*${GEOMETRY}* 2>/dev/null | head -1)
fi
if [ -z "$SAVED_DIR" ] || [ ! -d "$SAVED_DIR" ]; then
    if [ "$SKIP_EXT" == "true" ]; then
        echo "❌ Error: Step 2 was skipped and no extracted model found in extraction/extracted_models/ for $MODEL!"
    else
        echo "❌ Error: Could not locate extracted model directory in extraction/extracted_models/"
    fi
    exit 1
fi
MODEL_PATH=$(basename "$SAVED_DIR")
echo "ℹ️ Using extracted model at: $SAVED_DIR (Folder: $MODEL_PATH)"


if [ "$SKIP_DISTILL" == "true" ]; then
    echo "⏭️ Skipping Step 3 (Distillation) as requested."
else
    echo "--- Step 3: Distillation ($DIST_MODEL candidate expression, $DIST_ITERS iterations) ---"
    if [ "$DIST_TARGET" == "sef_split" ]; then
        CURRENT_TIME=$(date +"%Y%m%dT%H%M%S")
        SHARED_OUT_DIR="distillation/distilled_models/${CURRENT_TIME}_${MODEL}_${DIST_MODEL}_${SAMPLE_MODE}_${DIST_TARGET}_uqmodeldisc"
        mkdir -p "$SHARED_OUT_DIR"
        
        echo "Distilling DEV component into $SHARED_OUT_DIR..."
        python3 distillation/distill_uqmodeldisc.py \
            --saved_model_dir "$SAVED_DIR" \
            --material_model "$DIST_MODEL" \
            --n_iterations "$DIST_ITERS" \
            --distill_target "$DIST_TARGET" \
            --component "dev" \
            --override_out_dir "$SHARED_OUT_DIR" \
            --sample_mode "$SAMPLE_MODE" \
            --num_points "$NUM_POINTS" \
            --max_gamma "$MAX_GAMMA" \
            --sobol_threshold "$SOBOL_THRESHOLD" \
            --sobol_samples_factor "$SOBOL_FACTOR" \
            --seed "$SEED" \
            $SENSITIVITY_FLAG &
            
        echo "Distilling VOL component into $SHARED_OUT_DIR..."
        python3 distillation/distill_uqmodeldisc.py \
            --saved_model_dir "$SAVED_DIR" \
            --material_model "$DIST_MODEL" \
            --n_iterations "$DIST_ITERS" \
            --distill_target "$DIST_TARGET" \
            --component "vol" \
            --override_out_dir "$SHARED_OUT_DIR" \
            --sample_mode "$SAMPLE_MODE" \
            --num_points "$NUM_POINTS" \
            --max_gamma "$MAX_GAMMA" \
            --sobol_threshold "$SOBOL_THRESHOLD" \
            --sobol_samples_factor "$SOBOL_FACTOR" \
            --seed "$SEED" \
            $SENSITIVITY_FLAG &
            
        wait
        
        echo "Generating split validation plots..."
        python3 plots/plot_distilled_validation.py \
            --distilled_dir "$SHARED_OUT_DIR" \
            --saved_model_dir "$SAVED_DIR" \
            --material_model "$DIST_MODEL" \
            --distill_target "$DIST_TARGET"
            
    else
        python3 distillation/distill_uqmodeldisc.py \
            --saved_model_dir "$SAVED_DIR" \
            --material_model "$DIST_MODEL" \
            --n_iterations "$DIST_ITERS" \
            --distill_target "$DIST_TARGET" \
            --sample_mode "$SAMPLE_MODE" \
            --num_points "$NUM_POINTS" \
            --max_gamma "$MAX_GAMMA" \
            --sobol_threshold "$SOBOL_THRESHOLD" \
            --sobol_samples_factor "$SOBOL_FACTOR" \
            --seed "$SEED" \
            $SENSITIVITY_FLAG
    fi
    echo "✅ Step 3 (Distillation) completed."
fi


if [ "$SKIP_VAL" == "true" ]; then
    echo "⏭️ Skipping Step 4 (Stochastic Forward Sampling Validation) as requested."
    echo "🎉 Pipeline execution finished successfully!"
    exit 0
fi

DISTILLED_DIR=$(ls -td distillation/distilled_models/*_${MODEL}* 2>/dev/null | head -n 1 || echo "")
if [ -z "$DISTILLED_DIR" ] || [ ! -d "$DISTILLED_DIR" ]; then
    echo "❌ Error: Could not locate distilled model directory in distillation/distilled_models/ for validation!"
    exit 1
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

echo "--- Step 4: Stochastic Forward Sampling with Distilled Model ---"

# Restrict each process to a sensible number of threads and avoid JAX GPU OOM in parallel execution
export OMP_NUM_THREADS=4 
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45

# Start Validation in the background
python3 validation/forward_fem_distilled_piola_sample.py \
    --model_path "$MODEL_PATH" \
    --distilled_dir "$DISTILLED_DIR" \
    --material_model "$DIST_MODEL" \
    --n_sample "$VAL_SAMPLES" > "validation_distilled_${MODEL}.log" 2>&1 &
PID_VAL=$!

# Start Analysis in the background
python3 validation/forward_fem_distilled_piola_traction_sample.py \
    --model_path "$MODEL_PATH" \
    --distilled_dir "$DISTILLED_DIR" \
    --material_model "$DIST_MODEL" \
    --n_sample "$VAL_SAMPLES" > "analysis_distilled_${MODEL}.log" 2>&1 &
PID_ANA=$!

echo "Processes started: Validation (PID: $PID_VAL) and Analysis (PID: $PID_ANA)"
echo "Logs are being written to validation_distilled_${MODEL}.log and analysis_distilled_${MODEL}.log..."

# Wait for both background processes to finish
wait $PID_VAL $PID_ANA

cp "validation_distilled_${MODEL}.log" validation_distilled.log 2>/dev/null || true
cp "analysis_distilled_${MODEL}.log" analysis_distilled.log 2>/dev/null || true
cp "validation_distilled_${MODEL}.log" "$DISTILLED_DIR/" 2>/dev/null || true
cp "analysis_distilled_${MODEL}.log" "$DISTILLED_DIR/" 2>/dev/null || true

echo "Generating UQ verification displacement plots..."
python3 plots/uq_verification_disp.py \
    --model_path "$(basename "$DISTILLED_DIR")" \
    --validation_load_step_indices $VAL_LOAD_STEPS_INDICES \
    --n_sample "$VAL_SAMPLES"

echo "🎉 Full Pipeline (Datagen -> Extraction -> Distillation -> Distilled Forward FEM) Completed Successfully!"

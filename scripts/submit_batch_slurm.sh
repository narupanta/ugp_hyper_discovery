#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <recipe> [num_seeds]"
    echo "Example: $0 isihara 20"
    exit 1
fi

RECIPE=$1
NUM_SEEDS=${2:-20}

echo "Submitting batch extraction array for recipe '$RECIPE' with $NUM_SEEDS seeds..."

BATCH_TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# Parse config string from yaml
YAML_FILE="configs/recipes/${RECIPE}.yaml"
if [ ! -f "$YAML_FILE" ]; then
    YAML_FILE="configs/recipes/${RECIPE}_benchmark.yaml"
fi
CONFIG_STR=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(f\"{d.get('material_model_name', 'unk')}_{d.get('disp_noise', 'unk')}_{d.get('load_noise', 'unk')}_{d.get('target_load_true_top', 'unk')}_{d.get('asym_factor', 'unk')}_{d.get('n_ip', 'unk')}_{d.get('beta', 'unk')}_{d.get('is_fixed_reaction_force_noise', 'unk')}_fip{d.get('is_fixed_inducing_points', 1)}_{d.get('model_mode', 'isotropic')}_{d.get('geometry', 'block')}\")" 2>/dev/null || echo "$RECIPE")

BATCH_DIR="extraction/extracted_models/${BATCH_TIMESTAMP}_${CONFIG_STR}_batch"
echo "Models will be saved to $BATCH_DIR"

# Ensure slurm_logs directory exists
mkdir -p slurm_logs

# 1. Submit the array job
echo "Submitting array extraction job (1-$NUM_SEEDS)..."
ARRAY_JOB_ID=$(sbatch --parsable --array=1-$NUM_SEEDS scripts/slurm_extract_job.sbatch "$RECIPE" "$BATCH_DIR")

if [ -z "$ARRAY_JOB_ID" ]; then
    echo "❌ Failed to submit array job!"
    exit 1
fi
echo "✅ Array Job submitted with ID: $ARRAY_JOB_ID"

# 2. Submit the dependent summarization job
echo "Submitting summarization job dependent on array job completion..."
SUM_JOB_ID=$(sbatch --parsable --dependency=afterok:$ARRAY_JOB_ID scripts/slurm_summarize_job.sbatch "$BATCH_DIR")

if [ -z "$SUM_JOB_ID" ]; then
    echo "⚠️ Failed to submit summary job. You may need to run it manually later."
else
    echo "✅ Summarization Job submitted with ID: $SUM_JOB_ID"
    echo "Everything is queued! You can check the status using 'squeue -u \$USER'"
fi

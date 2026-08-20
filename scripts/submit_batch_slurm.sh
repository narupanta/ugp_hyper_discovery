#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <recipe> [num_seeds] [parallel_jobs]"
    echo "Example: $0 isihara 20 5"
    exit 1
fi

RECIPE=$1
NUM_SEEDS=${2:-20}
PARALLEL_JOBS=${3:-5}

echo "Submitting batch extraction array for recipe '$RECIPE' with $NUM_SEEDS seeds ($PARALLEL_JOBS concurrently)..."

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

# Submit the single job
echo "Submitting single SLURM job..."
JOB_ID=$(sbatch --parsable scripts/slurm_extract_job.sbatch "$RECIPE" "$BATCH_DIR" "$NUM_SEEDS" "$PARALLEL_JOBS")

if [ -z "$JOB_ID" ]; then
    echo "❌ Failed to submit job!"
    exit 1
fi
echo "✅ Job submitted with ID: $JOB_ID"
echo "Everything is queued! You can check the status using 'squeue -u \$USER'"

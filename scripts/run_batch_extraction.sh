#!/bin/bash
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <recipe> [num_seeds] [parallel_jobs]"
    echo "Example: $0 isihara 20 10"
    exit 1
fi

RECIPE=$1
NUM_SEEDS=${2:-20}
PARALLEL_JOBS=${3:-10}

echo "Running batch extraction for recipe '$RECIPE' with $NUM_SEEDS seeds ($PARALLEL_JOBS in parallel)..."

BATCH_TIMESTAMP=$(date +"%Y%m%dT%H%M%S")

# Parse config string from yaml
YAML_FILE="configs/recipes/${RECIPE}.yaml"
if [ ! -f "$YAML_FILE" ]; then
    YAML_FILE="configs/recipes/${RECIPE}_benchmark.yaml"
fi
CONFIG_STR=$(python3 -c "import yaml; d=yaml.safe_load(open('$YAML_FILE')); print(f\"{d.get('material_model_name', 'unk')}_{d.get('disp_noise', 'unk')}_{d.get('load_noise', 'unk')}_{d.get('target_load_true_top', 'unk')}_{d.get('asym_factor', 'unk')}_{d.get('n_ip', 'unk')}_{d.get('beta', 'unk')}_{d.get('is_fixed_reaction_force_noise', 'unk')}_fip{d.get('is_fixed_inducing_points', 1)}_{d.get('model_mode', 'isotropic')}_{d.get('geometry', 'block')}\")" 2>/dev/null || echo "$RECIPE")

BATCH_DIR="extraction/extracted_models/${BATCH_TIMESTAMP}_${CONFIG_STR}_batch"
echo "Saving models to $BATCH_DIR"

seq 1 $NUM_SEEDS | xargs -n 1 -P $PARALLEL_JOBS -I {} bash scripts/run_gen_extract_distill_val.sh $RECIPE --skip-gen --skip-distill --skip-val --batch-dir "$BATCH_DIR" --seed {}

echo "✅ Batch extraction finished! Models are saved in $BATCH_DIR"

echo "📊 Generating CSV summary..."
python3 scripts/summarize_batch_extraction.py "$BATCH_DIR"

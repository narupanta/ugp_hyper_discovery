#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=extract_batch
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de
#SBATCH --output=slurm_logs/batch_%j.out
#SBATCH --error=slurm_logs/batch_%j.err

RECIPE=$1
BATCH_DIR=$2
NUM_SEEDS=$3
PARALLEL_JOBS=$4

# Restrict JAX GPU memory so multiple parallel processes don't crash the GPU with OOM
# We allocate an even fraction of the GPU for each parallel worker
export XLA_PYTHON_CLIENT_MEM_FRACTION=$(awk "BEGIN {print 0.95 / $PARALLEL_JOBS}")

SCRIPT_PATH="./scripts/run_gen_extract_distill_val.sh"

echo "Launching batch extraction via Singularity: $SCRIPT_PATH"
echo "Running $NUM_SEEDS seeds with $PARALLEL_JOBS in parallel on a single SLURM node..."

# Run xargs via singularity
seq 1 $NUM_SEEDS | xargs -n 1 -P $PARALLEL_JOBS -I {} singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif "$SCRIPT_PATH" "$RECIPE" --skip-gen --skip-distill --skip-val --batch-dir "$BATCH_DIR" --seed {}

echo "✅ Batch extraction finished! Models are saved in $BATCH_DIR"

echo "📊 Generating CSV summary..."
singularity exec -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif python3 scripts/summarize_batch_extraction.py "$BATCH_DIR"

#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=20:00:00
#SBATCH --job-name=extract_array
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de
#SBATCH --output=slurm_logs/extract_%A_%a.out
#SBATCH --error=slurm_logs/extract_%A_%a.err

RECIPE=$1
BATCH_DIR=$2
SEED=$SLURM_ARRAY_TASK_ID

SCRIPT_PATH="./scripts/run_gen_extract_distill_val.sh"

echo "Task ID: $SEED"
echo "Launching extraction via Singularity: $SCRIPT_PATH"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif "$SCRIPT_PATH" "$RECIPE" --skip-gen --skip-distill --skip-val --batch-dir "$BATCH_DIR" --seed "$SEED"

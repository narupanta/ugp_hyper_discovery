#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --job-name=extract_summary
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm_logs/summary_%j.out
#SBATCH --error=slurm_logs/summary_%j.err

BATCH_DIR=$1

echo "Generating CSV summary for batch directory: $BATCH_DIR"
singularity exec -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif python3 scripts/summarize_batch_extraction.py "$BATCH_DIR"

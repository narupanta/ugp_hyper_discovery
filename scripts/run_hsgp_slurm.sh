#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=hsgp-ext
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# HSGP extraction on the cluster, same container setup as run_pipeline_slurm.sh.
# Usage: sbatch scripts/run_hsgp_slurm.sh <recipe | experiment dir> [--seeds "6 7 8"] [--hsgp_<option> <value> ...]
# The method needs float64; on GPUs without full-rate FP64 (consumer cards) the CPU can be as fast: to run on CPU
# only, remove the --gres line and set JAX_PLATFORMS=cpu below.
# export JAX_PLATFORMS=cpu

SCRIPT_PATH="./scripts/run_hsgp_pipeline.sh"
if [ ! -f "$SCRIPT_PATH" ] && [ -f "./run_hsgp_pipeline.sh" ]; then
    SCRIPT_PATH="./run_hsgp_pipeline.sh"
fi

echo "Launching HSGP pipeline via Singularity: $SCRIPT_PATH $*"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif bash "$SCRIPT_PATH" "$@"

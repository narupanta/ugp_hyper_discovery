#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=100:00:00
#SBATCH --job-name=gen-ext-dist-val
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

SCRIPT_PATH="./scripts/run_pipeline.sh"
if [ ! -f "$SCRIPT_PATH" ] && [ -f "./run_pipeline.sh" ]; then
    SCRIPT_PATH="./run_pipeline.sh"
fi

echo "Launching pipeline via Singularity: $SCRIPT_PATH"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif "$SCRIPT_PATH" "$@"

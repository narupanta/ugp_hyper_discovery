#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=study
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# scripts/run_study.sh on the cluster (same container as run_hsgp_slurm.sh); all arguments are passed through.
#   normal:    sbatch scripts/run_study_slurm.sh --mode normal --recipe configs/recipes_v2/<recipe>.yaml --seeds "1 2 3"
#   coverage:  sbatch scripts/run_study_slurm.sh --mode coverage --cov_experiment_dir results/<exp>
#   threshold: one array task per grid point, then the summary
#     sbatch scripts/run_study_slurm.sh --mode threshold --list_grid                  # (or locally) shows the indices
#     JOB=$(sbatch --parsable --array=0-11 scripts/run_study_slurm.sh --mode threshold)
#     sbatch --dependency=afterany:$JOB scripts/run_study_slurm.sh --mode threshold --collect_only
# The settings of each mode are at the top of scripts/run_study.sh (or override them here, e.g. --sweep_dir ...).

echo "Launching study via Singularity: scripts/run_study.sh $*"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif bash scripts/run_study.sh "$@"

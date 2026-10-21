#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=cov-study
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# Distillation coverage study (validation/distillation_coverage_study.py) on the cluster, same container setup as
# run_hsgp_slurm.sh. All arguments are passed on to the study script.
# Usage:
#   pilot: sbatch scripts/run_coverage_study_slurm.sh --experiment_dir results/<exp> --n_noise_seeds 1 --n_inits 5 \
#              --checkpoints 2000 4000 6000 8000 10000 --jobs 2
#   study: sbatch scripts/run_coverage_study_slurm.sh --experiment_dir results/<exp> --n_noise_seeds 10 --n_inits 5 \
#              --n_iterations 4000 --jobs 4 [--extract]
# Finished runs are skipped, so a job that hits the time limit can simply be resubmitted with the same arguments.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "Launching coverage study via Singularity: validation/distillation_coverage_study.py $*"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif python3 validation/distillation_coverage_study.py "$@"

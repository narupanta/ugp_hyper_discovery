#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=cov-2gpu
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# scripts/run_coverage_2gpu.sh on the cluster (same container as run_coverage_study_slurm.sh); all arguments are
# passed through. Resubmit the same command to continue a stopped study.
#   sbatch scripts/run_coverage_2gpu_slurm.sh --recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --n_seeds 10 --n_inits 5
#   sbatch scripts/run_coverage_2gpu_slurm.sh --experiment_dir results/<exp> --n_seeds 10 --n_inits 5 \
#          --sample_mode dataset_f_random --num_points 128

echo "Launching via Singularity: scripts/run_coverage_2gpu.sh $*"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif bash scripts/run_coverage_2gpu.sh "$@"

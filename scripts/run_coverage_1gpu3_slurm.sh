#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --job-name=cov-1gpu3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# scripts/run_coverage_2gpu.sh with 3 slots on ONE GPU (3 seeds at a time; extraction round-robin, distillation seeds
# 1 4 7 10 / 2 5 8 / 3 6 9 per init; each slot's JAX memory capped at 0.3 of the GPU). All arguments are passed
# through; resubmit the same command (with --experiment_dir) to continue a stopped study.
#   sbatch scripts/run_coverage_1gpu3_slurm.sh --recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --n_seeds 10 --n_inits 5
#   sbatch scripts/run_coverage_1gpu3_slurm.sh --experiment_dir results/<exp> --n_seeds 10 --n_inits 5 \
#          --sample_mode dataset_f_random --num_points 128

echo "Launching via Singularity: scripts/run_coverage_2gpu.sh --gpus '0 0 0' $*"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif bash scripts/run_coverage_2gpu.sh --gpus "0 0 0" "$@"

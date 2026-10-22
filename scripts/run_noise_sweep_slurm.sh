#!/bin/bash
#SBATCH --partition=iam
#SBATCH --qos=iam_qos
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --job-name=noise-sweep
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.pantapalin@tu-braunschweig.de

# Noise-threshold sweep (validation/noise_threshold_sweep.py) on the cluster, same container setup as run_hsgp_slurm.sh.
# All arguments are passed on to the sweep script; always give a fixed --sweep_dir so that all tasks share it.
#
# One array task per grid point (displacement x load noise, load-major order; --list_grid prints the indices):
#   ARGS="--base_recipe configs/recipes_v2/isihara_hsgp_residual_cv.yaml --force_noise_rule cv \
#         --noise_levels 1e-5 4e-5 7e-5 1e-4 --load_noise_levels 0.02 0.05 0.1 --seeds 1 2 3 \
#         --sweep_dir results/noise_sweep_isihara_cv"
#   JOB=$(sbatch --parsable --array=0-11 scripts/run_noise_sweep_slurm.sh $ARGS)
# Summary (table, per-seed CSV, plot) once every task has finished:
#   sbatch --dependency=afterany:$JOB scripts/run_noise_sweep_slurm.sh $ARGS --collect_only
# Without --array the whole grid runs sequentially in one job. Finished grid points are skipped on resubmission.

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

EXTRA=()
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [[ " $* " != *" --collect_only "* ]]; then
    EXTRA=(--grid_index "$SLURM_ARRAY_TASK_ID")
fi

echo "Launching noise sweep via Singularity: validation/noise_threshold_sweep.py $* ${EXTRA[*]}"
singularity exec --nv -B /home/npantapalin/work/projects/ugp_hyper_discovery:/home/mmdiscovery/shared --pwd /home/mmdiscovery/shared /home/npantapalin/work/container/ugp_hyper_discovery.sif python3 validation/noise_threshold_sweep.py "$@" "${EXTRA[@]}"

#!/bin/bash
# Submits a noise-threshold sweep on Slurm: one array task per grid point (displacement x load noise, all seeds),
# then the summary once every task has ended. Edit the settings below, then on the cluster:
#   bash scripts/submit_noise_sweep.sh
# Finished grid points in SWEEP_DIR are skipped on resubmission.
set -e
cd "$(dirname "$0")/.."

RECIPE="configs/recipes_v2/isihara_hsgp_residual_cv.yaml"
RULE="joint"                       # joint | joint_refit | evidence | cv | em | scaled | fixed | derived | from_reaction
REACTION_WEIGHT="100"              # joint / joint_refit / evidence / em: a number, or 'euclid' (N_free / N_BC)
DISP_NOISE=(1e-6 1e-5 4e-5 7e-5 1e-4)
LOAD_NOISE=(0.02)
SEEDS=(1 2 3 4 5 6 7 8 9 10)
HSGP_ARGS="--hsgp_sigma_cap 10"    # extra extraction options for every grid point
SWEEP_DIR="results/noise_sweep_$(basename "${RECIPE%.yaml}")_${RULE}_rw${REACTION_WEIGHT}"

ARGS=(--base_recipe "$RECIPE" --force_noise_rule "$RULE" --reaction_weight "$REACTION_WEIGHT"
      --noise_levels "${DISP_NOISE[@]}" --load_noise_levels "${LOAD_NOISE[@]}" --seeds "${SEEDS[@]}"
      --sweep_dir "$SWEEP_DIR" --hsgp_args "$HSGP_ARGS")

N=$(( ${#DISP_NOISE[@]} * ${#LOAD_NOISE[@]} ))
mkdir -p "$SWEEP_DIR"
echo "Sweep dir: $SWEEP_DIR ($N grid points x ${#SEEDS[@]} seeds)"

JOB=$(sbatch --parsable --array=0-$((N - 1)) scripts/run_noise_sweep_slurm.sh "${ARGS[@]}")
SUM=$(sbatch --parsable --dependency=afterany:"$JOB" scripts/run_noise_sweep_slurm.sh "${ARGS[@]}" --collect_only)
echo "Submitted array job $JOB (tasks 0-$((N - 1))) and summary job $SUM"
echo "Summary when done: $SWEEP_DIR/sweep_summary.md"

#!/usr/bin/env bash
#SBATCH --job-name=N2S.bm.eval           # Name of the process
#SBATCH --cpus-per-task=4                # Number of CPU cores (2 is reasonable)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=200G                       # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --time=0                         # Training limit in days: 0 means no limit.
#SBATCH --output=.slurm/brainmagick/stdout-%N-%A_%a.log
#SBATCH --error=.slurm/brainmagick/stderr-%N-%A_%a.log
#SBATCH --array=0-11%3                   # Do all runs, having at most 3 running at each time.

# Recommended way to run the script:
# % mkdir -p .slurm/brainmagick/
# % sbatch ~/brainmagick/run_eval_probs_hubert.slurm

# Load the bash and python environment:
source ~/.bashrc
VENV="bm_${PYTHON_VERSION}"
source ~/envs/${VENV}/bin/activate

RUNS=(
  # gwilliams2022
  59a1115e
  afb5ec0a
  f6ba5230

  # audio_mous
  2b3e7066
  6953a40e
  f442d549

  # broderick2019
  1cbb7be8
  7515bc62
  2415b073

  # brennan2019
  0cca7172
  24670c16
  141185ef
)

# Run the evaluation script in Slurm (full path recommended):
srun ~/brainmagick/run_eval_probs.sh "${RUNS[${SLURM_ARRAY_TASK_ID}]}"

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
#SBATCH --array=0-14%3                   # Do all runs, having at most 3 running at each time.

# Recommended way to run the script:
# % mkdir -p .slurm/brainmagick/
# % sbatch ~/brainmagick/run_eval_probs_bourguignon2020.slurm

# Load the bash and python environment:
source ~/.bashrc
VENV="bm_${PYTHON_VERSION}"
source ~/envs/${VENV}/bin/activate

RUNS=(
    72460be6
    3efc2eab
    9b9ac6b4
    0465d1be
    ed6a1149
    eea723c1
    95ceb3a7
    34540ac9
    6c44cd30
    b4448625
    b9230568
    a6056839
    8ef0bd6c
    c920fe9d
    43ec490c
)

# Run the evaluation script in Slurm (full path recommended):
srun ~/brainmagick/run_eval_probs.sh "${RUNS[${SLURM_ARRAY_TASK_ID}]}"

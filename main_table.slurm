#!/usr/bin/env bash
#SBATCH --job-name=N2S.bm                # Name of the process
#SBATCH --cpus-per-task=32               # Number of CPU cores (2 is reasonable)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # Number of GPUs (usually light processes only need 1) 
#SBATCH --mem=200G                       # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --time=0                         # Training limit in days: 0 means no limit.
#SBATCH --output=.slurm/brainmagick/stdout-%N-%A_%a.log
#SBATCH --error=.slurm/brainmagick/stderr-%N-%A_%a.log
#SBATCH --array=0-59%3                   # Do all runs, having at most 3 running at each time.

# Options based on: https://github.com/facebookresearch/brainmagick/blob/main/bm/grids/nmi/main_table.py

# Requirements:
# % dora grid nmi.main_table --dry_run --init

# Recommended way to run the script:
# % mkdir -p .slurm/brainmagick/
# % sbatch ~/brainmagick/main_table.slurm

# Load the bash and python environment:
source ~/.bashrc
VENV="bm_${PYTHON_VERSION}"
source ~/envs/${VENV}/bin/activate

RUNS=(
  34219380
  bcd967bc
  029557fd
  53f86c17
  ca882d84
  6e3bf7d7
  c5455d58
  87a001d2
  13767159
  c512a1a6
  557f5f8a
  ff8f3a2f
  a7e4621f
  8207605c
  68c1a1e5
  4395629c
  3b8b933e
  04966af2
  765c995a
  413c5356
  59c3fedd
  528367b9
  bce54762
  10815196
  14e245d9
  884343d2
  750d6954
  ca0c1506
  e1a95c9d
  f1cdb4e1
  adc8346b
  69712ead
  896a10b2
  426029bd
  0edfcb98
  ba38d581
  784dae52
  5be341d4
  0a01dee9
  9c6ed694
  b220992a
  f6a650ba
  48b35a15
  365153df
  c8968f15
  d977efac
  49b468a5
  71007c6f
  2c768010
  6bc9ab43
  a7028f25
  813077f4
  3d97b459
  5b8e222b
  868b0132
  5dcd85a2
  471a79d9
  34f042ed
  2fe8bdbf
  dc2a44b8
)

# Run the training script in Slurm (full path recommended):
srun ~/brainmagick/dora.sh run -f "${RUNS[${SLURM_ARRAY_TASK_ID}]}"

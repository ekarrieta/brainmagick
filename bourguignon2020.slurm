#!/usr/bin/env bash
#SBATCH --job-name=N2S.bcbl           # Name of the process
#SBATCH --cpus-per-task=32            # Number of CPU cores (2 is reasonable)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                  # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=200G                    # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --time=0                      # Training limit in days: 0 means no limit.
#SBATCH --output=.slurm/brainmagick/stdout-%N-%A_%a.log
#SBATCH --error=.slurm/brainmagick/stderr-%N-%A_%a.log
#SBATCH --array=0-14%2                # Do all runs, having at most 2 running at each time.

# Options based on: https://github.com/facebookresearch/brainmagick/blob/main/bm/grids/nmi/main_table.py

# Recommended way to run the script:
# % mkdir -p .slurm/brainmagick/
# % sbatch ~/brainmagick/bourguignon2020.slurm

# Python venv environment to load:
VENV="bm_${PYTHON_VERSION}"

# Load the bash and python environment:
source ~/.bashrc
source ~/envs/${VENV}/bin/activate

RUNS=(
  'dset.selections=[bourguignon2020]'
  'dset.selections=[bourguignon2020] optim.epochs=1 optim.max_batches=1 test.wer_random=true'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020]'
  'feature_model=deep_mel dset.features=[MelSpectrum] dset.selections=[bourguignon2020]'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020] optim.loss=mse'
  'dset.selections=[bourguignon2020] seed=2037'
  'dset.selections=[bourguignon2020] optim.epochs=1 optim.max_batches=1 seed=2037 test.wer_random=true'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020] seed=2037'
  'feature_model=deep_mel dset.features=[MelSpectrum] dset.selections=[bourguignon2020] seed=2037'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020] optim.loss=mse seed=2037'
  'dset.selections=[bourguignon2020] seed=2038'
  'dset.selections=[bourguignon2020] optim.epochs=1 optim.max_batches=1 seed=2038 test.wer_random=true'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020] seed=2038'
  'feature_model=deep_mel dset.features=[MelSpectrum] dset.selections=[bourguignon2020] seed=2038'
  'dset.features=[MelSpectrum] dset.selections=[bourguignon2020] optim.loss=mse seed=2038'
)

# Run the training script in Slurm (full path recommended):
srun ~/brainmagick/dora.sh run ${RUNS[${SLURM_ARRAY_TASK_ID}]}

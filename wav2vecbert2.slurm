#!/usr/bin/env bash
#SBATCH --job-name=N2S.bm.w2vB2          # Name of the process
#SBATCH --cpus-per-task=32               # Number of CPU cores (2 is reasonable)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1                     # Number of GPUs (usually light processes only need 1)
#SBATCH --mem=200G                       # RAM memory needed (8-16GB is reasonable for our servers, sometimes you'll need more)
#SBATCH --time=0                         # Training limit in days: 0 means no limit.
#SBATCH --output=.slurm/brainmagick/stdout-%N-%A_%a.log
#SBATCH --error=.slurm/brainmagick/stderr-%N-%A_%a.log
#SBATCH --array=0-11%3                    # Do all runs, having at most 3 running at each time.

# Trains HuBERT models as Speech module in the different datasets.

# Recommended way to run the script:
# % mkdir -p .slurm/brainmagick/
# % sbatch ~/brainmagick/wav2vecbert2.slurm

# Python venv environment to load:
VENV="bm_${PYTHON_VERSION}"

# Load the bash and python environment:
source ~/.bashrc
source ~/envs/${VENV}/bin/activate

# Run the training script in Slurm (full path recommended):
# We are going to add additional seeds to the ones used in the paper:
RUNS=(
  'dset.selections=[gwilliams2022] dset.features=[Wav2VecBert2Transformer]'
  'dset.selections=[gwilliams2022] dset.features=[Wav2VecBert2Transformer] seed=2037'
  'dset.selections=[gwilliams2022] dset.features=[Wav2VecBert2Transformer] seed=2038'
  'dset.force_uid_assignement=true dset.selections=["audio_mous"] dset.features=[Wav2VecBert2Transformer]'
  'dset.force_uid_assignement=true dset.selections=["audio_mous"] dset.features=[Wav2VecBert2Transformer] seed=2037'
  'dset.force_uid_assignement=true dset.selections=["audio_mous"] dset.features=[Wav2VecBert2Transformer] seed=2038'
  'dset.selections=["broderick2019"] dset.features=[Wav2VecBert2Transformer]'
  'dset.selections=["broderick2019"] dset.features=[Wav2VecBert2Transformer] seed=2037'
  'dset.selections=["broderick2019"] dset.features=[Wav2VecBert2Transformer] seed=2038'
  'dset.selections=["brennan2019"] dset.features=[Wav2VecBert2Transformer]'
  'dset.selections=["brennan2019"] dset.features=[Wav2VecBert2Transformer] seed=2037'
  'dset.selections=["brennan2019"] dset.features=[Wav2VecBert2Transformer] seed=2038'
)

# Run the training script in Slurm (full path recommended):
srun ~/brainmagick/dora.sh run ${RUNS[${SLURM_ARRAY_TASK_ID}]}

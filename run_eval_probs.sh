#!/usr/bin/env bash
# Simple script to evaluate a trained experiment.
#
# This is a simple dora wrapper because I am running Slurm with SSHFS, which
# has some restrictions. It may not be necessary for other cases.
#
# It also creates a log file to save all the output.


DIR="$(dirname "$(readlink -f "${0}")")"
LOG="run_eval_probs-${1}.log"

if [ x"${1}" = x ]
then
  python3 -m scripts.run_eval_probs --help
  exit 1
fi

pushd "${DIR}" || exit 255
python3 -m scripts.run_eval_probs "sigs=[\"${1}\"]" 2>&1 | tee "${LOG}"
popd || exit 255

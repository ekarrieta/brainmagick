#!/usr/bin/env bash
# Simple script to perform one training using dora.
#
# This is a simple dora wrapper because I am running Slurm with SSHFS, which
# has some restrictions. It may not be necessary for other cases.
#
# It also creates a log file to save all the output.

DIR="$(dirname "$(readlink -f "${0}")")"
TS="$(date +%Y-%m-%d-%H-%M-%S)"
LOG="dora-${TS}.log"

if [ x"${1}" = x ]
then
  dora run --help
  exit 1
fi

pushd "${DIR}" || exit 255
dora "${@}" 2>&1 | tee "${LOG}"
popd || exit 255

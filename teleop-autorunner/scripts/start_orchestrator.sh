#!/usr/bin/env bash
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-teleoperation}"
USER_HOME="${HOME:-/home/unitree}"
ORCHESTRATOR_PY="${ORCHESTRATOR_PY:-/opt/teleop_autorunner/orchestrator.py}"

for conda_sh in \
  "${USER_HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${USER_HOME}/anaconda3/etc/profile.d/conda.sh" \
  "${USER_HOME}/miniforge3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"
do
  if [ -f "$conda_sh" ]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
    break
  fi
done

if ! command -v conda >/dev/null 2>&1; then
  echo "conda no encontrado. Instalalo o ajustá start_orchestrator.sh" >&2
  exit 1
fi

conda activate "$CONDA_ENV"
exec python "$ORCHESTRATOR_PY"

#!/usr/bin/env bash
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-teleoperation}"
TELEOP_DIR="${TELEOP_DIR:-$HOME/xr_teleoperate/teleop}"

# Sin TTY (systemd) sudo no puede pedir password. install.sh instala NOPASSWD para este comando.
sudo -n /unitree/sbin/mscli stopservice video_hub_pc4 || true

for conda_sh in \
  "${HOME}/miniconda3/etc/profile.d/conda.sh" \
  "${HOME}/anaconda3/etc/profile.d/conda.sh" \
  "${HOME}/miniforge3/etc/profile.d/conda.sh" \
  "/opt/conda/etc/profile.d/conda.sh"
do
  if [ -f "$conda_sh" ]; then
    # shellcheck source=/dev/null
    source "$conda_sh"
    break
  fi
done

conda activate "$CONDA_ENV"

# Si un launch anterior dejó teleimager vivo, no intentes bindear :60000 de nuevo.
if python - <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("0.0.0.0", 60000))
except OSError:
    sys.exit(0)
s.close()
sys.exit(1)
PY
then
  echo "Puerto 60000 ocupado: reutilizo teleimager existente"
else
  (
    cd "${TELEOP_DIR}/teleimager" || exit 1
    exec teleimager-server --rs
  ) &
  sleep 3
fi

cd "${TELEOP_DIR}"
exec python teleop_hand_and_arm.py --motion --ee "${EE}" --img-server-ip "${IP}" --record

#!/usr/bin/env bash
set -eo pipefail

# No cortar el servicio si ya estaba parado
sudo /unitree/sbin/mscli stopservice video_hub_pc4 || true

# teleimager-server en background — exec dentro del subshell
# para que sea el proceso real, no un bash wrapper colgando
(
  cd "${HOME}/xr_teleoperate/teleop/teleimager" || exit 1
  exec teleimager-server --rs
) &

# Esperá a que levante antes de arrancar el teleop
sleep 3

cd /home/ubuntu/xr_teleoperate/teleop/
exec python teleop_hand_and_arm.py --motion --ee "${EE}" --img-server-ip "${IP}" --record

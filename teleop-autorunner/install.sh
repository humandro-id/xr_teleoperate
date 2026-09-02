#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_DIR=/opt/teleop_autorunner

# Scripts + orquestador
sudo mkdir -p "$INSTALL_DIR"
sudo cp -r "$REPO_DIR/scripts" "$INSTALL_DIR/"
sudo cp "$REPO_DIR/orchestrator.py" "$INSTALL_DIR/"
sudo chmod +x "$INSTALL_DIR"/scripts/*.sh "$INSTALL_DIR/orchestrator.py"

# Env por robot (solo si no existe, no piso config local)
sudo mkdir -p /etc/robot
if [ ! -f /etc/robot/robot.env ]; then
    sudo cp "$REPO_DIR/config/teleop_autorunner.env" /etc/robot/robot.env
    echo ">> Editá /etc/robot/robot.env con las rutas de este robot"
fi

# sudo sin password solo para parar/arrancar video_hub (el servicio no tiene TTY)
SUDOERS_SRC="$REPO_DIR/systemd/teleop-mscli.sudoers"
SUDOERS_DST=/etc/sudoers.d/teleop-mscli
sudo cp "$SUDOERS_SRC" "$SUDOERS_DST"
sudo chmod 0440 "$SUDOERS_DST"
if ! sudo visudo -cf "$SUDOERS_DST"; then
    sudo rm -f "$SUDOERS_DST"
    echo ">> sudoers inválido, no se instaló $SUDOERS_DST" >&2
    exit 1
fi

# Services
sudo cp "$REPO_DIR"/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now teleop_orchestrator.service

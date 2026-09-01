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

# Services
sudo cp "$REPO_DIR"/systemd/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now teleop_orchestrator.service

#!/usr/bin/env python3
"""Orquestador NATS para lanzar la teleoperación XR.

Escucha `{MODEL_KEY}.{ROBOT_ID}.command.launch-teleop` y, al recibir
`{"command": "teleop"}`, arranca el proceso de teleop de inmediato.
No usa ROS: no hay filtro de hardware ni lectura de estado.
"""

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from pathlib import Path

from nats.aio.client import Client as NATS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("teleop_orchestrator")

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_LAUNCH_SCRIPT = SCRIPT_DIR / "scripts" / "launch_teleop.sh"

IP_NATS = os.getenv("IP_NATS")
ROBOT_ID = os.getenv("ROBOT_ID")
MODEL_KEY = os.getenv("MODEL_KEY", "unitree_g1")


def build_launch_cmd():
    """Siempre el script completo: video_hub off + teleimager + teleop."""
    launch_script = os.getenv("TELEOP_LAUNCH_SCRIPT", str(DEFAULT_LAUNCH_SCRIPT))
    return ["/bin/bash", launch_script]


class TeleopOrchestrator:
    def __init__(self):
        if not IP_NATS or not ROBOT_ID:
            raise SystemExit("Faltan variables de entorno IP_NATS y/o ROBOT_ID")

        self.ip_nats = IP_NATS
        self.robot_id = ROBOT_ID
        self.model_key = MODEL_KEY
        self.nats_subject = f"{self.model_key}.{self.robot_id}.command.launch-teleop"
        self.nats_status = f"{self.model_key}.{self.robot_id}.command.launch-teleop.status"
        self.launch_cmd = build_launch_cmd()

        self.nc = NATS()
        self.process = None
        self._launch_lock = asyncio.Lock()

    def _process_running(self):
        return self.process is not None and self.process.poll() is None

    async def publish_status(self, status, extra=None):
        payload = {"status": status}
        if extra:
            payload.update(extra)
        await self.nc.publish(self.nats_status, json.dumps(payload).encode("utf-8"))
        logger.info("Estado publicado en %s: %s", self.nats_status, payload)

    async def launch_teleop(self):
        async with self._launch_lock:
            if self._process_running():
                logger.warning("La teleoperación ya está corriendo (pid=%s)", self.process.pid)
                await self.publish_status("already_running", {"pid": self.process.pid})
                return

            logger.info("Lanzando launch_teleop.sh (video_hub off + teleimager + teleop): %s", self.launch_cmd)
            try:
                # Hereda EE, IP y el resto del EnvironmentFile de systemd.
                self.process = subprocess.Popen(
                    self.launch_cmd,
                    env=os.environ.copy(),
                    start_new_session=True,
                )
            except Exception as exc:
                logger.exception("No se pudo lanzar la teleoperación")
                await self.publish_status("error", {"error": str(exc)})
                return

            await self.publish_status("ok", {"pid": self.process.pid})
            logger.info("Teleoperación lanzada (pid=%s)", self.process.pid)

    async def stop_teleop(self):
        if not self._process_running():
            logger.info("No hay proceso de teleoperación para detener")
            await self.publish_status("stopped")
            return

        logger.info("Deteniendo teleoperación (pid=%s)", self.process.pid)
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=3)
        except Exception:
            logger.exception("Error al detener la teleoperación")
        finally:
            self.process = None
        try:
            subprocess.run(
                ["sudo", "-n", "/unitree/sbin/mscli", "startservice", "video_hub_pc4"],
                check=False,
                timeout=10,
            )
        except Exception:
            logger.exception("No se pudo reactivar video_hub_pc4")
        await self.publish_status("stopped")

    async def on_nats_message(self, msg):
        logger.info("Mensaje recibido en %s", msg.subject)
        try:
            payload = json.loads(msg.data.decode("utf-8"))
        except json.JSONDecodeError:
            logger.error("El mensaje no es un JSON válido")
            return

        command = payload.get("command")
        if command == "teleop":
            # El frontend resuelve el botón con "armed" apenas llega el comando.
            await self.publish_status("armed")
            await self.launch_teleop()
        elif command in ("stop", "quit"):
            await self.stop_teleop()
        else:
            logger.warning("Comando desconocido: %r. Ignorando.", command)

    async def connect_and_listen(self):
        await self.nc.connect(self.ip_nats)
        await self.nc.subscribe(self.nats_subject, cb=self.on_nats_message)
        logger.info("Conectado a NATS en %s", self.ip_nats)
        logger.info("Suscrito a %s", self.nats_subject)
        logger.info("Comando de lanzamiento: %s", self.launch_cmd)

    async def shutdown(self):
        if self._process_running():
            await self.stop_teleop()
        try:
            await self.nc.close()
        except Exception:
            pass


async def main():
    orchestrator = TeleopOrchestrator()
    stop_event = asyncio.Event()

    def _request_stop():
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _request_stop)
        except NotImplementedError:
            signal.signal(sig, lambda *_: _request_stop())

    await orchestrator.connect_and_listen()
    logger.info("Orquestador listo. Esperando comando teleop...")
    await stop_event.wait()
    await orchestrator.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)

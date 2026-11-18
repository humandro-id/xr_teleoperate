#!/usr/bin/env python3
"""Hold G1 arms + BrainCo hands in a fixed pose so you can test locomotion.

Default is motion mode (rt/arm_sdk): the walking policy keeps the legs,
this script only overlays arms/hands. The robot must already be in AI /
regular loco mode (not debug).

Usage:
    python hold_nav_pose.py
    python hold_nav_pose.py --config config/grasp_and_hold.yaml
    python hold_nav_pose.py --config config/grasp_and_hold_trajectory.yaml

Acepta YAML de steps (hombros/codos/manos), pose simple, waypoints 26-DOF o trajectory 26-DOF:
    q = [left_arm(7), left_hand(6), right_arm(7), right_hand(6)]

Un solo proceso publica arm_sdk. Las poses se cambian acá, sin bajar los brazos:
    1 = put_mameluco.yaml
    2 = put_table.yaml
    3 = carry_mameluco.yaml
    4 = grasp_and_hold.yaml
    p = otorgar la pose actual a la policy (recién ahí acepta comandos)
    o / c = abrir / cerrar manos
    q = soltar overlay (única forma de bajar los brazos)
    Ctrl+C = congelar el movimiento actual

La policy espera /tmp/g1_arm_policy_ready, lee /tmp/g1_arm_start.yaml
y escribe objetivos en /tmp/g1_arm_goal.yaml.
"""

import argparse
import contextlib
import io
import os
import re
import select
import signal
import sys
import termios
import threading
import time
import tty

import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)

from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_

from teleop.robot_control.robot_arm import G1_29_ArmController, G1_29_JointIndex
from teleop.robot_control.robot_hand_brainco import (
    Brainco_Left_Hand_JointIndex,
    Brainco_Right_Hand_JointIndex,
    _fill_hand_cmds,
    brainco_Num_Motors,
    kTopicbraincoLeftCommand,
    kTopicbraincoLeftState,
    kTopicbraincoRightCommand,
    kTopicbraincoRightState,
)

logger = logging_mp.getLogger(__name__)

# Ctrl+C freezes the current motion. Overlay stays up until q.
_freeze = {"requested": False}
HOLD_PID_PATH = "/tmp/g1_hold_nav_pose.pid"
GOAL_PATH = "/tmp/g1_arm_goal.yaml"
STATE_PATH = "/tmp/g1_arm_state.yaml"
START_PATH = "/tmp/g1_arm_start.yaml"
READY_PATH = "/tmp/g1_arm_policy_ready"
_policy = {"enabled": False}


def _on_sigint(signum, frame):
    _freeze["requested"] = True

ARM_NAMES = [
    "shoulder_pitch", "shoulder_roll", "shoulder_yaw",
    "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw",
]
HAND_NAMES = ["thumb", "thumb_aux", "index", "middle", "ring", "pinky"]

# Recorte dentro de un brazo (7): pitch, roll, yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
ARM_GROUPS = {
    "hombros": (0, 3),
    "shoulders": (0, 3),
    "codos": (3, 4),
    "codo": (3, 4),
    "elbow": (3, 4),
    "elbows": (3, 4),
    "munecas": (4, 7),
    "muñecas": (4, 7),
    "wrists": (4, 7),
}


def _as_vec(value, size, label):
    arr = np.atleast_1d(np.asarray(value, dtype=float).reshape(-1))
    if arr.size != size:
        raise ValueError(f"{label} espera {size} valor(es), hay {arr.size}")
    return arr


def _as_hand(value, label):
    """6 valores 0-1, o un solo número para todos los dedos (0 abierto, 1 cerrado)."""
    arr = np.atleast_1d(np.asarray(value, dtype=float).reshape(-1))
    if arr.size == 1:
        arr = np.repeat(arr, 6)
    if arr.size != 6:
        raise ValueError(
            f"{label} espera 6 valores {HAND_NAMES} o 1 número 0-1, hay {arr.size}"
        )
    return np.clip(arr, 0.0, 1.0)


def _side_values(spec, size, label):
    if spec is None:
        return None, None
    if isinstance(spec, dict):
        left = _as_vec(spec["left"], size, f"{label}.left") if "left" in spec else None
        right = _as_vec(spec["right"], size, f"{label}.right") if "right" in spec else None
        return left, right
    both = _as_vec(spec, size, label)
    return both, both.copy()


def parse_step(raw, default_move_time):
    """Un paso: solo las articulaciones que aparecen se mueven, a ESOS números."""
    arm_patches = []  # (offset, values) offset 0=izq, 7=der
    hand_left = hand_right = None

    if "left_arm" in raw:
        arm_patches.append((0, _as_vec(raw["left_arm"], 7, "left_arm")))
    if "right_arm" in raw:
        arm_patches.append((7, _as_vec(raw["right_arm"], 7, "right_arm")))

    for name, (start, end) in ARM_GROUPS.items():
        if name not in raw:
            continue
        n = end - start
        left, right = _side_values(raw[name], n, name)
        if left is not None:
            arm_patches.append((start, left))
        if right is not None:
            arm_patches.append((7 + start, right))

    manos = raw.get("manos", raw.get("hands"))
    if manos is not None:
        if isinstance(manos, dict) and ("left" in manos or "right" in manos):
            if "left" in manos:
                hand_left = _as_hand(manos["left"], "manos.left")
            if "right" in manos:
                hand_right = _as_hand(manos["right"], "manos.right")
        else:
            both = _as_hand(manos, "manos")
            hand_left, hand_right = both, both.copy()
    if "left_hand" in raw:
        hand_left = _as_hand(raw["left_hand"], "left_hand")
    if "right_hand" in raw:
        hand_right = _as_hand(raw["right_hand"], "right_hand")

    if not arm_patches and hand_left is None and hand_right is None:
        raise ValueError(
            f"Paso {raw.get('name', '?')!r}: poné hombros, codos, munecas o manos con su objetivo"
        )
    return {
        "name": str(raw.get("name", "")),
        "move_time": float(raw.get("move_time", default_move_time)),
        "hold": float(raw.get("hold", 0.0)),
        "arm_patches": arm_patches,
        "hand_left": hand_left,
        "hand_right": hand_right,
    }


def apply_step(step, arm, left_hand, right_hand):
    goal_arm = arm.copy()
    for offset, values in step["arm_patches"]:
        goal_arm[offset:offset + values.size] = values
    goal_lh = left_hand.copy() if step["hand_left"] is None else step["hand_left"]
    goal_rh = right_hand.copy() if step["hand_right"] is None else step["hand_right"]
    return goal_arm, goal_lh, goal_rh


Q26_SIZE = 26  # left_arm(7) + left_hand(6) + right_arm(7) + right_hand(6)


def split_q26(q, label="q"):
    arr = np.asarray(q, dtype=float).reshape(-1)
    if arr.size != Q26_SIZE:
        raise ValueError(f"{label} espera {Q26_SIZE} valores (brazo L 7 + mano L 6 + brazo R 7 + mano R 6), hay {arr.size}")
    arm = np.concatenate([arr[0:7], arr[13:20]])
    left_hand = np.clip(arr[7:13], 0.0, 1.0)
    right_hand = np.clip(arr[20:26], 0.0, 1.0)
    return arm, left_hand, right_hand


def pack_q26(arm, left_hand, right_hand):
    return np.concatenate([arm[:7], left_hand, arm[7:14], right_hand])


def _ease(a, kind):
    a = min(1.0, max(0.0, float(a)))
    if kind == "cosine":
        return 0.5 - 0.5 * np.cos(np.pi * a)
    return a


def parse_timed_samples(rows, label, name_key="name"):
    samples = []
    prev_t = None
    for i, row in enumerate(rows):
        t = float(row["t"])
        if prev_t is not None and t < prev_t:
            raise ValueError(f"{label}[{i}] t={t} va hacia atrás (anterior {prev_t})")
        prev_t = t
        arm, lh, rh = split_q26(row["q"], f"{label}[{i}].q")
        samples.append({
            "name": str(row.get(name_key, row.get("name", f"{label}{i}"))),
            "t": t,
            "arm": arm,
            "lh": lh,
            "rh": rh,
        })
    if not samples:
        raise ValueError(f"{label} está vacío")
    t0 = samples[0]["t"]
    for sample in samples:
        sample["t"] = sample["t"] - t0
    return samples


def sample_timed(samples, t, interp):
    if t <= samples[0]["t"]:
        s = samples[0]
        return s["arm"].copy(), s["lh"].copy(), s["rh"].copy(), s["name"]
    if t >= samples[-1]["t"]:
        s = samples[-1]
        return s["arm"].copy(), s["lh"].copy(), s["rh"].copy(), s["name"]
    for i in range(len(samples) - 1):
        a, b = samples[i], samples[i + 1]
        if a["t"] <= t <= b["t"]:
            span = b["t"] - a["t"]
            alpha = 0.0 if span <= 1e-9 else _ease((t - a["t"]) / span, interp)
            return (
                (1.0 - alpha) * a["arm"] + alpha * b["arm"],
                (1.0 - alpha) * a["lh"] + alpha * b["lh"],
                (1.0 - alpha) * a["rh"] + alpha * b["rh"],
                b["name"],
            )
    s = samples[-1]
    return s["arm"].copy(), s["lh"].copy(), s["rh"].copy(), s["name"]


def load_motion(path):
    """Load steps YAML, simple pose, waypoints (26-DOF) or dense trajectory (26-DOF)."""
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if cfg.get("trajectory"):
        return {
            "kind": "timed",
            "interp": "linear",
            "samples": parse_timed_samples(cfg["trajectory"], "trajectory"),
        }
    if cfg.get("waypoints"):
        return {
            "kind": "timed",
            "interp": "cosine",
            "samples": parse_timed_samples(cfg["waypoints"], "waypoints"),
        }
    return {"kind": "steps", "steps": load_pose(path)}


def play_motion(arm_ctrl, hands, live, motion, dt, tau):
    if motion["kind"] == "timed":
        run_timed(arm_ctrl, hands, live, motion["samples"], motion["interp"], dt, tau)
        return
    run_sequence(arm_ctrl, hands, live, motion["steps"], dt, tau)


def load_pose(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    default_move_time = float(cfg.get("move_time", 3.0))
    if cfg.get("steps"):
        return [parse_step(step, default_move_time) for step in cfg["steps"]]
    left_arm = np.array(cfg["left_arm"], dtype=float)
    right_arm = np.array(cfg["right_arm"], dtype=float)
    left_hand = np.clip(np.array(cfg["left_hand"], dtype=float), 0.0, 1.0)
    right_hand = np.clip(np.array(cfg["right_hand"], dtype=float), 0.0, 1.0)
    if left_arm.size != 7 or right_arm.size != 7:
        raise ValueError("left_arm / right_arm must have 7 values")
    if left_hand.size != 6 or right_hand.size != 6:
        raise ValueError("left_hand / right_hand must have 6 values")
    return [parse_step({
        "name": "pose",
        "move_time": default_move_time,
        "left_arm": left_arm,
        "right_arm": right_arm,
        "left_hand": left_hand,
        "right_hand": right_hand,
    }, default_move_time)]


def dump_pose(arm_q, left_hand, right_hand, move_time=3.0):
    return {
        "move_time": move_time,
        "left_arm": [round(float(v), 4) for v in arm_q[:7]],
        "right_arm": [round(float(v), 4) for v in arm_q[7:]],
        "left_hand": [round(float(v), 4) for v in left_hand],
        "right_hand": [round(float(v), 4) for v in right_hand],
    }


def _stop_stale_background_holder():
    try:
        with open(HOLD_PID_PATH, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except (OSError, ValueError):
        pid = None
    if pid and pid != os.getpid():
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info("Apagué un holder viejo PID %d.", pid)
        except OSError:
            pass
    _clear_hold_pid()


def _write_hold_pid():
    with open(HOLD_PID_PATH, "w", encoding="utf-8") as f:
        f.write(str(os.getpid()))


def _clear_hold_pid():
    for path in (HOLD_PID_PATH, "/tmp/g1_hold_nav_pose.ready", READY_PATH):
        try:
            os.remove(path)
        except OSError:
            pass


def _write_yaml(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    os.replace(tmp, path)


def write_state(live):
    _write_yaml(STATE_PATH, dump_pose(live["arm"], live["lh"], live["rh"], move_time=0.0))


def grant_pose_to_policy(arm_ctrl, live, watcher):
    _snapshot_live(arm_ctrl, live)
    pose = dump_pose(live["arm"], live["lh"], live["rh"], move_time=0.0)
    write_state(live)
    _write_yaml(START_PATH, pose)
    with open(READY_PATH, "w", encoding="utf-8") as f:
        f.write("%d\n" % os.getpid())
    _policy["enabled"] = True
    if os.path.isfile(GOAL_PATH):
        watcher.mtime = os.path.getmtime(GOAL_PATH)
    else:
        watcher.mtime = None
    logger.info(
        "Pose otorgada a la policy. start=%s  ready=%s  goals=%s",
        START_PATH, READY_PATH, GOAL_PATH,
    )
    log_vec("brazo izq start", ARM_NAMES, live["arm"][:7])
    log_vec("brazo der start", ARM_NAMES, live["arm"][7:])
    log_vec("mano izq start", HAND_NAMES, live["lh"])
    log_vec("mano der start", HAND_NAMES, live["rh"])


def resolve_config(path):
    if os.path.isfile(path):
        return os.path.abspath(path)
    for candidate in (
        os.path.join(current_dir, path),
        os.path.join(current_dir, "config", os.path.basename(path)),
    ):
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    raise FileNotFoundError("No encuentro el YAML %s" % path)


def default_poses():
    names = [
        ("1", "put_mameluco", "put_mameluco.yaml"),
        ("2", "put_table", "put_table.yaml"),
        ("3", "carry_mameluco", "carry_mameluco.yaml"),
        ("4", "grasp_and_hold", "grasp_and_hold.yaml"),
    ]
    poses = []
    for key, label, fname in names:
        path = os.path.join(current_dir, "config", fname)
        if os.path.isfile(path):
            poses.append((key, label, path))
    return poses


class GoalWatcher:
    def __init__(self, path):
        self.path = path
        self.mtime = os.path.getmtime(path) if os.path.isfile(path) else None

    def poll(self):
        if not os.path.isfile(self.path):
            return None
        mtime = os.path.getmtime(self.path)
        if self.mtime is not None and mtime <= self.mtime:
            return None
        self.mtime = mtime
        return load_motion(self.path)


def run_sequence(arm_ctrl, hands, live, steps, dt, tau):
    arm_q = live["arm"].copy()
    left_q = live["lh"].copy()
    right_q = live["rh"].copy()
    for i, step in enumerate(steps, start=1):
        if _freeze["requested"]:
            break
        goal_arm, goal_lh, goal_rh = apply_step(step, arm_q, left_q, right_q)
        label = step["name"] or f"paso {i}"
        log_vec(f"{i}/{len(steps)} {label} izq", ARM_NAMES, goal_arm[:7])
        log_vec(f"{i}/{len(steps)} {label} der", ARM_NAMES, goal_arm[7:])
        if step["hand_left"] is not None:
            log_vec(f"{i}/{len(steps)} {label} mano izq", HAND_NAMES, goal_lh)
        if step["hand_right"] is not None:
            log_vec(f"{i}/{len(steps)} {label} mano der", HAND_NAMES, goal_rh)
        if (step["hand_left"] is not None or step["hand_right"] is not None) and not hands.ok:
            logger.error("Paso de manos ignorado por el hardware: no hay brainco_hand_server.")
        logger.info("Paso %d/%d '%s' en %.1fs", i, len(steps), label, step["move_time"])
        if not run_motion(
            arm_ctrl, hands, arm_q, left_q, right_q,
            goal_arm, goal_lh, goal_rh, step["move_time"], dt, tau, live,
        ):
            break
        arm_q, left_q, right_q = goal_arm, goal_lh, goal_rh
        live["arm"], live["lh"], live["rh"] = arm_q, left_q, right_q
        if step["hold"] > 0:
            if not hold_pose(arm_ctrl, hands, arm_q, left_q, right_q, dt, tau, step["hold"], live):
                break
    if _freeze["requested"]:
        _snapshot_live(arm_ctrl, live)
        arm_ctrl.ctrl_dual_arm(live["arm"], tau)
        hands.send(live["lh"], live["rh"])
        logger.info("Ctrl+C: me quedo en esta pose. 1/2/3/4 cambian YAML, q suelta.")
        _freeze["requested"] = False
    write_state(live)


def run_timed(arm_ctrl, hands, live, samples, interp, dt, tau, lead_in=1.0):
    """Play a 26-DOF timed sequence. Lead-in blends from the live pose to q[0]."""
    first = samples[0]
    last = samples[-1]
    names = " → ".join(s["name"] for s in samples)
    logger.info("Secuencia 26-DOF [%s]  t=0..%.2fs  interp=%s", names, last["t"], interp)
    log_vec("inicio izq", ARM_NAMES, first["arm"][:7])
    log_vec("inicio der", ARM_NAMES, first["arm"][7:])
    log_vec("final izq", ARM_NAMES, last["arm"][:7])
    log_vec("final der", ARM_NAMES, last["arm"][7:])

    need_lead = np.linalg.norm(live["arm"] - first["arm"]) > 0.08 or (
        np.linalg.norm(live["lh"] - first["lh"]) > 0.08
        or np.linalg.norm(live["rh"] - first["rh"]) > 0.08
    )
    if need_lead and lead_in > 0:
        logger.info("Lead-in %.1fs desde la pose actual hasta '%s'", lead_in, first["name"])
        if not run_motion(
            arm_ctrl, hands, live["arm"], live["lh"], live["rh"],
            first["arm"], first["lh"], first["rh"], lead_in, dt, tau, live,
        ):
            if _freeze["requested"]:
                _snapshot_live(arm_ctrl, live)
                logger.info("Ctrl+C: me quedo en esta pose.")
                _freeze["requested"] = False
            write_state(live)
            return

    t0 = time.time()
    last_name = None
    while True:
        if _freeze["requested"]:
            _snapshot_live(arm_ctrl, live)
            arm_ctrl.ctrl_dual_arm(live["arm"], tau)
            hands.send(live["lh"], live["rh"])
            logger.info("Ctrl+C: me quedo en esta pose. 1/2/3/4 cambian YAML, q suelta.")
            _freeze["requested"] = False
            write_state(live)
            return
        t = time.time() - t0
        arm_q, left_q, right_q, name = sample_timed(samples, t, interp)
        if name != last_name:
            logger.info("Waypoint '%s' t=%.2f/%.2f", name, min(t, last["t"]), last["t"])
            last_name = name
        live["arm"], live["lh"], live["rh"] = arm_q, left_q, right_q
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        hands.send(left_q, right_q)
        if t >= last["t"]:
            break
        time.sleep(dt)
    live["arm"], live["lh"], live["rh"] = last["arm"], last["lh"], last["rh"]
    arm_ctrl.ctrl_dual_arm(last["arm"], tau)
    hands.send(last["lh"], last["rh"])
    logger.info("Secuencia 26-DOF terminada, hold en '%s'.", last["name"])
    write_state(live)


def log_vec(label, names, values):
    pretty = ", ".join(f"{n}={v:.3f}" for n, v in zip(names, values))
    logger.info("%s: %s", label, pretty)


def _dds_read(sub, timeout=0.05):
    """Read with a short timeout so Ctrl+C can interrupt; hide SDK timeout prints."""
    with contextlib.redirect_stdout(io.StringIO()):
        return sub.Read(timeout)


class BraincoHold:
    def __init__(self, hz=100.0):
        self.left_pub = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.left_pub.Init()
        self.right_pub = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.right_pub.Init()
        self.left_sub = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.left_sub.Init()
        self.right_sub = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.right_sub.Init()

        self.left_msg = MotorCmds_()
        self.left_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]
        self.right_msg = MotorCmds_()
        self.right_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(brainco_Num_Motors)]
        for cmd in self.left_msg.cmds + self.right_msg.cmds:
            cmd.q = 0.0
            cmd.dq = 1.0

        self._lock = threading.Lock()
        self.left_q = np.zeros(brainco_Num_Motors)
        self.right_q = np.zeros(brainco_Num_Motors)
        self.ok = False
        self._hz = hz
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    def _publish_loop(self):
        dt = 1.0 / self._hz
        while self._running:
            with self._lock:
                left_q = self.left_q.copy()
                right_q = self.right_q.copy()
            _fill_hand_cmds(self.left_msg, Brainco_Left_Hand_JointIndex, left_q)
            _fill_hand_cmds(self.right_msg, Brainco_Right_Hand_JointIndex, right_q)
            self.left_pub.Write(self.left_msg)
            self.right_pub.Write(self.right_msg)
            time.sleep(dt)

    def read_state(self, timeout=3.0):
        left = np.zeros(brainco_Num_Motors)
        right = np.zeros(brainco_Num_Motors)
        got_l = got_r = False
        t0 = time.time()
        while time.time() - t0 < timeout and not (got_l and got_r):
            left_msg = _dds_read(self.left_sub)
            right_msg = _dds_read(self.right_sub)
            if left_msg is not None:
                for idx, jid in enumerate(Brainco_Left_Hand_JointIndex):
                    left[idx] = left_msg.states[jid].q
                got_l = True
            if right_msg is not None:
                for idx, jid in enumerate(Brainco_Right_Hand_JointIndex):
                    right[idx] = right_msg.states[jid].q
                got_r = True
            time.sleep(0.02)
        self.ok = got_l and got_r
        if not self.ok:
            logger.error(
                "No hay DDS de BrainCo: las manos NO se van a mover. "
                "En otra terminal: cd ~/brainco_hand_service/bin && sudo ./brainco_hand_server"
            )
        else:
            logger.info("BrainCo DDS ok.")
            with self._lock:
                self.left_q = left.copy()
                self.right_q = right.copy()
        return left, right

    def send(self, left_q, right_q):
        with self._lock:
            self.left_q = np.asarray(left_q, dtype=float).copy()
            self.right_q = np.asarray(right_q, dtype=float).copy()


def interpolate(start, goal, t, duration):
    if duration <= 0:
        return goal
    a = min(1.0, max(0.0, t / duration))
    a = 0.5 - 0.5 * np.cos(np.pi * a)
    return (1.0 - a) * start + a * goal


def run_motion(arm_ctrl, hands, start_arm, start_lh, start_rh,
               goal_arm, goal_lh, goal_rh, duration, dt, tau, live):
    t0 = time.time()
    while True:
        if _freeze["requested"]:
            return False
        t = time.time() - t0
        arm_q = interpolate(start_arm, goal_arm, t, duration)
        left_q = interpolate(start_lh, goal_lh, t, duration)
        right_q = interpolate(start_rh, goal_rh, t, duration)
        live["arm"], live["lh"], live["rh"] = arm_q, left_q, right_q
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        hands.send(left_q, right_q)
        if t >= duration:
            break
        time.sleep(dt)
    live["arm"], live["lh"], live["rh"] = goal_arm, goal_lh, goal_rh
    arm_ctrl.ctrl_dual_arm(goal_arm, tau)
    hands.send(goal_lh, goal_rh)
    return True


def hold_pose(arm_ctrl, hands, arm_q, left_q, right_q, dt, tau, duration=None, live=None):
    t0 = time.time()
    while duration is None or (time.time() - t0) < duration:
        if _freeze["requested"]:
            return False
        if live is not None:
            live["arm"], live["lh"], live["rh"] = arm_q, left_q, right_q
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        hands.send(left_q, right_q)
        time.sleep(dt)
    return True


def _read_key(timeout):
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        time.sleep(timeout)
        return None
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        mode = termios.tcgetattr(fd)
        mode[3] &= ~termios.ISIG
        termios.tcsetattr(fd, termios.TCSADRAIN, mode)
        try:
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
        except InterruptedError:
            return None
        if ready:
            return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return None


def move_hands_to(arm_ctrl, hands, arm_q, left_q, right_q, target, dt, tau, duration, live):
    goal = np.clip(np.full(6, float(target)), 0.0, 1.0)
    if run_motion(
        arm_ctrl, hands, arm_q, left_q, right_q,
        arm_q, goal, goal, duration, dt, tau, live,
    ):
        live["lh"], live["rh"] = goal, goal.copy()


def _snapshot_live(arm_ctrl, live):
    try:
        live["arm"] = arm_ctrl.get_current_dual_arm_q()
    except Exception:
        pass


def interactive_hold(arm_ctrl, hands, live, dt, tau, release_time, poses, watcher):
    signal.signal(signal.SIGINT, _on_sigint)
    by_key = {key: (label, path) for key, label, path in poses}
    for key, label, path in poses:
        logger.info("Tecla %s = %s (%s)", key, label, os.path.basename(path))
    logger.info(
        "o/c = manos. p = otorgar pose a la policy. q = soltar overlay. Ctrl+C congela."
    )
    write_state(live)
    while True:
        if _freeze["requested"]:
            _freeze["requested"] = False
            logger.info("Ctrl+C: sigo en esta pose. 1/2/3/4 YAML, p = policy, q suelta.")
        arm_q, left_q, right_q = live["arm"], live["lh"], live["rh"]
        arm_ctrl.ctrl_dual_arm(arm_q, tau)
        hands.send(left_q, right_q)
        incoming = None
        if _policy["enabled"]:
            try:
                incoming = watcher.poll()
            except Exception as e:
                logger.error("No pude leer %s: %s", GOAL_PATH, e)
        if incoming:
            logger.info("Goal de policy en %s", GOAL_PATH)
            _freeze["requested"] = False
            play_motion(arm_ctrl, hands, live, incoming, dt, tau)
            continue
        try:
            key = _read_key(dt)
        except KeyboardInterrupt:
            continue
        if not key:
            continue
        key = key.lower()
        if key in ("\x03", "\x1b"):
            logger.info("Ctrl+C: sigo en esta pose. 1/2/3/4 YAML, p = policy, q suelta.")
            continue
        if key in by_key:
            label, path = by_key[key]
            logger.info("Cambio a %s", label)
            _freeze["requested"] = False
            play_motion(arm_ctrl, hands, live, load_motion(path), dt, tau)
            continue
        if key == "o":
            logger.info("Abriendo manos...")
            move_hands_to(arm_ctrl, hands, live["arm"], live["lh"], live["rh"], 0.0, dt, tau, 1.2, live)
            write_state(live)
        elif key == "c":
            logger.info("Cerrando manos...")
            move_hands_to(arm_ctrl, hands, live["arm"], live["lh"], live["rh"], 1.0, dt, tau, 1.2, live)
            write_state(live)
        elif key == "p":
            grant_pose_to_policy(arm_ctrl, live, watcher)
        elif key == "q":
            logger.info("q: suelto overlay de brazos.")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            _policy["enabled"] = False
            release_arm_sdk(arm_ctrl, duration=release_time)
            _clear_hold_pid()
            return


def _load_cyclonedds_xml():
    uri = (os.environ.get("CYCLONEDDS_URI") or "").strip()
    if not uri:
        return None, None
    if uri.startswith("<"):
        return uri, "CYCLONEDDS_URI (inline)"
    path = uri[7:] if uri.startswith("file://") else uri
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read(), path
    return None, None


def init_dds(sim, network_interface):
    """Initialize DDS using CYCLONEDDS_URI when possible.

    unitree_sdk2py Domain(id, xml) ignores the env var. With wlan0+eth0 UP,
    AutoDetermine often binds WiFi and rt/lowstate never arrives.
    """
    if sim:
        ChannelFactoryInitialize(1, networkInterface=network_interface)
        return

    from unitree_sdk2py.core import channel_config as cc

    xml, source = _load_cyclonedds_xml()
    if xml is None:
        xml = """<?xml version="1.0" encoding="UTF-8" ?>
<CycloneDDS>
  <Domain Id="any">
    <General>
      <Interfaces>
        <NetworkInterface name="$IFACE" priority="default" multicast="default"/>
      </Interfaces>
      <AllowMulticast>spdp</AllowMulticast>
      <DontRoute>true</DontRoute>
    </General>
    <Discovery>
      <Peers>
        <Peer Address="192.168.123.161"/>
        <Peer Address="192.168.123.164"/>
      </Peers>
    </Discovery>
  </Domain>
</CycloneDDS>
"""
        iface = network_interface or "eth0"
        xml = xml.replace("$IFACE", iface)
        source = f"fallback {iface} + AllowMulticast=spdp"
    elif network_interface:
        xml, n = re.subn(
            r'(<NetworkInterface\b[^>]*\bname=")[^"]*"',
            rf'\g<1>{network_interface}"',
            xml,
            count=1,
        )
        if n:
            source = f"{source} (interface={network_interface})"

    cc.ChannelConfigAutoDetermine = xml
    logger.info("DDS: %s", source)
    ChannelFactoryInitialize(0)


def release_arm_sdk(arm_ctrl, duration=5.0):
    if not arm_ctrl.motion_mode:
        return
    duration = max(0.1, float(duration))
    dt = 0.02
    steps = max(2, int(round(duration / dt)))
    logger.info("Soltando overlay de brazos en %.1fs (arm_sdk weight → 0)...", duration)
    try:
        for weight in np.linspace(1.0, 0.0, num=steps):
            arm_ctrl.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = float(weight)
            time.sleep(dt)
    except KeyboardInterrupt:
        arm_ctrl.msg.motor_cmd[G1_29_JointIndex.kNotUsedJoint0].q = 0.0
        time.sleep(0.05)


def main():
    parser = argparse.ArgumentParser(description="Hold G1 arms + BrainCo hands for navigation tests")
    parser.add_argument(
        "--config",
        default=None,
        help="YAML opcional (steps, waypoints o trajectory); después 1/2/3/4 cambian de pose",
    )
    parser.add_argument("--network-interface", default=None)
    parser.add_argument("--sim", action="store_true")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use debug/lowcmd instead of arm_sdk. Do not use this while walking.",
    )
    parser.add_argument("--print-current", action="store_true", help="Print current q and exit")
    parser.add_argument("--save-current", default=None, help="Write current q to this YAML and exit")
    parser.add_argument("--hz", type=float, default=50.0)
    parser.add_argument(
        "--release-time",
        type=float,
        default=3.0,
        help="Seconds to fade arm_sdk weight to 0 on q (default: 3)",
    )
    args = parser.parse_args()

    init_dds(args.sim, args.network_interface)

    motion_mode = not args.debug
    try:
        arm_ctrl = G1_29_ArmController(
            motion_mode=motion_mode,
            simulation_mode=args.sim,
            hold_current=True,
        )
    except TimeoutError as e:
        logger.error("%s", e)
        sys.exit(1)

    signal.signal(signal.SIGINT, _on_sigint)

    tau = np.zeros(14)
    current_arm = arm_ctrl.get_current_dual_arm_q()
    arm_ctrl.ctrl_dual_arm(current_arm, tau)
    _stop_stale_background_holder()
    _write_hold_pid()

    hands = BraincoHold()
    current_left_hand, current_right_hand = hands.read_state(timeout=3.0)
    log_vec("brazo izq actual", ARM_NAMES, current_arm[:7])
    log_vec("brazo der actual", ARM_NAMES, current_arm[7:])
    log_vec("mano izq actual", HAND_NAMES, current_left_hand)
    log_vec("mano der actual", HAND_NAMES, current_right_hand)

    if args.print_current or args.save_current:
        pose = dump_pose(current_arm, current_left_hand, current_right_hand)
        if args.save_current:
            os.makedirs(os.path.dirname(os.path.abspath(args.save_current)) or ".", exist_ok=True)
            with open(args.save_current, "w", encoding="utf-8") as f:
                yaml.safe_dump(pose, f, sort_keys=False, allow_unicode=True)
            logger.info("Postura actual guardada en %s", args.save_current)
        else:
            print(yaml.safe_dump(pose, sort_keys=False, allow_unicode=True))
        _clear_hold_pid()
        return

    poses = default_poses()
    dt = 1.0 / args.hz
    live = {
        "arm": current_arm.copy(),
        "lh": current_left_hand.copy(),
        "rh": current_right_hand.copy(),
    }
    write_state(live)
    logger.info(
        "Overlay activo. Modo: %s. Los brazos no se bajan hasta q.",
        "motion/arm_sdk (navegación)" if motion_mode else "debug/lowcmd",
    )
    if args.config:
        path = resolve_config(args.config)
        logger.info("Arranco con %s", path)
        play_motion(arm_ctrl, hands, live, load_motion(path), dt, tau)
    while True:
        try:
            interactive_hold(
                arm_ctrl, hands, live, dt, tau, args.release_time,
                poses, GoalWatcher(GOAL_PATH),
            )
            break
        except KeyboardInterrupt:
            logger.info("Ctrl+C no baja los brazos. 1/2/3/4 cambian YAML, q suelta.")
    logger.info("Listo.")


if __name__ == "__main__":
    main()

"""
Brainco Hand Controller with UDCAP haptic glove input via OSC.

Receives per-finger flex from the shared UDCAP OSC bridge and maps the
result directly to Brainco 6-DOF motor commands via DDS, bypassing the
landmark-based hand-retargeting pipeline used by `Brainco_Controller`.

This mirrors `Inspire_Controller_UDCAP` but speaks the Brainco DDS
protocol (`rt/brainco/{left,right}/{cmd,state}`) and obeys the Brainco
motor order:
    [0:thumb, 1:thumb-aux, 2:index, 3:middle, 4:ring, 5:pinky]
with convention 0.0 = fully open, 1.0 = fully closed (same direction
as UDCAP flex, so no inversion is required).

The Meta Quest controllers continue to provide arm/wrist tracking while
the UDCAP gloves provide finger tracking.

Usage:
    python teleop_hand_and_arm.py --input-mode controller --ee brainco_uc
"""

import threading
import time
from enum import IntEnum
from multiprocessing import Process, Array

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

from teleop.robot_control.udcap_osc_bridge import (
    UDCAPOSCBridge, DEFAULT_UDCAP_HOST, DEFAULT_UDCAP_PORT,
)

# ── Brainco DDS topics (same as Brainco_Controller) ──────────────────────
Brainco_Num_Motors = 6

kTopicBraincoLeftCommand  = "rt/brainco/left/cmd"
kTopicBraincoLeftState    = "rt/brainco/left/state"
kTopicBraincoRightCommand = "rt/brainco/right/cmd"
kTopicBraincoRightState   = "rt/brainco/right/state"


# Brainco motor order (per official docs):
#   0: thumb (flex)        1: thumb-aux (rotation/opposition)
#   2: index               3: middle
#   4: ring                5: pinky
class _BraincoLeftIndex(IntEnum):
    kLeftHandThumb     = 0
    kLeftHandThumbAux  = 1
    kLeftHandIndex     = 2
    kLeftHandMiddle    = 3
    kLeftHandRing      = 4
    kLeftHandPinky     = 5


class _BraincoRightIndex(IntEnum):
    kRightHandThumb    = 0
    kRightHandThumbAux = 1
    kRightHandIndex    = 2
    kRightHandMiddle   = 3
    kRightHandRing     = 4
    kRightHandPinky    = 5


# UDCAP finger → Brainco motor index. Brainco uses 0=open, 1=closed
# (same direction as UDCAP flex_dict), so no inversion is required.
UDCAP_TO_BRAINCO_IDX = {
    "Thumb":  0,  # thumb flex → DOF 0
    "Index":  2,
    "Middle": 3,
    "Ring":   4,
    "Pinky":  5,
}

# Brainco DOF 1 is thumb-aux (opposition/rotation). It is driven from
# the UDCAP thumb-spread channel. Set to True if you need to invert
# the spread direction (more spread → more open instead of more closed).
INVERT_THUMB_SPREAD = False


def _udcap_to_brainco_mapper(side, flex_dict, thumb_spread):
    """Translate UDCAP flex/spread → Brainco 6-DOF (0.0=open, 1.0=closed)."""
    brainco = [0.0] * Brainco_Num_Motors
    for finger, idx in UDCAP_TO_BRAINCO_IDX.items():
        brainco[idx] = float(np.clip(flex_dict[finger], 0.0, 1.0))
    spread = thumb_spread if not INVERT_THUMB_SPREAD else (1.0 - thumb_spread)
    brainco[1] = float(np.clip(spread, 0.0, 1.0))
    return brainco


# ---------------------------------------------------------------------------
#  Main controller (same process/thread model as Brainco_Controller)
# ---------------------------------------------------------------------------

class Brainco_Controller_UDCAP:
    """Brainco hand controller driven by UDCAP haptic gloves via OSC."""

    def __init__(
        self,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        simulation_mode=False,
        udcap_host=DEFAULT_UDCAP_HOST,
        udcap_port=DEFAULT_UDCAP_PORT,
    ):
        logger_mp.info("Initialize Brainco_Controller_UDCAP...")
        self.fps = fps
        self.simulation_mode = simulation_mode

        # ---- DDS init (may already be initialised by robot_arm) -----------
        try:
            if self.simulation_mode:
                ChannelFactoryInitialize(1)
            else:
                ChannelFactoryInitialize(0)
        except Exception:
            pass

        # ---- DDS publishers (Brainco commands) ----------------------------
        self.LeftHandCmd_pub = ChannelPublisher(
            kTopicBraincoLeftCommand, MotorCmds_)
        self.LeftHandCmd_pub.Init()
        self.RightHandCmd_pub = ChannelPublisher(
            kTopicBraincoRightCommand, MotorCmds_)
        self.RightHandCmd_pub.Init()

        # ---- DDS subscribers (Brainco state feedback) ---------------------
        self.LeftHandState_sub = ChannelSubscriber(
            kTopicBraincoLeftState, MotorStates_)
        self.LeftHandState_sub.Init()
        self.RightHandState_sub = ChannelSubscriber(
            kTopicBraincoRightState, MotorStates_)
        self.RightHandState_sub.Init()

        # ---- Shared arrays ------------------------------------------------
        self.left_hand_state_array  = Array('d', Brainco_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Brainco_Num_Motors, lock=True)

        self.uc_left_mapped  = Array('d', Brainco_Num_Motors, lock=True)
        self.uc_right_mapped = Array('d', Brainco_Num_Motors, lock=True)
        for arr in (self.uc_left_mapped, self.uc_right_mapped):
            with arr.get_lock():
                for i in range(Brainco_Num_Motors):
                    arr[i] = 0.0  # open (Brainco convention)

        # ---- OSC UDCAP subscriber -----------------------------------------
        try:
            self.osc_bridge = UDCAPOSCBridge(
                self.uc_left_mapped, self.uc_right_mapped,
                mapper=_udcap_to_brainco_mapper,
                host=udcap_host, port=udcap_port,
                log_tag="UDCAP→Brainco",
            )
        except Exception as e:
            logger_mp.error(f"[UDCAP] OSC bridge init failed: {e}")
            self.osc_bridge = None

        # ---- DDS state subscriber thread ----------------------------------
        self._hand_sub_ready = False
        self._state_thread = threading.Thread(
            target=self._subscribe_hand_state, daemon=True)
        self._state_thread.start()

        wait_count = 0
        while not self._hand_sub_ready:
            time.sleep(0.1)
            wait_count += 1
            if wait_count % 10 == 0:
                logger_mp.warning("[Brainco-UDCAP] Waiting to subscribe dds...")
            if wait_count > 100:
                logger_mp.warning("[Brainco-UDCAP] Timeout waiting for hand states. Proceeding.")
                break
        logger_mp.info("[Brainco-UDCAP] Subscribe dds ok.")

        # ---- Control process ----------------------------------------------
        proc = Process(
            target=self.control_process,
            args=(
                self.uc_left_mapped, self.uc_right_mapped,
                self.left_hand_state_array, self.right_hand_state_array,
                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
            ),
        )
        proc.daemon = True
        proc.start()

        logger_mp.info("Initialize Brainco_Controller_UDCAP OK!\n")

    # ---- DDS state feedback -----------------------------------------------

    def _subscribe_hand_state(self):
        logger_mp.info("[Brainco-UDCAP] DDS state subscribe thread started.")
        while True:
            left_msg  = self.LeftHandState_sub.Read()
            right_msg = self.RightHandState_sub.Read()
            if left_msg is not None and right_msg is not None:
                self._hand_sub_ready = True
                with self.left_hand_state_array.get_lock():
                    for idx, jid in enumerate(_BraincoLeftIndex):
                        self.left_hand_state_array[idx] = left_msg.states[jid].q
                with self.right_hand_state_array.get_lock():
                    for idx, jid in enumerate(_BraincoRightIndex):
                        self.right_hand_state_array[idx] = right_msg.states[jid].q
            time.sleep(0.002)

    # ---- DDS command publishing -------------------------------------------

    def _init_cmd_messages(self):
        left_cmd  = MotorCmds_()
        left_cmd.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(_BraincoLeftIndex))]
        right_cmd = MotorCmds_()
        right_cmd.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(_BraincoRightIndex))]
        for idx, jid in enumerate(_BraincoLeftIndex):
            left_cmd.cmds[jid].q  = 0.0
            left_cmd.cmds[jid].dq = 1.0
        for idx, jid in enumerate(_BraincoRightIndex):
            right_cmd.cmds[jid].q  = 0.0
            right_cmd.cmds[jid].dq = 1.0
        return left_cmd, right_cmd

    def _send_hand_command(self, left_cmd, right_cmd, left_q, right_q):
        for idx, jid in enumerate(_BraincoLeftIndex):
            left_cmd.cmds[jid].q = float(left_q[idx])
        for idx, jid in enumerate(_BraincoRightIndex):
            right_cmd.cmds[jid].q = float(right_q[idx])
        self.LeftHandCmd_pub.Write(left_cmd)
        self.RightHandCmd_pub.Write(right_cmd)

    # ---- Control loop (runs in child Process) -----------------------------

    def control_process(
        self,
        uc_left_mapped, uc_right_mapped,
        left_hand_state_array, right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        logger_mp.info("[Brainco-UDCAP] Control process started.")
        self.running = True
        left_cmd, right_cmd = self._init_cmd_messages()

        try:
            while self.running:
                t0 = time.time()

                with uc_left_mapped.get_lock():
                    left_q = np.array(uc_left_mapped[:])
                with uc_right_mapped.get_lock():
                    right_q = np.array(uc_right_mapped[:])

                state_data = np.concatenate((
                    np.array(left_hand_state_array[:]),
                    np.array(right_hand_state_array[:]),
                ))

                action_data = np.concatenate((left_q, right_q))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:]  = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(left_cmd, right_cmd, left_q, right_q)

                sleep_time = max(0, (1.0 / self.fps) - (time.time() - t0))
                time.sleep(sleep_time)
        finally:
            logger_mp.info("[Brainco-UDCAP] Control process closed.")

"""
Inspire Hand Controller with UDCAP haptic glove input via OSC.

Receives per-finger flex from the shared UDCAP OSC bridge and maps the
result directly to Inspire 6-DOF motor commands via DDS.

This mirrors the SenseGlove integration (`Inspire_Controller_SenseGlove`)
but replaces the ROS2/SenseGlove bridge with the OSC server defined in
`udcap_osc_bridge.UDCAPOSCBridge`.

The Meta Quest controllers continue to provide arm/wrist tracking while
the UDCAP gloves provide finger tracking.

Usage:
    python teleop_hand_and_arm.py --input-mode controller --ee inspire_ftp_uc
"""

import threading
import time
from multiprocessing import Process, Array

import numpy as np

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from inspire_sdkpy import inspire_dds, inspire_hand_defaut

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

from teleop.robot_control.udcap_osc_bridge import (
    UDCAPOSCBridge, DEFAULT_UDCAP_HOST, DEFAULT_UDCAP_PORT, UDCAP_FINGERS,
)

# ── Inspire DDS topics (same as FTP controller) ───────────────────────────
Inspire_Num_Motors = 6

kTopicInspireFTPLeftCommand  = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState    = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState   = "rt/inspire_hand/state/r"

# Mapping from UDCAP finger to Inspire DOF index.
# Inspire motor order: [0:pinky, 1:ring, 2:middle, 3:index, 4:thumb_bend, 5:thumb_rotation]
UDCAP_TO_INSPIRE_DOF = {
    "Pinky":  0,
    "Ring":   1,
    "Middle": 2,
    "Index":  3,
    "Thumb":  4,  # thumb flex → thumb_bend; thumb_rotation handled via spread
}


def _udcap_to_inspire_mapper(side, flex_dict, thumb_spread):
    """Translate UDCAP flex/spread → Inspire 6-DOF (1.0=open, 0.0=closed)."""
    inspire = [1.0] * Inspire_Num_Motors
    for finger, dof in UDCAP_TO_INSPIRE_DOF.items():
        inspire[dof] = float(np.clip(1.0 - flex_dict[finger], 0.0, 1.0))
    # Thumb rotation: more spread → more open (DOF 5 = 1.0).
    inspire[5] = float(np.clip(1.0 - thumb_spread, 0.0, 1.0))
    return inspire


# ---------------------------------------------------------------------------
#  Main controller (same process/thread model as Inspire_Controller_SenseGlove)
# ---------------------------------------------------------------------------

class Inspire_Controller_UDCAP:
    """Inspire FTP hand controller driven by UDCAP haptic gloves via OSC."""

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
        logger_mp.info("Initialize Inspire_Controller_UDCAP...")
        self.fps = fps
        self.simulation_mode = simulation_mode

        # ---- DDS init (may already be initialised by robot_arm) -----------
        try:
            if self.simulation_mode:
                ChannelFactoryInitialize(1, "enp39s0")
            else:
                ChannelFactoryInitialize(0)
        except Exception:
            pass

        # ---- DDS publishers (Inspire commands) ----------------------------
        self.LeftHandCmd_pub = ChannelPublisher(
            kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
        self.LeftHandCmd_pub.Init()
        self.RightHandCmd_pub = ChannelPublisher(
            kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
        self.RightHandCmd_pub.Init()

        # ---- DDS subscribers (Inspire state feedback) ---------------------
        self.LeftHandState_sub = ChannelSubscriber(
            kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
        self.LeftHandState_sub.Init()
        self.RightHandState_sub = ChannelSubscriber(
            kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
        self.RightHandState_sub.Init()

        # ---- Shared arrays ------------------------------------------------
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        # Kept for parity with SenseGlove; UDCAP has no haptic feedback channel.
        self.left_hand_force_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_force_array = Array('d', Inspire_Num_Motors, lock=True)

        self.uc_left_mapped  = Array('d', Inspire_Num_Motors, lock=True)
        self.uc_right_mapped = Array('d', Inspire_Num_Motors, lock=True)
        for arr in (self.uc_left_mapped, self.uc_right_mapped):
            with arr.get_lock():
                for i in range(Inspire_Num_Motors):
                    arr[i] = 1.0  # open

        # ---- OSC UDCAP subscriber -----------------------------------------
        try:
            self.osc_bridge = UDCAPOSCBridge(
                self.uc_left_mapped, self.uc_right_mapped,
                mapper=_udcap_to_inspire_mapper,
                host=udcap_host, port=udcap_port,
                log_tag="UDCAP→Inspire",
            )
        except Exception as e:
            logger_mp.error(f"[UDCAP] OSC bridge init failed: {e}")
            self.osc_bridge = None

        # ---- DDS state subscriber thread ----------------------------------
        self._state_thread = threading.Thread(
            target=self._subscribe_hand_state, daemon=True)
        self._state_thread.start()

        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0:
                logger_mp.info("[UDCAP] Waiting for Inspire hand state DDS...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning("[UDCAP] Timeout waiting for hand states. Proceeding.")
                break

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

        logger_mp.info("Initialize Inspire_Controller_UDCAP OK!\n")

    # ---- DDS state feedback -----------------------------------------------

    def _subscribe_hand_state(self):
        logger_mp.info("[UDCAP] DDS state subscribe thread started.")
        while True:
            left_msg = self.LeftHandState_sub.Read()
            if left_msg is not None:
                if hasattr(left_msg, 'angle_act') and len(left_msg.angle_act) == Inspire_Num_Motors:
                    with self.left_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_state_array[i] = left_msg.angle_act[i] / 1000.0
                if hasattr(left_msg, 'force_act') and len(left_msg.force_act) == Inspire_Num_Motors:
                    with self.left_hand_force_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.left_hand_force_array[i] = float(left_msg.force_act[i])

            right_msg = self.RightHandState_sub.Read()
            if right_msg is not None:
                if hasattr(right_msg, 'angle_act') and len(right_msg.angle_act) == Inspire_Num_Motors:
                    with self.right_hand_state_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_state_array[i] = right_msg.angle_act[i] / 1000.0
                if hasattr(right_msg, 'force_act') and len(right_msg.force_act) == Inspire_Num_Motors:
                    with self.right_hand_force_array.get_lock():
                        for i in range(Inspire_Num_Motors):
                            self.right_hand_force_array[i] = float(right_msg.force_act[i])

            time.sleep(0.002)

    # ---- DDS command publishing -------------------------------------------

    def _send_hand_command(self, left_scaled, right_scaled):
        left_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        left_cmd.angle_set = left_scaled
        left_cmd.mode = 0b0001
        self.LeftHandCmd_pub.Write(left_cmd)

        right_cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        right_cmd.angle_set = right_scaled
        right_cmd.mode = 0b0001
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
        logger_mp.info("[UDCAP] Control process started.")
        self.running = True

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

                scaled_left  = [int(np.clip(v * 1000, 0, 1000)) for v in left_q]
                scaled_right = [int(np.clip(v * 1000, 0, 1000)) for v in right_q]

                action_data = np.concatenate((left_q, right_q))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:]  = state_data
                        dual_hand_action_array[:] = action_data

                self._send_hand_command(scaled_left, scaled_right)

                sleep_time = max(0, (1.0 / self.fps) - (time.time() - t0))
                time.sleep(sleep_time)
        finally:
            logger_mp.info("[UDCAP] Control process closed.")

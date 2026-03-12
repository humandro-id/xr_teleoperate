"""
Inspire Hand Controller with SenseGlove Nova 2 input via ROS2.

Subscribes to SenseGlove joint_states via ROS2 and maps finger joint
angles directly to Inspire hand motor commands via DDS — bypassing the
landmark-based retargeting pipeline used by the VR hand-tracking path.

Usage:
    python teleop_hand_and_arm.py --input-mode controller --ee inspire_ftp_sg

The Meta Quest controllers provide arm/wrist tracking while the
SenseGlove Nova 2 gloves provide finger tracking.
"""

from unitree_sdk2py.core.channel import (
    ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize,
)
from inspire_sdkpy import inspire_dds, inspire_hand_defaut
import numpy as np
import threading
import time
from multiprocessing import Process, Array

from std_msgs.msg import Float64MultiArray
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6

kTopicInspireFTPLeftCommand  = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState    = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState   = "rt/inspire_hand/state/r"

DEFAULT_LEFT_GLOVE_TOPIC  = "/senseglove/glove00799/lh/joint_states"
DEFAULT_RIGHT_GLOVE_TOPIC = "/senseglove/glove00768/rh/joint_states"

# Typical max flexion (rad) per finger for the SenseGlove Nova 2.
# Tune these if the mapping feels too sensitive or too sluggish.
SENSEGLOVE_JOINT_RANGES = {
    # Index
    'index_mcp':  (0.0, 1.2),
    'index_pip':  (0.0, 1.6),
    'index_dip':  (0.0, 1.5),
    # Middle
    'middle_mcp': (0.0, 1.1),
    'middle_pip': (0.0, 1.6),
    'middle_dip': (0.0, 1.4),
    # Ring
    'ring_mcp':   (0.0, 1.0),
    'ring_pip':   (0.0, 1.5),
    'ring_dip':   (0.0, 1.3),
    # Pinky
    'pinky_mcp':  (0.0, 1.0),
    'pinky_pip':  (0.0, 1.5),
    'pinky_dip':  (0.0, 1.5),  # dead on some units — safely ignored
    # Thumb flexion → Inspire DOF 4 (thumb_bend)
    'thumb_pip':  (0.0, 0.6),
    'thumb_dip':  (0.0, 1.4),
}

# thumb_mcp uses negative values: more negative = more closed.
# Maps to Inspire DOF 5 (thumb_rotation).
THUMB_MCP_RANGE = (-0.7, 0.0)  # (most_closed, most_open)

# Joints to ignore (haptic brakes, palm sensors, strap)
_IGNORED_SUFFIXES = ('brake', 'palm_pinky', 'palm_index', 'palm_strap')

# ── Haptic feedback config ────────────────────────────────────────────
# SenseGlove Nova 2 haptics_controller joint names (9 per hand).
# Order must match the haptics_controller/controller_state output.
LEFT_HAPTICS_JOINTS = [
    'l_thumb_brake', 'l_index_brake', 'l_middle_brake', 'l_ring_brake',
    'l_thumb_dip', 'l_index_dip',
    'l_palm_index', 'l_palm_pinky', 'l_palm_strap',
]
RIGHT_HAPTICS_JOINTS = [
    'r_thumb_brake', 'r_index_brake', 'r_middle_brake', 'r_ring_brake',
    'r_thumb_dip', 'r_index_dip',
    'r_palm_index', 'r_palm_pinky', 'r_palm_strap',
]
NUM_HAPTICS_JOINTS = 9

#LEFT_HAPTICS_TOPIC  = '/senseglove/glove00801/lh/haptics_controller/joint_trajectory'
#RIGHT_HAPTICS_TOPIC = '/senseglove/glove00804/rh/haptics_controller/joint_trajectory'
LEFT_HAPTICS_TOPIC  = '/senseglove/glove00799/lh/haptics_commands'
RIGHT_HAPTICS_TOPIC = '/senseglove/glove00768/rh/haptics_commands'

# Inspire DOF → SenseGlove brake index mapping
# Inspire: [0:pinky, 1:ring, 2:middle, 3:index, 4:thumb_bend, 5:thumb_rot]
# Brakes:  [0:thumb, 1:index, 2:middle, 3:ring]  (no pinky brake on Nova 2)
INSPIRE_TO_BRAKE = {
    1: 3,  # ring   → ring_brake   (idx 3)
    2: 2,  # middle → middle_brake (idx 2)
    3: 1,  # index  → index_brake  (idx 1)
    4: 0,  # thumb  → thumb_brake  (idx 0)
}

# Inspire force_act: 0–4096.  Dead zone below 200, full brake at 4096.
HAPTIC_FORCE_DEADZONE = 200
HAPTIC_FORCE_MAX = 4096# Inspire force_act: 0–4096.  Dead zone below 200, full brake at 4096.




# ---------------------------------------------------------------------------
#  ROS2 bridge – runs in the main process, writes to shared Arrays
# ---------------------------------------------------------------------------

class SenseGloveROS2Bridge:
    """Subscribes to SenseGlove joint_states, writes Inspire-mapped
    6-DOF values into shared Arrays, and publishes haptic brake
    feedback based on Inspire motor force (force_act)."""

    def __init__(self, left_topic, right_topic,
                 left_mapped_array, right_mapped_array,
                 left_force_array=None, right_force_array=None):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node('senseglove_inspire_bridge')
        self._left_arr = left_mapped_array
        self._right_arr = right_mapped_array
        self._left_force_arr = left_force_array
        self._right_force_arr = right_force_array
        self._names_logged_l = False
        self._names_logged_r = False
        self._latest_left_msg = None
        self._latest_right_msg = None

        # ── Finger tracking subscribers ──
        # SensorDataQoS: BEST_EFFORT, KEEP_LAST, depth=5, VOLATILE
        sub_qos = qos_profile_sensor_data
        sub_qos.depth = 1
        self._node.create_subscription(
            JointState, left_topic, self._cb_left, sub_qos)
        self._node.create_subscription(
            JointState, right_topic, self._cb_right, sub_qos)

        # ── Haptic feedback publishers ──
        pub_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
        )
        self._left_haptics_pub = self._node.create_publisher(
            Float64MultiArray, LEFT_HAPTICS_TOPIC, pub_qos)
        self._right_haptics_pub = self._node.create_publisher(
            Float64MultiArray, RIGHT_HAPTICS_TOPIC, pub_qos)

        # Timer to process latest SenseGlove msgs + publish haptics at 20 Hz
        self._node.create_timer(0.05, self._process_tick)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        logger_mp.info(
            f"[SenseGloveROS2Bridge] Subscribed to L={left_topic}  R={right_topic}")
        logger_mp.info(
            f"[SenseGloveROS2Bridge] Haptics publishing to L={LEFT_HAPTICS_TOPIC}  R={RIGHT_HAPTICS_TOPIC}")

    # ---- callbacks (lightweight: just store latest msg) --------------------

    def _cb_left(self, msg):
        if not self._names_logged_l:
            logger_mp.info(f"[SenseGlove L] joint names: {list(msg.name)}")
            self._names_logged_l = True
        self._latest_left_msg = msg

    def _cb_right(self, msg):
        if not self._names_logged_r:
            logger_mp.info(f"[SenseGlove R] joint names: {list(msg.name)}")
            self._names_logged_r = True
        self._latest_right_msg = msg

    # ---- process tick (mapping + haptics) ----------------------------------

    def _process_tick(self):
        """Process latest SenseGlove messages and publish haptic feedback."""
        # Map latest SenseGlove data to Inspire shared arrays
        left_msg = self._latest_left_msg
        if left_msg is not None:
            mapped = self._map_to_inspire(left_msg)
            with self._left_arr.get_lock():
                self._left_arr[:] = mapped

        right_msg = self._latest_right_msg
        if right_msg is not None:
            mapped = self._map_to_inspire(right_msg)
            with self._right_arr.get_lock():
                self._right_arr[:] = mapped

        # Publish haptic feedback from Inspire motor forces
        if self._left_force_arr is not None:
            with self._left_force_arr.get_lock():
                left_force = list(self._left_force_arr[:])
            with self._right_force_arr.get_lock():
                right_force = list(self._right_force_arr[:])

            self._left_haptics_pub.publish(
                self._build_haptics_msg(LEFT_HAPTICS_JOINTS, self._force_to_brakes(left_force)))
            self._right_haptics_pub.publish(
                self._build_haptics_msg(RIGHT_HAPTICS_JOINTS, self._force_to_brakes(right_force)))

    @staticmethod
    def _force_to_brakes(force_act):
        """Convert Inspire force_act[6] to SenseGlove 9-element haptics array.

        Mapping: force < 200 → brake 0,  force 4096 → brake 100.
        Linear interpolation between deadzone and max.
        """
        brakes = [0.0] * NUM_HAPTICS_JOINTS
        usable_range = HAPTIC_FORCE_MAX - HAPTIC_FORCE_DEADZONE

        for inspire_dof, brake_idx in INSPIRE_TO_BRAKE.items():
            raw = abs(force_act[inspire_dof])
            if raw <= HAPTIC_FORCE_DEADZONE:
                brakes[brake_idx] = 0.0
            else:
                brakes[brake_idx] = float(np.clip(
                    (raw - HAPTIC_FORCE_DEADZONE) / usable_range * 100.0,
                    0.0, 100.0))
        return brakes

    def _build_haptics_msg(self, joint_names, values):
        msg = Float64MultiArray()
        msg.data = list(values)
        return msg

    # ---- joint mapping ----------------------------------------------------

    @staticmethod
    def _map_to_inspire(msg):
        """Map a SenseGlove JointState message to Inspire 6-DOF values.

        Returns list of 6 floats in Inspire motor order:
            [pinky, ring, middle, index, thumb_bend, thumb_rotation]
        Values are normalised to [0, 1] where 1 = fully open.
        """
        joint_dict = dict(zip(msg.name, msg.position))
        result = np.ones(Inspire_Num_Motors, dtype=np.float64)

        # Collect flexion angles per finger
        finger_norm = {f: [] for f in ('pinky', 'ring', 'middle', 'index')}
        thumb_flex_norm = []
        thumb_rotation_val = None

        for name, pos in joint_dict.items():
            lo = name.lower()
            # Skip haptic-brake, palm-sensor and strap joints
            if any(lo.endswith(s) for s in _IGNORED_SUFFIXES):
                continue

            # Strip l_/r_ prefix → canonical name (e.g. "index_mcp")
            parts = lo.split('_', 1)
            if len(parts) == 2 and parts[0] in ('l', 'r'):
                canonical = parts[1]
            else:
                canonical = lo

            # ── thumb_mcp → Inspire DOF 5 (thumb_rotation) ──
            # Negative values; more negative = more closed.
            if canonical == 'thumb_mcp':
                thumb_rotation_val = pos
                continue

            # ── thumb_pip / thumb_dip → Inspire DOF 4 (thumb_bend) ──
            if canonical.startswith('thumb_'):
                jrange = SENSEGLOVE_JOINT_RANGES.get(canonical)
                if jrange:
                    open_v, closed_v = jrange
                    rng = closed_v - open_v
                    if rng > 0.01:
                        thumb_flex_norm.append(
                            np.clip((pos - open_v) / rng, 0.0, 1.0))
                continue

            # ── four fingers → Inspire DOF 0-3 ──
            for fname in finger_norm:
                if canonical.startswith(fname + '_'):
                    jrange = SENSEGLOVE_JOINT_RANGES.get(canonical)
                    if jrange:
                        open_v, closed_v = jrange
                        rng = closed_v - open_v
                        if rng > 0.01:
                            finger_norm[fname].append(
                                np.clip((pos - open_v) / rng, 0.0, 1.0))
                    break

                # Normalized flex values are 0=open, 1=closed → invert to Inspire convention
        for i, fname in enumerate(('pinky', 'ring', 'middle', 'index')):
            if finger_norm[fname]:
                result[i] = 1.0 - np.mean(finger_norm[fname])

        if thumb_flex_norm:
            result[4] = 1.0 - np.mean(thumb_flex_norm)

        if thumb_rotation_val is not None:
            closed_v, open_v = THUMB_MCP_RANGE  # (-0.7, 0.0)
            rng = open_v - closed_v
            if rng > 0.01:
                result[5] = np.clip(
                    (thumb_rotation_val - closed_v) / rng, 0.0, 1.0)

        return result.tolist()

    # ---- lifecycle --------------------------------------------------------

    def _spin(self):
        try:
            self._rclpy.spin(self._node)
        except Exception as e:
            logger_mp.error(f"[SenseGloveROS2Bridge] spin error: {e}")

    def shutdown(self):
        try:
            self._node.destroy_node()
        except Exception:
            pass
        try:
            if self._rclpy.ok():
                self._rclpy.shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
#  Main controller (same process/thread model as Inspire_Controller_FTP)
# ---------------------------------------------------------------------------

class Inspire_Controller_SenseGlove:
    """Inspire FTP hand controller driven by SenseGlove Nova 2 via ROS2."""

    def __init__(
        self,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
        fps=100.0,
        simulation_mode=False,
        left_glove_topic=DEFAULT_LEFT_GLOVE_TOPIC,
        right_glove_topic=DEFAULT_RIGHT_GLOVE_TOPIC,
    ):
        logger_mp.info("Initialize Inspire_Controller_SenseGlove...")
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
        self.left_hand_force_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_force_array = Array('d', Inspire_Num_Motors, lock=True)

        self.sg_left_mapped  = Array('d', Inspire_Num_Motors, lock=True)
        self.sg_right_mapped = Array('d', Inspire_Num_Motors, lock=True)
        for arr in (self.sg_left_mapped, self.sg_right_mapped):
            with arr.get_lock():
                for i in range(Inspire_Num_Motors):
                    arr[i] = 1.0  # open

        # ---- ROS2 SenseGlove subscriber + haptic publisher -----------------
        try:
            self.ros2_bridge = SenseGloveROS2Bridge(
                left_glove_topic, right_glove_topic,
                self.sg_left_mapped, self.sg_right_mapped,
                self.left_hand_force_array, self.right_hand_force_array,
            )
        except Exception as e:
            print(e)

        # ---- DDS state subscriber thread ----------------------------------
        self._state_thread = threading.Thread(
            target=self._subscribe_hand_state, daemon=True)
        self._state_thread.start()

        wait_count = 0
        while not (any(self.left_hand_state_array) or any(self.right_hand_state_array)):
            if wait_count % 100 == 0:
                logger_mp.info("[SenseGlove] Waiting for Inspire hand state DDS...")
            time.sleep(0.01)
            wait_count += 1
            if wait_count > 500:
                logger_mp.warning("[SenseGlove] Timeout waiting for hand states. Proceeding.")
                break

        # ---- Control process ----------------------------------------------
        proc = Process(
            target=self.control_process,
            args=(
                self.sg_left_mapped, self.sg_right_mapped,
                self.left_hand_state_array, self.right_hand_state_array,
                dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
            ),
        )
        proc.daemon = True
        proc.start()

        logger_mp.info("Initialize Inspire_Controller_SenseGlove OK!\n")

    # ---- DDS state feedback -----------------------------------------------

    def _subscribe_hand_state(self):
        logger_mp.info("[SenseGlove] DDS state subscribe thread started.")
        force_logged = False
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
                    if not force_logged:
                        logger_mp.info(f"[SenseGlove] force_act L: {list(left_msg.force_act)}")

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
                    if not force_logged:
                        logger_mp.info(f"[SenseGlove] force_act R: {list(right_msg.force_act)}")
                        force_logged = True

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
        sg_left_mapped, sg_right_mapped,
        left_hand_state_array, right_hand_state_array,
        dual_hand_data_lock=None,
        dual_hand_state_array=None,
        dual_hand_action_array=None,
    ):
        logger_mp.info("[SenseGlove] Control process started.")
        self.running = True

        try:
            while self.running:
                t0 = time.time()

                with sg_left_mapped.get_lock():
                    left_q = np.array(sg_left_mapped[:])
                with sg_right_mapped.get_lock():
                    right_q = np.array(sg_right_mapped[:])

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
            logger_mp.info("[SenseGlove] Control process closed.")

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
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_
from inspire_sdkpy import inspire_dds, inspire_hand_defaut
import numpy as np
import threading
import time
from multiprocessing import Process, Array

from std_msgs.msg import Float64MultiArray
from teleop.robot_control.senseglove_profile import DEFAULT_PROFILE, load_profile
import logging_mp
logger_mp = logging_mp.get_logger(__name__)

Inspire_Num_Motors = 6

# ── Real Inspire FTP hand (hardware) topics ──────────────────────────
kTopicInspireFTPLeftCommand  = "rt/inspire_hand/ctrl/l"
kTopicInspireFTPRightCommand = "rt/inspire_hand/ctrl/r"
kTopicInspireFTPLeftState    = "rt/inspire_hand/state/l"
kTopicInspireFTPRightState   = "rt/inspire_hand/state/r"
kTopicInspireFTPRightTouch   = "rt/inspire_hand/touch/r"
kTopicInspireFTPLeftTouch   = "rt/inspire_hand/touch/l"

# ── unitree_sim_isaaclab (simulation) topics ─────────────────────────
# The simulator's InspireDDS subscribes to a single MotorCmds_ message
# with 12 motors: indices 0-5 = RIGHT hand, 6-11 = LEFT hand, in order
# [pinky, ring, middle, index, thumb_bend, thumb_rotation].
# q values are normalised floats in [0, 1] where 1.0 = fully open
# (the sim denormalises them to joint radians internally).
kTopicInspireSimCommand = "rt/inspire/cmd"
kTopicInspireSimState   = "rt/inspire/state"
SIM_RIGHT_HAND_OFFSET = 0   # cmds[0..5]  → right hand
SIM_LEFT_HAND_OFFSET  = 6   # cmds[6..11] → left hand
SIM_NUM_MOTORS_TOTAL  = 12

# Tunables (finger ranges, thumb saturation, FFB, touch, mount offset, timing,
# glove namespaces) live in SenseGlove profiles: see senseglove_profile.py and
# teleop/config/senseglove_profiles.yaml.

# Thumb flexion joints → Inspire DOF 4 (thumb_bend / proximal_pitch).
# Nova 2 CMC flexion can come out positive or negative; we use abs() and
# adapt to the travel the user can actually reach.
THUMB_FLEXION_JOINTS = ('thumb_mcp', 'thumb_pip', 'thumb_dip')

# Joints to ignore (haptic brakes of other fingers, palm sensors, strap).
# thumb_brake is NOT ignored: it carries thumb abduction.
_IGNORED_SUFFIXES = ('brake', 'palm_pinky', 'palm_index', 'palm_strap')
_THUMB_BRAKE_CANONICAL = 'thumb_brake'

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

# Finger → (Inspire DOF whose force_act drives it, SenseGlove brake index)
# Inspire: [0:pinky, 1:ring, 2:middle, 3:index, 4:thumb_bend, 5:thumb_rot]
# Brakes:  [0:thumb, 1:index, 2:middle, 3:ring]  (no pinky brake on Nova 2)
FFB_FINGER_MAP = {
    'thumb':  (4, 0),
    'index':  (3, 1),
    'middle': (2, 2),
    'ring':   (1, 3),
}

#Number of touch values to send to gloves {pinky_palm, index_palm, thumb_top, index_top}
NUM_TOUCH_DATA = 4

# Nova 2 LRA motor (vibration_commands order) → index in the touch array
VIBRATION_MOTOR_TOUCH_IDX = {
    'thumb_tip':  2,
    'index_tip':  3,
    'palm_index': 1,
    'palm_pinky': 0,
}

def _euler_to_se3(rx_deg, ry_deg, rz_deg, tx=0.0, ty=0.0, tz=0.0):
    """Build a 4x4 SE(3) matrix from Euler angles (XYZ intrinsic) + translation."""
    rx, ry, rz = np.radians(rx_deg), np.radians(ry_deg), np.radians(rz_deg)
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T

_SG_MOUNT_OFFSET = {}


def set_senseglove_mount_offset(mount_cfg):
    pos = mount_cfg['pos']
    _SG_MOUNT_OFFSET['right'] = _euler_to_se3(*mount_cfg['right_euler_deg'], *pos)
    _SG_MOUNT_OFFSET['left'] = _euler_to_se3(*mount_cfg['left_euler_deg'], *pos)


set_senseglove_mount_offset(DEFAULT_PROFILE['mount_offset'])


def apply_senseglove_mount_offset(wrist_pose, is_right = True):
    """Apply SenseGlove controller mount compensation to a 4x4 wrist pose."""
    return wrist_pose @ _SG_MOUNT_OFFSET['right' if is_right else 'left']


class AdaptiveRange:
    """Expanding min/max so the user's reachable travel maps to [0, 1]."""

    def __init__(self, inner_frac, min_span, hi_frac=None):
        self.lo = None
        self.hi = None
        self.inner_frac = inner_frac
        self.hi_frac = inner_frac if hi_frac is None else hi_frac
        self.min_span = min_span

    def update(self, x):
        x = float(x)
        if self.lo is None:
            self.lo = self.hi = x
            return
        self.lo = min(self.lo, x)
        self.hi = max(self.hi, x)

    def normalize(self, x, invert=False):
        """0 at observed lo, 1 at observed hi. Saturates before extrema."""
        if self.lo is None or self.hi is None:
            return 0.5
        span = self.hi - self.lo
        if span < self.min_span:
            return 0.5
        lo = self.lo + self.inner_frac * span
        hi = self.hi - self.hi_frac * span
        if hi - lo < 1e-4:
            return 0.5
        n = float(np.clip((float(x) - lo) / (hi - lo), 0.0, 1.0))
        return 1.0 - n if invert else n

    def span_str(self):
        if self.lo is None:
            return "unseen"
        return f"{self.lo:.3f}:{self.hi:.3f}"


# ---------------------------------------------------------------------------
#  ROS2 bridge – runs in the main process, writes to shared Arrays
# ---------------------------------------------------------------------------

class SenseGloveROS2Bridge:
    """Subscribes to SenseGlove joint_states, writes Inspire-mapped
    6-DOF values into shared Arrays, and publishes haptic brake
    feedback based on Inspire motor force (force_act)."""

    def __init__(self, profile,
                 left_mapped_array, right_mapped_array,
                 left_force_array=None, right_force_array=None,
                 left_touch_array=None, right_touch_array=None):
        import rclpy
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_sensor_data
        from sensor_msgs.msg import JointState

        self._rclpy = rclpy

        if not rclpy.ok():
            rclpy.init()

        self._profile = profile
        self._finger_ranges = profile['mapping']['finger_ranges']
        thumb_cfg = profile['mapping']['thumb']
        self._yaw_invert = thumb_cfg['yaw_invert']
        self._ffb = profile['force_feedback']
        self._vib = profile['vibration']
        left_topic, left_haptics_topic, left_vib_topic = profile.glove_topics('left')
        right_topic, right_haptics_topic, right_vib_topic = profile.glove_topics('right')

        self._node = rclpy.create_node('senseglove_inspire_bridge')
        self._left_arr = left_mapped_array
        self._right_arr = right_mapped_array
        self._left_force_arr = left_force_array
        self._right_force_arr = right_force_array
        self._left_touch_arr = left_touch_array
        self._right_touch_arr = right_touch_array
        self._names_logged_l = False
        self._names_logged_r = False
        self._latest_left_msg = None
        self._latest_right_msg = None

        def _yaw():
            return AdaptiveRange(thumb_cfg['adapt_inner_frac'], thumb_cfg['adapt_min_span'])

        def _pitch():
            return AdaptiveRange(thumb_cfg['adapt_inner_frac'], thumb_cfg['adapt_min_span'],
                                 hi_frac=thumb_cfg['pitch_close_frac'])

        self._yaw_range = {'left': _yaw(), 'right': _yaw()}
        self._pitch_range = {'left': _pitch(), 'right': _pitch()}
        self._last_thumb_log = 0.0

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
            Float64MultiArray, left_haptics_topic, pub_qos)
        self._right_haptics_pub = self._node.create_publisher(
            Float64MultiArray, right_haptics_topic, pub_qos)
        self._left_vib_pub = self._node.create_publisher(
            Float64MultiArray, left_vib_topic, pub_qos)
        self._right_vib_pub = self._node.create_publisher(
            Float64MultiArray, right_vib_topic, pub_qos)

        # Timer to process latest SenseGlove msgs + publish haptics
        self._node.create_timer(
            1.0 / float(profile['timing']['bridge_rate_hz']), self._process_tick)

        self._spin_thread = threading.Thread(target=self._spin, daemon=True)
        self._spin_thread.start()
        logger_mp.info(
            f"[SenseGloveROS2Bridge] Subscribed to L={left_topic}  R={right_topic}")
        logger_mp.info(
            f"[SenseGloveROS2Bridge] Haptics publishing to L={left_haptics_topic}  R={right_haptics_topic}")
        logger_mp.info(
            f"[SenseGloveROS2Bridge] Vibration publishing to L={left_vib_topic}  R={right_vib_topic}")

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
            mapped = self._map_to_inspire(left_msg, 'left')
            with self._left_arr.get_lock():
                self._left_arr[:] = mapped

        right_msg = self._latest_right_msg
        if right_msg is not None:
            mapped = self._map_to_inspire(right_msg, 'right')
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

        # Publish vibration from Inspire touch sensors
        if self._vib['enabled'] and self._left_touch_arr is not None:
            with self._left_touch_arr.get_lock():
                left_touch = list(self._left_touch_arr[:])
            with self._right_touch_arr.get_lock():
                right_touch = list(self._right_touch_arr[:])

            self._left_vib_pub.publish(
                self._build_haptics_msg(None, self._touch_to_vibration(left_touch)))
            self._right_vib_pub.publish(
                self._build_haptics_msg(None, self._touch_to_vibration(right_touch)))

    def _touch_to_vibration(self, touch_levels):
        """Touch levels (0-100) → vibration_commands
        [thumb_tip, index_tip, palm_index, palm_pinky, frequency_hz, pulse_s]."""
        levels = [
            float(np.clip(touch_levels[touch_idx] * float(self._vib['gain'][motor]), 0.0, 100.0))
            for motor, touch_idx in VIBRATION_MOTOR_TOUCH_IDX.items()
        ]
        return levels + [float(self._vib['frequency_hz']), float(self._vib['pulse_s'])]

    def _force_to_brakes(self, force_act):
        """Convert Inspire force_act[6] to SenseGlove 9-element haptics array.

        Mapping: force <= deadzone → brake 0, force >= max → brake 100,
        linear in between, then scaled by the per-finger gain.
        """
        brakes = [0.0] * NUM_HAPTICS_JOINTS
        if not self._ffb['enabled']:
            return brakes

        deadzone = float(self._ffb['deadzone'])
        usable_range = max(float(self._ffb['max']) - deadzone, 1.0)

        for finger, (inspire_dof, brake_idx) in FFB_FINGER_MAP.items():
            raw = abs(force_act[inspire_dof])
            if raw <= deadzone:
                continue
            level = (raw - deadzone) / usable_range * 100.0
            brakes[brake_idx] = float(np.clip(
                level * float(self._ffb['gain'][finger]), 0.0, 100.0))
        return brakes

    def _build_haptics_msg(self, joint_names, values):
        msg = Float64MultiArray()
        msg.data = list(values)
        return msg

    # ---- joint mapping ----------------------------------------------------

    def _map_to_inspire(self, msg, side):
        """Map a SenseGlove JointState message to Inspire 6-DOF values.

        Returns list of 6 floats in Inspire motor order:
            [pinky, ring, middle, index, thumb_bend, thumb_rotation]
        Values are normalised to [0, 1] where 1 = fully open.

        Thumb yaw/pitch ranges are learned from the motion the glove actually
        reports (the Nova 2 never reaches the URDF joint limits).
        """
        joint_dict = dict(zip(msg.name, msg.position))
        result = np.ones(Inspire_Num_Motors, dtype=np.float64)

        finger_norm = {f: [] for f in ('pinky', 'ring', 'middle', 'index')}
        thumb_flex_abs = []
        thumb_raw = {}
        thumb_rotation_val = None

        for name, pos in joint_dict.items():
            lo = name.lower()

            parts = lo.split('_', 1)
            if len(parts) == 2 and parts[0] in ('l', 'r'):
                canonical = parts[1]
            else:
                canonical = lo

            if canonical == _THUMB_BRAKE_CANONICAL:
                thumb_rotation_val = pos
                thumb_raw['brake'] = pos
                continue

            if any(lo.endswith(s) for s in _IGNORED_SUFFIXES):
                continue

            if canonical in THUMB_FLEXION_JOINTS:
                thumb_flex_abs.append(abs(pos))
                thumb_raw[canonical] = pos
                continue

            for fname in finger_norm:
                if canonical.startswith(fname + '_'):
                    jrange = self._finger_ranges.get(canonical)
                    if jrange:
                        open_v, closed_v = jrange
                        rng = closed_v - open_v
                        if rng > 0.01:
                            finger_norm[fname].append(
                                np.clip((pos - open_v) / rng, 0.0, 1.0))
                    break

        for i, fname in enumerate(('pinky', 'ring', 'middle', 'index')):
            if finger_norm[fname]:
                result[i] = 1.0 - np.mean(finger_norm[fname])

        if thumb_flex_abs:
            curl = max(thumb_flex_abs)
            self._pitch_range[side].update(curl)
            # High curl → Inspire closed (0)
            result[4] = self._pitch_range[side].normalize(curl, invert=True)

        if thumb_rotation_val is not None:
            self._yaw_range[side].update(thumb_rotation_val)
            result[5] = self._yaw_range[side].normalize(
                thumb_rotation_val, invert=self._yaw_invert[side])

        now = time.time()
        if now - self._last_thumb_log > 2.0:
            self._last_thumb_log = now
            logger_mp.info(
                f"[SG {side} {self._profile.id}] raw brake={thumb_raw.get('brake', float('nan')):.3f} "
                f"mcp={thumb_raw.get('thumb_mcp', float('nan')):.3f} "
                f"pip={thumb_raw.get('thumb_pip', float('nan')):.3f} "
                f"dip={thumb_raw.get('thumb_dip', float('nan')):.3f} | "
                f"pitch={result[4]:.2f} yaw={result[5]:.2f} "
                f"pitch_span={self._pitch_range[side].span_str()} "
                f"yaw_span={self._yaw_range[side].span_str()}"
            )

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
        fps=None,
        simulation_mode=False,
        profile=None,
    ):
        logger_mp.info("Initialize Inspire_Controller_SenseGlove...")
        if profile is None:
            profile = load_profile()
        self.profile = profile
        self.fps = float(profile['timing']['control_fps']) if fps is None else fps
        self.simulation_mode = simulation_mode
        self._touch_cfg = profile['touch']
        self._force_set, self._speed_set = profile.inspire_settings()
        self._settings_resend_s = float(profile['inspire']['resend_s'])
        set_senseglove_mount_offset(profile['mount_offset'])
        logger_mp.info(
            f"[SenseGlove] Profile {profile.id} (hash {profile.params_hash}): "
            f"{profile.description}")
        logger_mp.info(
            f"[SenseGlove] Inspire force_set={self._force_set or 'hand default'} "
            f"speed_set={self._speed_set or 'hand default'}")

        # ---- DDS init (may already be initialised by robot_arm) -----------
        # NOTE: the simulator calls ChannelFactoryInitialize(1) with the
        # default network interface. Using the same signature here avoids
        # publisher/subscriber discovery mismatches across interfaces.
        try:
            if self.simulation_mode:
                ChannelFactoryInitialize(1)
            else:
                ChannelFactoryInitialize(0)
        except Exception:
            pass

        if self.simulation_mode:
            # ---- Simulation: single MotorCmds_ publisher (rt/inspire/cmd) --
            self.SimHandCmd_pub = ChannelPublisher(
                kTopicInspireSimCommand, MotorCmds_)
            self.SimHandCmd_pub.Init()

            # ---- Simulation: MotorStates_ subscriber (rt/inspire/state) ----
            self.SimHandState_sub = ChannelSubscriber(
                kTopicInspireSimState, MotorStates_)
            self.SimHandState_sub.Init()
        else:
            # ---- Real hand: FTP per-hand publishers -------------------------
            self.LeftHandCmd_pub = ChannelPublisher(
                kTopicInspireFTPLeftCommand, inspire_dds.inspire_hand_ctrl)
            self.LeftHandCmd_pub.Init()
            self.RightHandCmd_pub = ChannelPublisher(
                kTopicInspireFTPRightCommand, inspire_dds.inspire_hand_ctrl)
            self.RightHandCmd_pub.Init()

            # ---- Real hand: FTP per-hand state subscribers ------------------
            self.LeftHandState_sub = ChannelSubscriber(
                kTopicInspireFTPLeftState, inspire_dds.inspire_hand_state)
            self.LeftHandState_sub.Init()
            self.RightHandState_sub = ChannelSubscriber(
                kTopicInspireFTPRightState, inspire_dds.inspire_hand_state)
            self.RightHandState_sub.Init()

            # ---- Real hand: FTP per-hand touch subscribers -----------------
            self.LeftHandTouch_sub = ChannelSubscriber(
                kTopicInspireFTPLeftTouch, inspire_dds.inspire_hand_touch)
            self.LeftHandTouch_sub.Init()
            self.RightHandTouch_sub = ChannelSubscriber(
                kTopicInspireFTPRightTouch, inspire_dds.inspire_hand_touch)
            self.RightHandTouch_sub.Init()


        # ---- Shared arrays ------------------------------------------------
        self.left_hand_state_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_state_array = Array('d', Inspire_Num_Motors, lock=True)
        self.left_hand_force_array  = Array('d', Inspire_Num_Motors, lock=True)
        self.right_hand_force_array = Array('d', Inspire_Num_Motors, lock=True)

        self.sg_left_mapped  = Array('d', Inspire_Num_Motors, lock=True)
        self.sg_right_mapped = Array('d', Inspire_Num_Motors, lock=True)

        self.left_hand_touch_array = Array('d', NUM_TOUCH_DATA, lock=True)
        self.right_hand_touch_array = Array('d', NUM_TOUCH_DATA, lock=True)
        for arr in (self.sg_left_mapped, self.sg_right_mapped):
            with arr.get_lock():
                for i in range(Inspire_Num_Motors):
                    arr[i] = 1.0  # open

        # ---- ROS2 SenseGlove subscriber + haptic publisher -----------------
        # In simulation the Inspire hand has no force_act feedback, so we
        # skip haptics publishing entirely (force arrays would stay at 0).
        try:
            self.ros2_bridge = SenseGloveROS2Bridge(
                profile,
                self.sg_left_mapped, self.sg_right_mapped,
                None if self.simulation_mode else self.left_hand_force_array,
                None if self.simulation_mode else self.right_hand_force_array,
                None if self.simulation_mode else self.left_hand_touch_array,
                None if self.simulation_mode else self.right_hand_touch_array,
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
        if self.simulation_mode:
            self._subscribe_hand_state_sim()
        else:
            self._subscribe_hand_state_real()

    def _subscribe_hand_state_sim(self):
        """Read MotorStates_ from the simulator (rt/inspire/state).

        The sim publishes 12 motor states with q already normalised to
        [0, 1] (1 = open): indices 0-5 = right hand, 6-11 = left hand.
        There is no force feedback in simulation.
        """
        logger_mp.info("[SenseGlove] DDS state subscribe thread started (SIM mode).")
        while True:
            msg = self.SimHandState_sub.Read()
            if msg is not None and len(msg.states) >= SIM_NUM_MOTORS_TOTAL:
                with self.right_hand_state_array.get_lock():
                    for i in range(Inspire_Num_Motors):
                        self.right_hand_state_array[i] = float(
                            msg.states[SIM_RIGHT_HAND_OFFSET + i].q)
                with self.left_hand_state_array.get_lock():
                    for i in range(Inspire_Num_Motors):
                        self.left_hand_state_array[i] = float(
                            msg.states[SIM_LEFT_HAND_OFFSET + i].q)
            time.sleep(0.002)

    def _subscribe_hand_state_real(self):
        logger_mp.info("[SenseGlove] DDS state subscribe thread started.")
        force_logged = False
        while True:
            left_msg = self.LeftHandState_sub.Read()
            left_touch_msg = self.LeftHandTouch_sub.Read()
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
                        pass
                        #logger_mp.info(f"[SenseGlove] force_act L: {list(left_msg.force_act)}")
            if left_touch_msg is not None:
                with self.left_hand_touch_array.get_lock():
                    left_touch_data = self._process_touch_data(left_touch_msg)
                    for i in range(NUM_TOUCH_DATA):
                        self.left_hand_touch_array[i] = left_touch_data[i]           
                    #logger_mp.info(f"[SenseGlove] touch L: {list(left_touch_data)}")

            right_msg = self.RightHandState_sub.Read()
            right_touch_msg = self.RightHandTouch_sub.Read()
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
                        #logger_mp.info(f"[SenseGlove] force_act R: {list(right_msg.force_act)}")
                        force_logged = True
            if right_touch_msg is not None:
                with self.right_hand_touch_array.get_lock():
                    right_touch_data = self._process_touch_data(right_touch_msg)
                    for i in range(NUM_TOUCH_DATA):
                        self.right_hand_touch_array[i] = right_touch_data[i]
                    #logger_mp.info(f"[SenseGlove] touch R: {list(right_touch_data)}")
            time.sleep(0.002)

    def _process_touch_data(self, touch_data):
        """Map touch sensor array data to mean values in a compact array for 
           the SenseGlove
        """
        touch_array = np.zeros(NUM_TOUCH_DATA)
        if hasattr(touch_data, 'palm_touch'):
            palm_touch_array = np.array(touch_data.palm_touch).reshape(8, 14)
            palm_touch_pinky_mean = np.mean(np.sort(np.reshape(palm_touch_array[:, :7], -1))[-5:])
            palm_touch_index_mean = np.mean(np.sort(np.reshape(palm_touch_array[:, 7:], -1))[-5:])

            touch_array[0] = palm_touch_pinky_mean 
            touch_array[1] = palm_touch_index_mean 
        if hasattr(touch_data, 'fingerfive_top_touch'):
            thumb_touch_mean = np.mean(np.sort(touch_data.fingerfive_top_touch)[-5:])
            touch_array[2] = thumb_touch_mean
        if hasattr(touch_data, 'fingerfour_top_touch'):
            index_touch_mean = np.mean(np.sort(touch_data.fingerfour_top_touch)[-5:])
            touch_array[3] = index_touch_mean

        deadzone = float(self._touch_cfg['deadzone'])
        usable_range = max(float(self._touch_cfg['max']) - deadzone, 1.0)

        touch_array = np.clip(
                    (touch_array - deadzone) / usable_range * 100.0, 0.0, 100.0)

        return touch_array

            


    # ---- DDS command publishing -------------------------------------------

    def _build_hand_cmd(self, angles, with_settings):
        cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        cmd.angle_set = angles
        cmd.mode = 0b0001
        # Force/speed go through Modbus registers: only send them now and then
        if with_settings:
            if self._force_set is not None:
                cmd.force_set = list(self._force_set)
                cmd.mode |= 0b0100
            if self._speed_set is not None:
                cmd.speed_set = list(self._speed_set)
                cmd.mode |= 0b1000
        return cmd

    def _send_hand_command(self, left_scaled, right_scaled, with_settings=False):
        """Real Inspire FTP hand: angle_set in 0-1000 per hand."""
        self.LeftHandCmd_pub.Write(self._build_hand_cmd(left_scaled, with_settings))
        self.RightHandCmd_pub.Write(self._build_hand_cmd(right_scaled, with_settings))

    def _send_hand_command_sim(self, left_q, right_q):
        """unitree_sim_isaaclab: single MotorCmds_ on rt/inspire/cmd.

        left_q / right_q: floats in [0, 1] (1 = open), Inspire motor order
        [pinky, ring, middle, index, thumb_bend, thumb_rotation] — the same
        order the sim expects (right = cmds[0..5], left = cmds[6..11]).
        """
        for i in range(Inspire_Num_Motors):
            self.sim_hand_msg.cmds[SIM_RIGHT_HAND_OFFSET + i].q = float(
                np.clip(right_q[i], 0.0, 1.0))
            self.sim_hand_msg.cmds[SIM_LEFT_HAND_OFFSET + i].q = float(
                np.clip(left_q[i], 0.0, 1.0))
        self.SimHandCmd_pub.Write(self.sim_hand_msg)

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

        if self.simulation_mode:
            # pre-build the 12-motor MotorCmds_ message once (all open)
            self.sim_hand_msg = MotorCmds_()
            self.sim_hand_msg.cmds = [
                unitree_go_msg_dds__MotorCmd_() for _ in range(SIM_NUM_MOTORS_TOTAL)
            ]
            for i in range(SIM_NUM_MOTORS_TOTAL):
                self.sim_hand_msg.cmds[i].q = 1.0  # open

        has_settings = self._force_set is not None or self._speed_set is not None
        last_settings_t = None

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

                action_data = np.concatenate((left_q, right_q))
                if dual_hand_state_array is not None and dual_hand_action_array is not None:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:]  = state_data
                        dual_hand_action_array[:] = action_data

                if self.simulation_mode:
                    # sim expects normalised floats [0, 1] directly
                    self._send_hand_command_sim(left_q, right_q)
                else:
                    # real FTP hand expects integers 0-1000
                    scaled_left  = [int(np.clip(v * 1000, 0, 1000)) for v in left_q]
                    scaled_right = [int(np.clip(v * 1000, 0, 1000)) for v in right_q]
                    send_settings = has_settings and (
                        last_settings_t is None
                        or t0 - last_settings_t >= self._settings_resend_s)
                    if send_settings:
                        last_settings_t = t0
                    self._send_hand_command(scaled_left, scaled_right, send_settings)

                sleep_time = max(0, (1.0 / self.fps) - (time.time() - t0))
                time.sleep(sleep_time)
        finally:
            logger_mp.info("[SenseGlove] Control process closed.")

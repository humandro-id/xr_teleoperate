from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber, ChannelFactoryInitialize # dds
from unitree_sdk2py.idl.unitree_go.msg.dds_ import MotorCmds_, MotorStates_                           # idl
from unitree_sdk2py.idl.default import unitree_go_msg_dds__MotorCmd_

from teleop.robot_control.hand_retargeting import HandRetargeting, HandType
import numpy as np
from enum import IntEnum
import threading
import time
from multiprocessing import Process, Array

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

brainco_Num_Motors = 6
# q = close position [0, 1], tau = grip force [0, 1] for index..pinky (thumb stays 0).
kTopicbraincoLeftCommand = "rt/brainco/left/cmd"
kTopicbraincoLeftState = "rt/brainco/left/state"
kTopicbraincoRightCommand = "rt/brainco/right/cmd"
kTopicbraincoRightState = "rt/brainco/right/state"


def _init_brainco_publishers(simulation_mode=False):
    """Create DDS publishers inside the process that will Write() them.

    ChannelPublisher created in the parent is not reliable after Process fork.
    """
    try:
        ChannelFactoryInitialize(1 if simulation_mode else 0)
    except Exception:
        pass
    left_pub = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
    left_pub.Init()
    right_pub = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
    right_pub.Init()
    return left_pub, right_pub


def _wait_brainco_dds(is_ready, label, timeout=5.0):
    t0 = time.time()
    while not is_ready():
        if time.time() - t0 > timeout:
            logger_mp.error(
                f"[{label}] No llega estado DDS de las manos BrainCo. "
                "Arrancá el servicio en el robot: "
                "cd ~/brainco_hand_service/bin && sudo ./brainco_hand_server"
            )
            return False
        time.sleep(0.1)
        logger_mp.warning(f"[{label}] Waiting to subscribe dds...")
    logger_mp.info(f"[{label}] Subscribe dds ok.")
    return True

def _fill_hand_cmds(msg, joint_index_enum, q_target):
    for idx, joint_id in enumerate(joint_index_enum):
        msg.cmds[joint_id].q = float(q_target[idx])
        msg.cmds[joint_id].dq = 1.0
        # Thumb: position only. Other fingers: more close → more allowed current.
        msg.cmds[joint_id].tau = float(np.clip(q_target[idx], 0.0, 1.0)) if idx >= 2 else 0.0


class Brainco_Controller_ctrl:
    def __init__(self, left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in, 
                       dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False,
                       xr_motion_data_ready_in = None):
        logger_mp.info("Initialize Brainco_Controller_ctrl...")
        self.fps = fps
        self.hand_sub_ready = False
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND_Unit_Test)

        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', brainco_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', brainco_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        _wait_brainco_dds(lambda: self.hand_sub_ready, "Brainco_Controller_ctrl")

        hand_control_thread = threading.Thread(target=self.control_process, args=(left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in, 
                                                                          self.left_hand_state_array, self.right_hand_state_array, dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array,
                                                                          xr_motion_data_ready_in), daemon=True)
        hand_control_thread.start()

        logger_mp.info("Initialize Brainco_Controller_ctrl OK!\n")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None:
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
                self.hand_sub_ready = True
            if right_hand_msg is not None:
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q
                self.hand_sub_ready = True
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        _fill_hand_cmds(self.left_hand_msg, Brainco_Left_Hand_JointIndex, left_q_target)
        _fill_hand_cmds(self.right_hand_msg, Brainco_Right_Hand_JointIndex, right_q_target)

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
    
    def control_process(self, left_gripper_trigger_in, left_gripper_squeeze_in, right_gripper_trigger_in, right_gripper_squeeze_in,
                              left_hand_state_array, right_hand_state_array, dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None,
                              xr_motion_data_ready_in = None):
        self.running = True

        left_q_target  = np.full(brainco_Num_Motors, 0.0, dtype=float)
        right_q_target = np.full(brainco_Num_Motors, 0.0, dtype=float)

        # initialize brainco hand's cmd msg
        self.left_hand_msg  = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))]

        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0

        try:
            while self.running:
                start_time = time.time()
                # trigger value range: [10.0, 0.0], 10.0 means no press, 0.0 means full press
                # squeeze value range: [0.0, 1.0],   0.0 means no press, 1.0 means full press
                with left_gripper_trigger_in.get_lock():
                    left_trigger_value = left_gripper_trigger_in.value
                with left_gripper_squeeze_in.get_lock():
                    left_squeeze_value = left_gripper_squeeze_in.value
                with right_gripper_trigger_in.get_lock():
                    right_trigger_value = right_gripper_trigger_in.value
                with right_gripper_squeeze_in.get_lock():
                    right_squeeze_value = right_gripper_squeeze_in.value
                if xr_motion_data_ready_in is not None:
                    with xr_motion_data_ready_in.get_lock():
                        xr_motion_data_ready = xr_motion_data_ready_in.value
                else:
                    xr_motion_data_ready = True

                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if xr_motion_data_ready:
                    # Hardware order: [Thumb flex, Thumb aux, Index, Middle, Ring, Pinky]
                    # In the official document, the angles are in the range [0, 1] ==> 0.0: fully open  1.0: fully closed
                    left_triger_value = (10.0 - left_trigger_value) / 10.0
                    left_q_target[0]  = np.clip(left_triger_value / 0.5, 0.0, 1.0)              # thumb flex
                    left_q_target[1]  = np.clip((left_triger_value - 0.5) / 0.5, 0.0, 0.7)      # thumb-aux
                    left_q_target[2]  = np.clip(left_triger_value, 0.0, 1.0)                    # index
                    left_q_target[3]  = np.clip(left_triger_value, 0.0, 1.0)                    # middle
                    left_q_target[4]  = np.clip(left_triger_value, 0.0, 1.0)                    # ring
                    left_q_target[5]  = np.clip(left_triger_value, 0.0, 1.0)                    # pinky

                    right_triger_value = (10.0 - right_trigger_value) / 10.0
                    right_q_target[0] = np.clip(right_triger_value / 0.5, 0.0, 1.0)             # thumb flex
                    right_q_target[1] = np.clip((right_triger_value - 0.5) / 0.5, 0.0, 0.7)     # thumb-aux
                    right_q_target[2] = np.clip(right_triger_value, 0.0, 1.0)                   # index
                    right_q_target[3] = np.clip(right_triger_value, 0.0, 1.0)                   # middle
                    right_q_target[4] = np.clip(right_triger_value, 0.0, 1.0)                   # ring
                    right_q_target[5] = np.clip(right_triger_value, 0.0, 1.0)                   # pinky

                # get dual hand state
                action_data = np.concatenate((left_q_target, right_q_target))
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data

                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Brainco_Controller_ctrl has been closed.")


class Brainco_Controller_hand:
    def __init__(self, left_hand_array, right_hand_array, dual_hand_data_lock = None, dual_hand_state_array = None,
                       dual_hand_action_array = None, fps = 100.0, Unit_Test = False, simulation_mode = False, xr_motion_data_ready_in = None):
        logger_mp.info("Initialize Brainco_Controller_hand...")
        self.fps = fps
        self.hand_sub_ready = False
        self.Unit_Test = Unit_Test
        self.simulation_mode = simulation_mode

        if not self.Unit_Test:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND)
        else:
            self.hand_retargeting = HandRetargeting(HandType.BRAINCO_HAND_Unit_Test)


        # initialize handcmd publisher and handstate subscriber
        self.LeftHandCmb_publisher = ChannelPublisher(kTopicbraincoLeftCommand, MotorCmds_)
        self.LeftHandCmb_publisher.Init()
        self.RightHandCmb_publisher = ChannelPublisher(kTopicbraincoRightCommand, MotorCmds_)
        self.RightHandCmb_publisher.Init()

        self.LeftHandState_subscriber = ChannelSubscriber(kTopicbraincoLeftState, MotorStates_)
        self.LeftHandState_subscriber.Init()
        self.RightHandState_subscriber = ChannelSubscriber(kTopicbraincoRightState, MotorStates_)
        self.RightHandState_subscriber.Init()

        # Shared Arrays for hand states
        self.left_hand_state_array  = Array('d', brainco_Num_Motors, lock=True)  
        self.right_hand_state_array = Array('d', brainco_Num_Motors, lock=True)

        # initialize subscribe thread
        self.subscribe_state_thread = threading.Thread(target=self._subscribe_hand_state)
        self.subscribe_state_thread.daemon = True
        self.subscribe_state_thread.start()

        _wait_brainco_dds(lambda: self.hand_sub_ready, "Brainco_Controller_hand")

        hand_control_thread = threading.Thread(target=self.control_process, args=(left_hand_array, right_hand_array,  self.left_hand_state_array, self.right_hand_state_array,
                                                                          dual_hand_data_lock, dual_hand_state_array, dual_hand_action_array, xr_motion_data_ready_in), daemon=True)
        hand_control_thread.start()

        logger_mp.info("Initialize Brainco_Controller_hand OK!")

    def _subscribe_hand_state(self):
        while True:
            left_hand_msg  = self.LeftHandState_subscriber.Read()
            right_hand_msg = self.RightHandState_subscriber.Read()
            if left_hand_msg is not None:
                for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
                    self.left_hand_state_array[idx] = left_hand_msg.states[id].q
                self.hand_sub_ready = True
            if right_hand_msg is not None:
                for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
                    self.right_hand_state_array[idx] = right_hand_msg.states[id].q
                self.hand_sub_ready = True
            time.sleep(0.002)

    def ctrl_dual_hand(self, left_q_target, right_q_target):
        """
        Set current left, right hand motor state target q
        """
        _fill_hand_cmds(self.left_hand_msg, Brainco_Left_Hand_JointIndex, left_q_target)
        _fill_hand_cmds(self.right_hand_msg, Brainco_Right_Hand_JointIndex, right_q_target)

        self.LeftHandCmb_publisher.Write(self.left_hand_msg)
        self.RightHandCmb_publisher.Write(self.right_hand_msg)
    
    def control_process(self, left_hand_array, right_hand_array, left_hand_state_array, right_hand_state_array,
                              dual_hand_data_lock = None, dual_hand_state_array = None, dual_hand_action_array = None, xr_motion_data_ready_in = None):
        self.running = True

        left_q_target  = np.full(brainco_Num_Motors, 0.0, dtype=float)
        right_q_target = np.full(brainco_Num_Motors, 0.0, dtype=float)

        # initialize brainco hand's cmd msg
        self.left_hand_msg  = MotorCmds_()
        self.left_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Left_Hand_JointIndex))]
        self.right_hand_msg = MotorCmds_()
        self.right_hand_msg.cmds = [unitree_go_msg_dds__MotorCmd_() for _ in range(len(Brainco_Right_Hand_JointIndex))]

        for idx, id in enumerate(Brainco_Left_Hand_JointIndex):
            self.left_hand_msg.cmds[id].q = 0.0
            self.left_hand_msg.cmds[id].dq = 1.0
        for idx, id in enumerate(Brainco_Right_Hand_JointIndex):
            self.right_hand_msg.cmds[id].q = 0.0
            self.right_hand_msg.cmds[id].dq = 1.0

        try:
            while self.running:
                start_time = time.time()
                # get dual hand state
                with left_hand_array.get_lock():
                    left_hand_data  = np.array(left_hand_array[:]).reshape(25, 3).copy()
                with right_hand_array.get_lock():
                    right_hand_data = np.array(right_hand_array[:]).reshape(25, 3).copy()
                if xr_motion_data_ready_in is not None:
                    with xr_motion_data_ready_in.get_lock():
                        xr_motion_data_ready = xr_motion_data_ready_in.value
                else:
                    xr_motion_data_ready = True

                # Read left and right q_state from shared arrays
                state_data = np.concatenate((np.array(left_hand_state_array[:]), np.array(right_hand_state_array[:])))

                if xr_motion_data_ready:
                    ref_left_value = left_hand_data[self.hand_retargeting.left_indices[1,:]] - left_hand_data[self.hand_retargeting.left_indices[0,:]]
                    ref_right_value = right_hand_data[self.hand_retargeting.right_indices[1,:]] - right_hand_data[self.hand_retargeting.right_indices[0,:]]

                    try:
                        left_q_target  = self.hand_retargeting.left_retargeting.retarget(ref_left_value)[self.hand_retargeting.left_dex_retargeting_to_hardware]
                        right_q_target = self.hand_retargeting.right_retargeting.retarget(ref_right_value)[self.hand_retargeting.right_dex_retargeting_to_hardware]
                    except Exception as e:
                        logger_mp.error(f"[Brainco_Controller_hand] retarget failed: {e}")
                        time.sleep(0.05)
                        continue

                    # Hardware cmd is [0, 1] = fully open / fully closed.
                    # URDF hard-stops: thumb flex 1.05, thumb aux 1.52, fingers 1.47.
                    # DexPilot almost never reaches those on a human fist, so we
                    # use a smaller effective close angle to saturate the motors.
                    def normalize(val, min_val, max_val):
                        return np.clip((val - min_val) / (max_val - min_val), 0.0, 1.0)

                    # idx: thumb flex, thumb aux, index, middle, ring, pinky
                    close_rad = (0.80, 1.35, 1.00, 1.00, 1.00, 1.00)
                    # A fully open human thumb does not retarget to 0 rad: it sits
                    # around 0.2-0.34. Without this offset the thumb is never
                    # commanded below ~0.40 no matter how wide the hand opens.
                    # Measured over 7999 frames of recorded teleop. Other fingers
                    # do reach 0, so they keep an offset of 0.
                    open_rad = (0.20, 0.00, 0.00, 0.00, 0.00, 0.00)
                    for idx in range(brainco_Num_Motors):
                        left_q_target[idx]  = normalize(left_q_target[idx], open_rad[idx], close_rad[idx])
                        right_q_target[idx] = normalize(right_q_target[idx], open_rad[idx], close_rad[idx])

                # get dual hand action
                action_data = np.concatenate((left_q_target, right_q_target))    
                if dual_hand_state_array and dual_hand_action_array:
                    with dual_hand_data_lock:
                        dual_hand_state_array[:] = state_data
                        dual_hand_action_array[:] = action_data
                if xr_motion_data_ready and not getattr(self, "_logged_first_hand_cmd", False):
                    logger_mp.info(f"[Brainco_Controller_hand] first cmd L={np.round(left_q_target, 3)} R={np.round(right_q_target, 3)}")
                    self._logged_first_hand_cmd = True
                self.ctrl_dual_hand(left_q_target, right_q_target)
                current_time = time.time()
                time_elapsed = current_time - start_time
                sleep_time = max(0, (1 / self.fps) - time_elapsed)
                time.sleep(sleep_time)
        finally:
            logger_mp.info("Brainco_Controller_hand has been closed.")

# according to the official documentation, https://www.brainco-hz.com/docs/revolimb-hand/product/parameters.html
# the motor sequence is as shown in the table below
# ┌──────┬───────┬────────────┬────────┬────────┬────────┬────────┐
# │ Id   │   0   │     1      │   2    │   3    │   4    │   5    │
# ├──────┼───────┼────────────┼────────┼────────┼────────┼────────┤
# │Joint │ thumb │ thumb-aux  |  index │ middle │  ring  │  pinky │
# └──────┴───────┴────────────┴────────┴────────┴────────┴────────┘
class Brainco_Right_Hand_JointIndex(IntEnum):
    kRightHandThumb = 0
    kRightHandThumbAux = 1
    kRightHandIndex = 2
    kRightHandMiddle = 3
    kRightHandRing = 4
    kRightHandPinky = 5

class Brainco_Left_Hand_JointIndex(IntEnum):
    kLeftHandThumb = 0
    kLeftHandThumbAux = 1
    kLeftHandIndex = 2
    kLeftHandMiddle = 3
    kLeftHandRing = 4
    kLeftHandPinky = 5
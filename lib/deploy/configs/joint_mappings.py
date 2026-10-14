G1_DEFAULT_POSITIONS = {
    # Left leg
    0: -0.1,   # LeftHipPitch
    1: 0.0,    # LeftHipRoll
    2: 0.0,    # LeftHipYaw
    3: 0.3,    # LeftKnee
    4: -0.2,   # LeftAnklePitch
    5: 0.0,    # LeftAnkleRoll
    # Right leg
    6: -0.1,   # RightHipPitch
    7: 0.0,    # RightHipRoll
    8: 0.0,    # RightHipYaw
    9: 0.3,    # RightKnee
    10: -0.2,  # RightAnklePitch
    11: 0.0,   # RightAnkleRoll
    # Waist
    12: 0.0,   # WaistYaw
    13: 0.0,   # WaistRoll (INVALID)
    14: 0.0,   # WaistPitch (INVALID)
    # Left arm
    15: 0.0,   # LeftShoulderPitch
    16: 0.0,   # LeftShoulderRoll
    17: 0.0,   # LeftShoulderYaw
    18: 0.0,   # LeftElbow
    19: 0.0,   # LeftWristRoll
    20: 0.0,   # LeftWristPitch (INVALID)
    21: 0.0,   # LeftWristYaw (INVALID)
    # Right arm
    22: 0.0,   # RightShoulderPitch
    23: 0.0,   # RightShoulderRoll
    24: 0.0,   # RightShoulderYaw
    25: 0.0,   # RightElbow
    26: 0.0,   # RightWristRoll
    27: 0.0,   # RightWristPitch (INVALID)
    28: 0.0,   # RightWristYaw (INVALID)
}


REAL_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# print(env.scene['robot'].data.default_joint_pos)
SIM_DEFAULT_POSITIONS = [-0.1000, -0.1000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,  0.0000,
          0.0000,  0.3000,  0.3000,  0.3000,  0.3000, -0.2000, -0.2000,  0.2500,
         -0.2500,  0.0000,  0.0000,  0.0000,  0.0000,  0.9700,  0.9700,  0.1500,
         -0.1500,  0.0000,  0.0000,  0.0000,  0.0000]

SIM_SCALE = 0.125

# print(env.scene['robot'].joint_names)
SIMULATION_ORDER = ['left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
              'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
              'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
              'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
              'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
              'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
              'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
              'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint',
              'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint',
              'right_wrist_yaw_joint']


ACTION_ID_2_MOTOR_ID = {sim_id: REAL_ORDER.index(sim_name) for sim_id, sim_name in enumerate(SIMULATION_ORDER)}
MOTOR_ID_2_ACTION_ID = {REAL_ORDER.index(sim_name): sim_id for sim_id, sim_name in enumerate(SIMULATION_ORDER)}


# Dex3 hand joint names in teleop / DDS order: [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1].
DEX3_LEFT_REAL_ORDER = [
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
]

DEX3_RIGHT_REAL_ORDER = [
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

# Dex3 normal-mode closed poses in teleop / DDS order:
# [Thumb0, Thumb1, Thumb2, Middle0, Middle1, Index0, Index1].
DEX3_NORMAL_LEFT_CLOSED_Q = (0.0, 0.92, 1.74, -1.57, -1.74, -1.57, -1.74)
DEX3_NORMAL_RIGHT_CLOSED_Q = (0.0, -0.92, -1.74, 1.57, 1.74, 1.57, 1.74)


SIM_DEFAULT_SCALE_MULS = [0.9394, 0.9394, 0.9091, 0.6061, 0.6061, 0.1806, 0.9576, 0.9576, 0.1806,
        0.5152, 0.5152, 1.0000, 1.0000, 0.2424, 0.2424, 0.6667, 0.6667, 0.0909,
        0.0909, 0.9091, 0.9091, 0.5455, 0.5455, 0.6848, 0.6848, 0.5606, 0.5606,
        0.5606, 0.5606]

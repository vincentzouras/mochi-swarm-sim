from enum import IntEnum, auto

SERVO = "motors_servo"
THRUST_RIGHT = "motor_right_thrust"
THRUST_LEFT = "motor_left_thrust"
ULTRASONIC = "ultrasonic"
AXLE = "motors_axle"
IMU_POS = "imu_pos"
IMU_LIN_VEL = "imu_lin_vel"
IMU_ANG_VEL = "imu_ang_vel"
IMU_QUAT = "imu_quat"
CAMERA = "nicla_vision"


class Action(IntEnum):
    ARMED = 0
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


class State(IntEnum):
    Z_ALTITUDE = 0
    Z_ALTITUDE_VEL = auto()
    X_ROLL = auto()
    Y_PITCH = auto()
    Z_YAW = auto()
    X_ROLL_RATE = auto()
    Y_PITCH_RATE = auto()
    Z_YAW_RATE = auto()

    NICLA_FLAG = auto()  # 1 if detected, 0 if not
    NICLA_X = auto()  # 0.0 to 1.0 (Center is 0.5)
    NICLA_Y = auto()  # 0.0 to 1.0
    NICLA_W = auto()  # Width of blob (used for distance approx)
    NICLA_H = auto()  # Height of blob
    TARGET_HEIGHT = auto()  # Desired height for the active target (meters)

    NUM_STATES = auto()

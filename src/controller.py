from enum import Enum, auto
import mujoco as mj
import numpy as np
from mujoco.glfw import glfw

MAX_THRUST = 1.0
SERVO = "motors_servo"
THRUST_RIGHT = "motor_right_thrust"
THRUST_LEFT = "motor_left_thrust"
ACCELEROMETER = "accelerometer"
ULTRASONIC = "ultrasonic"
GYRO = "gyro"
BAROMETER = "barometer"
AXLE = "motors_axle"


class Action(Enum):
    FORWARD = auto()
    BACKWARD = auto()
    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()


KEY_BINDINGS = {
    glfw.KEY_W: Action.FORWARD,
    glfw.KEY_S: Action.BACKWARD,
    glfw.KEY_A: Action.LEFT,
    glfw.KEY_D: Action.RIGHT,
    glfw.KEY_SPACE: Action.UP,
    glfw.KEY_LEFT_SHIFT: Action.DOWN,
}


class Controller:
    """
    Handles all flight logic and sensor data parsing.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.action_states = {action: False for action in Action}

    def update_key_state(self, key, action):
        """
        Called by the simulation's keyboard callback to update our internal state.
        """
        is_pressed = action != glfw.RELEASE
        if key in KEY_BINDINGS:
            self.action_states[KEY_BINDINGS[key]] = is_pressed

    def control_step(self, model, data):
        """
        This is the main MuJoCo control callback.
        It reads internal state (self.action_states) and sets controls.
        """
        data.ctrl[:] = 0  # reset controls

        if self.action_states[Action.UP]:
            data.actuator(SERVO).ctrl = 0.0
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.action_states[Action.DOWN]:
            current_angle = data.joint(AXLE).qpos[0]
            target_angle = (
                np.pi
                if abs(current_angle - np.pi) < abs(current_angle + np.pi)
                else -np.pi
            )
            data.actuator(SERVO).ctrl = target_angle
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.action_states[Action.FORWARD]:
            data.actuator(SERVO).ctrl = np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.action_states[Action.BACKWARD]:
            data.actuator(SERVO).ctrl = -np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.action_states[Action.LEFT]:
            data.actuator(SERVO).ctrl = np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = 0.0
        elif self.action_states[Action.RIGHT]:
            data.actuator(SERVO).ctrl = np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = 0.0
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST

    def get_sensor_readings(self):
        """
        Called by the simulation's render loop to get formatted sensor data.
        """
        accel_data = self.data.sensor(ACCELEROMETER).data.copy()
        # self.data.sensordata[self.accel_start : self.accel_end].copy()
        accel_data[2] -= 9.81  # remove gravity

        gyro_data = self.data.sensor(GYRO).data.copy()
        ultrasonic_data = self.data.sensor(ULTRASONIC).data.copy()[0]
        pos_data = self.data.sensor(BAROMETER).data.copy()
        barometer_data = pos_data[2]

        return (
            f"Acel: {accel_data[0]:8.3f}, {accel_data[1]:8.3f}, {accel_data[2]:8.3f} m/s^2\n"
            f"Gyro: {gyro_data[0]:8.3f}, {gyro_data[1]:8.3f}, {gyro_data[2]:8.3f} rad/s\n"
            f"Ultra: {ultrasonic_data:8.3f} m\n"
            f"Baro: {barometer_data:8.3f} m"
        )

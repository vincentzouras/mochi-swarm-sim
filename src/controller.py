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


class Controller:
    """
    Handles all flight logic and sensor data parsing.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.key_states = {
            "space": False,
            "shift": False,
            "w": False,
            "s": False,
            "a": False,
            "d": False,
        }

    def update_key_state(self, key, action):
        """
        Called by the simulation's keyboard callback to update our internal state.
        """
        is_pressed = action != glfw.RELEASE

        if key == glfw.KEY_SPACE:
            self.key_states["space"] = is_pressed
        elif key == glfw.KEY_LEFT_SHIFT:
            self.key_states["shift"] = is_pressed
        elif key == glfw.KEY_W:
            self.key_states["w"] = is_pressed
        elif key == glfw.KEY_S:
            self.key_states["s"] = is_pressed
        elif key == glfw.KEY_A:
            self.key_states["a"] = is_pressed
        elif key == glfw.KEY_D:
            self.key_states["d"] = is_pressed

    def control_step(self, model, data):
        """
        This is the main MuJoCo control callback.
        It reads internal state (self.key_states) and sets controls.
        """
        data.ctrl[:] = 0  # reset controls

        if self.key_states["space"]:
            data.actuator(SERVO).ctrl = 0.0
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.key_states["shift"]:
            current_angle = data.joint(AXLE).qpos[0]
            target_angle = (
                np.pi
                if abs(current_angle - np.pi) < abs(current_angle + np.pi)
                else -np.pi
            )
            data.actuator(SERVO).ctrl = target_angle
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.key_states["w"]:
            data.actuator(SERVO).ctrl = np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.key_states["s"]:
            data.actuator(SERVO).ctrl = -np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = MAX_THRUST
        elif self.key_states["a"]:
            data.actuator(SERVO).ctrl = np.pi / 2
            data.actuator(THRUST_RIGHT).ctrl = MAX_THRUST
            data.actuator(THRUST_LEFT).ctrl = 0.0
        elif self.key_states["d"]:
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

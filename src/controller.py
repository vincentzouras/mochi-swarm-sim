import mujoco as mj
import numpy as np
from mujoco.glfw import glfw
import config


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

        # --- Actuator IDs ---
        self.motors_servo_id = self.model.actuator("motors_servo").id
        self.motor_right_thrust_id = self.model.actuator("motor_right_thrust").id
        self.motor_left_thrust_id = self.model.actuator("motor_left_thrust").id

        # --- Sensor IDs and Indices ---
        self.ultrasonic_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SENSOR, "ultrasonic"
        )

        self.accelerometer_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SENSOR, "accelerometer"
        )
        self.accel_start = self.model.sensor_adr[self.accelerometer_id]
        self.accel_end = self.accel_start + self.model.sensor_dim[self.accelerometer_id]

        self.gyro_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, "gyro")
        self.gyro_start = self.model.sensor_adr[self.gyro_id]
        self.gyro_end = self.gyro_start + self.model.sensor_dim[self.gyro_id]

        self.barometer_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_SENSOR, "barometer"
        )
        self.baro_start = self.model.sensor_adr[self.barometer_id]
        self.baro_end = self.baro_start + self.model.sensor_dim[self.barometer_id]

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
            data.ctrl[self.motors_servo_id] = 0.0
            data.ctrl[self.motor_right_thrust_id] = config.MAX_THRUST
            data.ctrl[self.motor_left_thrust_id] = config.MAX_THRUST
        elif self.key_states["shift"]:
            current_angle = data.joint("motors_axle").qpos[0]
            target_angle = (
                np.pi
                if abs(current_angle - np.pi) < abs(current_angle + np.pi)
                else -np.pi
            )
            data.ctrl[self.motors_servo_id] = target_angle
            data.ctrl[self.motor_right_thrust_id] = config.MAX_THRUST
            data.ctrl[self.motor_left_thrust_id] = config.MAX_THRUST
        elif self.key_states["w"]:
            data.ctrl[self.motors_servo_id] = np.pi / 2
            data.ctrl[self.motor_right_thrust_id] = config.MAX_THRUST
            data.ctrl[self.motor_left_thrust_id] = config.MAX_THRUST
        elif self.key_states["s"]:
            data.ctrl[self.motors_servo_id] = -np.pi / 2
            data.ctrl[self.motor_right_thrust_id] = config.MAX_THRUST
            data.ctrl[self.motor_left_thrust_id] = config.MAX_THRUST
        elif self.key_states["a"]:
            data.ctrl[self.motors_servo_id] = np.pi / 2
            data.ctrl[self.motor_right_thrust_id] = config.MAX_THRUST
            data.ctrl[self.motor_left_thrust_id] = 0.0
        elif self.key_states["d"]:
            data.ctrl[self.motors_servo_id] = np.pi / 2
            data.ctrl[self.motor_right_thrust_id] = 0.0
            data.ctrl[self.motor_left_thrust_id] = config.MAX_THRUST

    def get_sensor_readings(self):
        """
        Called by the simulation's render loop to get formatted sensor data.
        """
        accel_data = self.data.sensordata[self.accel_start : self.accel_end].copy()
        accel_data[2] -= 9.81  # remove gravity

        gyro_data = self.data.sensordata[self.gyro_start : self.gyro_end]
        ultrasonic_data = self.data.sensordata[
            self.model.sensor_adr[self.ultrasonic_id]
        ]
        barometer_data = self.data.sensordata[self.baro_end - 1]  # just the z value

        return (
            f"Acel: {accel_data[0]:8.3f}, {accel_data[1]:8.3f}, {accel_data[2]:8.3f} m/s^2\n"
            f"Gyro: {gyro_data[0]:8.3f}, {gyro_data[1]:8.3f}, {gyro_data[2]:8.3f} rad/s\n"
            f"Ultra: {ultrasonic_data:8.3f} m\n"
            f"Baro: {barometer_data:8.3f} m"
        )

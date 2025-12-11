import numpy as np
from mujoco.glfw import glfw
from .state.robot_state_machine import RobotStateMachine
from .state.manual_state import ManualState
from .state.exploration_state import ExplorationState
from .robot.differential import Differential
from scipy.spatial.transform import Rotation as R
from typing import Protocol
from .state.race_state import RaceState
from .definitions import (
    Action,
    State,
    SERVO,
    THRUST_RIGHT,
    THRUST_LEFT,
    IMU_POS,
    IMU_LIN_VEL,
    IMU_ANG_VEL,
    IMU_QUAT,
    CAMERA,
)


KEY_BINDINGS = {
    glfw.KEY_W: Action.FORWARD,
    glfw.KEY_S: Action.BACKWARD,
    glfw.KEY_A: Action.LEFT,
    glfw.KEY_D: Action.RIGHT,
    glfw.KEY_SPACE: Action.UP,
    glfw.KEY_LEFT_SHIFT: Action.DOWN,
    glfw.KEY_ENTER: Action.ARMED,
}


class Controller(Protocol):
    """
    Handles all flight logic and sensor data parsing.
    """

    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.action_states = {action: False for action in Action}
        self.state_machine = RobotStateMachine(ManualState())
        self.robot = Differential()
        self.senses = np.zeros(State.NUM_STATES)

        # targets are (body_name, height_meters)
        self.targets = [
            ("target_fr", 2.8),
            ("target_br", 3.8),
            ("target_bl", 1.8),
            ("target_fl", 0.8),
        ]
        self.target_idx = 0
        self.reach_threshold = 0.5  # Distance in meters to trigger switch

    def update_key_state(self, key, action):
        """
        Called by the simulation's keyboard callback to update our internal state.
        """
        is_pressed = action != glfw.RELEASE

        # State selection via number keys (press to select state; does not toggle ARMED)
        if action == glfw.PRESS:
            if key == glfw.KEY_1:
                self.state_machine.current_state = ManualState()
                print("[STATE SELECT] ManualState (1)")
                return
            if key == glfw.KEY_2:
                self.state_machine.current_state = ExplorationState()
                print("[STATE SELECT] ExplorationState (2)")
                return
            if key == glfw.KEY_3:
                self.state_machine.current_state = RaceState()
                print("[STATE SELECT] RaceState (3)")
                return

        if key in KEY_BINDINGS:
            mapped_action = KEY_BINDINGS[key]
            if mapped_action == Action.ARMED:
                # toggle
                if is_pressed:
                    self.action_states[Action.ARMED] = not self.action_states[
                        Action.ARMED
                    ]
                    print(
                        f"[ARMED] {'ON' if self.action_states[Action.ARMED] else 'OFF'}"
                    )
            else:
                # hold
                self.action_states[mapped_action] = is_pressed

    def control_step(self, model, data):
        """
        This is the main MuJoCo control callback.
        Coordinates state machine and robot controller
        """

        # --- Sense ---
        self._sense()

        # --- State Machine Update ---
        behavior_commands = self.state_machine.update(self.senses, self.action_states)

        # --- Pass behaviors to flight controller to get actuator commands ---
        actuator_commands = self.robot.control(self.senses, behavior_commands)

        # --- Apply actuator commands to simulation ---
        data.actuator(THRUST_LEFT).ctrl = actuator_commands[0]
        data.actuator(THRUST_RIGHT).ctrl = actuator_commands[1]
        data.actuator(SERVO).ctrl = actuator_commands[2]

    def _sense(self):
        """
        Returns a dictionary of current sensor readings.
        [z_altitude, z_altitude_vel, x_roll, y_pitch, z_yaw, x_roll_rate, y_pitch_rate, z_yaw_rate]
        """
        # take only z axis from imu_pos: [x, y, z]
        self.senses[State.Z_ALTITUDE] = self.data.sensor(IMU_POS).data.copy()[2]
        # take only z axis from imu_vel: [vx, vy, vz, wx, wy, wz]
        self.senses[State.Z_ALTITUDE_VEL] = self.data.sensor(IMU_LIN_VEL).data.copy()[2]

        # convert quat to euler angles
        quat = self.data.sensor(IMU_QUAT).data.copy()  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)  # in radians

        self.senses[State.X_ROLL] = roll
        self.senses[State.Y_PITCH] = pitch
        self.senses[State.Z_YAW] = yaw

        ang_vel = self.data.sensor(IMU_ANG_VEL).data.copy()
        self.senses[State.X_ROLL_RATE] = ang_vel[0]
        self.senses[State.Y_PITCH_RATE] = ang_vel[1]
        self.senses[State.Z_YAW_RATE] = ang_vel[2]

        self._sense_vision()

    def _sense_vision(self):
        """
        Virtual Nicla Sensor that cycles through targets.
        """
        try:
            # 1. Get the current active target (name, desired height)
            active_target_name, active_target_height = self.targets[self.target_idx]
            # publish the desired height into sensors so states can use it
            self.senses[State.TARGET_HEIGHT] = active_target_height

            target_id = self.model.body(active_target_name).id
            target_pos = self.data.xpos[target_id]

            # 2. Project 3D position to 2D camera frame
            cam_id = self.model.camera(CAMERA).id
            cam_pos = self.data.cam_xpos[cam_id]
            cam_mat = self.data.cam_xmat[cam_id].reshape(3, 3)

            vec = target_pos - cam_pos
            distance = np.linalg.norm(vec)  # Calculate distance

            # --- RACE LOGIC: switch target if reached ---
            if distance < self.reach_threshold:
                print(f"[RACE] Reached {active_target_name}! Switching to next.")
                # advance and immediately update the published target height
                self.target_idx = (self.target_idx + 1) % len(self.targets)
                next_name, next_height = self.targets[self.target_idx]
                self.senses[State.TARGET_HEIGHT] = next_height
                # Temporarily lose detection to force a re-scan behavior
                self.senses[State.NICLA_FLAG] = 0
                return

            local_vec = cam_mat.T @ vec

            if local_vec[2] > -0.1:  # Behind camera
                self.senses[State.NICLA_FLAG] = 0
                return

            fovy = self.model.cam_fovy[cam_id]
            f = 0.5 / np.tan(np.deg2rad(fovy) / 2)
            u = -local_vec[0] * f / local_vec[2] + 0.5
            v = -local_vec[1] * f / local_vec[2] + 0.5

            if 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0:
                self.senses[State.NICLA_FLAG] = 1
                self.senses[State.NICLA_X] = u
                self.senses[State.NICLA_Y] = v
                self.senses[State.NICLA_W] = 0.5 / distance if distance > 0 else 0
            else:
                self.senses[State.NICLA_FLAG] = 0

        except KeyError:
            self.senses[State.NICLA_FLAG] = 0

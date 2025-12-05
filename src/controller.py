import numpy as np
import pickle
import os
from mujoco.glfw import glfw
from .state.robot_state_machine import RobotStateMachine
from .state.manual_state import ManualState
from .state.exploration_state import ExplorationState
from .robot.differential import Differential
from scipy.spatial.transform import Rotation as R
from typing import Protocol
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

        # Data collection for (state, action, next_state) tuples
        self.collect_data = False
        self.data_buffer = []  # List of (state, action, next_state) tuples
        self._current_state = None
        self._current_action = None
        
        # Noise injection for data collection
        self.noise_enabled = False
        self.noise_std = np.array([0.0, 0.0, 0.0])  # [left_thrust, right_thrust, servo_angle]
        self.noise_seed = None
        if self.noise_seed is not None:
            np.random.seed(self.noise_seed)

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

        # --- Collect state (before action is computed) ---
        if self.collect_data:
            self._current_state = self.senses.copy()

        # --- State Machine Update ---
        behavior_commands = self.state_machine.update(self.senses, self.action_states)

        # --- Pass behaviors to flight controller to get actuator commands ---
        actuator_commands = self.robot.control(self.senses, behavior_commands)

        # --- Add noise to actuator commands if enabled ---
        if self.noise_enabled:
            noise = np.random.normal(0, self.noise_std, size=3)
            actuator_commands = actuator_commands + noise
            # Clamp to valid ranges: thrusts [0, 1], servo [-π, π]
            actuator_commands[0] = np.clip(actuator_commands[0], 0.0, 1.0)  # left_thrust
            actuator_commands[1] = np.clip(actuator_commands[1], 0.0, 1.0)  # right_thrust
            actuator_commands[2] = np.clip(actuator_commands[2], -np.pi, np.pi)  # servo_angle

        # --- Collect action (after noise injection, before applying to MuJoCo) ---
        if self.collect_data:
            self._current_action = actuator_commands.copy()

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

    def capture_next_state(self):
        """
        Call this after mj_step() to capture the next state.
        Forms (state, action, next_state) tuple and stores it if data collection is enabled.
        """
        if self.collect_data and self._current_state is not None and self._current_action is not None:
            # Capture next state (after physics step)
            self._sense()
            next_state = self.senses.copy()
            
            # Store the tuple
            self.data_buffer.append((
                self._current_state.copy(),
                self._current_action.copy(),
                next_state.copy()
            ))
            
            # Reset for next cycle
            self._current_state = None
            self._current_action = None

    def start_data_collection(self):
        """Enable data collection."""
        self.collect_data = True
        self.data_buffer = []

    def stop_data_collection(self):
        """Disable data collection."""
        self.collect_data = False

    def get_collected_data(self):
        """
        Returns the collected (state, action, next_state) tuples.
        Returns: List of tuples, each containing (state, action, next_state) as numpy arrays
        """
        return self.data_buffer.copy()

    def clear_collected_data(self):
        """Clear the data buffer."""
        self.data_buffer = []

    def enable_noise(self, left_thrust_std=0.1, right_thrust_std=0.1, servo_angle_std=0.1, seed=None):
        """
        Enable noise injection on actuator commands.
        
        Args:
            left_thrust_std: Standard deviation of noise for left motor thrust (default: 0.1)
            right_thrust_std: Standard deviation of noise for right motor thrust (default: 0.1)
            servo_angle_std: Standard deviation of noise for servo angle in radians (default: 0.1)
            seed: Random seed for reproducibility (None = no seed)
        """
        self.noise_enabled = True
        self.noise_std = np.array([left_thrust_std, right_thrust_std, servo_angle_std])
        self.noise_seed = seed
        if seed is not None:
            np.random.seed(seed)
        print(f"[NOISE] Enabled with std: left_thrust={left_thrust_std:.3f}, "
              f"right_thrust={right_thrust_std:.3f}, servo_angle={servo_angle_std:.3f}")

    def disable_noise(self):
        """Disable noise injection on actuator commands."""
        self.noise_enabled = False
        print("[NOISE] Disabled")

    def save_collected_data(self, filename):
        """Save the collected data to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self.data_buffer, f)
        print(f"Saved {len(self.data_buffer)} data tuples to {filename}")

from typing import Tuple
import numpy as np
from .robot_state import RobotState, Behavior
from src.definitions import Action, State
from src import preferences


class RaceState(RobotState):
    def __init__(self):
        super().__init__()
        self.target_height = preferences.B_DEFAULT_HEIGHT
        self.target_yaw = 0.0

        self.search_yaw_rate = 0.5  # Speed of spin when searching
        self.track_yaw_gain = 2.0  # Speed of turn when tracking
        self.forward_speed = 0.4  # Speed when approaching
        self.fov_center_tol = 0.15  # Tolerance to start moving forward

    def update(
        self, sensors: np.ndarray, action_states: dict
    ) -> Tuple[np.ndarray, RobotState]:
        self.target_height = sensors[State.TARGET_HEIGHT]
        behavior_targets = np.zeros(Behavior.NUM_PARAMS)
        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        flag = sensors[State.NICLA_FLAG]
        current_yaw = sensors[State.Z_YAW]
        nicla_x = sensors[State.NICLA_X]  # 0.0 to 1.0

        if flag == 1:
            # --- TARGET DETECTED (APPROACH) ---
            try:
                self.target_height = sensors[State.TARGET_HEIGHT]
            except Exception:
                # Fallback to existing value if sensor missing
                pass
            # 1. Calculate yaw to center the target
            # (nicla_x - 0.5) is the error.
            yaw_error = nicla_x - 0.5
            self.target_yaw = current_yaw - (yaw_error * self.track_yaw_gain)

            # 2. Move forward if roughly centered
            if abs(yaw_error) < self.fov_center_tol:
                behavior_targets[Behavior.FX_FORWARD] = self.forward_speed
            else:
                behavior_targets[Behavior.FX_FORWARD] = 0.0  # Stop to turn

        else:
            # --- TARGET LOST (SEARCH) ---
            # Spin to find the next target
            self.target_yaw = current_yaw - (self.search_yaw_rate)
            behavior_targets[Behavior.FX_FORWARD] = 0.0

        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = self.target_yaw

        return behavior_targets, self

from typing import Tuple
import numpy as np
from .robot_state import RobotState, Behavior
from src.definitions import Action, State
from src import preferences
from .exploration_state import ExplorationState


class MoveToGoal(RobotState):
    def __init__(self, start_yaw, target_height=None):
        super().__init__()
        # Use the provided target height (from Controller) when available
        self.target_height = (
            target_height if target_height is not None else preferences.B_DEFAULT_HEIGHT
        )
        self.target_yaw = start_yaw

        self.x_strength = 2.0  # Gain for yaw correction
        self.fx_togoal = 0.4  # Forward thrust when aligned
        self.range_for_forward = 0.1  # 10% tolerance from center

    def update(
        self, sensors: np.ndarray, action_states: dict
    ) -> Tuple[np.ndarray, RobotState]:
        behavior_targets = np.zeros(Behavior.NUM_PARAMS)
        behavior_targets[Behavior.READY] = 1.0 if action_states[Action.ARMED] else 0.0

        # If target lost, go back to Exploration
        if sensors[State.NICLA_FLAG] == 0:
            return behavior_targets, ExplorationState()

        current_yaw = sensors[State.Z_YAW]
        nicla_x = sensors[State.NICLA_X]  # Normalized 0.0 to 1.0

        # Calculate yaw correction
        # nicla_x - 0.5 gives error from center.
        des_yaw_diff = (nicla_x - 0.5) * self.x_strength
        self.target_yaw = (
            current_yaw - des_yaw_diff
        )  # Sign might need flipping based on coord system

        forward_force = 0.0
        if abs(nicla_x - 0.5) < self.range_for_forward:
            forward_force = self.fx_togoal

            # Simple height tracking (optional, based on C++ "z_estimator")
            # If target is too high/low on screen, adjust altitude
            nicla_y = sensors[State.NICLA_Y]
            if nicla_y < 0.4:
                self.target_height += 0.01
            if nicla_y > 0.6:
                self.target_height -= 0.01

        behavior_targets[Behavior.Z_HEIGHT] = self.target_height
        behavior_targets[Behavior.Z_YAW] = self.target_yaw
        behavior_targets[Behavior.FX_FORWARD] = forward_force

        return behavior_targets, self

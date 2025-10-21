from typing import Tuple
import numpy as np
from .robot_state import RobotState, Behavior
from src.controller import Action


class ManualState(RobotState):
    def __init__(self, target_height: float = 1.5, target_yaw: float = 0.0):
        self.target_height = target_height
        self.target_yaw = target_yaw
        self.f_thrust = 0.5  # Forward thrust command
        self.t_thrust = 0.2  # Turning thrust command

    def update(
        self, sensors: dict, action_states: dict
    ) -> Tuple[np.ndarray, "RobotState"]:

        # equivalent to 'behave.params' in DiffController.ino
        behavior_params = np.zeros(Behavior.NUM_PARAMS.value)

        # translate key presses into desired forces

        behavior_params[Behavior.READY.value] = 0.0

        # Set default targets for the PID controller
        behavior_params[Behavior.FZ_HEIGHT.value] = self.target_height
        behavior_params[Behavior.TZ_YAW.value] = self.target_yaw

        if action_states.get(Action.UP):
            # GO UP: Increase target height
            self.target_height += 0.01
            behavior_params[Behavior.FZ_HEIGHT.value] = self.target_height
        elif action_states.get(Action.DOWN):
            # GO DOWN: Decrease target height
            self.target_height -= 0.01
            behavior_params[Behavior.FZ_HEIGHT.value] = self.target_height
        elif action_states.get(Action.FORWARD):
            # GO FORWARD: Apply positive Fx
            behavior_params[Behavior.FX_FORWARD.value] = self.f_thrust
        elif action_states.get(Action.BACKWARD):
            # GO BACKWARD: Apply negative Fx
            behavior_params[Behavior.FX_FORWARD.value] = -self.f_thrust
        elif action_states.get(Action.LEFT):
            # TURN LEFT: Apply positive Tz (to change target yaw)
            self.target_yaw += 0.02
            behavior_params[Behavior.TZ_YAW.value] = self.target_yaw
        elif action_states.get(Action.RIGHT):
            # TURN RIGHT: Apply negative Tz (to change target yaw)
            self.target_yaw -= 0.02
            behavior_params[Behavior.TZ_YAW.value] = self.target_yaw

        # Always stay in ManualState in this example
        return behavior_params, self

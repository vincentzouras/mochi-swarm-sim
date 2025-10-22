import numpy as np
from .robot_state import RobotState


class RobotStateMachine:
    """
    Manages the current state of the robot and handles state transitions.
    """

    def __init__(self, initial_state: RobotState):
        self.current_state = initial_state

    def update(self, sensors: np.ndarray, action_states: dict) -> np.ndarray:
        """
        Calls the current state's update and handles transitions.

        Args:
            sensors: A numpy array of current sensor readings.
            action_states: A dictionary of the current action states.

        Returns:
            A numpy array of high-level behavior commands from the current state.
        """
        behavior_params, next_state = self.current_state.update(sensors, action_states)
        if next_state != self.current_state:
            print(
                f"[STATE] {type(self.current_state).__name__} --> {type(next_state).__name__}"
            )
            self.current_state = next_state
        return behavior_params

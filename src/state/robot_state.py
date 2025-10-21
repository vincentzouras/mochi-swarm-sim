from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Tuple
import numpy as np


class Behavior(Enum):
    READY = auto()
    FX_FORWARD = auto()
    FZ_HEIGHT = auto()
    TX_ROLL = auto()
    TZ_YAW = auto()
    NUM_PARAMS = auto()


class RobotState(ABC):

    @abstractmethod
    def update(
        self, sensors: dict, action_states: dict
    ) -> Tuple[np.ndarray, "RobotState"]:
        """
        Updates the state logic.

        Args:
            sensors: A dictionary of current sensor readings.
            action_states: A dictionary of the current action states.

        Returns:
            A tuple containing:
            1. A numpy array of high-level behavior commands
               (e.g., [ready, Fx, Fz, Tx, Tz]).
            2. The next state (can be 'self' to remain in the current state).
        """
        pass

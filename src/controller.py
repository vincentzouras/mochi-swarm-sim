import numpy as np
from mujoco.glfw import glfw
from .state.robot_state_machine import RobotStateMachine
from .state.manual_state import ManualState
from .robot.differential import Differential
from scipy.spatial.transform import Rotation as R
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


class Controller:
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

    def update_key_state(self, key, action):
        """
        Called by the simulation's keyboard callback to update our internal state.
        """
        is_pressed = action != glfw.RELEASE
        if key in KEY_BINDINGS:
            mapped_action = KEY_BINDINGS[key]
            if mapped_action == Action.ARMED:
                # toggle
                if is_pressed:
                    self.action_states[Action.ARMED] = not self.action_states[
                        Action.ARMED
                    ]
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
        [z_altitude, z_altitude_vel, tx_roll, ty_pitch, tz_yaw, tx_roll_rate, ty_pitch_rate, tz_yaw_rate]
        """
        # take only z axis from imu_pos: [x, y, z]
        self.senses[State.Z_ALTITUDE] = self.data.sensor(IMU_POS).data.copy()[2]
        # take only z axis from imu_vel: [vx, vy, vz, wx, wy, wz]
        self.senses[State.Z_ALTITUDE_VEL] = self.data.sensor(IMU_LIN_VEL).data.copy()[2]

        # convert quat to euler angles
        quat = self.data.sensor(IMU_QUAT).data.copy()  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)  # in radians

        self.senses[State.TX_ROLL] = roll
        self.senses[State.TY_PITCH] = pitch
        self.senses[State.TZ_YAW] = yaw

        ang_vel = self.data.sensor(IMU_ANG_VEL).data.copy()
        self.senses[State.TX_ROLL_RATE] = ang_vel[0]
        self.senses[State.TY_PITCH_RATE] = ang_vel[1]
        self.senses[State.TZ_YAW_RATE] = ang_vel[2]

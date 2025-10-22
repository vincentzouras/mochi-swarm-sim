from enum import IntEnum, auto
import numpy as np
from src.definitions import State
from src import preferences
from src.state.robot_state import Behavior


class Control(IntEnum):
    FX = auto()
    FZ = auto()
    TX = auto()
    TZ = auto()


# Porting the C++ PD terms and state
class Differential:
    def __init__(self):
        # PID state variables
        self.z_integral = 0.0
        self.yaw_integral = 0.0
        self.yawrate_integral = 0.0

        # (Differential.cpp::getPreferences)

        # Enable flags for each feedback (height, yaw)
        self.zEn = preferences.Z_EN
        self.yawEn = preferences.YAW_EN
        # no flags for roll, pitch, or rotate because they were disabled in real setup anyways

        # Z PID gains
        self.kpz = preferences.KPZ
        self.kdz = preferences.KDZ
        self.kiz = preferences.KIZ

        # Integral windup limits
        self.z_int_low = preferences.Z_INT_LOW
        self.z_int_high = preferences.Z_INT_HIGH

        # Yaw PID gains
        self.kpyaw = preferences.KPYAW
        self.kppyaw = preferences.KPPYAW
        self.kdyaw = preferences.KDYAW
        self.kddyaw = preferences.KDDYAW
        self.kiyaw = preferences.KIYAW

        self.lx = preferences.LX  # Blimp radius
        self.dt = preferences.DT  # Simulation time step, adjust as needed

    def control(self, sensors: np.ndarray, controls_in: np.ndarray) -> np.ndarray:
        """
        (Differential::control)
        Takes high-level commands and returns low-level actuator outputs.
        """
        if controls_in[Behavior.READY] == 0:
            # Return [right_thrust, left_thrust, servo_angle]
            return np.array([0.0, 0.0, np.pi / 2])  # Motors off, servos up

        # 1. Apply Feedback (PID controllers)
        feedback_controls = self._add_feedback(sensors, controls_in)

        # 2. Calculate Actuator Outputs (Mixer)
        actuator_outputs = self._get_outputs(feedback_controls)

        return actuator_outputs

    def _add_feedback(self, sensors: np.ndarray, controls_in: np.ndarray) -> dict:
        """
        (Differential::addFeedback).
        Converts desired setpoints (e.g., height, yaw) into forces and torques.
        """
        # Get high-level commands
        fx = controls_in[Behavior.FX_FORWARD]  # we dont add feedback just need it
        fz_target = controls_in[Behavior.FZ_HEIGHT]
        tx = controls_in[Behavior.TX_ROLL]  # we dont add feedback just need it
        tz_target = controls_in[Behavior.TZ_YAW]

        # --- Z (Altitude) Feedback ---
        if self.zEn:
            # error calculation
            e_z = fz_target - sensors[State.Z_ALTITUDE]

            # integral update
            # integrate error over time, multiply by timestep, and apply integral gain
            self.z_integral += e_z * self.dt * self.kiz
            # clamp within bounds to prevent windup
            self.z_integral = np.clip(self.z_integral, self.z_int_low, self.z_int_high)

            # PID output calculation
            #              P                        D                      I
            fz_out = (
                (e_z * self.kpz)
                - (sensors[State.Z_ALTITUDE_VEL] * self.kdz)
                + self.z_integral
            )

        # --- Yaw Feedback (Cascading PID) ---
        #  cascading so we have separate PID for yaw and yaw rate
        if self.yawEn:
            # error calculation
            e_yaw = tz_target - sensors[State.TZ_YAW]
            # Normalize yaw error (wraps error to range [-pi, pi])
            e_yaw = np.atan2(np.sin(e_yaw), np.cos(e_yaw))
            # clamp to prevent windup
            e_yaw = np.clip(e_yaw, -np.pi / 5, np.pi / 5)

            # integral update for yaw
            self.yaw_integral += e_yaw * self.dt * self.kiyaw  # TODO: could be wrong
            # clamp within bounds to prevent windup
            self.yaw_integral = np.clip(self.yaw_integral, -np.pi / 5, np.pi / 5)

            # cascading PID integral update for yaw rate
            yaw_desired_rate = e_yaw + self.yaw_integral
            # reduce yaw control when moving forward
            scaling_factor = 1.0 - np.clip(abs(fx), 0.0, 1.0)
            kpyaw_max_increase = 0.04  # TODO: could make preference
            # adjust P term dynamically based on forward speed
            dynamic_kpyaw = self.kpyaw + scaling_factor * kpyaw_max_increase
            e_yawrate = yaw_desired_rate * dynamic_kpyaw - sensors[State.TZ_YAW_RATE]

            kdyaw_max_increase = 0.04  # TODO: could make preference
            # adjust D term dynamically based on forward speed
            dynamic_kdyaw = self.kdyaw + scaling_factor * kdyaw_max_increase

            # PID output calculation, final result of cascading PID with dynamic P and D
            tz_out = (
                yaw_desired_rate * self.kppyaw
                + e_yawrate * dynamic_kdyaw
                - sensors[State.TZ_YAW_RATE] * self.kddyaw
                + self.yawrate_integral  # this is zero, not used in real setup for differential
            )

        # Return a dictionary of the final forces and torques
        return {Control.FX: fx, Control.FZ: fz_out, Control.TX: tx, Control.TZ: tz_out}

    def _get_outputs(self, feedback_controls: dict) -> np.ndarray:
        """
        Differential drive mixer (Differential::getOutputs).
        It converts forces (fx, fz) and torque (tz) into
        motor thrusts (f1, f2) and servo angle (theta).
        """
        fx_target = np.clip(feedback_controls[Control.FX], -1.0, 1.0)  # forward force
        fz_target = np.clip(feedback_controls[Control.FZ], -1.0, 1.0)  # upward force
        tz_target = np.clip(feedback_controls[Control.TZ], -0.1, 0.1)  # yaw torque

        l = self.lx  # distance from center to motor (blimp radius)

        F_mag_sq = fx_target**2 + fz_target**2
        theta = np.atan2(fz_target, fx_target)  # angle of servo

        # omitting exponential moving average, because not used in real setup

        # thrusts for left and right motors
        f1 = 0.0
        f2 = 0.0

        if F_mag_sq > 0:
            # if basically no forward force, assign force so we can still calculate angle
            if abs(fx_target) < 0.00001:
                fx_target = 10 * abs(tz_target)
                theta = np.atan2(fz_target, abs(tz_target * 10))

            # if rotating in place with very small forward force
            if abs(tz_target / fx_target) > 0.1:
                fx_target = 0.2  # set min forward force
                scaled_tz_target = tz_target * abs(fx_target)
                tz_target = scaled_tz_target
                theta = np.atan2(fz_target, fx_target)

            if fx_target < 0:
                theta = np.pi - 0.01

            term1 = tz_target / (l * np.cos(theta))
            term2 = np.sqrt(F_mag_sq)
            f1 = 0.5 * (term1 + term2)
            f2 = 0.5 * (-term1 + term2)

        # Clamp outputs
        f1_out = np.clip(f1, 0.0, 1.0)  # Assuming 0 to 1 range for thrust
        f2_out = np.clip(f2, 0.0, 1.0)
        theta_out = np.clip(theta, -np.pi / 2, np.pi / 2)  # Servo range

        # Return [right_thrust, left_thrust, servo_angle]
        # Match this to your MuJoCo model's actuators
        return np.array([f1_out, f2_out, theta_out])

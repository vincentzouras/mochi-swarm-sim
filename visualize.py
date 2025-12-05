import numpy as np
import mujoco
import mujoco.viewer
import time
from mochi_env import MochiHoverEnv
from stable_baselines3 import PPO
from src.robot.differential import Differential
from src.state.robot_state import Behavior
from src.definitions import State


def run_visual_forever(env, agent_type="rl", model_path=None):
    """
    Runs the simulation in an infinite loop.
    Supports 'rl' (PPO) and 'pid' (Classical Control).
    """
    print(f"--- Visualizing {agent_type.upper()} Agent ---")
    print("Press ESC in the viewer to close.")

    # --- SETUP PID AGENT ---
    diff_drive = None
    behavior = None
    if agent_type == "pid":
        diff_drive = Differential()
        behavior = np.zeros(int(Behavior.NUM_PARAMS))
        behavior[Behavior.READY] = 1
        behavior[Behavior.Z_HEIGHT] = 2.0
        behavior[Behavior.Z_YAW] = 0.0

    # --- SETUP RL AGENT ---
    model = None
    if agent_type == "rl":
        if model_path is None:
            raise ValueError("Need model_path for RL")
        model = PPO.load(model_path)

    # --- LAUNCH VIEWER ---
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # Initial Reset with Seed for consistency
        obs, _ = env.reset(seed=42)

        while viewer.is_running():
            step_start = time.time()

            # --- GET ACTION ---
            if agent_type == "pid":
                # 1. Map Gym Obs -> PID Sensors
                # Obs: [z, vz, roll, pitch, yaw, wx, wy, wz]
                sensors = np.zeros(State.NUM_STATES)
                sensors[State.Z_ALTITUDE] = obs[0]
                sensors[State.Z_ALTITUDE_VEL] = obs[1]
                sensors[State.Z_YAW] = obs[4]
                sensors[State.Z_YAW_RATE] = obs[7]

                # 2. Get PID Output [left, right, servo]
                # values are 0.0 to 1.0 for thrust, -pi to pi for servo
                pid_out = diff_drive.control(sensors, behavior)

                # 3. SQUASH TO 1D (The Adaptation)
                # Average the left/right thrust to get magnitude
                thrust_mag = (pid_out[0] + pid_out[1]) / 2.0
                servo_angle = pid_out[2]

                # If PID wants to point UP, action is positive.
                # If PID wants to point DOWN, action is negative.
                direction = 1.0 if servo_angle >= 0 else -1.0

                # Final 1D action [-1, 1]
                action = [thrust_mag * direction]

            else:
                # RL Agent (Naturally outputs 1D now)
                action, _ = model.predict(obs, deterministic=True)

            # --- STEP ENV ---
            obs, reward, terminated, truncated, info = env.step(action)
            z_height = obs[0]

            # --- UPDATE VIEWER ---
            viewer.sync()

            # --- CHECK CRASH ---
            if terminated or truncated:
                if z_height > 5.0:
                    print(f"RESET: Hit Ceiling! (z={z_height:.2f})")
                elif z_height < 0.1:
                    print(f"RESET: Hit Floor! (z={z_height:.2f})")
                else:
                    print(f"RESET: Terminated (Reward: {reward:.2f})")

                time.sleep(1.0)
                obs, _ = env.reset(seed=42)  # Keep seed consistent

            # --- CAP FPS ---
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    env = MochiHoverEnv()

    # --- TOGGLE THIS TO SWITCH AGENTS ---

    # AGENT = "pid"
    AGENT = "pid"

    # Path to your best RL model
    MODEL_PATH = "models/PPO/230000.zip"

    run_visual_forever(env, agent_type=AGENT, model_path=MODEL_PATH)

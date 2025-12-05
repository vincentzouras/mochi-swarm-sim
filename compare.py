import numpy as np
import matplotlib.pyplot as plt
from mochi_env import MochiHoverEnv
from stable_baselines3 import PPO
from src.robot.differential import Differential
from src.state.robot_state import Behavior
from src.definitions import State


def run_pid_episode(env, steps=1000):
    """Runs the classical PID controller on the environment"""
    env.reset()
    diff_drive = Differential()

    # Mocking behavior commands for [FX, FZ, TX, TZ, Z_HEIGHT, Z_YAW, READY]
    # We want Z=2.0, Yaw=0.0
    # Allocate behavior vector based on enum-defined size
    behavior = np.zeros(int(Behavior.NUM_PARAMS))
    behavior[Behavior.READY] = 1
    behavior[Behavior.Z_HEIGHT] = 2.0
    behavior[Behavior.Z_YAW] = 0.0

    obs, _ = env.reset()
    z_history = []

    for _ in range(steps):
        # Create sensor array expected by Differential.py
        # obs is [z, vz, roll, pitch, yaw, wx, wy, wz]
        # State mapping based on your definitions.py might differ slightly, checking controller.py...
        # Controller.py: [Z_ALTITUDE, Z_ALTITUDE_VEL, X_ROLL, Y_PITCH, Z_YAW, ...]

        sensors = np.zeros(State.NUM_STATES)
        sensors[State.Z_ALTITUDE] = obs[0]
        sensors[State.Z_ALTITUDE_VEL] = obs[1]
        sensors[State.Z_YAW] = obs[4]
        sensors[State.Z_YAW_RATE] = obs[7]

        # Get PID outputs
        # output is [left_thrust, right_thrust, servo_angle]
        actions = diff_drive.control(sensors, behavior)

        # Convert PID [0, 1] / [-pi, pi] to Gym Action Space [-1, 1]
        # Gym Thrust: -1 is 0, 1 is 1
        gym_left = actions[0] * 2 - 1
        gym_right = actions[1] * 2 - 1
        # Gym Servo: -1 is -pi, 1 is pi
        gym_servo = actions[2] / np.pi

        obs, _, terminated, _, _ = env.step([gym_left, gym_right, gym_servo])
        z_history.append(obs[0])

        if terminated:
            break

    return z_history


def run_rl_episode(env, model_path, steps=1000):
    """Runs the PPO Agent"""
    model = PPO.load(model_path)
    obs, _ = env.reset()
    z_history = []

    for _ in range(steps):
        action, _ = model.predict(obs)
        obs, _, terminated, _, _ = env.step(action)
        z_history.append(obs[0])
        if terminated:
            break
    return z_history


# --- Main Execution ---
env = MochiHoverEnv()
pid_trace = run_pid_episode(env)

# CHANGE THIS to your latest saved model path after training
model_path = "models/PPO/100000.zip"
try:
    rl_trace = run_rl_episode(env, model_path)

    plt.plot(pid_trace, label="PID Controller", color="blue", linestyle="--")
    plt.plot(rl_trace, label="PPO (RL)", color="orange")
    plt.axhline(y=2.0, color="r", linestyle=":", label="Target Height")
    plt.legend()
    plt.title("Step Response: PID vs RL")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Altitude (m)")
    plt.show()
except Exception as e:
    print(f"Could not run RL trace (did you train yet?): {e}")
    plt.plot(pid_trace, label="PID Controller")
    plt.show()

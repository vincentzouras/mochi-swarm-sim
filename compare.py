import numpy as np
import matplotlib.pyplot as plt
from mochi_env import MochiHoverEnv
from stable_baselines3 import PPO
from src.robot.differential import Differential
from src.state.robot_state import Behavior
from src.definitions import State

# CONSTANT SEED for fair comparison
SEED = 42


def run_pid_episode(env, steps=1000):
    """Runs the classical PID controller on the environment"""
    # RESET WITH SEED
    obs, _ = env.reset(seed=SEED)

    diff_drive = Differential()

    # Setup Target: Z=2.0, Yaw=0.0
    behavior = np.zeros(int(Behavior.NUM_PARAMS))
    behavior[Behavior.READY] = 1
    behavior[Behavior.Z_HEIGHT] = 2.0
    behavior[Behavior.Z_YAW] = 0.0

    z_history = []

    for _ in range(steps):
        # 1. Map Gym Obs to PID Sensors
        # CAUTION: The new environment returns only [z, vz]
        # But PID expects full state. We fill the rest with zeros since we assume
        # perfect vertical flight in this 1D constraint.
        sensors = np.zeros(State.NUM_STATES)
        sensors[State.Z_ALTITUDE] = obs[0]  # z
        sensors[State.Z_ALTITUDE_VEL] = obs[1]  # vz
        # All other sensors (yaw, roll, etc.) remain 0.0

        # 2. Get PID Output [left, right, servo]
        # left/right are 0.0 to 1.0. Servo is -pi to pi.
        pid_actions = diff_drive.control(sensors, behavior)

        thrust_mag = pid_actions[0]  # Magnitude (0 to 1)
        servo_angle = pid_actions[2]

        # 3. Convert to New "1D Env" Action Space [-1, 1]
        # Logic: If PID points servo UP (>0), action is positive.
        #        If PID points servo DOWN (<0), action is negative.
        direction = 1.0 if servo_angle >= 0 else -1.0

        # Result: 0 thrust -> 0 action. 1.0 thrust UP -> 1.0 action.
        final_action = thrust_mag * direction

        # 4. Step Env
        obs, _, terminated, _, _ = env.step([final_action])
        z_history.append(obs[0])

        if terminated:
            break

    return z_history


def run_rl_episode(env, model_path, steps=1000):
    """Runs the PPO Agent"""
    model = PPO.load(model_path)

    # RESET WITH SAME SEED
    obs, _ = env.reset(seed=SEED)
    z_history = []

    for _ in range(steps):
        # Deterministic=True ensures the RL doesn't "jitter" randomly
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, _, _ = env.step(action)
        z_history.append(obs[0])
        if terminated:
            break
    return z_history


# --- Main Execution ---
env = MochiHoverEnv()

print("Running PID...")
pid_trace = run_pid_episode(env)

# UPDATE THIS PATH to your latest trained model
# Note: You MUST use a model trained on the new 2-input environment!
model_path = "models/PPO2/300000.zip"

print("Running RL...")
try:
    rl_trace = run_rl_episode(env, model_path)

    plt.figure(figsize=(10, 5))
    plt.plot(
        pid_trace, label="PID Controller", color="blue", linestyle="--", linewidth=2
    )
    plt.plot(rl_trace, label="PPO (RL)", color="orange", linewidth=2)
    plt.axhline(y=2.0, color="r", linestyle=":", label="Target Height")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.title("Flight Performance: PID vs PPO (State Space: z, vz)")
    plt.xlabel("Simulation Steps")
    plt.ylabel("Altitude (m)")
    plt.show()
except Exception as e:
    print(f"Could not run RL trace: {e}")

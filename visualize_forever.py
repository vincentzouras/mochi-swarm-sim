import numpy as np
import mujoco
import mujoco.viewer
import time
from mochi_env import MochiHoverEnv
from stable_baselines3 import PPO

# For logging and plotting
import csv
import matplotlib.pyplot as plt


def run_visual_forever(env, model_path, log_csv_path=None, plot_path=None):
    """
    Runs the trained RL agent in an infinite loop.
    Resets automatically on crash.
    """
    print(f"--- Visualizing RL Agent (Infinite Loop) ---")
    print(f"Loading model: {model_path}")
    print("Press ESC in the viewer to close.")

    # Load Model
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Did you delete the 'models' folder and retrain?")
        return

    # --- Logging setup ---
    log_rewards = []
    log_timesteps = []
    global_timestep = 0
    cumulative_reward = 0.0
    if log_csv_path is not None:
        # Write header
        with open(log_csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "reward"])

    # Launch Viewer
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:

        # Initial Reset
        obs, _ = env.reset(seed=42)

        while viewer.is_running():
            step_start = time.time()

            # --- GET ACTION ---
            action, _ = model.predict(obs, deterministic=True)

            # --- STEP ENV ---
            obs, reward, terminated, truncated, info = env.step(action)
            z_height = obs[0]

            # --- LOG CUMULATIVE REWARD ---
            cumulative_reward += reward
            log_rewards.append(cumulative_reward)
            log_timesteps.append(global_timestep)
            if log_csv_path is not None:
                with open(log_csv_path, mode="a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([global_timestep, cumulative_reward])
            global_timestep += 1

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

                # Pause briefly so you can see what happened
                time.sleep(1.0)
                obs, _ = env.reset()

            # --- CAP FPS ---
            time_until_next_step = env.dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    # --- Plot cumulative reward vs timestep at end ---
    if plot_path is not None and log_timesteps and log_rewards:
        plt.figure(figsize=(10, 6))
        plt.plot(log_timesteps, log_rewards, color="orange", linewidth=1.5)
        plt.title("Agent Cumulative Reward vs Timestep (Visualization Run)")
        plt.xlabel("Timestep")
        plt.ylabel("Cumulative Reward")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.show()


if __name__ == "__main__":
    env = MochiHoverEnv()

    # You can change these paths as needed
    log_csv = "logs/visualize_reward_log.csv"
    plot_png = "visualize_reward_curve.png"
    run_visual_forever(
        env,
        model_path="models/PPO2/300000.zip",
        log_csv_path=log_csv,
        plot_path=plot_png,
    )

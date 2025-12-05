from stable_baselines3 import PPO
from mochi_env import MochiHoverEnv
from reward_callback import PerStepRewardLogger
import os

# Create log dir
models_dir = "models/PPO2"
log_dir = "logs2"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize Environment
env = MochiHoverEnv()
env.reset()

# Initialize Agent
# MlpPolicy = Standard Neural Network
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# Per-timestep reward CSV logger
reward_csv = os.path.join(log_dir, "step_rewards.csv")
reward_logger = PerStepRewardLogger(csv_path=reward_csv)

# Train in loop
TIMESTEPS = 10000
for i in range(1, 31):
    model.learn(
        total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=reward_logger
    )

    save_path = f"{models_dir}/{TIMESTEPS*i}"
    model.save(save_path)
    print(f"Saved model at {save_path}")

env.close()

from stable_baselines3 import PPO
from mochi_env import MochiHoverEnv
import os

# Create log dir
models_dir = "models/PPO"
log_dir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Initialize Environment
env = MochiHoverEnv()
env.reset()

# Initialize Agent
# MlpPolicy means "Multi-Layer Perceptron" (standard neural net)
# Use CPU for MLP policies to avoid poor GPU utilization warning
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, device="cpu")

# Train
TIMESTEPS = 10000
for i in range(1, 30):
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    model.save(f"{models_dir}/{TIMESTEPS*i}")
    print(f"Saved model at step {TIMESTEPS*i}")

env.close()

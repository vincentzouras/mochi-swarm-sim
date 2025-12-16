# load the model for blimp dynamics 

import pickle as pkl
import torch
from saviolo_et_al_mlp import DiscreteQuadDynamicsNN
from mppi_controller import BlimpMPPI
import numpy as np
from blimp_gym import BlimpGymEnv

# Load learnt dynamics model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sav_model = DiscreteQuadDynamicsNN()
sav_model.to(device)
print(device)

sav_model.load_model("trained_models/trained_model_noisy_spiral_0_05.pth")

# Initialize MPPI controller
mppi = BlimpMPPI(
    dynamics_model=sav_model,
    horizon=60,           # Prediction horizon (60 * 0.01s = 0.6s lookahead)
    num_samples=4000,     # Number of samples (reduce for faster computation)
    lambda_=1.0,          # Temperature parameter
    sigma=[0.05, 0.05, 1.0],           # Action noise standard deviation
    dt=0.01,              # Time step (should match your simulation dt)
    device=device,
    pos_weight=[0.0, 0.0, 1.0],  # Only care about Z (altitude) for hover
    vel_weight=0.5,              # penalize vertical velocity to encourage hover
)

# Create environment instance

env = BlimpGymEnv(
    model_xml_path="models/mochi.xml",
    headless=False,  # Set False to see GUI
    dt=0.05,
    max_episode_steps=None
)

# Reset environment
obs, info = env.reset()
print(f"Initial Position: {info['position']}")
target_position = np.array([0.0, 0.0, 3.0])

# Step through environment using MPPI controller
for i in range(10000):
    action = mppi.compute_action(obs, info['position'], target_position)
    # print(f"Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action.cpu().numpy())
    if terminated or truncated:
        obs, info = env.reset()
    if i % 100 == 0:
        print(f"Step {i}: Position = {info['position']}")

# Cleanup
env.close()
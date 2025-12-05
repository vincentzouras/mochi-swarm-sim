import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt


def extract_tensorboard_data(log_dir):
    # Find the latest event file
    subdirs = [
        os.path.join(log_dir, d)
        for d in os.listdir(log_dir)
        if os.path.isdir(os.path.join(log_dir, d))
    ]
    latest_subdir = max(subdirs, key=os.path.getmtime)
    event_file = [
        os.path.join(latest_subdir, f)
        for f in os.listdir(latest_subdir)
        if "tfevents" in f
    ][0]

    print(f"Loading logs from: {event_file}")

    ea = EventAccumulator(event_file)
    ea.Reload()

    # Extract Reward Data
    if "rollout/ep_rew_mean" not in ea.Tags()["scalars"]:
        print("No reward data found! Did you finish training?")
        return [], []

    events = ea.Scalars("rollout/ep_rew_mean")
    steps = [e.step for e in events]
    rewards = [e.value for e in events]

    # Convert to cumulative reward over time
    cumulative_rewards = []
    total = 0.0
    for r in rewards:
        total += r
        cumulative_rewards.append(total)

    return steps, cumulative_rewards


# --- MAIN ---
log_dir = "logs/PPO_0"  # Check your logs folder name, it might be PPO_1, PPO_2 etc.
try:
    steps, rewards = extract_tensorboard_data("logs")

    plt.figure(figsize=(10, 6))
    plt.plot(steps, rewards, linewidth=2.5, color="purple")

    plt.title("Training Progress: Mean Reward vs Timesteps", fontsize=14)
    plt.xlabel("Timesteps", fontsize=12)
    plt.ylabel("Mean Episode Reward", fontsize=12)
    plt.grid(True, alpha=0.3)

    # Highlight the "Breakthrough" point
    # plt.axvline(x=212000, color="green", linestyle="--", label="Convergence Point")
    plt.legend()

    plt.tight_layout()
    plt.savefig("reward_curve.png")
    plt.show()

except Exception as e:
    print(f"Could not parse logs: {e}")
    print("Try using 'tensorboard --logdir logs' in your terminal instead.")

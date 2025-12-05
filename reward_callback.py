from stable_baselines3.common.callbacks import BaseCallback
import csv
import os
from typing import Optional


class PerStepRewardLogger(BaseCallback):
    """
    Stable-Baselines3 callback that logs per-timestep rewards to a CSV file.

    Columns: timestep, episode, step_in_episode, reward, cumulative_reward, done
    """

    def __init__(self, csv_path: str, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.csv_path = csv_path
        self._csv_file: Optional[object] = None
        self._writer: Optional[csv.writer] = None
        self._episode_idx: int = 0
        self._step_in_episode: int = 0
        self._cumulative_reward: float = 0.0

    def _on_training_start(self) -> None:
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        # Open CSV and write header
        self._csv_file = open(self.csv_path, mode="w", newline="")
        self._writer = csv.writer(self._csv_file)
        self._writer.writerow(
            [
                "timestep",
                "episode",
                "step_in_episode",
                "reward",
                "cumulative_reward",
                "done",
            ]
        )

    def _on_step(self) -> bool:
        # Rewards and dones are provided by the algorithm during rollout collection
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is None or self._writer is None:
            # Nothing to log
            return True

        # Handle vectorized/non-vectorized case (SB3 wraps single env in DummyVecEnv)
        try:
            reward_val = float(rewards[0])
        except Exception:
            reward_val = float(rewards)

        done_bool = False
        if dones is not None:
            try:
                done_bool = bool(dones[0])
            except Exception:
                done_bool = bool(dones)

        # Update episode tracking
        self._step_in_episode += 1
        self._cumulative_reward += reward_val

        # Write row
        self._writer.writerow(
            [
                int(self.num_timesteps),
                int(self._episode_idx),
                int(self._step_in_episode),
                float(reward_val),
                float(self._cumulative_reward),
                int(done_bool),
            ]
        )
        # Flush to ensure data is written even on long runs
        self._csv_file.flush()

        # Reset counters when an episode ends
        if done_bool:
            self._episode_idx += 1
            self._step_in_episode = 0
            self._cumulative_reward = 0.0

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.close()
            finally:
                self._csv_file = None
                self._writer = None

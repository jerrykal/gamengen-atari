import io
import os

import gymnasium as gym
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from stable_baselines3.common.type_aliases import GymStepReturn


class DataCollectWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, rank: int, output_dir: str) -> None:
        super().__init__(env)
        assert self.render_mode == "rgb_array", "render_mode must be rgb_array for data collection"

        if os.path.isdir(output_dir) and os.listdir(output_dir):
            raise ValueError(f"Output directory {output_dir} already exists and is not empty")
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir
        self.rank = rank
        self.episode_id = 0
        self.step_id = 0

        self._reset_episode_data()

    def _reset_episode_data(self) -> None:
        self.episode_data = {
            "rank": self.rank,
            "episode_id": self.episode_id,
            "frame": [],
            "action": [],
            "step_id": [],
        }
        self.episode_id += 1
        self.step_id = 0

    def _dump_episode_data(self) -> None:
        """Save the episode data to a parquet file"""
        table = pa.Table.from_pandas(pd.DataFrame(self.episode_data))
        pq.write_table(table, os.path.join(self.output_dir, f"rank_{self.rank}_episode_{self.episode_id}.parquet"))

    def step(self, action: int) -> GymStepReturn:
        """Collect frame and action data during the episode"""
        frame = Image.fromarray(self.render())
        buffer = io.BytesIO()
        frame.save(buffer, format="PNG")
        self.episode_data["frame"].append(buffer.getvalue())
        self.episode_data["action"].append(action)
        self.episode_data["step_id"].append(self.step_id)

        observation, reward, terminated, truncated, info = self.env.step(action)
        self.step_id += 1

        if terminated or truncated:
            self._dump_episode_data()
            self._reset_episode_data()

        return observation, reward, terminated, truncated, info

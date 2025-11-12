import io
import os
from collections.abc import Mapping, Sequence
from typing import Any

import datasets
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from datasets.splits import Split
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class GameplayDataset(Dataset):
    """
    A dataset class for handling gameplay trajectories.

    Each example in the dataset is a sequence of contiguous gameplay steps, represented as (frame, action) pairs.
    When an episode ends, the sequence is followed by steps from a new episode.
    Multiple examples, each of length `context_length`, are concatenated together to form a single dataset example.
    """

    def __init__(
        self,
        path: str,
        name: str | None = None,
        data_dir: str | Sequence[str] | Mapping[str, str | Sequence[str]] | None = None,
        data_files: str | None = None,
        split: str | Split = "train",
        context_length: int = 16,
        width: int = 256,
        height: int = 256,
        **load_dataset_kwargs,
    ) -> None:
        self.context_length = context_length

        assert split is not None, "Dataset split must be specified"
        if os.path.exists(os.path.join(path, "dataset_info.json")) or os.path.exists(
            os.path.join(path, "dataset_dict.json")
        ):
            # Load the dataset from disk if it exists
            self.dataset: datasets.Dataset = load_from_disk(path)
            if isinstance(self.dataset, DatasetDict):
                self.dataset = self.dataset[split]
        else:
            self.dataset: datasets.Dataset = load_dataset(
                path=path,
                name=name,
                data_dir=data_dir,
                data_files=data_files,
                split=split,
                **load_dataset_kwargs,
            )
        self.dataset = self.dataset.select_columns(["frame", "action", "step_id"])
        self._action_dim = max(self.dataset["action"]) + 1

        # NOTE: The images are resized to 256x256, which differs from the original paper.
        #       The original paper pads the VizDoom resolution of 320x240 to 320x256 instead.
        self._frame_transform = transforms.Compose(
            [
                transforms.Resize((width, height), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.dataset.set_transform(self._transform)

    @property
    def action_dim(self) -> int:
        return self._action_dim

    def _transform(self, example: dict[str, Any]) -> dict[str, Any]:
        frames = torch.stack([self._frame_transform(Image.open(io.BytesIO(frame))) for frame in example["frame"]])
        actions = torch.tensor(example["action"])
        return {
            "pixel_values": frames,
            "input_ids": actions,
            # step_id is used to determine the beginning of an episode during __getitem__.
            "step_id": example["step_id"],
        }

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        step_id = self.dataset[idx]["step_id"]
        start_idx = idx - (self.context_length if step_id > self.context_length else step_id)

        pixel_values = self.dataset[start_idx : idx + 1]["pixel_values"]
        input_ids = self.dataset[start_idx : idx + 1]["input_ids"]

        # Insert padding when step_id is smaller than the context length.
        # This occurs at the start of an episode when the frame count is below the context length.
        if step_id < self.context_length:
            pad_len = self.context_length - step_id
            pad_pixel_values = torch.zeros(
                pad_len,
                *pixel_values.shape[1:],
                dtype=pixel_values.dtype,
                device=pixel_values.device,
            )

            # NOTE: input_ids are padded with zeros, which are considered as the default value for NOOP
            pad_input_ids = torch.zeros(pad_len, dtype=input_ids.dtype, device=input_ids.device)

            pixel_values = torch.cat([pad_pixel_values, pixel_values], dim=0)
            input_ids = torch.cat([pad_input_ids, input_ids], dim=0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }

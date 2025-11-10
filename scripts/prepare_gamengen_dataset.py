"""
This script prepares GameNGen dataset from data collected via RL agent gameplay.
"""

import glob
import os
import random
from argparse import ArgumentParser

from datasets import concatenate_datasets, load_dataset


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the directory containing the data files(required).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output the hf dataset(required).",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.9,
        help="Size of the train split, should be between 0.0 and 1.0(default: 0.9).",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Whether to not shuffle the dataset(default: False).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling(default: 42).",
    )
    args = parser.parse_args()

    if os.path.isdir(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError(f"Output directory {args.output_dir} already exists and is not empty.")
    os.makedirs(args.output_dir, exist_ok=True)

    # Data are shuffled at file level, since each file contains gameplay trajectory
    # for a single episode. This ensures that trajectory from the same episode are
    # contiguous in the dataset.
    data_files = sorted(glob.glob(os.path.join(args.data_dir, "*.parquet")))
    if not args.no_shuffle:
        random.seed(args.seed)
        random.shuffle(data_files)

    dataset = load_dataset(
        "parquet",
        data_files=data_files,
        split="train",
    )
    if args.train_size < 1.0:
        dataset = dataset.train_test_split(
            train_size=args.train_size,
            shuffle=False,
        )

        # The trajectory from the last episode in the train set could be split between the train and test sets.
        # Therefore, we move it from the test set to the train set.
        episode_end = dataset["test"]["step_id"].index(0)
        dataset["train"] = concatenate_datasets([dataset["train"], dataset["test"].select(range(episode_end))])
        dataset["test"] = dataset["test"].select(range(episode_end, len(dataset["test"])))

        print(f"Number of examples in train split: {len(dataset['train'])}")
        print(f"Number of examples in test split: {len(dataset['test'])}")
        print(f"Saving dataset to {args.output_dir}")
        dataset.save_to_disk(args.output_dir)
    else:
        print(f"Total number of examples: {len(dataset)}")
        print(f"Saving dataset to {args.output_dir}")
        dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    main()

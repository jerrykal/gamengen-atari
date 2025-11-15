# GameNGen Atari

An unofficial implementation of [GameNGen](https://gamengen.github.io/) simulating various classic Atari games via Diffusion Model.

## Getting Started

### Prerequisites

- Install [`uv`](https://github.com/astral-sh/uv) and ensure it is available on your `PATH`.

### Installation

To install all required dependencies, run:

```bash
uv sync
```

This will set up your environment according to the project's `pyproject.toml` configuration.

## Training

First, activate the virtual environment created by `uv`. For detailed instructions, refer to the [uv documentation](https://docs.astral.sh/uv/pip/environments/).

### Collect Gameplay Data

This project uses a customized version of the `rl-baselines3-zoo` framework, which primarily adds a wrapper around the environment to capture gameplay videos and agent actions during RL training.

To collect gameplay data with `rl-baselines3-zoo`, run:

```bash
python -m rl_zoo3.train --algo ppo --env PongNoFrameskip-v4 --data-collect-dir data
```

This command will create a dataset of game episodes in the `data` directory, which can be used for subsequent training.

### Preprocess dataset

Before training, you need to preprocess the raw recorded episodes into a format suitable for training the diffusion model. This script compiles and optionally shuffles the episode files into a final dataset directory.

```bash
python scripts/prepare_gamengen_dataset.py --data-dir data/ppo/PongNoFrameskip-v4_1 --output-dir data/gamengen/pong
```

### Train Action-conditioned Diffusion Model

Once your dataset is ready, you can begin training the GameNGen model. The following command will launch the training script with the default settings as defined in `scripts/train_gamengen.sh`. This script will handle model initialization, configuration, and checkpointing.

```bash
bash scripts/train_gamengen.sh
```

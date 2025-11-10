import os
from typing import Any, Callable

import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from rl_zoo3.data_collect.wrapper import DataCollectWrapper


def make_data_collect_vec_env(
    env_id: str | Callable[..., gym.Env],
    n_envs: int = 1,
    seed: int | None = None,
    start_index: int = 0,
    data_collect_dir: str | None = None,
    monitor_dir: str | None = None,
    wrapper_class: Callable[[gym.Env], gym.Env] | None = None,
    env_kwargs: dict[str, Any] | None = None,
    vec_env_cls: type[DummyVecEnv] | type[SubprocVecEnv] | None = None,
    vec_env_kwargs: dict[str, Any] | None = None,
    monitor_kwargs: dict[str, Any] | None = None,
    wrapper_kwargs: dict[str, Any] | None = None,
) -> VecEnv:
    env_kwargs = env_kwargs or {}
    vec_env_kwargs = vec_env_kwargs or {}
    monitor_kwargs = monitor_kwargs or {}
    wrapper_kwargs = wrapper_kwargs or {}
    assert vec_env_kwargs is not None  # for mypy

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert monitor_kwargs is not None
            assert wrapper_kwargs is not None
            assert env_kwargs is not None

            if isinstance(env_id, str):
                # if the render mode was not specified, we set it to `rgb_array` as default.
                kwargs = {"render_mode": "rgb_array"}
                kwargs.update(env_kwargs)
                try:
                    env = gym.make(env_id, **kwargs)  # type: ignore[arg-type]
                except TypeError:
                    env = gym.make(env_id, **env_kwargs)
            else:
                if data_collect_dir is not None:
                    env_kwargs["render_mode"] = "rgb_array"
                env = env_id(**env_kwargs)
                # Patch to support gym 0.21/0.26 and gymnasium
                env = _patch_env(env)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)

            # Wrap the env in a Monitor wrapper
            # to have additional training information
            monitor_path = os.path.join(monitor_dir, str(rank)) if monitor_dir is not None else None
            # Create the monitor folder if needed
            if monitor_path is not None and monitor_dir is not None:
                os.makedirs(monitor_dir, exist_ok=True)
            env = Monitor(env, filename=monitor_path, **monitor_kwargs)

            # Wrap the env in a DataCollectWrapper to collect (frame, action) pair data
            if data_collect_dir is not None:
                env = DataCollectWrapper(env, rank=rank, output_dir=data_collect_dir)

            # Optionally, wrap the environment with the provided wrapper
            if wrapper_class is not None:
                env = wrapper_class(env, **wrapper_kwargs)
            return env

        return _init

    # No custom VecEnv is passed
    if vec_env_cls is None:
        # Default: use a DummyVecEnv
        vec_env_cls = DummyVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)], **vec_env_kwargs)
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env

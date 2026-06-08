from __future__ import annotations

from typing import Any, Literal

import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices, VecEnvObs, VecEnvStepReturn

from pingpong_rl2.envs import PingPongKeepUpGymEnv

VectorMode = Literal["async", "sync"]


def make_env_factory(env_kwargs: dict[str, object] | None = None):
    # multiprocessing worker가 pickle할 수 있도록 env 생성 인자를 복사한 thunk를 만든다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/gym_env.py:17
    environment_kwargs = {} if env_kwargs is None else dict(env_kwargs)

    def _thunk() -> PingPongKeepUpGymEnv:
        return PingPongKeepUpGymEnv(**environment_kwargs)

    return _thunk


def make_gym_vector_env(
    num_envs: int,
    env_kwargs: dict[str, object] | None = None,
    vector_mode: VectorMode = "async",
    context: str = "spawn",
):
    # Gymnasium vector env를 만들되 autoreset은 끄고 SB3 adapter가 terminal info를 보존하게 한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/training/vector_env.py:49
    if num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {num_envs}.")
    env_fns = [make_env_factory(env_kwargs=env_kwargs) for _ in range(num_envs)]
    if vector_mode == "async":
        return AsyncVectorEnv(
            env_fns,
            shared_memory=False,
            copy=True,
            context=context,
            autoreset_mode=AutoresetMode.DISABLED,
        )
    if vector_mode == "sync":
        return SyncVectorEnv(env_fns, autoreset_mode=AutoresetMode.DISABLED)
    raise ValueError(f"vector_mode must be 'async' or 'sync', got {vector_mode!r}.")


class SB3AsyncVectorEnvAdapter(VecEnv):
    # Gymnasium Async/SyncVectorEnv를 SB3 VecEnv 인터페이스로 감싸 episode 종료 처리를 맞춘다.
    # LINK: pingpong_rl2/scripts/run_ppo_learning.py:113
    def __init__(self, vector_env: AsyncVectorEnv | SyncVectorEnv):
        # autoreset을 직접 수행해야 terminal_observation과 reset_infos를 SB3 형식으로 남길 수 있다.
        if vector_env.metadata.get("autoreset_mode") != AutoresetMode.DISABLED:
            raise ValueError("SB3AsyncVectorEnvAdapter requires AsyncVectorEnv with autoreset disabled.")
        self.vector_env = vector_env
        super().__init__(
            num_envs=vector_env.num_envs,
            observation_space=vector_env.single_observation_space,
            action_space=vector_env.single_action_space,
        )
        self.metadata = dict(vector_env.metadata)
        self._pending_actions: np.ndarray | None = None

    @staticmethod
    def _split_infos(vector_infos: dict[str, Any], num_envs: int) -> list[dict[str, Any]]:
        # Gymnasium vector info의 columnar dict와 mask를 SB3가 기대하는 env별 dict 목록으로 분해한다.
        info_list: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        for key, value in vector_infos.items():
            if key.startswith("_"):
                continue
            mask_key = f"_{key}"
            mask = None
            if mask_key in vector_infos:
                mask = np.asarray(vector_infos[mask_key], dtype=bool)

            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == num_envs:
                for env_index in range(num_envs):
                    if mask is not None and not bool(mask[env_index]):
                        continue
                    info_list[env_index][key] = value[env_index]
                continue

            if isinstance(value, (list, tuple)) and len(value) == num_envs:
                for env_index in range(num_envs):
                    if mask is not None and not bool(mask[env_index]):
                        continue
                    info_list[env_index][key] = value[env_index]
                continue

            for env_index in range(num_envs):
                info_list[env_index][key] = value
        return info_list

    def reset(self) -> VecEnvObs:
        # SB3 seed/options 저장소를 Gymnasium reset 인자로 옮기고 reset_infos를 env별로 보관한다.
        options = None
        if any(bool(option) for option in self._options):
            first_option = self._options[0]
            if any(option != first_option for option in self._options[1:]):
                raise ValueError("SB3AsyncVectorEnvAdapter only supports shared reset options.")
            options = dict(first_option)
        seeds = self._seeds if any(seed is not None for seed in self._seeds) else None
        observations, infos = self.vector_env.reset(seed=seeds, options=options)
        self.reset_infos = self._split_infos(infos, self.num_envs)
        self._reset_seeds()
        self._reset_options()
        return np.asarray(observations)

    def step_async(self, actions: np.ndarray) -> None:
        # AsyncVectorEnv는 비동기 step을 그대로 쓰고, SyncVectorEnv는 action만 저장해 step_wait에서 실행한다.
        if hasattr(self.vector_env, "step_async"):
            self.vector_env.step_async(actions)
            return
        self._pending_actions = np.asarray(actions)

    def step_wait(self) -> VecEnvStepReturn:
        # terminated/truncated를 done으로 합치고, 끝난 env만 reset해 다음 observation batch를 이어 붙인다.
        if hasattr(self.vector_env, "step_wait"):
            observations, rewards, terminations, truncations, vector_infos = self.vector_env.step_wait()
        else:
            if self._pending_actions is None:
                raise RuntimeError("step_wait called before step_async.")
            observations, rewards, terminations, truncations, vector_infos = self.vector_env.step(self._pending_actions)
            self._pending_actions = None
        dones = np.asarray(np.logical_or(terminations, truncations), dtype=bool)
        info_list = self._split_infos(vector_infos, self.num_envs)
        observations = np.asarray(observations)
        rewards = np.asarray(rewards, dtype=np.float32)
        for env_index in range(self.num_envs):
            info_list[env_index]["TimeLimit.truncated"] = bool(truncations[env_index] and not terminations[env_index])
            if dones[env_index]:
                info_list[env_index]["terminal_observation"] = np.array(observations[env_index], copy=True)

        if np.any(dones):
            observations, reset_infos = self.vector_env.reset(options={"reset_mask": dones})
            observations = np.asarray(observations)
            split_reset_infos = self._split_infos(reset_infos, self.num_envs)
            for env_index, done in enumerate(dones):
                if done:
                    self.reset_infos[env_index] = split_reset_infos[env_index]

        return observations, rewards, dones, info_list

    def close(self) -> None:
        self.vector_env.close()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        values = list(self.vector_env.get_attr(attr_name))
        return [values[index] for index in self._get_indices(indices)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        # 일부 env만 바꿀 때도 Gymnasium vector env에는 전체 env 값 목록을 다시 전달해야 한다.
        if indices is None:
            self.vector_env.set_attr(attr_name, value)
            return
        target_indices = list(self._get_indices(indices))
        current_values = list(self.vector_env.get_attr(attr_name))
        if isinstance(value, (list, tuple)) and len(value) == len(target_indices):
            for target_index, target_value in zip(target_indices, value, strict=True):
                current_values[target_index] = target_value
        else:
            for target_index in target_indices:
                current_values[target_index] = value
        self.vector_env.set_attr(attr_name, current_values)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        results = list(self.vector_env.call(method_name, *method_args, **method_kwargs))
        return [results[index] for index in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class: type, indices: VecEnvIndices = None) -> list[bool]:
        return [False for _ in self._get_indices(indices)]

    def get_images(self) -> list[np.ndarray | None]:
        if self.render_mode != "rgb_array":
            return [None for _ in range(self.num_envs)]
        rendered = self.vector_env.render()
        if rendered is None:
            return [None for _ in range(self.num_envs)]
        return list(rendered)


def make_sb3_async_vector_env(
    num_envs: int,
    env_kwargs: dict[str, object] | None = None,
    seed: int | None = None,
    context: str = "spawn",
) -> SB3AsyncVectorEnvAdapter:
    # 단일 env는 sync로 가볍게 돌리고, 여러 env는 async worker로 병렬 rollout을 만든다.
    # LINK: pingpong_rl2/scripts/benchmark_vector_env.py:31
    vector_mode: VectorMode = "sync" if num_envs == 1 else "async"
    vector_env = make_gym_vector_env(
        num_envs=num_envs,
        env_kwargs=env_kwargs,
        vector_mode=vector_mode,
        context=context,
    )
    adapter = SB3AsyncVectorEnvAdapter(vector_env)
    if seed is not None:
        adapter.seed(seed)
    return adapter

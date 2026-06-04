from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl3.defaults import (
    DEFAULT_PPO_BATCH_SIZE,
    DEFAULT_PPO_CLIP_RANGE,
    DEFAULT_PPO_ENT_COEF,
    DEFAULT_PPO_GAE_LAMBDA,
    DEFAULT_PPO_GAMMA,
    DEFAULT_PPO_LEARNING_RATE,
    DEFAULT_PPO_LOG_STD_INIT,
    DEFAULT_PPO_N_STEPS,
    DEFAULT_PPO_TOTAL_TIMESTEPS,
)
from pingpong_rl3.training import make_sb3_async_vector_env
from pingpong_rl3.utils import ARTIFACTS_ROOT, resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the pingpong_rl3 two-ball keep-up PPO policy.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "keep2_v1.json")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> dict[str, Any]:
    resolved_path = resolve_input_path(config_path)
    with resolved_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    run_name = args.run_name or str(config.get("run_name", "keep2_v1"))
    total_steps = int(args.total_steps or config.get("total_timesteps", DEFAULT_PPO_TOTAL_TIMESTEPS))
    num_envs = int(args.num_envs or config.get("num_envs", 8))
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))

    env_kwargs = dict(config.get("env", {}))
    ppo_config = dict(config.get("ppo", {}))
    policy_kwargs = dict(ppo_config.get("policy_kwargs", {}))
    policy_kwargs.setdefault("log_std_init", float(ppo_config.get("log_std_init", DEFAULT_PPO_LOG_STD_INIT)))

    vector_env = VecMonitor(make_sb3_async_vector_env(num_envs=num_envs, env_kwargs=env_kwargs, seed=seed))
    if args.resume_from is not None:
        model = PPO.load(str(resolve_input_path(args.resume_from)), env=vector_env, device=args.device or "auto")
        model.set_random_seed(seed)
    else:
        model = PPO(
            "MlpPolicy",
            vector_env,
            n_steps=int(ppo_config.get("n_steps", DEFAULT_PPO_N_STEPS)),
            batch_size=int(ppo_config.get("batch_size", DEFAULT_PPO_BATCH_SIZE)),
            learning_rate=float(ppo_config.get("learning_rate", DEFAULT_PPO_LEARNING_RATE)),
            gamma=float(ppo_config.get("gamma", DEFAULT_PPO_GAMMA)),
            gae_lambda=float(ppo_config.get("gae_lambda", DEFAULT_PPO_GAE_LAMBDA)),
            clip_range=float(ppo_config.get("clip_range", DEFAULT_PPO_CLIP_RANGE)),
            ent_coef=float(ppo_config.get("ent_coef", DEFAULT_PPO_ENT_COEF)),
            vf_coef=float(ppo_config.get("vf_coef", 0.5)),
            max_grad_norm=float(ppo_config.get("max_grad_norm", 0.5)),
            policy_kwargs=policy_kwargs,
            seed=seed,
            verbose=int(ppo_config.get("verbose", 1)),
            device=args.device or ppo_config.get("device", "auto"),
        )

    run_dir = ARTIFACTS_ROOT / "ppo_runs" / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = run_dir / "training_config.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "run_name": run_name,
                "config_path": str(resolve_input_path(args.config)),
                "total_timesteps": total_steps,
                "num_envs": num_envs,
                "seed": seed,
                "resume_from": None if args.resume_from is None else str(resolve_input_path(args.resume_from)),
                "env": env_kwargs,
                "ppo": ppo_config,
            },
            file,
            indent=2,
        )

    model.learn(
        total_timesteps=total_steps,
        reset_num_timesteps=args.resume_from is None,
        progress_bar=bool(ppo_config.get("progress_bar", False)),
    )
    model_path = run_dir / f"{run_name}_model.zip"
    model.save(str(model_path))
    vector_env.close()
    print(f"saved_model={model_path}")
    print(f"training_config={summary_path}")


if __name__ == "__main__":
    main()

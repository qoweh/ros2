from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl3.envs import TwoBallKeepUpGymEnv
from pingpong_rl3.utils import resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained pingpong_rl3 two-ball PPO model.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "keep2_v1.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_env_kwargs(config_path: Path) -> dict[str, Any]:
    with resolve_input_path(config_path).open("r", encoding="utf-8") as file:
        return dict(json.load(file).get("env", {}))


def main() -> None:
    args = parse_args()
    env = TwoBallKeepUpGymEnv(**load_env_kwargs(args.config))
    model = PPO.load(str(resolve_input_path(args.model_path)), device=args.device)
    failure_counts: Counter[str] = Counter()
    episode_returns: list[float] = []
    step_counts: list[int] = []
    useful_counts: list[int] = []
    contact_counts: list[int] = []
    episode_rows: list[dict[str, object]] = []

    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        done = False
        info: dict[str, object] = {}
        episode_return = 0.0
        while not done:
            action, _ = model.predict(observation, deterministic=not args.stochastic)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            done = bool(terminated or truncated)
        failure_reason = str(info.get("failure_reason") or "time_limit")
        failure_counts[failure_reason] += 1
        steps = int(info.get("elapsed_steps", 0))
        useful_bounces = int(info.get("useful_bounces", 0))
        contacts = int(info.get("contact_count", 0))
        episode_returns.append(episode_return)
        step_counts.append(steps)
        useful_counts.append(useful_bounces)
        contact_counts.append(contacts)
        episode_rows.append(
            {
                "episode": episode + 1,
                "return": episode_return,
                "steps": steps,
                "contacts": contacts,
                "useful_bounces": useful_bounces,
                "failure_reason": failure_reason,
            }
        )

    summary = {
        "episodes": args.episodes,
        "model_path": str(resolve_input_path(args.model_path)),
        "deterministic": not args.stochastic,
        "seed": args.seed,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "max_steps": int(max(step_counts, default=0)),
        "mean_useful_bounces": float(np.mean(useful_counts)) if useful_counts else 0.0,
        "max_useful_bounces": int(max(useful_counts, default=0)),
        "mean_contacts": float(np.mean(contact_counts)) if contact_counts else 0.0,
        "max_contacts": int(max(contact_counts, default=0)),
        "failure_counts": dict(failure_counts),
        "episodes_detail": episode_rows,
    }
    print(json.dumps({key: value for key, value in summary.items() if key != "episodes_detail"}, indent=2))
    if args.output is not None:
        output_path = resolve_input_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        print(f"analysis_output={output_path}")
    env.close()


if __name__ == "__main__":
    main()

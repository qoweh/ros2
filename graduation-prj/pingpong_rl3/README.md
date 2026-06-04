# pingpong_rl3

Two-ball ping-pong keep-up RL project built from the useful parts of `pingpong_rl2`, but cleaned up around one purpose: train a PPO policy that keeps two balls airborne with one racket.

## Structure

- `assets/scene.xml`: MuJoCo scene with the Franka racket and two ping-pong balls.
- `src/pingpong_rl3/envs/two_ball_sim.py`: small MuJoCo wrapper for two freejoint balls.
- `src/pingpong_rl3/envs/two_ball_keepup_env.py`: 2-ball keep-up task, reward, reset randomization, action/observation logic.
- `scripts/train.py`: config-driven PPO training. No checkpoint callback.
- `scripts/analyze.py`: episode summary for a saved model.
- `scripts/viewer.py`: MuJoCo viewer for zero-action or trained-policy playback.
- `configs/keep2_v1.json`: first 2-ball training config.

## Commands

Run from `pingpong_rl3/`.

```bash
pip install -e ".[dev]"
```

```bash
pytest -q
```

```bash
python scripts/train.py --config configs/keep2_v1.json
```

```bash
python scripts/analyze.py \
  --model-path artifacts/ppo_runs/keep2_v1/keep2_v1_model.zip \
  --config configs/keep2_v1.json \
  --episodes 100 \
  --output artifacts/ppo_runs/keep2_v1/analysis/summary.json
```

```bash
mjpython scripts/viewer.py \
  --model-path artifacts/ppo_runs/keep2_v1/keep2_v1_model.zip \
  --config configs/keep2_v1.json \
  --episodes 20
```

## Notes

- `max_episode_steps=0` disables the time limit.
- Contact reward is counted only when a racket contact starts during the current control step.
- Short smoke runs verified the single-env and async vector PPO paths in `mujoco_env`.

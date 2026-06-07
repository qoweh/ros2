from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pingpong_rl2.envs import PingPongSim
from pingpong_rl2.utils import resolve_input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ping-pong ball/racket XML material and rebound sanity.")
    parser.add_argument("--scene-path", type=Path, default=Path("assets/scene.xml"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--drop-height", type=float, default=0.30)
    parser.add_argument("--initial-velocity-z", type=float, default=0.0)
    parser.add_argument("--max-substeps", type=int, default=2500)
    return parser.parse_args()


def geom_summary(sim: PingPongSim, geom_name: str) -> dict[str, object]:
    geom = sim.model.geom(geom_name)
    geom_id = geom.id
    body_id = int(sim.model.geom_bodyid[geom_id])
    return {
        "name": geom_name,
        "body_name": sim.model.body(body_id).name,
        "size": np.asarray(sim.model.geom_size[geom_id], dtype=float).tolist(),
        "body_mass": float(sim.model.body_mass[body_id]),
        "friction": np.asarray(sim.model.geom_friction[geom_id], dtype=float).tolist(),
        "solref": np.asarray(sim.model.geom_solref[geom_id], dtype=float).tolist(),
        "solimp": np.asarray(sim.model.geom_solimp[geom_id], dtype=float).tolist(),
    }


def run_static_racket_drop(sim: PingPongSim, *, drop_height: float, initial_velocity_z: float, max_substeps: int) -> dict[str, object]:
    # 고정된 라켓 위로 공을 떨어뜨려 XML contact 파라미터의 유효 반발계수를 추정한다.
    # LINK: pingpong_rl2/src/pingpong_rl2/envs/pingpong_sim.py:13
    sim.reset()
    sim.reset_ball_above_racket(
        height=drop_height,
        xy_offset=(0.0, 0.0),
        velocity=(0.0, 0.0, initial_velocity_z),
    )

    previous_ball_velocity = sim.ball_velocity
    previous_racket_velocity = sim.racket_velocity
    contact_started = False
    was_contact_active = False
    contact_normal: np.ndarray | None = None
    pre_contact_relative_normal_speed: float | None = None
    contact_substep: int | None = None

    for substep in range(1, max_substeps + 1):
        sim.step(joint_targets=sim.home_joint_targets, n_substeps=1)
        matching_contact = sim._matching_contact(("ball_geom", "racket_head"))
        contact_active = matching_contact is not None
        if matching_contact is not None and not contact_started:
            contact_started = True
            contact_substep = substep
            contact_normal = np.asarray(matching_contact["contact_normal_racket_to_ball"], dtype=float)
            normal_norm = float(np.linalg.norm(contact_normal))
            if normal_norm > 1.0e-9:
                contact_normal = contact_normal / normal_norm
            previous_relative_velocity = previous_ball_velocity - previous_racket_velocity
            pre_contact_relative_normal_speed = float(np.dot(previous_relative_velocity, contact_normal))
        if contact_started and was_contact_active and not contact_active and contact_normal is not None:
            post_relative_velocity = sim.ball_velocity - sim.racket_velocity
            post_contact_relative_normal_speed = float(np.dot(post_relative_velocity, contact_normal))
            effective_restitution = (
                post_contact_relative_normal_speed / max(abs(pre_contact_relative_normal_speed or 0.0), 1.0e-6)
            )
            return {
                "contact_observed": True,
                "contact_substep": contact_substep,
                "contact_end_substep": substep,
                "pre_contact_relative_normal_speed": pre_contact_relative_normal_speed,
                "post_contact_relative_normal_speed": post_contact_relative_normal_speed,
                "effective_normal_restitution": float(effective_restitution),
                "final_ball_velocity": sim.ball_velocity.tolist(),
            }
        previous_ball_velocity = sim.ball_velocity
        previous_racket_velocity = sim.racket_velocity
        was_contact_active = contact_active

    return {
        "contact_observed": contact_started,
        "contact_substep": contact_substep,
        "contact_end_substep": None,
        "pre_contact_relative_normal_speed": pre_contact_relative_normal_speed,
        "post_contact_relative_normal_speed": None,
        "effective_normal_restitution": None,
        "final_ball_velocity": sim.ball_velocity.tolist(),
    }


def main() -> None:
    # scene XML의 geom/material 값과 drop test 결과를 JSON으로 출력한다.
    args = parse_args()
    scene_path = resolve_input_path(args.scene_path)
    sim = PingPongSim(scene_path=scene_path)
    rebound_trials = [
        run_static_racket_drop(
            sim,
            drop_height=args.drop_height,
            initial_velocity_z=args.initial_velocity_z,
            max_substeps=args.max_substeps,
        )
        for _ in range(args.episodes)
    ]
    restitution_values = [
        float(trial["effective_normal_restitution"])
        for trial in rebound_trials
        if trial["effective_normal_restitution"] is not None
    ]
    summary = {
        "scene_path": str(scene_path.resolve()),
        "gravity": np.asarray(sim.model.opt.gravity, dtype=float).tolist(),
        "timestep": float(sim.model.opt.timestep),
        "control_dt": float(sim.control_dt),
        "n_substeps_per_control_step": int(sim.n_substeps),
        "ball_geom": geom_summary(sim, "ball_geom"),
        "racket_head_geom": geom_summary(sim, "racket_head"),
        "drop_test": {
            "episodes": args.episodes,
            "drop_height": args.drop_height,
            "initial_velocity_z": args.initial_velocity_z,
            "mean_effective_normal_restitution": (
                float(np.mean(restitution_values)) if restitution_values else None
            ),
            "min_effective_normal_restitution": (
                float(np.min(restitution_values)) if restitution_values else None
            ),
            "max_effective_normal_restitution": (
                float(np.max(restitution_values)) if restitution_values else None
            ),
            "trials": rebound_trials,
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

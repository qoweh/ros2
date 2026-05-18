from __future__ import annotations

import argparse
import math
import subprocess
import sys
import time

import mujoco.viewer
import numpy as np

from pingpong_rl.controllers import JointPositionController, RacketCartesianController
from pingpong_rl.defaults import DEFAULT_BALL_HEIGHT, DEFAULT_CONTROL_DT, DEFAULT_VIEWER_RESET_HEIGHT
from pingpong_rl.envs import PingPongSim
from pingpong_rl.utils.paths import SCENE_XML_PATH


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the MuJoCo ping-pong scene viewer.")
    parser.add_argument(
        "--mode",
        choices=("interactive", "passive"), 
        default="passive",
        help="Default to the passive scripted loop so the project home pose/reset is applied; use interactive for the raw stock MuJoCo viewer.",
    )
    parser.add_argument(
        "--demo-controller",
        choices=("hold", "joint", "ee"),
        default="hold",
        help="Passive mode only: hold the home pose, run the existing joint demo, or run the new EE demo.",
    )
    parser.add_argument(
        "--ball-height",
        type=float,
        default=DEFAULT_BALL_HEIGHT,
        help="Ball spawn height above the racket in meters.",
    )
    parser.add_argument(
        "--reset-height",
        type=float,
        default=DEFAULT_VIEWER_RESET_HEIGHT,
        help="Passive mode only: reset the ball when it falls below this z height.",
    )
    parser.add_argument("--demo-joint", type=int, choices=range(1, 8), help="Passive mode only: 1-based Franka joint index to move with a sine wave.")
    parser.add_argument("--demo-amplitude", type=float, default=0.2, help="Passive mode only: sine-wave amplitude in radians.")
    parser.add_argument("--demo-frequency", type=float, default=0.25, help="Passive mode only: sine-wave frequency in Hz.")
    parser.add_argument(
        "--ee-axis",
        choices=("x", "y", "z"),
        default="z",
        help="Passive mode only: Cartesian axis for the EE sine-wave demo.",
    )
    parser.add_argument("--control-dt", type=float, default=DEFAULT_CONTROL_DT, help="Controller period in seconds.")
    return parser.parse_args(argv)


def _make_sim(control_dt: float, ball_height: float) -> PingPongSim:
    sim = PingPongSim(control_dt=control_dt)
    sim.reset(ball_height=ball_height)
    return sim


def _run_interactive(args: argparse.Namespace) -> None:
    _ = args
    command = [sys.executable, "-m", "mujoco.viewer", f"--mjcf={SCENE_XML_PATH}"]
    raise SystemExit(subprocess.run(command, check=False).returncode)


def _ee_demo_target_position(
    anchor_position: np.ndarray,
    axis: str,
    amplitude: float,
    frequency: float,
    time_seconds: float,
) -> np.ndarray:
    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    offset = np.zeros(3, dtype=float)
    offset[axis_index] = amplitude * math.sin(2.0 * math.pi * frequency * time_seconds)
    return anchor_position + offset


def _passive_viewer_is_running(viewer: mujoco.viewer.Handle) -> bool:
    simulate = viewer._get_sim()
    return bool(getattr(simulate, "run", 1))


def _run_passive(args: argparse.Namespace) -> None:
    sim = _make_sim(args.control_dt, args.ball_height)
    joint_controller = JointPositionController(sim.home_joint_targets)
    ee_controller = RacketCartesianController(sim)
    ee_anchor_position = sim.racket_position.copy()

    with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
        while viewer.is_running():
            viewer.sync()
            if not _passive_viewer_is_running(viewer):
                time.sleep(sim.model.opt.timestep * sim.n_substeps)
                continue

            joint_targets = sim.home_joint_targets
            if args.demo_controller == "joint":
                joint_controller.reset()
                if args.demo_joint is not None:
                    joint_index = args.demo_joint - 1
                    offset = args.demo_amplitude * math.sin(2.0 * math.pi * args.demo_frequency * sim.data.time)
                    joint_controller.add_joint_offset(joint_index, offset)
                joint_targets = joint_controller.targets
            elif args.demo_controller == "ee":
                ee_target = _ee_demo_target_position(
                    ee_anchor_position,
                    args.ee_axis,
                    args.demo_amplitude,
                    args.demo_frequency,
                    sim.data.time,
                )
                ee_controller.set_target_position(ee_target)
                joint_targets = ee_controller.compute_joint_targets()

            sim.step(joint_targets=joint_targets)

            if sim.ball_position[2] < args.reset_height:
                sim.reset(ball_height=args.ball_height)
                joint_controller.reset()
                ee_controller.reset()
                ee_anchor_position = sim.racket_position.copy()

            viewer.sync()
            time.sleep(sim.model.opt.timestep * sim.n_substeps)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.mode == "interactive":
        _run_interactive(args)
        return
    _run_passive(args)


if __name__ == "__main__":
    main()
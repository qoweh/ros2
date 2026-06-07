from __future__ import annotations

from pathlib import Path

from pingpong_rl2.utils import PPO_RUNS_ROOT, resolve_input_path, resolve_output_path

def build_run_dir(run_name: str, output_dir: Path | None) -> Path:
    if output_dir is None:
        run_dir = PPO_RUNS_ROOT / run_name
    else:
        run_dir = resolve_output_path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def default_model_path(run_dir: Path, run_name: str) -> Path:
    return run_dir / f"{run_name}_model.zip"


def resolve_starting_model(args: argparse.Namespace, run_dir: Path, resolved_run_name: str) -> tuple[str, Path | None]:
    if args.reset_model and args.resume_from is not None:
        raise ValueError("--reset-model and --resume-from cannot be used together.")

    if args.reset_model:
        return "new", None

    if args.resume_from is not None:
        resume_path = resolve_input_path(args.resume_from)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume model not found: {resume_path}")
        return "resume", resume_path

    existing_model_path = default_model_path(run_dir, resolved_run_name)
    if existing_model_path.is_file():
        return "resume", existing_model_path
    return "new", None


def build_session_monitor_path(run_dir: Path) -> Path:
    session_index = 1
    while True:
        candidate = run_dir / f"monitor_{session_index:03d}.monitor.csv"
        if not candidate.exists():
            return candidate
        session_index += 1



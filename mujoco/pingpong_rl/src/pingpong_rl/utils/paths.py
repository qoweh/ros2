from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = PACKAGE_ROOT
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DOCS_ROOT = PACKAGE_ROOT / "docs"
ARTIFACT_ROOT = DOCS_ROOT / "etc"
PPO_RUNS_ROOT = ARTIFACT_ROOT / "ppo_runs"
ROLLOUT_ANALYSIS_ROOT = ARTIFACT_ROOT / "rollout_analysis"
ASSET_ROOT = PACKAGE_ROOT / "assets"
SCENE_XML_PATH = ASSET_ROOT / "scene.xml"


def resolve_input_path(path: Path) -> Path:
	if path.is_absolute():
		return path.resolve()

	cwd_path = (Path.cwd() / path).resolve()
	if cwd_path.exists():
		return cwd_path

	return (PACKAGE_ROOT / path).resolve()


def resolve_output_path(path: Path) -> Path:
	if path.is_absolute():
		return path.resolve()
	return (PACKAGE_ROOT / path).resolve()
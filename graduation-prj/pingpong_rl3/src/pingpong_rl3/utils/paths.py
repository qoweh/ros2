from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ASSETS_ROOT = PROJECT_ROOT / "assets"
CONFIGS_ROOT = PROJECT_ROOT / "configs"
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
SCENE_XML_PATH = ASSETS_ROOT / "scene.xml"


def resolve_input_path(path: Path | str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    rooted_candidate = PROJECT_ROOT / candidate
    if rooted_candidate.exists():
        return rooted_candidate
    return candidate

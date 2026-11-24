"""Loading of the YAML files in configs/."""

from pathlib import Path

import yaml

# Repository root: src/ecg_mi/config.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict:
    """Read a YAML file. Relative paths are resolved against the repository root."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)

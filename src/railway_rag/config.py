from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load YAML config and normalize relative paths against repo root."""
    path = Path(config_path).resolve()
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    base_dir = path.parent.parent.resolve()
    paths = config.setdefault("paths", {})

    for key, value in list(paths.items()):
        if isinstance(value, str):
            value_path = Path(value)
            paths[key] = str((base_dir / value_path).resolve() if not value_path.is_absolute() else value_path)

    for section in ("regulation", "glossary"):
        document = config.get("documents", {}).get(section, {})
        file_value = document.get("file")
        if file_value:
            file_path = Path(file_value)
            document["file"] = str((base_dir / file_path).resolve() if not file_path.is_absolute() else file_path)

    config["base_dir"] = str(base_dir)
    return config


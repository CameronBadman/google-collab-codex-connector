from __future__ import annotations

import json
import tomllib
from pathlib import Path

from colab_runner import __version__


def test_release_metadata_stays_in_sync() -> None:
    root = Path(__file__).resolve().parents[1]
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    plugin = json.loads(
        (root / ".codex-plugin" / "plugin.json").read_text(encoding="utf-8")
    )

    assert __version__ == pyproject["project"]["version"] == plugin["version"]
    assert plugin["name"] == "colab-runner"
    assert "google-colab-cli==0.6.0" in pyproject["project"]["dependencies"]
    assert "nbformat>=5.10,<6" in pyproject["project"]["dependencies"]

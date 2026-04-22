"""
riskam.provenance

Reproducibility metadata stamp for experiment artefacts.

Populated into each ``results.json`` so a result can be traced back to the
code + environment that produced it. All lookups are best-effort and return
``None`` on failure rather than raising — a missing git binary or a package
not declared in ``importlib.metadata`` should not abort an experiment.
"""

from __future__ import annotations

import datetime
import socket
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent


# Packages whose versions affect numerical reproducibility.
_TRACKED_PACKAGES: tuple[str, ...] = (
    "torch",
    "ultralytics",
    "numpy",
    "opencv-python",
)


def _git(args: list[str]) -> str | None:
    try:
        res = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return res.stdout.strip()


def _pkg_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def reproducibility_metadata() -> dict:
    """Return a snapshot of the code + environment state.

    Keys:
      git_sha          commit of HEAD, or None if not in a repo / no git
      git_dirty        True if the working tree has uncommitted changes
      hostname         machine hostname
      timestamp_utc    ISO-8601 UTC
      python_version   e.g. "3.12.3"
      packages         dict of tracked-package name → version | None
    """
    sha = _git(["rev-parse", "HEAD"])
    porcelain = _git(["status", "--porcelain"])
    return {
        "git_sha": sha,
        "git_dirty": (bool(porcelain) if porcelain is not None else None),
        "hostname": socket.gethostname(),
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "python_version": sys.version.split()[0],
        "packages": {name: _pkg_version(name) for name in _TRACKED_PACKAGES},
    }

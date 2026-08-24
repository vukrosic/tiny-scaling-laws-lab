#!/usr/bin/env python3
"""Install pinned dependencies, selecting the CPU-only PyTorch wheel."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def installed(distribution: str, version: str) -> bool:
    try:
        actual = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return False
    return actual.split("+")[0] == version


def install(label: str, *arguments: str) -> None:
    print(f"[setup] Installing {label}. This is required only on the first run.", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *arguments],
        check=True,
    )


def main() -> int:
    print("[setup] Checking dependencies...", flush=True)
    if not (installed("numpy", "2.3.2") and installed("Pillow", "11.3.0")):
        install("NumPy and Pillow", "-r", str(ROOT / "requirements.txt"))
    if not installed("torch", "2.13.0"):
        arguments = ["torch==2.13.0"]
        if sys.platform.startswith(("linux", "win")):
            arguments.extend(["--index-url", "https://download.pytorch.org/whl/cpu"])
        install("CPU PyTorch", *arguments)
    print("[setup] Dependencies ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

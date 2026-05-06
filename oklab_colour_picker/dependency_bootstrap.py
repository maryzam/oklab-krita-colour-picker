"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys


NUMPY_REQUIREMENT = "numpy>=1.26,<3"
ENSUREPIP_TIMEOUT_SECONDS = 120
PIP_INSTALL_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class InstallResult:
    success: bool
    message: str


def install_numpy(vendor_path: str, *, requirement: str = NUMPY_REQUIREMENT) -> InstallResult:
    Path(vendor_path).mkdir(parents=True, exist_ok=True)

    python = find_krita_python()
    if python is None:
        return InstallResult(
            False,
            "Could not locate Krita's Python interpreter. "
            "Install NumPy manually using Krita's bundled python (see README).",
        )

    try:
        if not _ensure_pip_available(python):
            return InstallResult(
                False,
                f"pip is unavailable in {python} and `ensurepip` did not bootstrap it.",
            )

        completed = subprocess.run(
            [
                python,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--only-binary=:all:",
                "--target",
                vendor_path,
                requirement,
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_INSTALL_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "pip install timed out. Check your network connection and retry.")
    except OSError as exc:
        return InstallResult(False, f"Could not run {python}: {exc}")

    if completed.returncode == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, _format_process_failure(completed))


def find_krita_python() -> str | None:
    """Locate a Python executable that matches Krita's runtime.

    On Linux Krita usually runs under system Python, so ``sys.executable`` is
    already python. On Windows ``sys.executable`` is ``krita.exe`` and the
    bundled interpreter sits next to it. On macOS the bundle ships
    ``krita_python`` alongside ``krita`` inside ``Contents/MacOS``.
    """
    executable = sys.executable
    if executable and _looks_like_python(Path(executable).name):
        return executable

    if not executable:
        return None

    here = Path(executable).parent
    candidates = [
        here / "python.exe",
        here / "python3.exe",
        here / "python3",
        here / "python",
        here / "krita_python",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _ensure_pip_available(python: str) -> bool:
    if _python_can_import(python, "pip"):
        return True

    try:
        subprocess.run(
            [python, "-m", "ensurepip", "--upgrade"],
            check=False,
            capture_output=True,
            text=True,
            timeout=ENSUREPIP_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False

    return _python_can_import(python, "pip")


def _python_can_import(python: str, module: str) -> bool:
    try:
        completed = subprocess.run(
            [python, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=ENSUREPIP_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


def _looks_like_python(executable_name: str) -> bool:
    name = executable_name.lower()
    return name.startswith("python") or name == "krita_python"


def _format_process_failure(completed: subprocess.CompletedProcess) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if output:
        return output
    return f"pip exited with status {completed.returncode}."

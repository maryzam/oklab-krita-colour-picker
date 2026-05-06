"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import runpy
import subprocess
import sys
from urllib import request


NUMPY_REQUIREMENT = "numpy>=1.26,<3"
PYPI_PIP_JSON_URL = "https://pypi.org/pypi/pip/json"


@dataclass(frozen=True)
class InstallResult:
    success: bool
    message: str


def install_numpy(vendor_path: str, *, requirement: str = NUMPY_REQUIREMENT) -> InstallResult:
    Path(vendor_path).mkdir(parents=True, exist_ok=True)

    pip_ready = _pip_is_available()
    if not pip_ready:
        pip_ready = _bootstrap_pip_with_ensurepip()
    if not pip_ready:
        pip_ready = _bootstrap_pip_with_wheel(Path(vendor_path).parent)
    if not pip_ready:
        return InstallResult(False, "Could not bootstrap pip in Krita's Python environment.")

    pip_args = ["install", "--only-binary=:all:", "--target", vendor_path, requirement]
    try:
        if _can_run_python_module_subprocess(sys.executable):
            completed = subprocess.run(
                [sys.executable, "-m", "pip", *pip_args],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0:
                return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
            return InstallResult(False, _format_process_failure(completed))

        exit_code = _run_pip_in_process(["pip", *pip_args])
        if exit_code == 0:
            return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
        return InstallResult(False, f"pip exited with status {exit_code}.")
    except Exception as exc:
        return InstallResult(False, f"NumPy installation failed: {exc}")


def _pip_is_available() -> bool:
    importlib.invalidate_caches()
    return importlib.util.find_spec("pip") is not None


def _bootstrap_pip_with_ensurepip() -> bool:
    if _can_run_python_module_subprocess(sys.executable):
        completed = subprocess.run(
            [sys.executable, "-m", "ensurepip", "--upgrade"],
            check=False,
            capture_output=True,
            text=True,
        )
        return completed.returncode == 0 and _pip_is_available()

    try:
        import ensurepip

        ensurepip.bootstrap(upgrade=True)
    except Exception:
        return False
    return _pip_is_available()


def _bootstrap_pip_with_wheel(cache_dir: Path) -> bool:
    try:
        wheel = _find_or_download_pip_wheel(cache_dir)
    except Exception:
        return False

    sys.path.append(str(wheel))
    importlib.invalidate_caches()
    return _pip_is_available()


def _find_or_download_pip_wheel(cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for wheel in sorted(cache_dir.glob("pip-*.whl")):
        return wheel

    with request.urlopen(PYPI_PIP_JSON_URL) as response:
        metadata = json.loads(response.read())

    for artifact in metadata["urls"]:
        if artifact.get("packagetype") != "bdist_wheel":
            continue
        pip_url = artifact["url"]
        wheel_path = cache_dir / Path(pip_url).name
        with request.urlopen(pip_url) as response:
            wheel_path.write_bytes(response.read())
        return wheel_path

    raise RuntimeError("No pip wheel found in PyPI metadata.")


def _run_pip_in_process(argv: list[str]) -> int:
    original_argv = sys.argv
    original_exit = sys.exit

    def _exit(code=0):
        raise SystemExit(code)

    try:
        sys.argv = argv
        sys.exit = _exit
        try:
            runpy.run_module("pip", run_name="__main__")
        except SystemExit as exc:
            if exc.code is None:
                return 0
            if isinstance(exc.code, int):
                return exc.code
            return 1
        return 0
    finally:
        sys.argv = original_argv
        sys.exit = original_exit


def _can_run_python_module_subprocess(executable: str) -> bool:
    if not executable:
        return False
    executable_name = Path(executable).name.lower()
    return executable_name.startswith("python") or executable_name == "py.exe"


def _format_process_failure(completed: subprocess.CompletedProcess) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if output:
        return output
    return f"pip exited with status {completed.returncode}."

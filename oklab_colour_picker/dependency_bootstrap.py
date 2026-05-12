"""Opt-in runtime dependency installation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from urllib import error, parse, request


NUMPY_REQUIREMENT = "numpy>=1.26,<3"
PIP_IMPORT_TIMEOUT_SECONDS = 120
PIP_INSTALL_TIMEOUT_SECONDS = 600
PIP_METADATA_URL = "https://pypi.org/pypi/pip/json"
PIP_DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class InstallResult:
    success: bool
    message: str


@dataclass(frozen=True)
class PipSubprocessBootstrap:
    available: bool
    env: dict[str, str] | None = None


@dataclass(frozen=True)
class PipWheelInfo:
    url: str
    sha256: str


class PipBootstrapError(Exception):
    """Raised when pip cannot be bootstrapped from a downloaded wheel."""


def install_numpy(vendor_path: str, *, requirement: str = NUMPY_REQUIREMENT) -> InstallResult:
    vendor = Path(vendor_path)
    vendor.mkdir(parents=True, exist_ok=True)

    python = find_krita_python()
    if python is not None:
        subprocess_result = _install_via_subprocess(python, vendor, requirement)
        if subprocess_result is not None:
            return subprocess_result
        # The interpreter or pip bootstrap is unusable; let the in-process
        # path try. We only fall through on infrastructure failures, never on
        # genuine pip install failures (e.g. "no matching wheel"), since
        # retrying those in-process would just repeat the same error.
    return _install_in_process(vendor, requirement)


def find_krita_python() -> str | None:
    """Locate a Python executable that matches Krita's runtime.

    On Linux Krita usually runs under system Python, so ``sys.executable`` is
    already python. On Windows ``sys.executable`` is ``krita.exe`` and the
    bundled interpreter sits next to it. On macOS the bundle ships
    ``krita_python`` alongside ``krita`` inside ``Contents/MacOS``. Some Krita
    builds ship without a Python interpreter at all; in that case this returns
    ``None`` and the caller must fall back to in-process pip.
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


def _install_via_subprocess(python: str, vendor_path: Path, requirement: str) -> InstallResult | None:
    """Run pip via the discovered interpreter.

    Returns ``None`` when the interpreter or pip bootstrap cannot be exercised
    (interpreter not runnable, pip bootstrap fails) so the caller can fall back
    to in-process pip. Returns an ``InstallResult`` for any actual pip-install
    outcome — success, timeout, or a normal pip failure such as
    "no matching wheel" — since retrying those in-process would recur.
    """
    try:
        pip_bootstrap = _pip_bootstrap_for_subprocess(python, vendor_path)
    except PipBootstrapError:
        return None

    if not pip_bootstrap.available:
        return None

    kwargs = {
        "check": False,
        "capture_output": True,
        "text": True,
        "timeout": PIP_INSTALL_TIMEOUT_SECONDS,
    }
    if pip_bootstrap.env is not None:
        kwargs["env"] = pip_bootstrap.env

    try:
        completed = subprocess.run(
            [python, "-m", "pip", *_pip_install_args(str(vendor_path), requirement)],
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(False, "pip install timed out. Check your network connection and retry.")
    except OSError:
        return None

    if completed.returncode == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, _format_process_failure(completed))


def _install_in_process(vendor_path: Path, requirement: str) -> InstallResult:
    """Run pip inside Krita's own interpreter when no python executable is reachable.

    Used on Krita builds that bundle a Python runtime without exposing a
    standalone python.exe. We mutate sys.argv/sys.exit while pip's CLI runs and
    restore them afterwards.
    """
    try:
        pip_available = _ensure_pip_available_in_process(vendor_path)
    except PipBootstrapError as exc:
        return InstallResult(False, f"pip is unavailable and could not be downloaded from PyPI: {exc}")

    if not pip_available:
        return InstallResult(
            False,
            "pip is unavailable in Krita's bundled Python and could not be bootstrapped from PyPI.",
        )

    try:
        exit_code = _run_pip_in_process(["pip", *_pip_install_args(str(vendor_path), requirement)])
    except Exception as exc:
        return InstallResult(False, f"NumPy installation failed: {exc}")

    if exit_code == 0:
        return InstallResult(True, "NumPy installed. Restart Krita to load the colour selector.")
    return InstallResult(False, f"pip exited with status {exit_code}.")


def _pip_install_args(vendor_path: str, requirement: str) -> list[str]:
    return [
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--target",
        vendor_path,
        requirement,
    ]


def _pip_bootstrap_for_subprocess(python: str, vendor_path: Path) -> PipSubprocessBootstrap:
    if _python_can_import(python, "pip"):
        return PipSubprocessBootstrap(available=True)

    pip_wheel = _ensure_pip_wheel(vendor_path)
    env = os.environ.copy()
    env["PYTHONPATH"] = _prepend_path(str(pip_wheel), env.get("PYTHONPATH"))
    if _python_can_import(python, "pip", env=env):
        return PipSubprocessBootstrap(available=True, env=env)
    return PipSubprocessBootstrap(available=False)


def _ensure_pip_available_in_process(vendor_path: Path) -> bool:
    if _can_import("pip"):
        return True

    pip_wheel = _ensure_pip_wheel(vendor_path)
    pip_wheel_path = str(pip_wheel)
    if pip_wheel_path not in sys.path:
        sys.path.append(pip_wheel_path)

    return _can_import("pip")


def _ensure_pip_wheel(vendor_path: Path) -> Path:
    cache_dir = vendor_path.parent / "pip-bootstrap"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        with request.urlopen(PIP_METADATA_URL, timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            metadata = json.loads(response.read().decode("utf-8"))
        wheel_info = _pip_wheel_info(metadata)
        wheel_path = cache_dir / _filename_from_url(wheel_info.url)
        if wheel_path.exists() and _sha256(wheel_path) == wheel_info.sha256:
            return wheel_path
        with request.urlopen(wheel_info.url, timeout=PIP_DOWNLOAD_TIMEOUT_SECONDS) as response:
            wheel_bytes = response.read()
        digest = hashlib.sha256(wheel_bytes).hexdigest()
        if digest != wheel_info.sha256:
            raise PipBootstrapError("downloaded pip wheel failed SHA-256 verification")
        temp_path = wheel_path.with_name(wheel_path.name + ".tmp")
        temp_path.write_bytes(wheel_bytes)
        os.replace(temp_path, wheel_path)
    except (OSError, error.URLError, json.JSONDecodeError, KeyError, StopIteration) as exc:
        raise PipBootstrapError(str(exc)) from exc

    return wheel_path


def _pip_wheel_info(metadata: dict) -> PipWheelInfo:
    try:
        file_info = next(url for url in metadata["urls"] if url.get("packagetype") == "bdist_wheel")
        return PipWheelInfo(url=file_info["url"], sha256=file_info["digests"]["sha256"])
    except (KeyError, StopIteration) as exc:
        raise PipBootstrapError("PyPI pip metadata did not include a wheel URL and SHA-256 digest") from exc


def _filename_from_url(url: str) -> str:
    return Path(parse.urlparse(url).path).name


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepend_path(path: str, existing: str | None) -> str:
    if not existing:
        return path
    return path + os.pathsep + existing


def _can_import(module: str) -> bool:
    import importlib.util

    importlib.invalidate_caches()
    return importlib.util.find_spec(module) is not None


def _python_can_import(python: str, module: str, *, env: dict[str, str] | None = None) -> bool:
    try:
        completed = subprocess.run(
            [python, "-c", f"import {module}"],
            check=False,
            capture_output=True,
            text=True,
            timeout=PIP_IMPORT_TIMEOUT_SECONDS,
            env=env,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return completed.returncode == 0


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


def _looks_like_python(executable_name: str) -> bool:
    name = executable_name.lower()
    return name.startswith("python") or name == "krita_python"


def _format_process_failure(completed: subprocess.CompletedProcess) -> str:
    output = (completed.stderr or completed.stdout or "").strip()
    if output:
        return output
    return f"pip exited with status {completed.returncode}."

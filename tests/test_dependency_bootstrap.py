import hashlib
import os
import subprocess
from pathlib import Path

import pytest

from oklab_colour_picker import dependency_bootstrap


KRITA_PYTHON = "/fake/krita/bin/python.exe"
PIP_WHEEL_BYTES = b"wheel-content"
PIP_WHEEL_SHA256 = hashlib.sha256(PIP_WHEEL_BYTES).hexdigest()
PIP_WHEEL_URL = "https://example.test/pip-25.0-py3-none-any.whl"


def _completed(args, returncode=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(args, returncode, stdout=stdout, stderr=stderr)


def _pip_bootstrap(env=None, available=True):
    return dependency_bootstrap.PipSubprocessBootstrap(available=available, env=env)


def _pip_metadata(sha256=PIP_WHEEL_SHA256, url=PIP_WHEEL_URL):
    return (
        b'{"urls": ['
        b'{"packagetype": "sdist", "url": "https://example.test/pip.tar.gz"}, '
        + (
            '{"packagetype": "bdist_wheel", "url": "%s", "digests": {"sha256": "%s"}}'
            % (url, sha256)
        ).encode("utf-8")
        + b"]}"
    )


class FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return self.body


def test_install_numpy_invokes_krita_python_with_target_and_upgrade(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed(args, returncode=0, stdout="ok")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_pip_bootstrap_for_subprocess", lambda _python, _vendor: _pip_bootstrap())
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert calls == [
        (
            [
                KRITA_PYTHON,
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--only-binary=:all:",
                "--target",
                str(tmp_path),
                dependency_bootstrap.NUMPY_REQUIREMENT,
            ],
            {
                "check": False,
                "capture_output": True,
                "text": True,
                "timeout": dependency_bootstrap.PIP_INSTALL_TIMEOUT_SECONDS,
            },
        )
    ]


def test_install_numpy_falls_back_to_in_process_when_no_python_executable(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda _vendor: True)

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_in_process",
        lambda argv: captured.append(argv) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured == [
        [
            "pip",
            "install",
            "--upgrade",
            "--only-binary=:all:",
            "--target",
            str(tmp_path),
            dependency_bootstrap.NUMPY_REQUIREMENT,
        ]
    ]


def test_install_numpy_in_process_reports_pip_exit_code(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda _vendor: True)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_in_process", lambda _argv: 2)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "status 2" in result.message


def test_install_numpy_in_process_reports_missing_pip(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda _vendor: False)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "PyPI" in result.message


def test_install_numpy_in_process_reports_pip_download_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: None)

    def fail_bootstrap(_vendor):
        raise dependency_bootstrap.PipBootstrapError("network unavailable")

    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", fail_bootstrap)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "network unavailable" in result.message


def test_install_numpy_surfaces_pip_stderr_on_failure(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        return _completed(args, returncode=1, stderr="ERROR: no matching wheel")

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_pip_bootstrap_for_subprocess", lambda _python, _vendor: _pip_bootstrap())
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "no matching wheel" in result.message


def test_install_numpy_reports_timeout(tmp_path, monkeypatch):
    def fake_run(args, **kwargs):
        raise subprocess.TimeoutExpired(args, kwargs.get("timeout", 0))

    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_pip_bootstrap_for_subprocess", lambda _python, _vendor: _pip_bootstrap())
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "timed out" in result.message.lower()


def test_install_numpy_falls_through_when_subprocess_pip_bootstrap_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(
        dependency_bootstrap,
        "_pip_bootstrap_for_subprocess",
        lambda _python, _vendor: _pip_bootstrap(available=False),
    )
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_available_in_process", lambda _vendor: True)

    captured = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_run_pip_in_process",
        lambda argv: captured.append(argv) or 0,
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert captured and captured[0][0] == "pip"


def test_install_numpy_does_not_retry_in_process_after_pip_install_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_pip_bootstrap_for_subprocess", lambda _python, _vendor: _pip_bootstrap())
    monkeypatch.setattr(
        dependency_bootstrap.subprocess,
        "run",
        lambda args, **kwargs: _completed(args, returncode=1, stderr="ERROR: no matching wheel"),
    )

    in_process_called = []
    monkeypatch.setattr(
        dependency_bootstrap,
        "_install_in_process",
        lambda *args, **kwargs: in_process_called.append(args) or dependency_bootstrap.InstallResult(True, "unexpected"),
    )

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is False
    assert "no matching wheel" in result.message
    assert in_process_called == []


def test_pip_bootstrap_for_subprocess_short_circuits_when_pip_imports(monkeypatch, tmp_path):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed(args, returncode=0)

    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    bootstrap = dependency_bootstrap._pip_bootstrap_for_subprocess(KRITA_PYTHON, tmp_path)

    assert bootstrap == dependency_bootstrap.PipSubprocessBootstrap(available=True)
    assert len(calls) == 1
    assert calls[0][0][1:] == ["-c", "import pip"]


def test_pip_bootstrap_for_subprocess_adds_downloaded_wheel_to_pythonpath(monkeypatch, tmp_path):
    calls = []
    pip_wheel = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        if (kwargs.get("env") or {}).get("PYTHONPATH") == str(pip_wheel):
            return _completed(args, returncode=0)
        return _completed(args, returncode=1)

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_wheel", lambda _vendor: pip_wheel)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    bootstrap = dependency_bootstrap._pip_bootstrap_for_subprocess(KRITA_PYTHON, tmp_path)

    assert bootstrap.available is True
    assert bootstrap.env is not None
    assert bootstrap.env["PYTHONPATH"] == str(pip_wheel)
    assert [call[0] for call in calls] == [[KRITA_PYTHON, "-c", "import pip"]] * 2
    assert calls[0][1]["env"] is None
    assert calls[1][1]["env"] == bootstrap.env


def test_pip_bootstrap_for_subprocess_preserves_existing_pythonpath(monkeypatch, tmp_path):
    pip_wheel = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"
    monkeypatch.setenv("PYTHONPATH", "existing")
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_wheel", lambda _vendor: pip_wheel)
    monkeypatch.setattr(
        dependency_bootstrap,
        "_python_can_import",
        lambda _python, _module, env=None: env is not None,
    )

    bootstrap = dependency_bootstrap._pip_bootstrap_for_subprocess(KRITA_PYTHON, tmp_path)

    assert bootstrap.available is True
    assert bootstrap.env is not None
    assert bootstrap.env["PYTHONPATH"] == str(pip_wheel) + os.pathsep + "existing"


def test_ensure_pip_available_in_process_adds_downloaded_wheel_to_sys_path(monkeypatch, tmp_path):
    calls = []
    pip_wheel = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"

    def fake_can_import(module):
        calls.append(module)
        return len(calls) > 1

    monkeypatch.setattr(dependency_bootstrap, "_can_import", fake_can_import)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_wheel", lambda _vendor: pip_wheel)
    monkeypatch.setattr(dependency_bootstrap.sys, "path", [])

    assert dependency_bootstrap._ensure_pip_available_in_process(tmp_path) is True
    assert dependency_bootstrap.sys.path == [str(pip_wheel)]


def test_ensure_pip_wheel_downloads_pip_wheel(monkeypatch, tmp_path):
    opened = []

    def fake_urlopen(url, **kwargs):
        opened.append((url, kwargs))
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(_pip_metadata())
        if url == PIP_WHEEL_URL:
            return FakeResponse(PIP_WHEEL_BYTES)
        raise AssertionError(url)

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    wheel = dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages")

    assert wheel == tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"
    assert wheel.read_bytes() == PIP_WHEEL_BYTES
    assert not wheel.with_name(wheel.name + ".tmp").exists()
    assert opened == [
        (
            dependency_bootstrap.PIP_METADATA_URL,
            {"timeout": dependency_bootstrap.PIP_DOWNLOAD_TIMEOUT_SECONDS},
        ),
        (
            PIP_WHEEL_URL,
            {"timeout": dependency_bootstrap.PIP_DOWNLOAD_TIMEOUT_SECONDS},
        ),
    ]


def test_ensure_pip_wheel_reuses_cached_wheel(monkeypatch, tmp_path):
    cached = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"
    cached.parent.mkdir()
    cached.write_bytes(PIP_WHEEL_BYTES)
    opened = []

    def fake_urlopen(url, **kwargs):
        opened.append((url, kwargs))
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(_pip_metadata())
        raise AssertionError("unexpected wheel download")

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    assert dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages") == cached
    assert opened == [
        (
            dependency_bootstrap.PIP_METADATA_URL,
            {"timeout": dependency_bootstrap.PIP_DOWNLOAD_TIMEOUT_SECONDS},
        )
    ]


def test_ensure_pip_wheel_replaces_invalid_cached_wheel(monkeypatch, tmp_path):
    cached = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"
    cached.parent.mkdir()
    cached.write_bytes(b"truncated")
    opened = []

    def fake_urlopen(url, **kwargs):
        opened.append((url, kwargs))
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(_pip_metadata())
        if url == PIP_WHEEL_URL:
            return FakeResponse(PIP_WHEEL_BYTES)
        raise AssertionError(url)

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    assert dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages") == cached
    assert cached.read_bytes() == PIP_WHEEL_BYTES
    assert [call[0] for call in opened] == [dependency_bootstrap.PIP_METADATA_URL, PIP_WHEEL_URL]


def test_ensure_pip_wheel_rejects_download_with_wrong_hash(monkeypatch, tmp_path):
    def fake_urlopen(url, **_kwargs):
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(_pip_metadata(sha256="0" * 64))
        if url == PIP_WHEEL_URL:
            return FakeResponse(PIP_WHEEL_BYTES)
        raise AssertionError(url)

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="SHA-256"):
        dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages")

    assert not (tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl").exists()


def test_ensure_pip_wheel_keeps_invalid_cache_when_replacement_download_fails(monkeypatch, tmp_path):
    cached = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"
    cached.parent.mkdir()
    cached.write_bytes(b"truncated")

    def fake_urlopen(url, **_kwargs):
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(_pip_metadata())
        if url == PIP_WHEEL_URL:
            raise OSError("download failed")
        raise AssertionError(url)

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="download failed"):
        dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages")

    assert cached.read_bytes() == b"truncated"


def test_ensure_pip_wheel_reports_missing_digest_cleanly(monkeypatch, tmp_path):
    metadata = (
        b'{"urls": ['
        b'{"packagetype": "bdist_wheel", "url": "https://example.test/pip-25.0-py3-none-any.whl"}'
        b"]}"
    )

    def fake_urlopen(url, **_kwargs):
        if url == dependency_bootstrap.PIP_METADATA_URL:
            return FakeResponse(metadata)
        raise AssertionError(url)

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fake_urlopen)

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="metadata did not include"):
        dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages")


def test_ensure_pip_wheel_reports_download_failure(monkeypatch, tmp_path):
    def fail_urlopen(*_args, **_kwargs):
        raise OSError("network unavailable")

    monkeypatch.setattr(dependency_bootstrap.request, "urlopen", fail_urlopen)

    with pytest.raises(dependency_bootstrap.PipBootstrapError, match="network unavailable"):
        dependency_bootstrap._ensure_pip_wheel(tmp_path / "site-packages")


def test_install_numpy_subprocess_uses_downloaded_pip_wheel_when_pip_missing(tmp_path, monkeypatch):
    calls = []
    pip_wheel = tmp_path / "pip-bootstrap" / "pip-25.0-py3-none-any.whl"

    def fake_python_can_import(_python, _module, env=None):
        return env is not None and env.get("PYTHONPATH") == str(pip_wheel)

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return _completed(args, returncode=0)

    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.setattr(dependency_bootstrap, "find_krita_python", lambda: KRITA_PYTHON)
    monkeypatch.setattr(dependency_bootstrap, "_ensure_pip_wheel", lambda _vendor: pip_wheel)
    monkeypatch.setattr(dependency_bootstrap, "_python_can_import", fake_python_can_import)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == [
        KRITA_PYTHON,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--only-binary=:all:",
        "--target",
        str(tmp_path),
        dependency_bootstrap.NUMPY_REQUIREMENT,
    ]
    assert kwargs["timeout"] == dependency_bootstrap.PIP_INSTALL_TIMEOUT_SECONDS
    assert kwargs["env"]["PYTHONPATH"] == str(pip_wheel)


def test_find_krita_python_uses_sys_executable_when_already_python(monkeypatch):
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "/usr/bin/python3.10")

    assert dependency_bootstrap.find_krita_python() == "/usr/bin/python3.10"


def test_find_krita_python_locates_sibling_python_exe(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")
    python = bin_dir / "python.exe"
    python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() == str(python)


def test_find_krita_python_locates_macos_krita_python(tmp_path, monkeypatch):
    macos_dir = tmp_path / "Krita.app" / "Contents" / "MacOS"
    macos_dir.mkdir(parents=True)
    krita = macos_dir / "krita"
    krita.write_text("")
    krita_python = macos_dir / "krita_python"
    krita_python.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() == str(krita_python)


def test_find_krita_python_returns_none_when_no_candidate(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    krita = bin_dir / "krita.exe"
    krita.write_text("")

    monkeypatch.setattr(dependency_bootstrap.sys, "executable", str(krita))

    assert dependency_bootstrap.find_krita_python() is None

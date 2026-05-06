import subprocess
import sys

from oklab_colour_picker import dependency_bootstrap


def test_install_numpy_uses_pip_subprocess_with_vendor_target(tmp_path, monkeypatch):
    calls = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(dependency_bootstrap, "_pip_is_available", lambda: True)
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", sys.executable)
    monkeypatch.setattr(dependency_bootstrap.subprocess, "run", fake_run)

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert calls == [
        (
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--only-binary=:all:",
                "--target",
                str(tmp_path),
                dependency_bootstrap.NUMPY_REQUIREMENT,
            ],
            {"check": False, "capture_output": True, "text": True},
        )
    ]


def test_install_numpy_tries_ensurepip_before_wheel_bootstrap(tmp_path, monkeypatch):
    steps = []

    monkeypatch.setattr(dependency_bootstrap, "_pip_is_available", lambda: False)
    monkeypatch.setattr(dependency_bootstrap, "_bootstrap_pip_with_ensurepip", lambda: steps.append("ensurepip") or True)
    monkeypatch.setattr(dependency_bootstrap, "_bootstrap_pip_with_wheel", lambda _cache_dir: steps.append("wheel") or True)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_in_process", lambda _argv: 0)
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "krita.exe")

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert steps == ["ensurepip"]


def test_install_numpy_falls_back_to_pip_wheel_when_ensurepip_fails(tmp_path, monkeypatch):
    steps = []

    monkeypatch.setattr(dependency_bootstrap, "_pip_is_available", lambda: False)
    monkeypatch.setattr(dependency_bootstrap, "_bootstrap_pip_with_ensurepip", lambda: steps.append("ensurepip") or False)
    monkeypatch.setattr(dependency_bootstrap, "_bootstrap_pip_with_wheel", lambda _cache_dir: steps.append("wheel") or True)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_in_process", lambda _argv: 0)
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "krita.exe")

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert steps == ["ensurepip", "wheel"]


def test_krita_executable_uses_in_process_pip(tmp_path, monkeypatch):
    pip_args = []

    monkeypatch.setattr(dependency_bootstrap, "_pip_is_available", lambda: True)
    monkeypatch.setattr(dependency_bootstrap, "_run_pip_in_process", lambda argv: pip_args.append(argv) or 0)
    monkeypatch.setattr(dependency_bootstrap.sys, "executable", "krita.exe")

    result = dependency_bootstrap.install_numpy(str(tmp_path))

    assert result.success is True
    assert pip_args == [
        [
            "pip",
            "install",
            "--only-binary=:all:",
            "--target",
            str(tmp_path),
            dependency_bootstrap.NUMPY_REQUIREMENT,
        ]
    ]

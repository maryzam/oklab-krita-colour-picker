#!/usr/bin/env python3
"""Deterministic development-loop checks for the rewrite.

The checks intentionally avoid third-party dependencies so they can run from
Git hooks before the project has a full Python package/test setup.
"""

from __future__ import annotations

import argparse
import ast
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PREFIX = "legacy-plugin/"
PY_SOURCE_DIRS = ("lab_colour_picker", "tests", "scripts")
KRITA_ALLOWED = {
    Path("lab_colour_picker/plugin.py"),
    Path("lab_colour_picker/controller.py"),
    Path("lab_colour_picker/krita_adapter.py"),
}
PURE_NO_QT = {
    Path("lab_colour_picker/color_math.py"),
    Path("lab_colour_picker/selector_models.py"),
}
SET_FOREGROUND_ALLOWED = {
    Path("lab_colour_picker/controller.py"),
    Path("lab_colour_picker/krita_adapter.py"),
}


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def rel(path: Path) -> Path:
    return path.resolve().relative_to(ROOT)


def tracked_files() -> list[Path]:
    return [ROOT / line for line in run_git(["ls-files"]).splitlines() if line]


def staged_files() -> list[Path]:
    lines = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).splitlines()
    return [ROOT / line for line in lines if line]


def candidate_files(scope: str) -> list[Path]:
    files = staged_files() if scope == "staged" else tracked_files()
    return [path for path in files if path.exists()]


def python_files(scope: str) -> list[Path]:
    files = []
    for path in candidate_files(scope):
        rp = rel(path)
        if path.suffix == ".py" and rp.parts and rp.parts[0] in PY_SOURCE_DIRS:
            files.append(path)
    return files


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def check_no_legacy(scope: str) -> int:
    bad = [rel(path).as_posix() for path in candidate_files(scope) if rel(path).as_posix().startswith(LEGACY_PREFIX)]
    if not bad:
        return 0
    fail("legacy-plugin files must remain untracked:")
    for path in bad:
        print(f"  {path}", file=sys.stderr)
    return 1


def check_python_compile(scope: str) -> int:
    errors = 0
    for path in python_files(scope):
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            fail(f"Python syntax check failed for {rel(path)}")
            print(exc.msg, file=sys.stderr)
            errors += 1
    return errors


def check_import_rules(scope: str) -> int:
    errors = 0
    for path in python_files(scope):
        rp = rel(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rp))
        except SyntaxError as exc:
            fail(f"Cannot parse {rp}: {exc}")
            errors += 1
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
                if any(module == "krita" or module.startswith("krita.") for module in modules) and rp not in KRITA_ALLOWED:
                    fail(f"{rp}: Krita imports are only allowed in plugin/controller adapter files")
                    errors += 1
                if any(module.startswith(("legacy_plugin", "legacy-plugin")) for module in modules):
                    fail(f"{rp}: imports from legacy plugin are forbidden")
                    errors += 1
                if rp in PURE_NO_QT and any(module.startswith(("PyQt5", "krita")) for module in modules):
                    fail(f"{rp}: pure model/math modules must not import Qt or Krita")
                    errors += 1

            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if (module == "krita" or module.startswith("krita.")) and rp not in KRITA_ALLOWED:
                    fail(f"{rp}: Krita imports are only allowed in plugin/controller adapter files")
                    errors += 1
                if module.startswith(("legacy_plugin", "legacy-plugin")):
                    fail(f"{rp}: imports from legacy plugin are forbidden")
                    errors += 1
                if rp in PURE_NO_QT and module.startswith(("PyQt5", "krita")):
                    fail(f"{rp}: pure model/math modules must not import Qt or Krita")
                    errors += 1

    return errors


def check_forbidden_calls(scope: str) -> int:
    errors = 0
    for path in python_files(scope):
        rp = rel(path)
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(rp))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "pixelColor":
                fail(f"{rp}: selection must not read colours from QImage.pixelColor")
                errors += 1
            if isinstance(func, ast.Attribute) and func.attr == "setForeGroundColor" and rp not in SET_FOREGROUND_ALLOWED:
                fail(f"{rp}: setForeGroundColor is only allowed behind the controller/Krita adapter boundary")
                errors += 1
    return errors


def check_formatting(scope: str) -> int:
    errors = 0
    for path in candidate_files(scope):
        if not path.is_file():
            continue
        data = path.read_bytes()
        rp = rel(path)
        if b"\r\n" in data:
            fail(f"{rp}: CRLF line endings are not allowed")
            errors += 1
        if data and not data.endswith(b"\n"):
            fail(f"{rp}: file must end with a newline")
            errors += 1
        for line_no, line in enumerate(data.splitlines(), start=1):
            if line.rstrip(b" \t") != line:
                fail(f"{rp}:{line_no}: trailing whitespace is not allowed")
                errors += 1
    return errors


def run_pytest() -> int:
    tests_dir = ROOT / "tests"
    if not tests_dir.exists():
        print("No tests/ directory yet; skipping pytest.")
        return 0
    return subprocess.call([sys.executable, "-m", "pytest"], cwd=ROOT)


def install_hooks() -> int:
    subprocess.check_call(["git", "config", "core.hooksPath", ".githooks"], cwd=ROOT)
    print("Configured Git to use .githooks for this repository.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=("tracked", "staged"), default="tracked")
    parser.add_argument("--pytest", action="store_true", help="Run pytest after static checks.")
    parser.add_argument("--install-hooks", action="store_true", help="Set core.hooksPath=.githooks.")
    args = parser.parse_args()

    if args.install_hooks:
        return install_hooks()

    os.chdir(ROOT)
    errors = 0
    errors += check_no_legacy(args.scope)
    errors += check_formatting(args.scope)
    errors += check_python_compile(args.scope)
    errors += check_import_rules(args.scope)
    errors += check_forbidden_calls(args.scope)
    if args.pytest:
        errors += run_pytest()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

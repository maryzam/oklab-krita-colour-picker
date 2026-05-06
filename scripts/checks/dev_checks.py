#!/usr/bin/env python3
"""Deterministic development-loop checks for the rewrite.

The checks intentionally avoid third-party dependencies so they can run from
Git hooks before the project has a full Python package/test setup.
"""

from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_PREFIX = "legacy-plugin/"
KRITA_ALLOWED = {
    Path("oklab_colour_picker/plugin.py"),
    Path("oklab_colour_picker/controller.py"),
    Path("oklab_colour_picker/krita_adapter.py"),
}
PURE_NO_QT = {
    Path("oklab_colour_picker/color_math.py"),
    Path("oklab_colour_picker/renderers.py"),
    Path("oklab_colour_picker/selector_models.py"),
}
SET_FOREGROUND_ALLOWED = {
    Path("oklab_colour_picker/controller.py"),
    Path("oklab_colour_picker/krita_adapter.py"),
}


@dataclass(frozen=True)
class SourceFile:
    """A file snapshot from either the working tree or the staged index."""

    path: Path
    data: bytes

    @property
    def suffix(self) -> str:
        return self.path.suffix

    @property
    def posix(self) -> str:
        return self.path.as_posix()

    @property
    def is_legacy(self) -> bool:
        return self.posix.startswith(LEGACY_PREFIX)

    @property
    def is_test(self) -> bool:
        return bool(self.path.parts) and self.path.parts[0] == "tests"

    @property
    def is_binary(self) -> bool:
        return b"\0" in self.data


def run_git(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def git_blob(path: Path) -> bytes:
    return subprocess.check_output(["git", "show", f":{path.as_posix()}"], cwd=ROOT)


def tracked_paths() -> list[Path]:
    return [Path(line) for line in run_git(["ls-files"]).splitlines() if line]


def staged_paths() -> list[Path]:
    lines = run_git(["diff", "--cached", "--name-only", "--diff-filter=ACMR"]).splitlines()
    return [Path(line) for line in lines if line]


def source_files(scope: str) -> list[SourceFile]:
    sources = []
    paths = staged_paths() if scope == "staged" else tracked_paths()
    for path in paths:
        if scope == "staged":
            data = git_blob(path)
        else:
            full_path = ROOT / path
            if not full_path.is_file():
                continue
            data = full_path.read_bytes()
        sources.append(SourceFile(path=path, data=data))
    return sources


def python_sources(sources: list[SourceFile]) -> list[SourceFile]:
    return [source for source in sources if source.suffix == ".py" and not source.is_legacy]


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)


def check_no_legacy(sources: list[SourceFile]) -> int:
    bad = [source.posix for source in sources if source.is_legacy]
    if not bad:
        return 0
    fail("legacy-plugin files must remain untracked:")
    for path in bad:
        print(f"  {path}", file=sys.stderr)
    return 1


def check_python_rules(sources: list[SourceFile]) -> int:
    errors = 0
    for source in python_sources(sources):
        rp = source.path
        try:
            tree = ast.parse(source.data, filename=source.posix)
            compile(tree, source.posix, "exec")
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
                if any(module.startswith("legacy_plugin") for module in modules):
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
                if module.startswith("legacy_plugin"):
                    fail(f"{rp}: imports from legacy plugin are forbidden")
                    errors += 1
                if rp in PURE_NO_QT and module.startswith(("PyQt5", "krita")):
                    fail(f"{rp}: pure model/math modules must not import Qt or Krita")
                    errors += 1

            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # Name-based AST guardrail by design: it catches direct calls in
            # production code, but it is not a type-aware semantic analysis.
            if isinstance(func, ast.Attribute) and func.attr == "pixelColor":
                fail(f"{rp}: selection must not read colours from QImage.pixelColor")
                errors += 1
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "setForeGroundColor"
                and rp not in SET_FOREGROUND_ALLOWED
                and not source.is_test
            ):
                fail(f"{rp}: setForeGroundColor is only allowed behind the controller/Krita adapter boundary")
                errors += 1
    return errors


def check_formatting(sources: list[SourceFile]) -> int:
    errors = 0
    for source in sources:
        if source.is_binary:
            continue
        data = source.data
        rp = source.path
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
    sources = source_files(args.scope)
    errors = 0
    errors += check_no_legacy(sources)
    errors += check_formatting(sources)
    errors += check_python_rules(sources)
    if args.pytest:
        errors += run_pytest()
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

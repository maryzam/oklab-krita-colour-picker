import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
KRITA_IMPORT_ALLOWED = {
    Path("oklab_colour_picker/plugin.py"),
    Path("oklab_colour_picker/controller.py"),
    Path("oklab_colour_picker/krita_adapter.py"),
}
PURE_NO_QT_OR_KRITA = {
    Path("oklab_colour_picker/color_math.py"),
    Path("oklab_colour_picker/renderers.py"),
    Path("oklab_colour_picker/selector_models.py"),
}


def test_krita_imports_are_limited_to_boundary_files():
    offenders = []
    for path, tree in _project_python_asts():
        for module in _imported_modules(tree):
            if _is_krita_module(module) and path not in KRITA_IMPORT_ALLOWED:
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_widgets_do_not_import_krita():
    offenders = []
    widgets_dir = ROOT / "oklab_colour_picker" / "widgets"
    assert widgets_dir.exists()

    for full_path in sorted(widgets_dir.rglob("*.py")):
        path = full_path.relative_to(ROOT)
        tree = ast.parse(full_path.read_text(), filename=path.as_posix())
        for module in _imported_modules(tree):
            if _is_krita_module(module):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_pure_color_math_has_no_qt_or_krita_imports():
    offenders = []
    for path in sorted(PURE_NO_QT_OR_KRITA):
        full_path = ROOT / path
        if not full_path.exists():
            continue
        tree = ast.parse(full_path.read_text(), filename=path.as_posix())
        for module in _imported_modules(tree):
            if module.startswith(("PyQt5", "PySide", "krita")):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def _project_python_asts():
    for full_path in sorted((ROOT / "oklab_colour_picker").rglob("*.py")):
        path = full_path.relative_to(ROOT)
        yield path, ast.parse(full_path.read_text(), filename=path.as_posix())


def _imported_modules(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def _is_krita_module(module):
    return module == "krita" or module.startswith("krita.")

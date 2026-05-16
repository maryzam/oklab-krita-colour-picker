import ast
import importlib.util
import sys
from pathlib import Path

import pytest


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
SET_FOREGROUND_ALLOWED = {
    Path("oklab_colour_picker/controller.py"),
    Path("oklab_colour_picker/krita_adapter.py"),
}
LOWER_LAYER_FILES = {
    Path("oklab_colour_picker/color_math.py"),
    Path("oklab_colour_picker/renderers.py"),
    Path("oklab_colour_picker/selector_models.py"),
    Path("oklab_colour_picker/controller.py"),
}
LOWER_LAYER_TESTS = {
    "oklab_colour_picker.color_math": Path("tests/test_color_math.py"),
    "oklab_colour_picker.renderers": Path("tests/test_renderers.py"),
    "oklab_colour_picker.selector_models": Path("tests/test_selector_models.py"),
    "oklab_colour_picker.controller": Path("tests/test_controller.py"),
}
UI_LAYER_MODULE_PREFIXES = (
    "oklab_colour_picker.dock",
    "oklab_colour_picker.plugin",
    "oklab_colour_picker.widgets",
)


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


def test_krita_foreground_writes_stay_behind_controller_boundary():
    offenders = []
    for path, tree in _project_python_asts():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "setForeGroundColor"
                and path not in SET_FOREGROUND_ALLOWED
            ):
                offenders.append(path.as_posix())

    assert offenders == []


def test_selection_does_not_read_from_qimage_pixels():
    """Strict production tripwire for selector-by-rendered-pixel regressions."""

    offenders = []
    for path, tree in _project_python_asts():
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "pixelColor"
            ):
                offenders.append(path.as_posix())

    assert offenders == []


def test_lower_layers_do_not_import_widget_dock_or_plugin_layers():
    offenders = []
    for path, tree in _project_python_asts():
        if path not in LOWER_LAYER_FILES:
            continue
        for module in _project_import_references(tree, path):
            if _starts_with_any(module, UI_LAYER_MODULE_PREFIXES):
                offenders.append(f"{path}: {module}")

    assert offenders == []


def test_lower_layer_guard_constants_match_dev_check_runner():
    dev_checks = _load_dev_checks_module()

    assert LOWER_LAYER_FILES == dev_checks.LOWER_LAYER_FILES
    assert UI_LAYER_MODULE_PREFIXES == dev_checks.UI_LAYER_MODULE_PREFIXES


def test_relative_import_references_are_resolved_before_lower_layer_guard():
    tree = ast.parse("from .widgets import selector\nfrom . import dock\n")
    import_from_nodes = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    dev_checks = _load_dev_checks_module()

    imports = set(_project_import_references(tree, Path("oklab_colour_picker/color_math.py")))
    runner_imports = {
        imported
        for node in import_from_nodes
        for imported in dev_checks.project_import_references(node, Path("oklab_colour_picker/color_math.py"))
    }

    assert "oklab_colour_picker.widgets" in imports
    assert "oklab_colour_picker.widgets.selector" in imports
    assert "oklab_colour_picker.dock" in imports
    assert imports == runner_imports


def test_lower_layer_coverage_modules_exist_and_target_the_claimed_layer():
    missing = []
    untargeted = []
    for module, test_path in LOWER_LAYER_TESTS.items():
        full_path = ROOT / test_path
        if not full_path.exists():
            missing.append(test_path.as_posix())
            continue

        tree = ast.parse(full_path.read_text(), filename=test_path.as_posix())
        imports = set(_project_import_references(tree, test_path))
        if module not in imports:
            untargeted.append(f"{test_path}: {module}")

    assert missing == []
    assert untargeted == []


@pytest.mark.xfail(strict=True, reason="PR-1 / §2.3: selector widget still probes optional model helpers")
def test_selector_widget_uses_explicit_model_contract():
    path = ROOT / "oklab_colour_picker" / "widgets" / "selector.py"
    tree = ast.parse(path.read_text(), filename=path.relative_to(ROOT).as_posix())
    probes = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and node.args
        and _is_self_model_reference(node.args[0])
    ]

    assert probes == []


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


def _project_import_references(tree, source_path):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = ".".join(_relative_import_base(source_path, node.level))
                resolved_module = ".".join(part for part in (base, module) if part)
                if module:
                    yield resolved_module
                    for alias in node.names:
                        yield f"{resolved_module}.{alias.name}"
                    continue
                for alias in node.names:
                    yield f"{base}.{alias.name}"
                continue
            if module != "oklab_colour_picker":
                yield module
                continue
            for alias in node.names:
                yield f"{module}.{alias.name}"


def _is_krita_module(module):
    return module == "krita" or module.startswith("krita.")


def _starts_with_any(module, prefixes):
    return any(module == prefix or module.startswith(f"{prefix}.") for prefix in prefixes)


def _relative_import_base(source_path, level):
    module_parts = source_path.with_suffix("").parts
    if module_parts[-1] == "__init__":
        package_parts = module_parts[:-1]
    else:
        package_parts = module_parts[:-1]
    keep = max(0, len(package_parts) - level + 1)
    return package_parts[:keep]


def _is_self_model_reference(node):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "_model"
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _load_dev_checks_module():
    path = ROOT / "scripts" / "checks" / "dev_checks.py"
    spec = importlib.util.spec_from_file_location("dev_checks_for_import_discipline", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

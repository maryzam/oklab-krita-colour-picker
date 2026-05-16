import math
import sys

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

import oklab_colour_picker
from oklab_colour_picker import color_math
from oklab_colour_picker.dock import ColourPickerDockPanel, SelectorMode, connect_dock_visibility
from oklab_colour_picker.plugin import DOCK_FACTORY_ID, DOCK_TITLE, create_dock_widget_class, register_plugin
from oklab_colour_picker.widgets import HueLightnessSliceDiskWidget
import oklab_colour_picker.dock as dock_module
import oklab_colour_picker.plugin as plugin_module


def test_dock_panel_constructs_all_selector_views_and_switches_modes(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert [widget.objectName() for widget in panel.selector_widgets] == [
        "lightness-slice-selector",
        "hue-lightness-slice-selector",
        "lightness-chroma-slice-selector",
    ]
    assert panel.mode == SelectorMode.LIGHTNESS_SLICE

    assert isinstance(
        panel.selector_for_mode(SelectorMode.HUE_LIGHTNESS_SLICE),
        HueLightnessSliceDiskWidget,
    )

    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)

    assert panel.mode == SelectorMode.LIGHTNESS_CHROMA_SLICE
    assert panel.active_selector is panel.selector_for_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)


def test_dock_panel_initializes_only_active_selector_view(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert tuple(panel._selectors) == (SelectorMode.LIGHTNESS_SLICE,)
    assert panel._tabs.count() == len(SelectorMode)
    assert panel._tabs.widget(0) is panel.selector_for_mode(SelectorMode.LIGHTNESS_SLICE)
    assert panel._tabs.widget(1).objectName() == "hue-lightness-slice-selector-placeholder"


def test_dock_panel_initializes_only_active_selector_model(qtbot, monkeypatch):
    controller = FakeController()
    original = dock_module._model_for_colour
    model_calls = []

    def counted_model_for_colour(mode, colour):
        model_calls.append(mode)
        return original(mode, colour)

    monkeypatch.setattr(dock_module, "_model_for_colour", counted_model_for_colour)

    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert model_calls == [SelectorMode.LIGHTNESS_SLICE]


def test_dock_panel_uses_current_foreground_on_construction(qtbot):
    colour = color_math.oklch_to_oklab([0.58, 0.07, math.pi / 3.0])
    controller = FakeController(selected_colour=colour)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour)


def test_dock_panel_construction_does_not_synchronously_resync_foreground(qtbot):
    colour = color_math.oklch_to_oklab([0.58, 0.07, math.pi / 3.0])
    controller = FakeController(selected_colour=colour)

    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert controller.sync_count == 0


def test_selector_signals_update_controller_and_sibling_indicators(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)

    colour = active.model.color_at_position((40, 20), (active.width(), active.height()))
    assert colour is not None
    active.previewed.emit(colour.copy())
    active.committed.emit(colour.copy())

    np.testing.assert_allclose(controller.previews[-1], colour)
    np.testing.assert_allclose(controller.commits[-1], colour)
    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)


def test_click_on_achromatic_hue_lightness_slice_keeps_indicator_at_click(qtbot):
    # The dock loops set_selected_colour back to the source widget after
    # every previewed/committed signal. On a chroma=0 hue/lightness disk the
    # picked OKLab is greyscale, so model.position_for_color cannot recover
    # the click angle; the indicator must still report the click point
    # rather than snapping to the model's hue=0 fallback.
    grey = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    controller = FakeController(selected_colour=grey)
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS_SLICE)
    active = panel.active_selector
    active.resize(121, 121)

    click = QtCore.QPoint(60, 20)
    expected_colour = active.model.color_at_position(
        (click.x(), click.y()), (active.width(), active.height())
    )
    assert expected_colour is not None

    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress, click, QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton, QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease, click, QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton, QtCore.Qt.NoModifier,
    )
    QtCore.QCoreApplication.sendEvent(active, press)
    QtCore.QCoreApplication.sendEvent(active, release)

    indicator = active.indicator_position()
    assert indicator is not None
    assert indicator == pytest.approx((float(click.x()), float(click.y())))


def test_preview_reuses_equal_selector_models(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    active = panel.active_selector
    active.resize(120, 80)

    colour = active.model.color_at_position((40, 20), (active.width(), active.height()))
    assert colour is not None
    panel.set_selected_colour(colour)
    models = {mode: panel.selector_for_mode(mode).model for mode in SelectorMode}

    active.previewed.emit(colour.copy())

    for mode, model in models.items():
        assert panel.selector_for_mode(mode).model is model


def test_external_foreground_sync_updates_all_selector_views(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.resize(120, 80)
    colour = color_math.oklch_to_oklab([0.42, 0.06, math.pi / 4.0])

    controller.emit_foreground(colour)

    for widget in panel.selector_widgets:
        np.testing.assert_allclose(widget.selected_colour, colour)
        assert widget.indicator_position() is not None


def test_lazy_selector_uses_latest_colour_when_first_built(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    colour = color_math.oklch_to_oklab([0.42, 0.06, math.pi / 4.0])

    panel.set_selected_colour(colour)
    assert SelectorMode.LIGHTNESS_CHROMA_SLICE not in panel._selectors

    widget = panel.selector_for_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)

    np.testing.assert_allclose(widget.selected_colour, colour)
    assert widget.indicator_position() is not None


def test_indicator_maps_to_same_colour_after_resize(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.LIGHTNESS_CHROMA_SLICE)
    widget = panel.active_selector
    widget.resize(80, 40)
    colour = widget.model.color_at_position((30, 12), (80, 40))
    assert colour is not None
    widget.set_selected_colour(colour)

    widget.resize(160, 90)
    actual = widget.indicator_position()
    expected = widget.model.position_for_color(colour, (160, 90))

    assert actual is not None
    assert expected is not None
    assert actual == pytest.approx(expected, abs=1.0)


def test_selector_preview_then_commit_preserves_previous_swatch(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    colour_a = color_math.oklch_to_oklab([0.50, 0.05, math.pi / 6.0])
    colour_b = color_math.oklch_to_oklab([0.60, 0.08, math.pi / 3.0])
    panel.set_selected_colour(colour_a, committed=True)
    panel._readout_panel.set_previous_colour(colour_a)

    selector = panel.active_selector
    selector.previewed.emit(np.asarray(colour_b, dtype=float).copy())
    selector.committed.emit(np.asarray(colour_b, dtype=float).copy())

    np.testing.assert_allclose(panel._readout_panel._previous_oklab, colour_a, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour_b, atol=1e-6)


def test_readout_commit_signal_preserves_previous_swatch(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    colour_a = color_math.oklch_to_oklab([0.50, 0.05, math.pi / 6.0])
    colour_b = color_math.oklch_to_oklab([0.60, 0.08, math.pi / 3.0])
    panel.set_selected_colour(colour_a, committed=True)
    panel._readout_panel.set_previous_colour(colour_a)

    panel._readout_panel.committed.emit(np.asarray(colour_b, dtype=float).copy())

    np.testing.assert_allclose(panel._readout_panel._previous_oklab, colour_a, atol=1e-6)
    np.testing.assert_allclose(panel._readout_panel._current_oklab, colour_b, atol=1e-6)


def test_qdock_visibility_signal_reaches_controller(qtbot):
    controller = FakeController()
    dock = QtWidgets.QDockWidget()
    qtbot.addWidget(dock)

    connect_dock_visibility(dock, controller)
    dock.visibilityChanged.emit(False)
    dock.visibilityChanged.emit(True)

    assert controller.visibility == [False, True]


def test_visibility_connection_can_be_disconnected(qtbot):
    controller = FakeController()
    dock = QtWidgets.QDockWidget()
    qtbot.addWidget(dock)

    connection = connect_dock_visibility(dock, controller)
    dock.visibilityChanged.emit(False)
    connection.disconnect()
    dock.visibilityChanged.emit(True)

    assert controller.visibility == [False]


def test_plugin_registers_krita_dock_factory():
    app = FakeKritaApp()
    api = FakeKritaApi(app)

    assert register_plugin(krita_instance=app, api=api) is True

    assert len(app.factories) == 1
    factory = app.factories[0]
    assert factory.identifier == DOCK_FACTORY_ID
    assert factory.area == FakeDockWidgetFactoryBase.DockRight


def test_vendor_site_packages_are_added_before_runtime_imports(tmp_path, monkeypatch):
    vendor_dir = tmp_path / plugin_module.VENDOR_ROOT_DIRECTORY_NAME / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    vendor_dir.mkdir(parents=True)
    monkeypatch.setattr(sys, "path", list(sys.path))

    plugin_module._add_vendor_site_packages(str(tmp_path))

    assert sys.path[0] == str(vendor_dir)


def test_vendor_site_packages_fall_back_next_to_plugin_package(tmp_path, monkeypatch):
    package_dir = tmp_path / "pykrita" / "oklab_colour_picker"
    package_dir.mkdir(parents=True)
    expected = package_dir / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME
    monkeypatch.setattr(plugin_module, "__file__", str(package_dir / "plugin.py"))

    assert plugin_module._vendor_site_packages_path() == str(expected)


def test_created_krita_dock_builds_panel_and_wires_visibility(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(FakeDockWidget, controller_factory=lambda: controller)
    dock = dock_class()
    qtbot.addWidget(dock)

    assert dock.windowTitle() == DOCK_TITLE
    assert isinstance(dock.widget(), ColourPickerDockPanel)

    dock.visibilityChanged.emit(False)
    assert controller.visibility == [False]


def test_created_krita_dock_syncs_foreground_on_canvas_change(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(FakeDockWidget, controller_factory=lambda: controller)
    dock = dock_class()
    qtbot.addWidget(dock)
    sync_count = controller.sync_count

    dock.canvasChanged(object())

    assert controller.sync_count == sync_count + 1


def test_dock_shows_friendly_message_when_numpy_is_missing(qtbot, monkeypatch):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.dock", fake_dock)

    dock_class = create_dock_widget_class(FakeDockWidget)
    dock = dock_class()
    qtbot.addWidget(dock)

    widget = dock.widget()
    assert isinstance(widget, QtWidgets.QWidget)
    assert widget.objectName() == "oklab-missing-dependency"
    assert "numpy" in widget.findChild(QtWidgets.QLabel).text().lower()
    assert widget.findChild(QtWidgets.QPushButton, "oklab-install-numpy").text() == "Install NumPy"


def test_install_numpy_action_requires_confirmation(qtbot, monkeypatch, tmp_path):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.dock", fake_dock)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.No)
    installer_calls = []

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=lambda vendor_path: installer_calls.append(vendor_path),
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    assert installer_calls == []


def test_install_numpy_action_runs_installer_when_confirmed(qtbot, monkeypatch, tmp_path):
    import types

    from oklab_colour_picker.dependency_bootstrap import InstallResult

    fake_dock = types.ModuleType("oklab_colour_picker.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.dock", fake_dock)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *args, **kwargs: QtWidgets.QMessageBox.Ok)
    installer_calls = []

    def fake_installer(vendor_path):
        installer_calls.append(vendor_path)
        return InstallResult(True, "NumPy installed.")

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=fake_installer,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    status = dock.widget().findChild(QtWidgets.QLabel, "oklab-install-status")
    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: bool(installer_calls) and "installed" in status.text().lower(), timeout=5000)
    expected_vendor = str(tmp_path / plugin_module.VENDOR_ROOT_DIRECTORY_NAME / plugin_module.VENDOR_SITE_PACKAGES_DIRECTORY_NAME)
    assert installer_calls == [expected_vendor]
    assert button.isEnabled()


def test_install_numpy_action_reports_installer_exception(qtbot, monkeypatch, tmp_path):
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.dock")

    def _raise_numpy_missing(_name):
        raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

    fake_dock.__getattr__ = _raise_numpy_missing
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.dock", fake_dock)
    monkeypatch.setattr(QtWidgets.QMessageBox, "question", lambda *args, **kwargs: QtWidgets.QMessageBox.Yes)
    captured = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda parent, title, message, *args, **kwargs: captured.append(message) or QtWidgets.QMessageBox.Ok,
    )

    def boom(_vendor_path):
        raise RuntimeError("network is down")

    dock_class = create_dock_widget_class(
        FakeDockWidget,
        app_data_location=str(tmp_path),
        dependency_installer=boom,
    )
    dock = dock_class()
    qtbot.addWidget(dock)

    button = dock.widget().findChild(QtWidgets.QPushButton, "oklab-install-numpy")
    qtbot.mouseClick(button, QtCore.Qt.LeftButton)

    qtbot.waitUntil(lambda: bool(captured), timeout=5000)
    assert "network is down" in captured[0]
    assert button.isEnabled()


def test_dock_propagates_unexpected_import_errors(qtbot, monkeypatch):
    import sys
    import types

    fake_dock = types.ModuleType("oklab_colour_picker.dock")

    def _raise_unknown(_name):
        raise ModuleNotFoundError("No module named 'something_else'", name="something_else")

    fake_dock.__getattr__ = _raise_unknown
    monkeypatch.setitem(sys.modules, "oklab_colour_picker.dock", fake_dock)

    dock_class = create_dock_widget_class(FakeDockWidget)
    with pytest.raises(ModuleNotFoundError):
        dock_class()


def test_package_exports_register_plugin():
    assert oklab_colour_picker.__all__ == ["register_plugin"]
    assert oklab_colour_picker.register_plugin is register_plugin


class FakeController:
    def __init__(self, selected_colour=None):
        self.previews = []
        self.commits = []
        self.visibility = []
        self._foreground_listeners = []
        self._selected_colour = None if selected_colour is None else np.asarray(selected_colour, dtype=float).copy()
        self.sync_count = 0

    @property
    def selected_colour(self):
        return None if self._selected_colour is None else self._selected_colour.copy()

    def set_preview_colour(self, colour):
        self.previews.append(None if colour is None else np.asarray(colour, dtype=float).copy())

    def request_foreground_commit(self, colour):
        self.commits.append(None if colour is None else np.asarray(colour, dtype=float).copy())

    def set_dock_visible(self, visible):
        self.visibility.append(bool(visible))

    def sync_external_foreground(self):
        self.sync_count += 1
        return False

    def add_foreground_listener(self, listener):
        self._foreground_listeners.append(listener)

    def remove_foreground_listener(self, listener):
        self._foreground_listeners.remove(listener)

    def emit_foreground(self, colour):
        for listener in list(self._foreground_listeners):
            listener(np.asarray(colour, dtype=float).copy())


class FakeKritaApp:
    def __init__(self):
        self.factories = []

    def addDockWidgetFactory(self, factory):
        self.factories.append(factory)

    def getAppDataLocation(self):
        return "/tmp/fake-krita-app-data"


class FakeDockWidgetFactoryBase:
    DockRight = object()


class FakeDockWidgetFactory:
    def __init__(self, identifier, area, widget_class):
        self.identifier = identifier
        self.area = area
        self.widget_class = widget_class


class FakeKritaApi:
    def __init__(self, app):
        self.Krita = FakeKrita(app)
        self.DockWidget = FakeDockWidget
        self.DockWidgetFactory = FakeDockWidgetFactory
        self.DockWidgetFactoryBase = FakeDockWidgetFactoryBase


class FakeKrita:
    def __init__(self, app):
        self._app = app

    def instance(self):
        return self._app


class FakeDockWidget(QtWidgets.QDockWidget):
    def canvasChanged(self, canvas):
        pass

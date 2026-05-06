import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtWidgets

import lab_colour_picker
from lab_colour_picker import color_math
from lab_colour_picker.dock import ColourPickerDockPanel, SelectorMode, connect_dock_visibility
from lab_colour_picker.plugin import DOCK_FACTORY_ID, DOCK_TITLE, create_dock_widget_class, register_plugin


def test_dock_panel_constructs_all_selector_views_and_switches_modes(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)

    assert [widget.objectName() for widget in panel.selector_widgets] == [
        "lightness-slice-selector",
        "hue-lightness-selector",
        "chroma-lightness-selector",
    ]
    assert panel.mode == SelectorMode.LIGHTNESS_SLICE

    panel.set_mode(SelectorMode.HUE_LIGHTNESS)

    assert panel.mode == SelectorMode.HUE_LIGHTNESS
    assert panel.active_selector is panel.selector_for_mode(SelectorMode.HUE_LIGHTNESS)


def test_selector_signals_update_controller_and_sibling_indicators(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS)
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


def test_preview_reuses_equal_selector_models(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS)
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


def test_indicator_maps_to_same_colour_after_resize(qtbot):
    controller = FakeController()
    panel = ColourPickerDockPanel(controller)
    qtbot.addWidget(panel)
    panel.set_mode(SelectorMode.HUE_LIGHTNESS)
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


def test_created_krita_dock_builds_panel_and_wires_visibility(qtbot):
    controller = FakeController()
    dock_class = create_dock_widget_class(FakeDockWidget, controller_factory=lambda: controller)
    dock = dock_class()
    qtbot.addWidget(dock)

    assert dock.windowTitle() == DOCK_TITLE
    assert isinstance(dock.widget(), ColourPickerDockPanel)

    dock.visibilityChanged.emit(False)
    assert controller.visibility == [False]


def test_package_exports_register_plugin():
    assert lab_colour_picker.__all__ == ["register_plugin"]
    assert lab_colour_picker.register_plugin is register_plugin


class FakeController:
    def __init__(self):
        self.previews = []
        self.commits = []
        self.visibility = []
        self._foreground_listeners = []

    @property
    def selected_colour(self):
        return None

    def set_preview_colour(self, colour):
        self.previews.append(None if colour is None else np.asarray(colour, dtype=float).copy())

    def request_foreground_commit(self, colour):
        self.commits.append(None if colour is None else np.asarray(colour, dtype=float).copy())

    def set_dock_visible(self, visible):
        self.visibility.append(bool(visible))

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

"""Qt dock content for the OKLab colour picker."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
from PyQt5 import QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.controller import ChangeKind
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.widgets.readout_panel import ReadoutPanel
from oklab_colour_picker.widgets.selector import SelectorWidget


ColourListener = Callable[[np.ndarray, ChangeKind], None]


class DockController(Protocol):
    @property
    def selected_colour(self) -> np.ndarray | None:
        ...

    def set_preview_colour(self, oklab: Sequence[float] | None) -> None:
        ...

    def request_foreground_commit(self, oklab: Sequence[float] | None) -> None:
        ...

    def set_dock_visible(self, visible: bool) -> None:
        ...

    def add_colour_listener(self, listener: ColourListener) -> None:
        ...

    def remove_colour_listener(self, listener: ColourListener) -> None:
        ...


class SelectorMode(str, Enum):
    LIGHTNESS_SLICE = "lightness_slice"
    HUE_LIGHTNESS_SLICE = "hue_lightness_slice"
    LIGHTNESS_CHROMA_SLICE = "lightness_chroma_slice"


ModelFactory = Callable[[float, float, float], object]
WidgetFactory = Callable[[object, QtWidgets.QWidget], SelectorWidget]


@dataclass(frozen=True)
class ModeSpec:
    label: str
    object_name: str
    model_factory: ModelFactory
    widget_factory: WidgetFactory


def _lightness_slice_model(lightness: float, _chroma: float, _hue: float) -> object:
    return LightnessSliceModel(lightness=lightness)


def _hue_lightness_slice_model(_lightness: float, chroma: float, _hue: float) -> object:
    return HueLightnessSliceModel(chroma=chroma)


def _lightness_chroma_slice_model(_lightness: float, _chroma: float, hue: float) -> object:
    return LightnessChromaSliceModel(hue=hue)


def _selector_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    return SelectorWidget(model, parent)


def _lightness_slice_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    from oklab_colour_picker.widgets.lightness_slice_disk import LightnessSliceDiskWidget

    return LightnessSliceDiskWidget(model, parent)


def _hue_lightness_slice_widget(model: object, parent: QtWidgets.QWidget) -> SelectorWidget:
    from oklab_colour_picker.widgets.hue_lightness_slice_disk import HueLightnessSliceDiskWidget

    return HueLightnessSliceDiskWidget(model, parent)


MODE_SPECS = {
    SelectorMode.LIGHTNESS_SLICE: ModeSpec(
        "Hue/Chroma",
        "lightness-slice-selector",
        _lightness_slice_model,
        _lightness_slice_widget,
    ),
    SelectorMode.HUE_LIGHTNESS_SLICE: ModeSpec(
        "Hue/Lightness",
        "hue-lightness-slice-selector",
        _hue_lightness_slice_model,
        _hue_lightness_slice_widget,
    ),
    SelectorMode.LIGHTNESS_CHROMA_SLICE: ModeSpec(
        "Lightness/Chroma",
        "lightness-chroma-slice-selector",
        _lightness_chroma_slice_model,
        _selector_widget,
    ),
}

DEFAULT_COLOUR = np.array([0.5, 0.0, 0.0], dtype=float)


class ColourPickerDockPanel(QtWidgets.QWidget):
    """Build and synchronize the selector widgets shown inside the docker."""

    def __init__(self, controller: DockController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._view_seed_colour = _selected_or_default(controller.selected_colour)
        self._selector_modes = tuple(MODE_SPECS)
        self._tabs = QtWidgets.QTabWidget(self)
        self._selectors: dict[SelectorMode, QtWidgets.QWidget] = {}
        self._readout_panel = ReadoutPanel(self)
        self._readout_panel.previewed.connect(self._preview_colour)
        self._readout_panel.committed.connect(self._commit_colour)
        self._build_selector_tabs()
        self._build_layout()
        self._tabs.currentChanged.connect(self._ensure_selector_for_tab)
        self._seed_readout_panel(self._view_seed_colour)
        self._controller_subscription = ColourSubscription(controller, self._on_colour_changed)
        self.destroyed.connect(self._controller_subscription.disconnect)

    @property
    def selector_widgets(self) -> tuple[SelectorWidget, ...]:
        return tuple(self.selector_for_mode(mode) for mode in SelectorMode)

    @property
    def mode(self) -> SelectorMode:
        index = self._tabs.currentIndex()
        if 0 <= index < len(self._selector_modes):
            return self._selector_modes[index]
        return self._selector_modes[0]

    @property
    def active_selector(self) -> SelectorWidget:
        return self.selector_for_mode(self.mode)

    def selector_for_mode(self, mode: SelectorMode | str) -> SelectorWidget:
        return self._ensure_selector(SelectorMode(mode))

    def set_mode(self, mode: SelectorMode | str) -> None:
        selector_mode = SelectorMode(mode)
        self._ensure_selector(selector_mode)
        self._tabs.setCurrentIndex(self._tab_index_for_mode(selector_mode))

    def set_selected_colour(
        self, oklab: Sequence[float] | None, *, committed: bool = True
    ) -> None:
        if oklab is None:
            return
        self._show_on_views(
            _as_oklab(oklab), ChangeKind.COMMIT if committed else ChangeKind.PREVIEW
        )

    def _on_colour_changed(self, oklab: Sequence[float], kind: ChangeKind) -> None:
        self._show_on_views(_as_oklab(oklab), kind)

    def _show_on_views(self, colour: np.ndarray, kind: ChangeKind) -> None:
        self._view_seed_colour = colour
        for mode, widget in self._selectors.items():
            widget.apply_broadcast(
                colour, self._selector_model_factory(mode, colour)
            )
        self._readout_panel.set_current_colour(
            colour, committed=kind is not ChangeKind.PREVIEW
        )

    def _build_selector_tabs(self) -> None:
        for mode in self._selector_modes:
            if mode == self._selector_modes[0]:
                widget = self._ensure_selector(mode)
            else:
                widget = QtWidgets.QWidget(self)
                widget.setObjectName(f"{_mode_spec(mode).object_name}-placeholder")
            self._tabs.addTab(widget, _mode_spec(mode).label)

    def _ensure_selector_for_tab(self, index: int) -> None:
        if 0 <= index < len(self._selector_modes):
            self._ensure_selector(self._selector_modes[index])

    def _ensure_selector(self, mode: SelectorMode) -> SelectorWidget:
        existing = self._selectors.get(mode)
        if existing is not None:
            return existing

        seed = self._view_seed_colour
        widget = _build_selector_widget(mode, _model_for_colour(mode, seed), self)
        widget.setObjectName(_mode_spec(mode).object_name)
        self._seed_selector(widget, seed)
        widget.previewed.connect(self._preview_colour)
        widget.committed.connect(self._commit_colour)
        self._selectors[mode] = widget

        index = self._tab_index_for_mode(mode)
        if index < self._tabs.count():
            current_index = self._tabs.currentIndex()
            placeholder = self._tabs.widget(index)
            self._tabs.removeTab(index)
            self._tabs.insertTab(index, widget, _mode_spec(mode).label)
            if placeholder is not None:
                placeholder.deleteLater()
            if current_index == index:
                self._tabs.setCurrentIndex(index)
        return widget

    def _tab_index_for_mode(self, mode: SelectorMode) -> int:
        return self._selector_modes.index(mode)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)
        layout.addWidget(self._tabs)
        layout.addWidget(self._readout_panel)

    def _selector_model_factory(self, mode: SelectorMode, colour: np.ndarray) -> Callable[[], object]:
        return lambda: _model_for_colour(mode, colour)

    def _seed_selector(self, widget: SelectorWidget, colour: np.ndarray) -> None:
        widget.show_colour(colour, ChangeKind.INITIAL)

    def _seed_readout_panel(self, colour: np.ndarray) -> None:
        self._readout_panel.set_current_colour(colour)
        self._readout_panel.set_previous_colour(colour)

    def _preview_colour(self, oklab: Sequence[float] | None) -> None:
        self._controller.set_preview_colour(oklab)

    def _commit_colour(self, oklab: Sequence[float] | None) -> None:
        self._controller.request_foreground_commit(oklab)

def connect_dock_visibility(dock_widget, controller: DockController) -> "VisibilityConnection":
    dock_widget.visibilityChanged.connect(controller.set_dock_visible)
    return VisibilityConnection(dock_widget.visibilityChanged, controller.set_dock_visible)


class ColourSubscription:
    def __init__(self, controller: DockController, listener: ColourListener) -> None:
        self._controller = controller
        self._listener = listener
        self._connected = True
        self._controller.add_colour_listener(self._listener)

    def disconnect(self, *_args) -> None:
        if not self._connected:
            return
        try:
            self._controller.remove_colour_listener(self._listener)
        except (AttributeError, ValueError, RuntimeError):
            pass
        self._connected = False


class VisibilityConnection:
    def __init__(self, signal, slot) -> None:
        self._signal = signal
        self._slot = slot
        self._connected = True

    def disconnect(self) -> None:
        if not self._connected:
            return
        try:
            self._signal.disconnect(self._slot)
        except (TypeError, RuntimeError):
            pass
        self._connected = False


def _build_selector_widget(
    mode: SelectorMode,
    model: object,
    parent: QtWidgets.QWidget,
) -> SelectorWidget | QtWidgets.QWidget:
    return _mode_spec(mode).widget_factory(model, parent)


def _model_for_colour(mode: SelectorMode, oklab: Sequence[float]) -> object:
    lightness, chroma, hue = color_math.oklab_to_oklch(_as_oklab(oklab))
    lightness = float(np.clip(lightness, 0.0, 1.0))
    chroma = max(0.0, float(chroma))
    hue = float(hue % math.tau)
    return _mode_spec(mode).model_factory(lightness, chroma, hue)


def _mode_spec(mode: SelectorMode) -> ModeSpec:
    return MODE_SPECS[mode]


def _selected_or_default(oklab: Sequence[float] | None) -> np.ndarray:
    return DEFAULT_COLOUR.copy() if oklab is None else _as_oklab(oklab)


def _as_oklab(oklab: Sequence[float]) -> np.ndarray:
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()

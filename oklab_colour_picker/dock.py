"""Qt dock content for the OKLab colour picker."""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
from PyQt5 import QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.widgets.readout_panel import ReadoutPanel
from oklab_colour_picker.widgets.selector import SelectorWidget


ForegroundListener = Callable[[Sequence[float]], None]


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

    def add_foreground_listener(self, listener: ForegroundListener) -> None:
        ...

    def remove_foreground_listener(self, listener: ForegroundListener) -> None:
        ...


class SelectorMode(str, Enum):
    LIGHTNESS_SLICE = "lightness_slice"
    HUE_LIGHTNESS_SLICE = "hue_lightness_slice"
    LIGHTNESS_CHROMA_SLICE = "lightness_chroma_slice"


MODE_LABELS = {
    SelectorMode.LIGHTNESS_SLICE: "Hue/Chroma",
    SelectorMode.HUE_LIGHTNESS_SLICE: "Hue/Lightness",
    SelectorMode.LIGHTNESS_CHROMA_SLICE: "Lightness/Chroma",
}

MODE_OBJECT_NAMES = {
    SelectorMode.LIGHTNESS_SLICE: "lightness-slice-selector",
    SelectorMode.HUE_LIGHTNESS_SLICE: "hue-lightness-slice-selector",
    SelectorMode.LIGHTNESS_CHROMA_SLICE: "lightness-chroma-slice-selector",
}

DEFAULT_COLOUR = np.array([0.5, 0.0, 0.0], dtype=float)


class ColourPickerDockPanel(QtWidgets.QWidget):
    """Build and synchronize the selector widgets shown inside the docker."""

    def __init__(self, controller: DockController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        self._selected_colour = _selected_or_default(controller.selected_colour)
        self._selector_modes = tuple(SelectorMode)
        self._tabs = QtWidgets.QTabWidget(self)
        self._selectors: dict[SelectorMode, QtWidgets.QWidget] = {}
        self._readout_panel = ReadoutPanel(self)
        self._readout_panel.previewed.connect(self._preview_colour)
        self._readout_panel.committed.connect(self._commit_colour)
        self._foreground_listener = self.set_selected_colour
        self._build_selector_tabs()
        self._build_layout()
        self._tabs.currentChanged.connect(self._ensure_selector_for_tab)
        self._readout_panel.set_current_colour(self._selected_colour)
        # Seed previous swatch with the initial foreground so the first
        # display has a meaningful revert target (current == previous).
        self._readout_panel.set_previous_colour(self._selected_colour)
        self._controller.add_foreground_listener(self._foreground_listener)
        self.destroyed.connect(self._remove_foreground_listener)

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

        self._selected_colour = _as_oklab(oklab)
        for mode, widget in self._selectors.items():
            model = _model_for_colour(mode, self._selected_colour)
            if widget.model != model:
                widget.set_model(model)
            widget.set_selected_colour(self._selected_colour)
        self._readout_panel.set_current_colour(
            self._selected_colour, committed=committed
        )

    def _build_selector_tabs(self) -> None:
        for mode in self._selector_modes:
            if mode == self._selector_modes[0]:
                widget = self._ensure_selector(mode)
            else:
                widget = QtWidgets.QWidget(self)
                widget.setObjectName(f"{MODE_OBJECT_NAMES[mode]}-placeholder")
            self._tabs.addTab(widget, MODE_LABELS[mode])

    def _ensure_selector_for_tab(self, index: int) -> None:
        if 0 <= index < len(self._selector_modes):
            self._ensure_selector(self._selector_modes[index])

    def _ensure_selector(self, mode: SelectorMode) -> SelectorWidget:
        existing = self._selectors.get(mode)
        if existing is not None:
            return existing

        widget = _build_selector_widget(mode, _model_for_colour(mode, self._selected_colour), self)
        widget.setObjectName(MODE_OBJECT_NAMES[mode])
        widget.set_selected_colour(self._selected_colour)
        widget.previewed.connect(self._preview_colour)
        widget.committed.connect(self._commit_colour)
        self._selectors[mode] = widget

        index = self._tab_index_for_mode(mode)
        if index < self._tabs.count():
            current_index = self._tabs.currentIndex()
            placeholder = self._tabs.widget(index)
            # Replacing the placeholder may re-emit currentChanged; the
            # existing-widget early return above keeps that re-entry harmless.
            self._tabs.removeTab(index)
            self._tabs.insertTab(index, widget, MODE_LABELS[mode])
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

    def _preview_colour(self, oklab: Sequence[float] | None) -> None:
        if oklab is not None:
            self.set_selected_colour(oklab, committed=False)
        self._controller.set_preview_colour(oklab)

    def _commit_colour(self, oklab: Sequence[float] | None) -> None:
        if oklab is not None:
            self.set_selected_colour(oklab, committed=True)
        self._controller.request_foreground_commit(oklab)

    def _remove_foreground_listener(self) -> None:
        try:
            self._controller.remove_foreground_listener(self._foreground_listener)
        except (AttributeError, ValueError, RuntimeError):
            pass


def connect_dock_visibility(dock_widget, controller: DockController) -> "VisibilityConnection":
    dock_widget.visibilityChanged.connect(controller.set_dock_visible)
    return VisibilityConnection(dock_widget.visibilityChanged, controller.set_dock_visible)


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
    if mode == SelectorMode.HUE_LIGHTNESS_SLICE:
        from oklab_colour_picker.widgets.hue_lightness_slice_disk import HueLightnessSliceDiskWidget

        return HueLightnessSliceDiskWidget(model, parent)
    if mode == SelectorMode.LIGHTNESS_SLICE:
        from oklab_colour_picker.widgets.lightness_slice_disk import LightnessSliceDiskWidget

        return LightnessSliceDiskWidget(model, parent)
    return SelectorWidget(model, parent)


def _model_for_colour(mode: SelectorMode, oklab: Sequence[float]) -> object:
    lightness, chroma, hue = color_math.oklab_to_oklch(_as_oklab(oklab))
    lightness = float(np.clip(lightness, 0.0, 1.0))
    chroma = max(0.0, float(chroma))
    hue = float(hue % math.tau)
    if mode == SelectorMode.LIGHTNESS_SLICE:
        return LightnessSliceModel(lightness=lightness)
    if mode == SelectorMode.HUE_LIGHTNESS_SLICE:
        return HueLightnessSliceModel(chroma=chroma)
    if mode == SelectorMode.LIGHTNESS_CHROMA_SLICE:
        return LightnessChromaSliceModel(hue=hue)
    raise AssertionError(f"unhandled selector mode: {mode!r}")


def _selected_or_default(oklab: Sequence[float] | None) -> np.ndarray:
    return DEFAULT_COLOUR.copy() if oklab is None else _as_oklab(oklab)


def _as_oklab(oklab: Sequence[float]) -> np.ndarray:
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()

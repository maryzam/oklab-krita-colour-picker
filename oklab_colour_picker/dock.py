"""Qt dock content for the OKLab colour picker."""

from __future__ import annotations

import math
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np
from PyQt5 import QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    ChromaLightnessModel,
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.widgets import (
    HueLightnessSliceDiskWidget,
    HueRingTabWidget,
    LightnessSliceDiskWidget,
    ReadoutPanel,
    SelectorWidget,
)


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
    CHROMA_LIGHTNESS = "chroma_lightness"


MODE_LABELS = {
    SelectorMode.LIGHTNESS_SLICE: "Hue/Chroma",
    SelectorMode.HUE_LIGHTNESS_SLICE: "Hue/Lightness",
    SelectorMode.LIGHTNESS_CHROMA_SLICE: "Lightness/Chroma",
    SelectorMode.CHROMA_LIGHTNESS: "Hue Ring",
}

MODE_OBJECT_NAMES = {
    SelectorMode.LIGHTNESS_SLICE: "lightness-slice-selector",
    SelectorMode.HUE_LIGHTNESS_SLICE: "hue-lightness-slice-selector",
    SelectorMode.LIGHTNESS_CHROMA_SLICE: "lightness-chroma-slice-selector",
    SelectorMode.CHROMA_LIGHTNESS: "chroma-lightness-selector",
}

DEFAULT_COLOUR = np.array([0.5, 0.0, 0.0], dtype=float)


class ColourPickerDockPanel(QtWidgets.QWidget):
    """Build and synchronize the selector widgets shown inside the docker."""

    def __init__(self, controller: DockController, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._controller = controller
        # Pull current Krita foreground synchronously if the controller supports
        # it, so the initial paint reflects the real FG rather than the default.
        sync = getattr(controller, "sync_external_foreground", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass
        self._selected_colour = _selected_or_default(controller.selected_colour)
        self._tabs = QtWidgets.QTabWidget(self)
        self._selectors: dict[SelectorMode, SelectorWidget] = {}
        self._readout_panel = ReadoutPanel(self)
        self._readout_panel.previewed.connect(self._preview_colour)
        self._readout_panel.committed.connect(self._commit_colour)
        self._foreground_listener = self.set_selected_colour
        self._build_selectors()
        self._build_layout()
        self._readout_panel.set_current_colour(self._selected_colour)
        # Seed previous swatch with the initial foreground so the first
        # display has a meaningful revert target (current == previous).
        self._readout_panel.set_previous_colour(self._selected_colour)
        self._controller.add_foreground_listener(self._foreground_listener)
        self.destroyed.connect(self._remove_foreground_listener)

    @property
    def selector_widgets(self) -> tuple[SelectorWidget, ...]:
        return tuple(self._selectors[mode] for mode in SelectorMode)

    @property
    def mode(self) -> SelectorMode:
        current = self._tabs.currentWidget()
        for mode, widget in self._selectors.items():
            if widget is current:
                return mode
        return SelectorMode.LIGHTNESS_SLICE

    @property
    def active_selector(self) -> SelectorWidget:
        return self._selectors[self.mode]

    def selector_for_mode(self, mode: SelectorMode | str) -> SelectorWidget:
        return self._selectors[SelectorMode(mode)]

    def set_mode(self, mode: SelectorMode | str) -> None:
        widget = self.selector_for_mode(mode)
        self._tabs.setCurrentWidget(widget)

    def set_selected_colour(
        self, oklab: Sequence[float] | None, *, committed: bool = True
    ) -> None:
        if oklab is None:
            return

        self._selected_colour = _as_oklab(oklab)
        models = _models_for_colour(self._selected_colour)
        for mode, widget in self._selectors.items():
            if widget.model != models[mode]:
                widget.set_model(models[mode])
            widget.set_selected_colour(self._selected_colour)
        self._readout_panel.set_current_colour(
            self._selected_colour, committed=committed
        )

    def _build_selectors(self) -> None:
        for mode, model in _models_for_colour(self._selected_colour).items():
            widget = _build_selector_widget(mode, model, self)
            widget.setObjectName(MODE_OBJECT_NAMES[mode])
            widget.set_selected_colour(self._selected_colour)
            widget.previewed.connect(self._preview_colour)
            widget.committed.connect(self._commit_colour)
            self._selectors[mode] = widget
            self._tabs.addTab(widget, MODE_LABELS[mode])

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
) -> SelectorWidget | HueRingTabWidget:
    if mode == SelectorMode.CHROMA_LIGHTNESS:
        return HueRingTabWidget(model, parent)
    if mode == SelectorMode.HUE_LIGHTNESS_SLICE:
        return HueLightnessSliceDiskWidget(model, parent)
    if mode == SelectorMode.LIGHTNESS_SLICE:
        return LightnessSliceDiskWidget(model, parent)
    return SelectorWidget(model, parent)


def _models_for_colour(oklab: Sequence[float]) -> dict[SelectorMode, object]:
    lightness, chroma, hue = color_math.oklab_to_oklch(_as_oklab(oklab))
    hue = float(hue % math.tau)
    return {
        SelectorMode.LIGHTNESS_SLICE: LightnessSliceModel(lightness=float(np.clip(lightness, 0.0, 1.0))),
        SelectorMode.HUE_LIGHTNESS_SLICE: HueLightnessSliceModel(chroma=max(0.0, float(chroma))),
        SelectorMode.LIGHTNESS_CHROMA_SLICE: LightnessChromaSliceModel(hue=hue),
        SelectorMode.CHROMA_LIGHTNESS: ChromaLightnessModel(
            lightness=float(np.clip(lightness, 0.0, 1.0)),
            chroma=max(0.0, float(chroma)),
        ),
    }


def _selected_or_default(oklab: Sequence[float] | None) -> np.ndarray:
    return DEFAULT_COLOUR.copy() if oklab is None else _as_oklab(oklab)


def _as_oklab(oklab: Sequence[float]) -> np.ndarray:
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()

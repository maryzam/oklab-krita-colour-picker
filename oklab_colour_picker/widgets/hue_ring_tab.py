"""Composite widget for the Hue Ring tab.

Lays a thick OKLCh hue donut around a central panel containing the dynamic
hue label, the selected-colour swatch, and L/C gradient sliders. Slider drags
emit ``previewed``/``committed`` with a new OKLab triple; ring drags pass
through unchanged.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    CHROMA_LIGHTNESS_INNER_RADIUS_FRACTION,
    LIGHTNESS_CHART_CHROMA_MAX,
    ChromaLightnessModel,
)
from oklab_colour_picker.widgets.selector import SelectorWidget


class HueRingTabWidget(QtWidgets.QWidget):
    """Hue donut + central swatch/sliders for picking hue at a chosen L,C."""

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, model: ChromaLightnessModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._ring = SelectorWidget(model, self)
        self._ring.previewed.connect(self.previewed.emit)
        self._ring.committed.connect(self.committed.emit)

        self._panel = _CentralPanel(self)
        self._panel.lightness_previewed.connect(lambda v: self._emit_axis_change("L", v, commit=False))
        self._panel.lightness_committed.connect(lambda v: self._emit_axis_change("L", v, commit=True))
        self._panel.chroma_previewed.connect(lambda v: self._emit_axis_change("C", v, commit=False))
        self._panel.chroma_committed.connect(lambda v: self._emit_axis_change("C", v, commit=True))

        self._selected: np.ndarray | None = None
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(self._ring.minimumSize())
        self._sync_panel_from_model()

    @property
    def model(self) -> ChromaLightnessModel:
        return self._ring.model

    def set_model(self, model: ChromaLightnessModel) -> None:
        self._ring.set_model(model)
        self._sync_panel_from_model()

    def set_selected_colour(self, oklab: Sequence[float] | None) -> None:
        self._selected = None if oklab is None else np.asarray(oklab, dtype=float)
        self._ring.set_selected_colour(oklab)
        self._sync_panel_from_model()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._ring.setGeometry(0, 0, self.width(), self.height())
        outer_radius = (min(self.width(), self.height()) - 1) / 2.0
        inner_radius = outer_radius * CHROMA_LIGHTNESS_INNER_RADIUS_FRACTION
        side = max(0, int(inner_radius * math.sqrt(2)) - 6)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        self._panel.setGeometry(int(cx - side / 2), int(cy - side / 2), side, side)
        self._panel.setVisible(side >= 60)

    def _current_hue(self) -> float:
        if self._selected is None:
            return 0.0
        _, _, hue = color_math.oklab_to_oklch(self._selected)
        return float(hue % math.tau)

    def _sync_panel_from_model(self) -> None:
        model = self._ring.model
        hue = self._current_hue()
        self._panel.set_axes(lightness=model.lightness, chroma=model.chroma, hue=hue)
        if self._selected is not None:
            self._panel.set_swatch_colour(self._selected)

    def _emit_axis_change(self, axis: str, value: float, commit: bool) -> None:
        if self._selected is None:
            base_l, base_c, base_h = self.model.lightness, self.model.chroma, 0.0
        else:
            base_l, base_c, base_h = (float(v) for v in color_math.oklab_to_oklch(self._selected))
        if axis == "L":
            new_l = float(np.clip(value, 0.0, 1.0))
            new_c = base_c
        else:
            new_l = base_l
            new_c = max(0.0, float(value))
        oklab = np.asarray(color_math.oklch_to_oklab([new_l, new_c, base_h]), dtype=float)
        (self.committed if commit else self.previewed).emit(oklab)


class _CentralPanel(QtWidgets.QWidget):
    """Hue label + swatch + L/C gradient sliders shown inside the donut hole."""

    lightness_previewed = QtCore.pyqtSignal(float)
    lightness_committed = QtCore.pyqtSignal(float)
    chroma_previewed = QtCore.pyqtSignal(float)
    chroma_committed = QtCore.pyqtSignal(float)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._hue_label = QtWidgets.QLabel("H —", self)
        self._hue_label.setAlignment(QtCore.Qt.AlignCenter)
        bold = self._hue_label.font()
        bold.setBold(True)
        self._hue_label.setFont(bold)

        self._swatch = _Swatch(self)
        self._lightness_slider = _OklchGradientSlider("L", self)
        self._chroma_slider = _OklchGradientSlider("C", self)

        self._lightness_slider.previewed.connect(self.lightness_previewed)
        self._lightness_slider.committed.connect(self.lightness_committed)
        self._chroma_slider.previewed.connect(self.chroma_previewed)
        self._chroma_slider.committed.connect(self.chroma_committed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        layout.addWidget(self._hue_label)
        layout.addWidget(self._swatch, 1)
        layout.addWidget(self._lightness_slider)
        layout.addWidget(self._chroma_slider)

    def set_axes(self, lightness: float, chroma: float, hue: float) -> None:
        self._lightness_slider.set_value(lightness)
        self._lightness_slider.set_other_axis(chroma)
        self._chroma_slider.set_value(chroma)
        self._chroma_slider.set_other_axis(lightness)
        self._set_hue(hue)

    def set_swatch_colour(self, oklab: Sequence[float]) -> None:
        self._swatch.set_colour(oklab)

    def _set_hue(self, hue_radians: float) -> None:
        deg = (math.degrees(hue_radians) + 360.0) % 360.0
        self._hue_label.setText(f"H {deg:.0f}°")
        self._lightness_slider.set_hue(hue_radians)
        self._chroma_slider.set_hue(hue_radians)


class _Swatch(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(0, 0, 0)
        self.setMinimumHeight(20)

    def set_colour(self, oklab: Sequence[float]) -> None:
        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(np.asarray(oklab, dtype=float)))
        rgb = np.rint(srgb * 255.0).astype(int).tolist()
        self._color = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect(), self._color)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 1))
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        painter.end()


class _OklchGradientSlider(QtWidgets.QWidget):
    """Horizontal slider whose track is a live OKLCh gradient.

    ``axis="L"`` varies lightness in [0, 1] at fixed (chroma, hue);
    ``axis="C"`` varies chroma in [0, LIGHTNESS_CHART_CHROMA_MAX] at fixed
    (lightness, hue).
    """

    previewed = QtCore.pyqtSignal(float)
    committed = QtCore.pyqtSignal(float)

    _MARGIN = 4
    _TRACK_HEIGHT = 12

    def __init__(self, axis: str, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        if axis not in ("L", "C"):
            raise ValueError("axis must be 'L' or 'C'")
        self._axis = axis
        self._max = 1.0 if axis == "L" else LIGHTNESS_CHART_CHROMA_MAX
        self._value = 0.5 if axis == "L" else 0.1
        self._other = 0.1 if axis == "L" else 0.5
        self._hue = 0.0
        self._dragging = False
        self._buffer: np.ndarray | None = None
        self.setMinimumHeight(self._TRACK_HEIGHT + 2 * self._MARGIN)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_value(self, value: float) -> None:
        self._value = float(np.clip(value, 0.0, self._max))
        self.update()

    def set_other_axis(self, other: float) -> None:
        self._other = float(other)
        self.update()

    def set_hue(self, hue_radians: float) -> None:
        self._hue = float(hue_radians) % math.tau
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        rect = self._track_rect()
        if rect.width() < 2:
            painter.end()
            return

        image = self._gradient_image(rect.width(), rect.height())
        painter.drawImage(rect, image)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rect.adjusted(0, 0, -1, -1))

        x = self._value_to_x(self._value)
        thumb = QtCore.QRect(x - 3, rect.top() - 3, 7, rect.height() + 6)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2))
        painter.drawRect(thumb)
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 1))
        painter.drawRect(thumb)
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        self._dragging = True
        self._value = self._x_to_value(event.x())
        self.update()
        self.previewed.emit(self._value)
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._dragging:
            event.ignore()
            return
        self._value = self._x_to_value(event.x())
        self.update()
        self.previewed.emit(self._value)
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or not self._dragging:
            event.ignore()
            return
        self._dragging = False
        self._value = self._x_to_value(event.x())
        self.update()
        self.committed.emit(self._value)
        event.accept()

    def _track_rect(self) -> QtCore.QRect:
        return QtCore.QRect(
            self._MARGIN,
            (self.height() - self._TRACK_HEIGHT) // 2,
            max(2, self.width() - 2 * self._MARGIN),
            self._TRACK_HEIGHT,
        )

    def _value_to_x(self, value: float) -> int:
        rect = self._track_rect()
        if self._max <= 0.0:
            return rect.left()
        fraction = float(np.clip(value / self._max, 0.0, 1.0))
        return rect.left() + int(round(fraction * (rect.width() - 1)))

    def _x_to_value(self, x: int) -> float:
        rect = self._track_rect()
        fraction = (x - rect.left()) / max(1, rect.width() - 1)
        return float(np.clip(fraction, 0.0, 1.0)) * self._max

    def _gradient_image(self, width: int, height: int) -> QtGui.QImage:
        ts = np.linspace(0.0, 1.0, width)
        if self._axis == "L":
            lightness = ts
            chroma = np.full_like(ts, self._other)
        else:
            lightness = np.full_like(ts, self._other)
            chroma = ts * self._max
        hue = np.full_like(ts, self._hue)
        oklch = np.stack([lightness, chroma, hue], axis=-1)
        oklab = color_math.oklch_to_oklab(oklch)
        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
        rgb = np.rint(srgb * 255.0).astype(np.uint8)

        rgba_row = np.zeros((1, width, 4), dtype=np.uint8)
        rgba_row[0, :, :3] = rgb
        rgba_row[0, :, 3] = 255
        rgba = np.ascontiguousarray(np.repeat(rgba_row, height, axis=0))
        self._buffer = rgba
        return QtGui.QImage(
            rgba.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888
        )

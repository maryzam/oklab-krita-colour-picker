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
    CHROMA_EPSILON,
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
        self._ring.previewed.connect(self._on_ring_previewed)
        self._ring.committed.connect(self._on_ring_committed)
        # OKLab→OKLCh cannot recover hue when the model's chroma is zero
        # (default startup, or after a C-slider drag to 0): the ring's
        # emitted colour is achromatic. Watch the ring's mouse events
        # directly so we can lift the hue from the click geometry.
        self._ring.installEventFilter(self)

        self._panel = _CentralPanel(self)
        self._panel.lightness_previewed.connect(lambda v: self._emit_axis_change("L", v, commit=False))
        self._panel.lightness_committed.connect(lambda v: self._emit_axis_change("L", v, commit=True))
        self._panel.chroma_previewed.connect(lambda v: self._emit_axis_change("C", v, commit=False))
        self._panel.chroma_committed.connect(lambda v: self._emit_axis_change("C", v, commit=True))

        # Hue is undefined whenever the selected colour is achromatic
        # (chroma ~= 0), so OKLab→OKLCh would collapse the user's chosen hue
        # to 0. Track it separately and only refresh from OKLab when the
        # incoming colour actually carries hue information.
        self._chosen_hue: float = 0.0
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setMinimumSize(self._ring.minimumSize())
        self._sync_panel_from_model()

    @property
    def model(self) -> ChromaLightnessModel:
        return self._ring.model

    @property
    def selected_colour(self) -> np.ndarray | None:
        return self._ring.selected_colour

    def indicator_position(self) -> tuple[float, float] | None:
        return self._ring.indicator_position()

    def set_model(self, model: ChromaLightnessModel) -> None:
        self._ring.set_model(model)
        self._sync_panel_from_model()

    def set_selected_colour(self, oklab: Sequence[float] | None) -> None:
        if oklab is not None:
            _, chroma, hue = color_math.oklab_to_oklch(np.asarray(oklab, dtype=float))
            if float(chroma) > CHROMA_EPSILON:
                self._chosen_hue = float(hue) % math.tau
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

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._ring and event.type() in (
            QtCore.QEvent.MouseButtonPress,
            QtCore.QEvent.MouseMove,
            QtCore.QEvent.MouseButtonRelease,
        ):
            self._capture_hue_from_ring_position(event.pos())
        return super().eventFilter(obj, event)

    def _capture_hue_from_ring_position(self, pos: QtCore.QPoint) -> None:
        width, height = self._ring.width(), self._ring.height()
        if width <= 1 or height <= 1:
            return
        # Only honour clicks that fall on the donut band: the ring model's
        # ``color_at_position`` returns None for the centre, the corners, and
        # any out-of-gamut hue at the current (L, C). That keeps stray
        # corner clicks from rotating the saved hue.
        if self._ring.model.color_at_position((pos.x(), pos.y()), (width, height)) is None:
            return
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        dx = float(pos.x()) - center_x
        dy = center_y - float(pos.y())
        if math.hypot(dx, dy) <= 1e-9:
            return
        self._chosen_hue = math.atan2(dy, dx) % math.tau

    def _on_ring_previewed(self, oklab: object) -> None:
        self._capture_hue_if_chromatic(oklab)
        self.previewed.emit(oklab)

    def _on_ring_committed(self, oklab: object) -> None:
        self._capture_hue_if_chromatic(oklab)
        self.committed.emit(oklab)

    def _capture_hue_if_chromatic(self, oklab: object) -> None:
        if oklab is None:
            return
        _, chroma, hue = color_math.oklab_to_oklch(np.asarray(oklab, dtype=float))
        if float(chroma) > CHROMA_EPSILON:
            self._chosen_hue = float(hue) % math.tau

    def _sync_panel_from_model(self) -> None:
        model = self._ring.model
        self._panel.set_axes(lightness=model.lightness, chroma=model.chroma, hue=self._chosen_hue)
        selected = self._ring.selected_colour
        if selected is not None:
            self._panel.set_swatch_colour(selected)

    def _emit_axis_change(self, axis: str, value: float, commit: bool) -> None:
        selected = self._ring.selected_colour
        if selected is None:
            base_l, base_c = self.model.lightness, self.model.chroma
        else:
            base_l, base_c, _ = (float(v) for v in color_math.oklab_to_oklch(selected))
        base_h = self._chosen_hue
        if axis == "L":
            new_l = float(np.clip(value, 0.0, 1.0))
            new_c = base_c
        else:
            new_l = base_l
            new_c = max(0.0, float(value))
        # Clamp chroma against the per-(L, hue) gamut so we never emit OKLCh
        # outside sRGB; otherwise the adapter would clip silently and the
        # dock/swatch would drift away from the colour committed to Krita.
        max_c = float(color_math.max_chroma_for_lh(new_l, base_h))
        if new_c > max_c:
            new_c = max(0.0, max_c)
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
        gamut_max = float(color_math.max_chroma_for_lh(lightness, hue))
        effective_max = max(0.0, min(LIGHTNESS_CHART_CHROMA_MAX, gamut_max))
        self._chroma_slider.set_max(effective_max)
        self._chroma_slider.set_value(min(chroma, effective_max))
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
        self._gradient_cache_key: tuple | None = None
        self._gradient_cache_image: QtGui.QImage | None = None
        self._gradient_cache_buffer: np.ndarray | None = None
        self.setMinimumHeight(self._TRACK_HEIGHT + 2 * self._MARGIN)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

    def set_value(self, value: float) -> None:
        clamped = float(np.clip(value, 0.0, max(0.0, self._max)))
        if clamped == self._value:
            return
        self._value = clamped
        self.update()

    def set_max(self, maximum: float) -> None:
        new_max = max(0.0, float(maximum))
        if new_max == self._max:
            return
        self._max = new_max
        self._value = float(np.clip(self._value, 0.0, new_max))
        self.update()

    def set_other_axis(self, other: float) -> None:
        if float(other) == self._other:
            return
        self._other = float(other)
        self.update()

    def set_hue(self, hue_radians: float) -> None:
        new_hue = float(hue_radians) % math.tau
        if new_hue == self._hue:
            return
        self._hue = new_hue
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
        # Drag-only repaints (where the user moves the thumb) keep axis,
        # other, hue and max constant — recomputing the OKLCh→sRGB gradient
        # for every move would be wasted colour-math work, so cache by inputs.
        key = (self._axis, width, height, self._other, self._hue, self._max)
        if self._gradient_cache_key == key and self._gradient_cache_image is not None:
            return self._gradient_cache_image

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
        image = QtGui.QImage(
            rgba.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888
        )
        self._gradient_cache_key = key
        self._gradient_cache_buffer = rgba
        self._gradient_cache_image = image
        return image

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
from PyQt5 import QtCore, QtGui, QtWidgets, sip

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    CHROMA_EPSILON,
    LIGHTNESS_CHART_CHROMA_MAX,
    ChromaLightnessModel,
    chroma_lightness_band_width,
)
from oklab_colour_picker.widgets.selector import SelectorWidget


class HueRingTabWidget(QtWidgets.QWidget):
    """Hue donut + central swatch/sliders for picking hue at a chosen L,C."""

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, model: ChromaLightnessModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._ring = _HueRingSelector(model, self)
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
        # Mirror SelectorWidget's left-button drag lifecycle so achromatic
        # ring drags can preserve the latest valid hue even when the pointer
        # is released outside the selectable band.
        self._ring_drag_active: bool = False
        self._last_valid_drag_hue: float | None = None
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
        inner_radius = max(0.0, outer_radius - chroma_lightness_band_width(outer_radius))
        side = max(0, int(inner_radius * math.sqrt(2)) - 6)
        cx, cy = self.width() / 2.0, self.height() / 2.0
        self._panel.setGeometry(int(cx - side / 2), int(cy - side / 2), side, side)
        self._panel.setVisible(side >= 60)

    def eventFilter(self, obj: QtCore.QObject, event: QtCore.QEvent) -> bool:
        if obj is self._ring:
            event_type = event.type()
            if event_type == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    self._begin_hue_drag()
                    self._capture_hue_from_ring_position(event.pos())
            elif event_type == QtCore.QEvent.MouseMove:
                if self._ring_drag_active and (event.buttons() & QtCore.Qt.LeftButton):
                    self._capture_hue_from_ring_position(event.pos())
            elif event_type == QtCore.QEvent.MouseButtonRelease:
                if event.button() == QtCore.Qt.LeftButton and self._ring_drag_active:
                    self._end_hue_drag(event.pos())
        return super().eventFilter(obj, event)

    def _begin_hue_drag(self) -> None:
        self._ring_drag_active = True
        self._last_valid_drag_hue = None

    def _end_hue_drag(self, release_pos: QtCore.QPoint) -> None:
        # A valid release may arrive without a separate move event at that
        # position, so capture it before the drag cache is cleared.
        if self._ring_drag_position_selectable(release_pos):
            self._capture_hue_from_ring_position(release_pos)
        self._ring_drag_active = False
        self._last_valid_drag_hue = None

    def _capture_hue_from_ring_position(self, pos: QtCore.QPoint) -> None:
        width, height = self._ring.width(), self._ring.height()
        if width <= 1 or height <= 1:
            return
        if not self._ring_drag_position_selectable(pos):
            return
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        dx = float(pos.x()) - center_x
        dy = center_y - float(pos.y())
        if math.hypot(dx, dy) <= 1e-9:
            return
        hue = math.atan2(dy, dx) % math.tau
        self._chosen_hue = hue
        if self._ring_drag_active:
            self._last_valid_drag_hue = hue

    def _ring_drag_position_selectable(self, pos: QtCore.QPoint) -> bool:
        width, height = self._ring.width(), self._ring.height()
        if width <= 1 or height <= 1:
            return False
        position = (pos.x(), pos.y())
        size = (width, height)
        if self._ring.model.color_at_position(position, size) is not None:
            return True
        if not self._ring_drag_active or self._last_valid_drag_hue is None:
            return False
        snapper = getattr(self._ring.model, "snapped_color_at_position", None)
        return callable(snapper) and snapper(position, size) is not None

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
        layout.addWidget(self._swatch, 1)
        layout.addLayout(_labelled_row("L", self._lightness_slider))
        layout.addLayout(_labelled_row("C", self._chroma_slider))

    def set_axes(self, lightness: float, chroma: float, hue: float) -> None:
        self._lightness_slider.set_value(lightness)
        self._lightness_slider.set_other_axis(chroma)
        gamut_max = float(color_math.max_chroma_for_lh(lightness, hue))
        effective_max = max(0.0, min(LIGHTNESS_CHART_CHROMA_MAX, gamut_max))
        # The C slider's track length stays anchored to LIGHTNESS_CHART_CHROMA_MAX
        # so the thumb's x-position depends only on the actual chroma value.
        # Rescaling by per-hue gamut max would shift the thumb every time the
        # user rotates the ring even though chroma is unchanged.
        self._chroma_slider.set_gamut_max(effective_max)
        self._chroma_slider.set_value(chroma)
        self._chroma_slider.set_other_axis(lightness)
        self._set_hue(hue)

    def set_swatch_colour(self, oklab: Sequence[float]) -> None:
        self._swatch.set_colour(oklab)

    def _set_hue(self, hue_radians: float) -> None:
        deg = (math.degrees(hue_radians) + 360.0) % 360.0
        self._swatch.set_hue_text(f"H {deg:.0f}°")
        self._lightness_slider.set_hue(hue_radians)
        self._chroma_slider.set_hue(hue_radians)


class _Swatch(QtWidgets.QWidget):
    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QtGui.QColor(0, 0, 0)
        self._hue_text = ""
        self.setMinimumHeight(20)

    def set_colour(self, oklab: Sequence[float]) -> None:
        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(np.asarray(oklab, dtype=float)))
        rgb = np.rint(srgb * 255.0).astype(int).tolist()
        self._color = QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.update()

    def set_hue_text(self, text: str) -> None:
        if text == self._hue_text:
            return
        self._hue_text = text
        self.update()

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        if sip.isdeleted(self):
            return

        rect = self.rect()
        width = rect.width()
        hue_text = self._hue_text
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.fillRect(rect, self._color)
            painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 160), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if hue_text:
                self._paint_hue_text(painter, hue_text, width)
        finally:
            painter.end()

    def _paint_hue_text(self, painter: QtGui.QPainter, text: str, width: int) -> None:
        # Stroked text overlays the swatch — a dark halo plus a light fill keep
        # the label legible regardless of the swatch colour.
        font = painter.font()
        font.setBold(True)
        path = QtGui.QPainterPath()
        metrics = QtGui.QFontMetricsF(font)
        text_width = metrics.horizontalAdvance(text)
        x = (width - text_width) / 2.0
        y = metrics.ascent() + 4.0
        path.addText(QtCore.QPointF(x, y), font, text)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 220), 3.0))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawPath(path)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(255, 255, 255, 240))
        painter.drawPath(path)


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
        # For the C slider, the per-(L, hue) sRGB gamut max — chroma values
        # past this point are unreachable and rendered as a dimmed tail.
        # Defaults to _max so L sliders simply skip the overlay.
        self._gamut_max = self._max
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

    def set_gamut_max(self, gamut_max: float) -> None:
        new_gamut = float(np.clip(gamut_max, 0.0, self._max))
        if new_gamut == self._gamut_max:
            return
        self._gamut_max = new_gamut
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
        if self._gamut_max < self._max:
            gamut_x = self._value_to_x(self._gamut_max)
            tail = QtCore.QRect(gamut_x, rect.top(), rect.right() - gamut_x + 1, rect.height())
            painter.fillRect(tail, QtGui.QColor(0, 0, 0, 140))
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


def _labelled_row(text: str, slider: QtWidgets.QWidget) -> QtWidgets.QHBoxLayout:
    label = QtWidgets.QLabel(text, slider.parentWidget())
    label.setAlignment(QtCore.Qt.AlignCenter)
    label.setFixedWidth(12)
    bold = label.font()
    bold.setBold(True)
    label.setFont(bold)
    row = QtWidgets.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(label)
    row.addWidget(slider, 1)
    return row


class _HueRingSelector(SelectorWidget):
    """Hue ring with a radial-bar indicator instead of the default circle.

    The bar spans the full donut band (with a small overhang at each end so
    it reads as selecting the entire hue slice) and rotates with the current
    hue. The painter rotates around the ring centre so the bar's geometry is
    expressed in simple "distance along radius" coordinates.
    """

    _BAR_THICKNESS = 4.0
    _BAR_OVERHANG = 3.0

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        position = self.indicator_position()
        if position is None:
            return
        outer_radius = (min(self.width(), self.height()) - 1) / 2.0
        if outer_radius <= 0.0:
            return
        center_x = (self.width() - 1) / 2.0
        center_y = (self.height() - 1) / 2.0
        dx = position[0] - center_x
        dy = center_y - position[1]
        if math.hypot(dx, dy) <= 1e-9:
            return
        hue = math.atan2(dy, dx)
        band = chroma_lightness_band_width(outer_radius)
        inner_radius = max(0.0, outer_radius - band)
        bar_inner = inner_radius - self._BAR_OVERHANG
        bar_outer = outer_radius + self._BAR_OVERHANG
        rect = QtCore.QRectF(
            bar_inner,
            -self._BAR_THICKNESS / 2.0,
            bar_outer - bar_inner,
            self._BAR_THICKNESS,
        )

        painter.save()
        painter.translate(center_x, center_y)
        # atan2 used widget-y-flipped coords (dy = center_y - pos.y), so the
        # painter rotation has to flip back to Qt's y-down system.
        painter.rotate(-math.degrees(hue))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 2.0))
        painter.drawRect(rect)
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 1.0))
        painter.drawRect(rect)
        painter.restore()

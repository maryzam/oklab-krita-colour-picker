"""Expanded readout panel: swatches, L/C/H sliders, and hex field.

This widget is *additive* to the dock layout. It owns no colour state of its
own — every interaction emits :attr:`previewed` / :attr:`committed` so the
host panel can route through the existing controller path, the same way the
selector widgets do.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math, renderers


HEX_RE = re.compile(r"^\s*#?([0-9a-fA-F]{6})\s*$")

# Per-axis numeric step sizes (Shift x10). H is in degrees, displayed in
# degrees, stored on the model in radians.
_STEP_L = 0.01
_STEP_C = 0.005
_STEP_H = 1.0


def oklab_to_hex(oklab: Sequence[float]) -> str:
    """Return ``#rrggbb`` for the OKLab colour, clipping to sRGB if needed."""

    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(np.asarray(oklab, dtype=float)))
    r, g, b = (int(round(float(c) * 255.0)) for c in srgb)
    return f"#{r:02x}{g:02x}{b:02x}"


def hex_to_oklab(text: str) -> np.ndarray | None:
    """Parse ``#rrggbb`` / ``rrggbb`` to OKLab; ``None`` on malformed input."""

    match = HEX_RE.match(text or "")
    if not match:
        return None
    digits = match.group(1)
    srgb = np.array(
        [int(digits[i : i + 2], 16) / 255.0 for i in (0, 2, 4)],
        dtype=float,
    )
    return color_math.srgb_to_oklab(srgb)


def is_in_srgb_gamut(oklab: Sequence[float], *, epsilon: float = 1e-4) -> bool:
    srgb = color_math.oklab_to_srgb(np.asarray(oklab, dtype=float))
    return bool(color_math.in_srgb_gamut(srgb, epsilon=epsilon))


class _SwatchWidget(QtWidgets.QFrame):
    """Solid colour swatch with a thin border (Krita-style)."""

    clicked = QtCore.pyqtSignal()

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.Box)
        self.setLineWidth(1)
        self.setFixedHeight(28)
        self.setMinimumWidth(48)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self._colour = QtGui.QColor(0, 0, 0)
        self._enabled_click = False

    def set_colour(self, oklab: Sequence[float] | None) -> None:
        if oklab is None:
            self._colour = QtGui.QColor(0, 0, 0, 0)
        else:
            r, g, b = (int(round(float(c) * 255.0)) for c in color_math.clip_srgb(color_math.oklab_to_srgb(np.asarray(oklab, dtype=float))))
            self._colour = QtGui.QColor(r, g, b)
        self.update()

    def set_clickable(self, enabled: bool) -> None:
        self._enabled_click = bool(enabled)
        self.setCursor(QtCore.Qt.PointingHandCursor if enabled else QtCore.Qt.ArrowCursor)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        painter.fillRect(self.rect().adjusted(1, 1, -1, -1), self._colour)
        painter.end()
        super().paintEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if self._enabled_click and event.button() == QtCore.Qt.LeftButton and self.rect().contains(event.pos()):
            self.clicked.emit()
        super().mouseReleaseEvent(event)


class _GradientSlider(QtWidgets.QSlider):
    """Horizontal slider whose groove is replaced by an OKLCh axis gradient."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(1000)
        self.setSingleStep(1)
        self.setPageStep(10)
        self.setFixedHeight(20)
        self._track_image: QtGui.QImage | None = None
        self._track_buffer: np.ndarray | None = None
        self._track_cache_key: tuple | None = None

    def set_track(self, rgba: np.ndarray) -> None:
        self._track_buffer = rgba
        bytes_per_line = int(rgba.strides[0])
        self._track_image = QtGui.QImage(
            rgba.data,
            rgba.shape[1],
            rgba.shape[0],
            bytes_per_line,
            QtGui.QImage.Format_RGBA8888,
        )
        self.update()

    def cache_key(self) -> tuple | None:
        return self._track_cache_key

    def set_cache_key(self, key: tuple | None) -> None:
        self._track_cache_key = key

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        track_rect = self.rect().adjusted(0, 4, 0, -4)
        if self._track_image is not None:
            painter.drawImage(track_rect, self._track_image)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(track_rect)

        # Handle: a vertical bar with dark halo + light core, mirroring the
        # disk indicator style so the picker reads as one widget family.
        fraction = (self.value() - self.minimum()) / max(1, self.maximum() - self.minimum())
        x = track_rect.left() + int(round(fraction * (track_rect.width() - 1)))
        handle_rect = QtCore.QRect(x - 2, self.rect().top(), 4, self.rect().height())
        painter.fillRect(handle_rect, QtGui.QColor(0, 0, 0, 200))
        painter.fillRect(handle_rect.adjusted(1, 1, -1, -1), QtGui.QColor(255, 255, 255, 230))
        painter.end()


class _AxisRow(QtWidgets.QWidget):
    """One row: label, gradient slider, numeric spinbox."""

    valueChanged = QtCore.pyqtSignal(float, bool)  # (value, committed)

    def __init__(
        self,
        label: str,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._minimum = float(minimum)
        self._maximum = float(maximum)
        self._step = float(step)
        self._decimals = int(decimals)
        self._syncing = False

        label_widget = QtWidgets.QLabel(label, self)
        label_widget.setFixedWidth(14)
        label_widget.setAlignment(QtCore.Qt.AlignCenter)

        self.slider = _GradientSlider(self)
        self.spin = QtWidgets.QDoubleSpinBox(self)
        self.spin.setRange(minimum, maximum)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(72)
        self.spin.setAlignment(QtCore.Qt.AlignRight)

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(label_widget)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.spin)

        self.slider.valueChanged.connect(self._on_slider_changed)
        self.slider.sliderReleased.connect(self._on_slider_released)
        self.spin.editingFinished.connect(self._on_spin_committed)
        self.spin.valueChanged.connect(self._on_spin_value_changed)

    def set_value(self, value: float) -> None:
        self._syncing = True
        try:
            clamped = float(np.clip(value, self._minimum, self._maximum))
            self.spin.setValue(clamped)
            self.slider.setValue(self._value_to_slider(clamped))
        finally:
            self._syncing = False

    def value(self) -> float:
        return float(self.spin.value())

    def _value_to_slider(self, value: float) -> int:
        if self._maximum <= self._minimum:
            return 0
        fraction = (value - self._minimum) / (self._maximum - self._minimum)
        return int(round(fraction * self.slider.maximum()))

    def _slider_to_value(self, position: int) -> float:
        fraction = position / max(1, self.slider.maximum())
        return self._minimum + fraction * (self._maximum - self._minimum)

    def _on_slider_changed(self, position: int) -> None:
        if self._syncing:
            return
        value = self._slider_to_value(position)
        self._syncing = True
        try:
            self.spin.setValue(value)
        finally:
            self._syncing = False
        committed = not self.slider.isSliderDown()
        self.valueChanged.emit(value, committed)

    def _on_slider_released(self) -> None:
        self.valueChanged.emit(self.value(), True)

    def _on_spin_value_changed(self, value: float) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self.slider.setValue(self._value_to_slider(value))
        finally:
            self._syncing = False
        self.valueChanged.emit(value, False)

    def _on_spin_committed(self) -> None:
        if self._syncing:
            return
        self.valueChanged.emit(self.value(), True)


class ReadoutPanel(QtWidgets.QWidget):
    """Swatches + L/C/H sliders + hex readout.

    Emits :attr:`previewed` while a slider is being dragged or a spinbox/hex
    field is being edited; emits :attr:`committed` on slider release, spin
    Enter/blur, hex Enter, and previous-swatch click.
    """

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_oklab: np.ndarray | None = None
        self._previous_oklab: np.ndarray | None = None
        self._syncing = False

        self._previous_swatch = _SwatchWidget(self)
        self._current_swatch = _SwatchWidget(self)
        self._previous_swatch.set_clickable(True)
        self._previous_swatch.clicked.connect(self._on_previous_clicked)

        self._row_l = _AxisRow("L", 0.0, 1.0, _STEP_L, 3, self)
        self._row_c = _AxisRow("C", 0.0, color_math.SRGB_MAX_CHROMA, _STEP_C, 3, self)
        self._row_h = _AxisRow("H", 0.0, 360.0, _STEP_H, 1, self)

        self._row_l.valueChanged.connect(self._on_l_changed)
        self._row_c.valueChanged.connect(self._on_c_changed)
        self._row_h.valueChanged.connect(self._on_h_changed)

        self._hex_field = QtWidgets.QLineEdit(self)
        self._hex_field.setMaxLength(7)
        self._hex_field.setFixedWidth(96)
        hex_font = self._hex_field.font()
        hex_font.setStyleHint(QtGui.QFont.Monospace)
        hex_font.setFamily("monospace")
        self._hex_field.setFont(hex_font)
        # editingFinished covers Enter and focus-out; returnPressed would
        # double-fire on Enter and clobber the revert target.
        self._hex_field.editingFinished.connect(self._on_hex_committed)

        self._gamut_warning = QtWidgets.QLabel("out of gamut", self)
        self._gamut_warning.setObjectName("oklab-gamut-warning")
        warn_font = self._gamut_warning.font()
        warn_font.setBold(True)
        self._gamut_warning.setFont(warn_font)
        self._gamut_warning.setStyleSheet("color: #c0392b;")
        self._gamut_warning.setVisible(False)

        self._build_layout()
        # Initial slider tracks at a sensible default so the panel paints
        # something before the first colour arrives.
        self.set_current_colour(np.array([0.5, 0.0, 0.0], dtype=float))
        self._previous_oklab = None
        self._previous_swatch.set_colour(None)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        swatch_row = QtWidgets.QHBoxLayout()
        swatch_row.setSpacing(0)
        swatch_row.addWidget(self._previous_swatch, 1)
        swatch_row.addWidget(self._current_swatch, 1)
        layout.addLayout(swatch_row)

        layout.addWidget(self._row_l)
        layout.addWidget(self._row_c)
        layout.addWidget(self._row_h)

        hex_row = QtWidgets.QHBoxLayout()
        hex_label = QtWidgets.QLabel("Hex", self)
        hex_label.setFixedWidth(28)
        hex_row.addWidget(hex_label)
        hex_row.addWidget(self._hex_field)
        hex_row.addWidget(self._gamut_warning, 1)
        layout.addLayout(hex_row)

    # -- Public API ----------------------------------------------------

    def set_current_colour(
        self, oklab: Sequence[float] | None, *, committed: bool = True
    ) -> None:
        """Update sliders, swatch, and hex without emitting signals.

        When ``committed`` is False the display updates but the previous
        swatch is left alone — preview samples (mid-drag) must not clobber
        the revert target.
        """
        if oklab is None:
            return
        colour = np.asarray(oklab, dtype=float).copy()
        if committed and self._current_oklab is not None:
            self._previous_oklab = self._current_oklab.copy()
            self._previous_swatch.set_colour(self._previous_oklab)
        self._current_oklab = colour
        self._sync_widgets_to_colour(colour)

    def set_previous_colour(self, oklab: Sequence[float] | None) -> None:
        """Seed the previous-swatch directly (e.g. from initial Krita FG)."""
        if oklab is None:
            self._previous_oklab = None
            self._previous_swatch.set_colour(None)
            return
        self._previous_oklab = np.asarray(oklab, dtype=float).copy()
        self._previous_swatch.set_colour(self._previous_oklab)

    # -- Internal sync -------------------------------------------------

    def _sync_widgets_to_colour(self, oklab: np.ndarray) -> None:
        self._syncing = True
        try:
            l, c, h = color_math.oklab_to_oklch(oklab)
            self._row_l.set_value(float(l))
            self._row_c.set_value(float(c))
            self._row_h.set_value(math.degrees(float(h) % math.tau))
            self._current_swatch.set_colour(oklab)
            hex_text = oklab_to_hex(oklab)
            if self._hex_field.text().lower() != hex_text:
                self._hex_field.setText(hex_text)
            self._gamut_warning.setVisible(not is_in_srgb_gamut(oklab))
            self._refresh_tracks(float(l), float(c), float(h))
        finally:
            self._syncing = False

    def _refresh_tracks(self, lightness: float, chroma: float, hue: float) -> None:
        # Track widths can be 0 before the widget is laid out; skip then and
        # rely on the next sync (showEvent / resize) to populate.
        for axis, row, fixed in (
            (renderers.AXIS_L, self._row_l, (chroma, hue)),
            (renderers.AXIS_C, self._row_c, (lightness, hue)),
            (renderers.AXIS_H, self._row_h, (lightness, chroma)),
        ):
            slider = row.slider
            width = max(2, slider.width())
            height = max(2, slider.height() - 8)
            key = (axis, round(fixed[0], 4), round(fixed[1], 4), width, height)
            if slider.cache_key() == key:
                continue
            rgba = renderers.render_axis_track(
                axis, fixed, color_math.SRGB_MAX_CHROMA, (width, height)
            )
            slider.set_track(rgba)
            slider.set_cache_key(key)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self._current_oklab is not None:
            l, c, h = color_math.oklab_to_oklch(self._current_oklab)
            self._refresh_tracks(float(l), float(c), float(h))

    # -- Slot wiring ---------------------------------------------------

    def _current_lch(self) -> tuple[float, float, float]:
        if self._current_oklab is None:
            return 0.5, 0.0, 0.0
        l, c, h = color_math.oklab_to_oklch(self._current_oklab)
        return float(l), float(c), float(h)

    def _emit_from_lch(self, lightness: float, chroma: float, hue_rad: float, committed: bool) -> None:
        if self._syncing:
            return
        oklab = color_math.oklch_to_oklab([lightness, chroma, hue_rad])
        # Update internal state so the previous-swatch tracks the last colour
        # the user actually committed (not every intermediate preview).
        if committed:
            if self._current_oklab is not None:
                self._previous_oklab = self._current_oklab.copy()
                self._previous_swatch.set_colour(self._previous_oklab)
            self._current_oklab = oklab.copy()
        # Reflect the new colour in the swatch + hex + gamut warning + the
        # other two slider tracks immediately, without re-emitting.
        self._syncing = True
        try:
            self._current_swatch.set_colour(oklab)
            hex_text = oklab_to_hex(oklab)
            if self._hex_field.text().lower() != hex_text:
                self._hex_field.setText(hex_text)
            self._gamut_warning.setVisible(not is_in_srgb_gamut(oklab))
            self._refresh_tracks(lightness, chroma, hue_rad)
        finally:
            self._syncing = False
        (self.committed if committed else self.previewed).emit(oklab)

    def _on_l_changed(self, value: float, committed: bool) -> None:
        _, c, h = self._current_lch()
        self._emit_from_lch(value, c, h, committed)

    def _on_c_changed(self, value: float, committed: bool) -> None:
        l, _, h = self._current_lch()
        self._emit_from_lch(l, value, h, committed)

    def _on_h_changed(self, value_degrees: float, committed: bool) -> None:
        l, c, _ = self._current_lch()
        self._emit_from_lch(l, c, math.radians(value_degrees) % math.tau, committed)

    def _on_hex_committed(self) -> None:
        if self._syncing:
            return
        oklab = hex_to_oklab(self._hex_field.text())
        if oklab is None:
            # Restore the field to the current colour on malformed input.
            if self._current_oklab is not None:
                self._hex_field.setText(oklab_to_hex(self._current_oklab))
            return
        l, c, h = color_math.oklab_to_oklch(oklab)
        self._emit_from_lch(float(l), float(c), float(h), True)

    def _on_previous_clicked(self) -> None:
        if self._previous_oklab is None:
            return
        l, c, h = color_math.oklab_to_oklch(self._previous_oklab)
        self._emit_from_lch(float(l), float(c), float(h), True)

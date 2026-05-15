"""Expanded readout panel: unified swatch and L/C/H gradient sliders.

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

# Contrast-aware paint colours (dark vs light) for handle borders and overlay
# text drawn on top of arbitrary gradient pixels.
_DARK_INK = QtGui.QColor("#1e1e1e")
_LIGHT_INK = QtGui.QColor("#f2f2f2")

_HANDLE_WIDTH = 10
_HANDLE_BORDER = 2

_SWATCH_HEIGHT = 48
_CORNER_BUTTON_SIZE = 20

# Minimum chroma used when *rendering* the H slider rail. The hue track sweeps
# all hues at the current (L, C); for near-neutral colours every column
# collapses to the same grey, which makes the rail unreadable as a hue picker.
# We floor the rendering chroma to keep the rail colourful while the OOG
# checker continues to reflect the actual selected chroma.
_H_TRACK_CHROMA_FLOOR = 0.06


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


def _perceived_luminance(r: int, g: int, b: int) -> float:
    """Simple Rec.709 luma on 0-255 sRGB bytes; good enough for ink choice."""

    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def _ink_for(r: int, g: int, b: int) -> QtGui.QColor:
    return _DARK_INK if _perceived_luminance(r, g, b) > 0.55 else _LIGHT_INK


class _GradientSlider(QtWidgets.QSlider):
    """Horizontal slider with a custom-painted gradient track and hollow handle.

    Replaces Qt's default groove+handle painting because the track is already a
    cached RGBA image and the handle is a contrast-aware hollow rectangle that
    must reveal the underlying gradient (including the 4 px checkerboard for
    out-of-gamut regions). We subclass ``QSlider`` and override ``paintEvent``
    instead of routing through ``QProxyStyle`` because the renderer pipeline
    already gives us a cached RGBA track.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Horizontal, parent)
        self.setMinimum(0)
        self.setMaximum(1000)
        self.setSingleStep(1)
        self.setPageStep(10)
        # Match the spinbox sizeHint so slider + spinbox align vertically.
        probe = QtWidgets.QSpinBox()
        self.setFixedHeight(max(20, probe.sizeHint().height()))
        self._track_image: QtGui.QImage | None = None
        self._track_buffer: np.ndarray | None = None
        self._track_cache_key: tuple | None = None
        self._fallback_colour: QtGui.QColor | None = None

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

    def set_fallback_colour(self, colour: QtGui.QColor | None) -> None:
        # The fallback colour fills the inside of the hollow handle so it
        # always shows the clipped sRGB colour the user will actually paint
        # with. When the handle sits over an in-gamut track pixel the fill
        # matches the track underneath; over the OOG checker it shows the
        # solid clipped colour, making the fallback visible.
        if colour is None:
            self._fallback_colour = None
        else:
            self._fallback_colour = QtGui.QColor(colour)
        self.update()

    def _track_rect(self) -> QtCore.QRect:
        # Reserve a little horizontal padding so the handle never paints past
        # the slider edges; mapping below stays linear inside that band.
        pad = _HANDLE_WIDTH // 2
        return self.rect().adjusted(pad, 2, -pad, -2)

    def _handle_x_center(self, track_rect: QtCore.QRect) -> int:
        rng = max(1, self.maximum() - self.minimum())
        fraction = (self.value() - self.minimum()) / rng
        return track_rect.left() + int(round(fraction * (track_rect.width() - 1)))

    def _border_ink(self, x_center: int, track_rect: QtCore.QRect) -> QtGui.QColor:
        if self._track_buffer is None:
            return _DARK_INK
        buf = self._track_buffer
        # Map handle x in widget coords to a column in the cached track buffer.
        rel = (x_center - track_rect.left()) / max(1, track_rect.width() - 1)
        col = int(round(np.clip(rel, 0.0, 1.0) * (buf.shape[1] - 1)))
        row = buf.shape[0] // 2
        r, g, b = int(buf[row, col, 0]), int(buf[row, col, 1]), int(buf[row, col, 2])
        return _ink_for(r, g, b)

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        track_rect = self._track_rect()
        if self._track_image is not None:
            painter.drawImage(track_rect, self._track_image)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(track_rect)

        x = self._handle_x_center(track_rect)
        handle_rect = QtCore.QRect(
            x - _HANDLE_WIDTH // 2,
            self.rect().top(),
            _HANDLE_WIDTH,
            self.rect().height() - 1,
        )
        ink = self._border_ink(x, track_rect)
        if self._fallback_colour is not None:
            # Fill inside the border so OOG handles show a solid sample of the
            # clipped colour over the checker.
            inner = QtCore.QRectF(handle_rect).adjusted(
                _HANDLE_BORDER, _HANDLE_BORDER, -_HANDLE_BORDER, -_HANDLE_BORDER
            )
            if inner.width() > 0 and inner.height() > 0:
                painter.fillRect(inner, self._fallback_colour)
        pen = QtGui.QPen(ink, _HANDLE_BORDER)
        pen.setJoinStyle(QtCore.Qt.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(QtCore.Qt.NoBrush)
        # Inset by half the pen width so the stroke stays inside handle_rect.
        inset = _HANDLE_BORDER / 2
        painter.drawRect(QtCore.QRectF(handle_rect).adjusted(inset, inset, -inset, -inset))
        painter.end()

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton:
            super().mousePressEvent(event)
            return
        # Use the full widget height as the hit target, matching native
        # sliders while still mapping horizontally through the visible track.
        self.setSliderDown(True)
        self.setValue(self._value_at_x(event.x()))
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if not self.isSliderDown():
            super().mouseMoveEvent(event)
            return
        self.setValue(self._value_at_x(event.x()))
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() != QtCore.Qt.LeftButton or not self.isSliderDown():
            super().mouseReleaseEvent(event)
            return
        self.setValue(self._value_at_x(event.x()))
        self.setSliderDown(False)
        event.accept()

    def _value_at_x(self, x: int) -> int:
        track_rect = self._track_rect()
        fraction = (x - track_rect.left()) / max(1, track_rect.width() - 1)
        fraction = float(np.clip(fraction, 0.0, 1.0))
        return self.minimum() + int(round(fraction * (self.maximum() - self.minimum())))


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
        layout.setAlignment(QtCore.Qt.AlignVCenter)
        layout.addWidget(label_widget, 0, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.slider, 1, QtCore.Qt.AlignVCenter)
        layout.addWidget(self.spin, 0, QtCore.Qt.AlignVCenter)

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


class _UnifiedSwatch(QtWidgets.QWidget):
    """Big colour swatch with overlaid hex text, corner revert button, and
    out-of-gamut indicator.

    The widget paints the fill itself; child widgets (hex line edit, revert
    button, OOG label) are positioned absolutely in :meth:`resizeEvent` so the
    overlay controls can sit *on top* of the colour fill without an intermediate
    layout.
    """

    hex_committed = QtCore.pyqtSignal(str)
    revert_clicked = QtCore.pyqtSignal()
    _INK_STYLES: dict[str, tuple[str, str, str]] = {}

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(_SWATCH_HEIGHT)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        self.setMinimumWidth(48)
        self._colour = QtGui.QColor(0, 0, 0)
        self._hex_text = "#000000"
        self._oog_visible = False

        self._oog_label = QtWidgets.QLabel("⚠", self)
        oog_font = self._oog_label.font()
        oog_font.setBold(True)
        oog_font.setPointSizeF(oog_font.pointSizeF() + 1.0)
        self._oog_label.setFont(oog_font)
        self._oog_label.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self._oog_label.setToolTip("Out of sRGB gamut")
        self._oog_label.setVisible(False)

        self._hex_edit = QtWidgets.QLineEdit(self)
        self._hex_edit.setMaxLength(7)
        self._hex_edit.setAlignment(QtCore.Qt.AlignCenter)
        hex_font = self._hex_edit.font()
        hex_font.setStyleHint(QtGui.QFont.Monospace)
        hex_font.setFamily("monospace")
        hex_font.setPointSizeF(hex_font.pointSizeF() + 2.0)
        hex_font.setBold(True)
        self._hex_edit.setFont(hex_font)
        self._hex_edit.setFrame(False)
        self._hex_edit.setStyleSheet("QLineEdit { background: transparent; border: none; }")
        self._hex_edit.setReadOnly(True)
        self._hex_edit.setCursor(QtCore.Qt.IBeamCursor)
        self._hex_edit.installEventFilter(self)
        self._hex_edit.editingFinished.connect(self._on_hex_finished)
        self._editing = False
        self._edit_start_hex = self._hex_text
        self._suppress_finish = False
        self._ink_name: str | None = None

        self._revert_button = QtWidgets.QToolButton(self)
        self._revert_button.setText("↶")
        self._revert_button.setFixedSize(_CORNER_BUTTON_SIZE, _CORNER_BUTTON_SIZE)
        self._revert_button.setCursor(QtCore.Qt.PointingHandCursor)
        self._revert_button.setAutoRaise(True)
        self._revert_button.setEnabled(False)
        self._revert_button.setToolTip("No previous colour")
        self._revert_button.clicked.connect(self.revert_clicked.emit)

    # -- Public state --------------------------------------------------

    def set_colour(self, oklab: Sequence[float] | None) -> None:
        if oklab is None:
            self._colour = QtGui.QColor(0, 0, 0, 0)
            self._hex_text = "#000000"
        else:
            arr = np.asarray(oklab, dtype=float)
            r, g, b = (
                int(round(float(c) * 255.0))
                for c in color_math.clip_srgb(color_math.oklab_to_srgb(arr))
            )
            self._colour = QtGui.QColor(r, g, b)
            self._hex_text = f"#{r:02x}{g:02x}{b:02x}"
        if not self._editing and self._hex_edit.text().lower() != self._hex_text:
            self._suppress_finish = True
            try:
                self._hex_edit.setText(self._hex_text)
            finally:
                self._suppress_finish = False
        self._apply_ink_styles()
        self.update()

    def set_oog_visible(self, visible: bool) -> None:
        self._oog_visible = bool(visible)
        self._oog_label.setVisible(self._oog_visible)
        self._apply_ink_styles()

    def set_revert_target(self, hex_text: str | None) -> None:
        if hex_text is None:
            self._revert_button.setEnabled(False)
            self._revert_button.setToolTip("No previous colour")
            return
        self._revert_button.setEnabled(True)
        tip = (
            f"Revert to <b>{hex_text}</b> "
            f"<span style='background:{hex_text};'>&nbsp;&nbsp;&nbsp;&nbsp;</span>"
        )
        self._revert_button.setToolTip(tip)

    @property
    def hex_text(self) -> str:
        return self._hex_text

    # -- Painting / interactions --------------------------------------

    def _apply_ink_styles(self) -> None:
        r, g, b = self._colour.red(), self._colour.green(), self._colour.blue()
        ink = _ink_for(r, g, b)
        ink_name = ink.name()
        if ink_name == self._ink_name:
            return
        self._ink_name = ink_name
        hex_style, oog_style, revert_style = self._styles_for_ink(ink_name)
        self._hex_edit.setStyleSheet(hex_style)
        # Match the OOG icon and the revert glyph to the ink colour.
        self._oog_label.setStyleSheet(oog_style)
        self._revert_button.setStyleSheet(revert_style)

    @classmethod
    def _styles_for_ink(cls, ink_name: str) -> tuple[str, str, str]:
        styles = cls._INK_STYLES.get(ink_name)
        if styles is not None:
            return styles
        styles = (
            f"QLineEdit {{ background: transparent; border: none; color: {ink_name}; }}",
            f"color: {ink_name}; background: transparent;",
            f"QToolButton {{ color: {ink_name}; background: transparent; border: none; }}"
            f"QToolButton:hover {{ background: rgba(127,127,127,80); border-radius: 3px; }}"
            f"QToolButton:disabled {{ color: rgba(127,127,127,160); }}",
        )
        cls._INK_STYLES[ink_name] = styles
        return styles

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:  # type: ignore[override]
        painter = QtGui.QPainter(self)
        rect = self.rect().adjusted(0, 0, -1, -1)
        painter.fillRect(rect, self._colour)
        painter.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0, 120), 1))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.drawRect(rect)
        painter.end()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        margin = 4
        # OOG indicator top-left.
        self._oog_label.adjustSize()
        self._oog_label.move(margin, margin)
        # Revert button top-right.
        self._revert_button.move(
            self.width() - _CORNER_BUTTON_SIZE - margin, margin
        )
        # Hex edit centred horizontally, vertically centred in the swatch,
        # inset so it doesn't overlap the corner controls.
        edit_height = self._hex_edit.sizeHint().height()
        side_inset = _CORNER_BUTTON_SIZE + margin * 2
        self._hex_edit.setGeometry(
            side_inset,
            (self.height() - edit_height) // 2,
            max(40, self.width() - side_inset * 2),
            edit_height,
        )

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == QtCore.Qt.LeftButton and self._hex_edit.geometry().contains(event.pos()):
            self._enter_edit_mode()
            return
        super().mousePressEvent(event)

    def _enter_edit_mode(self) -> None:
        if self._editing:
            return
        self._editing = True
        self._edit_start_hex = self._hex_text
        self._hex_edit.setReadOnly(False)
        self._hex_edit.setFocus(QtCore.Qt.MouseFocusReason)
        self._hex_edit.selectAll()

    def _leave_edit_mode(self) -> None:
        self._editing = False
        self._hex_edit.setReadOnly(True)
        # Reset to the canonical text in case the user typed garbage.
        self._suppress_finish = True
        try:
            self._hex_edit.setText(self._hex_text)
        finally:
            self._suppress_finish = False

    def _on_hex_finished(self) -> None:
        if self._suppress_finish or not self._editing:
            return
        text = self._hex_edit.text()
        self._editing = False
        self._hex_edit.setReadOnly(True)
        if text.strip().lower() == self._edit_start_hex.lower():
            return
        self.hex_committed.emit(text)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._hex_edit and event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_Escape:
                self._leave_edit_mode()
                self.setFocus(QtCore.Qt.OtherFocusReason)
                return True
        return super().eventFilter(obj, event)


class ReadoutPanel(QtWidgets.QWidget):
    """Unified swatch + L/C/H gradient sliders.

    Emits :attr:`previewed` while a slider is being dragged or a spinbox/hex
    field is being edited; emits :attr:`committed` on slider release, spin
    Enter/blur, hex Enter, and revert-button click.
    """

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_oklab: np.ndarray | None = None
        self._previous_oklab: np.ndarray | None = None
        self._syncing = False

        self._swatch = _UnifiedSwatch(self)
        self._swatch.hex_committed.connect(self._on_hex_committed)
        self._swatch.revert_clicked.connect(self._on_previous_clicked)

        self._row_l = _AxisRow("L", 0.0, 1.0, _STEP_L, 3, self)
        self._row_c = _AxisRow("C", 0.0, color_math.SRGB_MAX_CHROMA, _STEP_C, 3, self)
        self._row_h = _AxisRow("H", 0.0, 360.0, _STEP_H, 1, self)

        self._row_l.valueChanged.connect(self._on_l_changed)
        self._row_c.valueChanged.connect(self._on_c_changed)
        self._row_h.valueChanged.connect(self._on_h_changed)

        self._build_layout()
        # Initial slider tracks at a sensible default so the panel paints
        # something before the first colour arrives.
        self.set_current_colour(np.array([0.5, 0.0, 0.0], dtype=float))
        self._previous_oklab = None
        self._swatch.set_revert_target(None)

    def _build_layout(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        layout.addWidget(self._swatch)
        layout.addWidget(self._row_l)
        layout.addWidget(self._row_c)
        layout.addWidget(self._row_h)

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
        if (
            committed
            and self._current_oklab is not None
            and not np.allclose(colour, self._current_oklab, atol=1e-6)
        ):
            self._previous_oklab = self._current_oklab.copy()
            self._swatch.set_revert_target(oklab_to_hex(self._previous_oklab))
        self._current_oklab = colour
        self._sync_widgets_to_colour(colour)

    def set_previous_colour(self, oklab: Sequence[float] | None) -> None:
        """Seed the revert target directly (e.g. from initial Krita FG)."""
        if oklab is None:
            self._previous_oklab = None
            self._swatch.set_revert_target(None)
            return
        self._previous_oklab = np.asarray(oklab, dtype=float).copy()
        self._swatch.set_revert_target(oklab_to_hex(self._previous_oklab))

    # -- Internal sync -------------------------------------------------

    def _sync_widgets_to_colour(self, oklab: np.ndarray) -> None:
        self._syncing = True
        try:
            l, c, h = color_math.oklab_to_oklch(oklab)
            self._row_l.set_value(float(l))
            self._row_c.set_value(float(c))
            self._row_h.set_value(math.degrees(float(h) % math.tau))
            self._swatch.set_colour(oklab)
            self._swatch.set_oog_visible(not is_in_srgb_gamut(oklab))
            self._refresh_tracks(float(l), float(c), float(h))
            self._refresh_handle_fallback(oklab)
        finally:
            self._syncing = False

    def _refresh_handle_fallback(self, oklab: np.ndarray) -> None:
        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(np.asarray(oklab, dtype=float)))
        r, g, b = (int(round(float(c) * 255.0)) for c in srgb)
        colour = QtGui.QColor(r, g, b)
        for row in (self._row_l, self._row_c, self._row_h):
            row.slider.set_fallback_colour(colour)

    def _refresh_tracks(self, lightness: float, chroma: float, hue: float) -> None:
        # Track widths can be 0 before the widget is laid out; skip then and
        # rely on the next sync (showEvent / resize) to populate.
        for axis, row, fixed in (
            (renderers.AXIS_L, self._row_l, (chroma, hue)),
            (renderers.AXIS_C, self._row_c, (lightness, hue)),
            (renderers.AXIS_H, self._row_h, (lightness, chroma)),
        ):
            slider = row.slider
            width = max(2, slider.width() - _HANDLE_WIDTH)
            height = max(2, slider.height() - 4)
            chroma_floor = _H_TRACK_CHROMA_FLOOR if axis == renderers.AXIS_H else 0.0
            key = (
                axis,
                round(fixed[0], 4),
                round(fixed[1], 4),
                width,
                height,
                chroma_floor,
            )
            if slider.cache_key() == key:
                continue
            rgba = renderers.render_axis_track(
                axis,
                fixed,
                color_math.SRGB_MAX_CHROMA,
                (width, height),
                hue_chroma_floor=chroma_floor,
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
        # Update internal state so the revert target tracks the last colour
        # the user actually committed (not every intermediate preview).
        if committed:
            if self._current_oklab is not None:
                self._previous_oklab = self._current_oklab.copy()
                self._swatch.set_revert_target(oklab_to_hex(self._previous_oklab))
            self._current_oklab = oklab.copy()
        # Reflect the new colour in the swatch + hex + gamut indicator + the
        # other two slider tracks immediately, without re-emitting.
        self._syncing = True
        try:
            self._swatch.set_colour(oklab)
            self._swatch.set_oog_visible(not is_in_srgb_gamut(oklab))
            self._refresh_tracks(lightness, chroma, hue_rad)
            self._refresh_handle_fallback(oklab)
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

    def _on_hex_committed(self, text: str) -> None:
        if self._syncing:
            return
        oklab = hex_to_oklab(text)
        if oklab is None:
            # Restore the swatch to the current colour on malformed input.
            if self._current_oklab is not None:
                self._swatch.set_colour(self._current_oklab)
            return
        l, c, h = color_math.oklab_to_oklch(oklab)
        self._emit_from_lch(float(l), float(c), float(h), True)

    def _on_previous_clicked(self) -> None:
        if self._previous_oklab is None:
            return
        l, c, h = color_math.oklab_to_oklch(self._previous_oklab)
        self._emit_from_lch(float(l), float(c), float(h), True)

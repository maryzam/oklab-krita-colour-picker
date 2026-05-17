"""Qt selector widget backed by pure selector models and NumPy renderers.

The widget is an **explicit** finite state machine (north-star §3). Every
interaction transition is named and tested instead of emerging from a tangle
of booleans. The states are:

- ``IDLE``     — rendering an externally pushed colour; no anchor. The
  indicator is a pure function of ``(colour, model, size)``.
- ``DRAGGING`` — pointer held; emitting ``previewed``. Anchor = cursor pixel.
- ``KEYBOARD`` — arrow/page navigation in flight; commit pending.
  Anchor = target pixel.
- ``PINNED``   — post-commit; holds the committed colour at the anchor pixel
  until something external supersedes it.

Inbound colours (``show_colour``) are arbitrated **locally** by the state
machine (INV-3): only ``PINNED`` swallows an echo, and only when it
quantizes-equal to the pinned colour (INV-4). The dock/controller never
special-case the source.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import renderers
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.selector_models import IndicatorSpec, SelectorModel


IDLE = "IDLE"
DRAGGING = "DRAGGING"
KEYBOARD = "KEYBOARD"
PINNED = "PINNED"

# States that own an anchor pixel and win over inbound broadcasts.
_ANCHORED_STATES = (DRAGGING, KEYBOARD, PINNED)
# States representing a local gesture in flight; inbound colours are ignored.
_IN_FLIGHT_STATES = (DRAGGING, KEYBOARD)


class SelectorWidget(QtWidgets.QWidget):
    """Paint and interact with a selector model via an explicit state machine.

    The widget owns only presentation/interaction state. Picking and indicator
    placement are delegated to the selector model; RGBA pixels come from the
    renderer. It never holds authoritative colour state (INV-5).
    """

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, model: SelectorModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._selected_colour: np.ndarray | None = None
        self._state: str = IDLE
        self._anchor: tuple[float, float] | None = None
        self._colour_before: np.ndarray | None = None
        self._last_valid: np.ndarray | None = None
        self._transition_log: list[str] = [IDLE]
        self._image_cache_key: tuple[SelectorModel, int, int] | None = None
        self._image_cache_buffer: np.ndarray | None = None
        self._image_cache: QtGui.QImage | None = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(32, 32)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # -- State machine surface ----------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._anchor

    @property
    def transition_log(self) -> tuple[str, ...]:
        """Ordered log of states entered (for no-flicker assertions, §4.1)."""

        return tuple(self._transition_log)

    def enter_state(self, name: str, *, anchor: tuple[float, float] | None = None) -> None:
        """Force a state transition. Test/orchestration hook for §3.2.

        Colour state is never mutated here — only interaction state. ``IDLE``
        drops the anchor (INV-1); the other states adopt ``anchor``.
        """

        if name not in (IDLE, DRAGGING, KEYBOARD, PINNED):
            raise ValueError(f"unknown selector state: {name!r}")
        self._set_state(name, anchor=None if name == IDLE else anchor)
        self.update()

    def _set_state(self, name: str, *, anchor: tuple[float, float] | None) -> None:
        self._anchor = None if name == IDLE else anchor
        if name != self._state:
            self._state = name
            self._transition_log.append(name)
        elif anchor is not None:
            # Same state, moved anchor (e.g. DRAGGING follows the cursor).
            self._anchor = anchor

    # -- Colour surface ------------------------------------------------

    @property
    def model(self) -> SelectorModel:
        return self._model

    @property
    def selected_colour(self) -> np.ndarray | None:
        return None if self._selected_colour is None else self._selected_colour.copy()

    def set_model(self, model: SelectorModel) -> None:
        if self._model is model:
            return
        self._model = model
        # Model change is an external reset: no anchor survives it (INV-1).
        self._set_state(IDLE, anchor=None)
        self._colour_before = None
        self._last_valid = None
        self._clear_image_cache()
        self.update()

    def show_colour(self, oklab: Sequence[float] | None, kind: object | None = None) -> None:
        """Absorb an inbound (broadcast) colour per the state machine (§3.2/§3.5).

        ``kind`` is informational only; it is never used to skip a view.
        Absorption is local: only ``PINNED`` swallows an echo, and only when
        the colour quantizes-equal to the pinned colour (INV-3 / INV-4).
        """

        if self._state in _IN_FLIGHT_STATES:
            # An in-flight local gesture wins; the inbound colour is ignored.
            return

        new_colour = _as_oklab(oklab)

        if self._state == PINNED:
            if new_colour is not None and self._quantized_equal(new_colour, self._selected_colour):
                return  # the echo — swallow it, stay PINNED (INV-3)
            # A genuinely different colour supersedes the pin.
            self._set_state(IDLE, anchor=None)

        self._selected_colour = new_colour
        self.update()

    # Backwards-compatible alias used by programmatic/seed callers.
    set_selected_colour = show_colour

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        if self._anchor is not None:
            # Anchored states draw at the anchor regardless of the model's
            # colour→position round-trip (INV-2).
            return self._anchor
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

    # -- Painting ------------------------------------------------------

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setClipRect(event.rect())

        try:
            image = self._selector_image()
        except ValueError:
            painter.end()
            return

        painter.drawImage(0, 0, image)
        self._paint_indicator(painter)
        painter.end()

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        if self._selected_colour is None:
            return

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)

        if self._anchor is not None:
            self._stroke_circle(painter, self._anchor, solid=True)
            return

        indicator = self._model.indicator_for_color(self._selected_colour, _widget_size(self))
        if indicator is None:
            return
        if indicator.snapped is not None and indicator.out_of_gamut:
            # Out of gamut on this slice: solid ring where the colour wants to
            # be, dashed ring at the snapped position the user paints with.
            self._stroke_circle(painter, indicator.desired, solid=True)
            self._stroke_circle(painter, indicator.snapped, solid=False)
            return
        self._stroke_circle(painter, indicator.desired, solid=True)

    # -- Mouse interaction --------------------------------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        # Any non-IDLE state → DRAGGING; this cancels a pending keyboard
        # commit *without* flushing it (last row of §3.2).
        self._colour_before = None if self._selected_colour is None else self._selected_colour.copy()
        self._last_valid = None
        self._set_state(DRAGGING, anchor=(float(event.pos().x()), float(event.pos().y())))
        self._preview_at(event.pos())
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._state != DRAGGING:
            event.ignore()
            return
        self._preview_at(event.pos())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or self._state != DRAGGING:
            event.ignore()
            return
        point = event.pos()
        colour = self._drag_colour_at(point)
        if colour is not None:
            self._selected_colour = colour.copy()
            self._set_state(PINNED, anchor=(float(point.x()), float(point.y())))
            self.update()
            self.committed.emit(colour.copy())
        elif self._last_valid is not None:
            self._selected_colour = self._last_valid.copy()
            self._set_state(PINNED, anchor=self._anchor)
            self.update()
            self.committed.emit(self._last_valid.copy())
        else:
            restored = self._colour_before
            self._selected_colour = restored
            self._set_state(IDLE, anchor=None)
            self.update()
            self.previewed.emit(None if restored is None else restored.copy())
        self._colour_before = None
        self._last_valid = None
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        # The anchor is stored in absolute widget pixels, so it is meaningless
        # under a new size. A resize while PINNED is a model/size change: drop
        # the anchor and let the model place the indicator (§3.2, INV-1).
        if self._state == PINNED:
            self._set_state(IDLE, anchor=None)
        super().resizeEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._flush_keyboard_commit()
        super().focusOutEvent(event)

    # -- Keyboard interaction -----------------------------------------

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if self._state == DRAGGING:
            event.ignore()
            return

        position = self.indicator_position()
        if position is None:
            position = ((self.width() - 1.0) / 2.0, (self.height() - 1.0) / 2.0)

        point = self._keyboard_target_position(position, event)
        if point is None:
            event.ignore()
            return

        colour = self._colour_at(point)
        if colour is None:
            event.ignore()
            return

        self._selected_colour = colour.copy()
        self._set_state(KEYBOARD, anchor=(float(point.x()), float(point.y())))
        self.update()
        self.previewed.emit(colour.copy())
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if not self._is_keyboard_navigation_key(event.key()) or self._state != KEYBOARD:
            event.ignore()
            return

        self._flush_keyboard_commit()
        event.accept()

    def _flush_keyboard_commit(self) -> None:
        if self._state != KEYBOARD:
            return
        colour = self._selected_colour
        self._set_state(PINNED, anchor=self._anchor)
        if colour is not None:
            self.committed.emit(colour.copy())

    # -- Picking helpers ----------------------------------------------

    def _preview_at(self, point: QtCore.QPoint) -> None:
        colour = self._drag_colour_at(point)
        if colour is not None:
            self._selected_colour = colour.copy()
            self._last_valid = self._selected_colour.copy()
            self._set_state(DRAGGING, anchor=(float(point.x()), float(point.y())))
            self.update()
            self.previewed.emit(colour.copy())
            return

        # A drag that began on (or moved onto) an invalid point and has not yet
        # reached any selectable colour keeps cancellation semantics until it
        # does (INV-6 fallback only applies once last_valid exists).
        if self._last_valid is not None:
            return

        self._selected_colour = None
        self.update()
        self.previewed.emit(None)

    def _colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        return self._model.color_at_position((point.x(), point.y()), _widget_size(self))

    def _drag_colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        colour = self._colour_at(point)
        if colour is not None:
            return colour
        if self._last_valid is None:
            return None
        return self._model.snapped_color_at_position((point.x(), point.y()), _widget_size(self))

    def _keyboard_target_position(
        self,
        position: tuple[float, float],
        event: QtGui.QKeyEvent,
    ) -> QtCore.QPoint | None:
        x, y = position
        step = _keyboard_step(self.size(), event.modifiers())
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            return self._nearest_valid_point(position, -step, 0.0)
        if key == QtCore.Qt.Key_Right:
            return self._nearest_valid_point(position, step, 0.0)
        if key == QtCore.Qt.Key_Up:
            return self._nearest_valid_point(position, 0.0, -step)
        if key == QtCore.Qt.Key_Down:
            return self._nearest_valid_point(position, 0.0, step)
        if key == QtCore.Qt.Key_Home:
            return self._nearest_valid_point(position, -x, 0.0)
        if key == QtCore.Qt.Key_End:
            return self._nearest_valid_point(position, self.width() - 1.0 - x, 0.0)
        if key == QtCore.Qt.Key_PageUp:
            return self._nearest_valid_point(position, 0.0, -y)
        if key == QtCore.Qt.Key_PageDown:
            return self._nearest_valid_point(position, 0.0, self.height() - 1.0 - y)
        return None

    def _nearest_valid_point(self, position: tuple[float, float], dx: float, dy: float) -> QtCore.QPoint | None:
        start_x, start_y = position
        steps = max(1, int(max(abs(dx), abs(dy))))
        fractions = np.arange(steps, -1, -1, dtype=float) / steps
        x = np.rint(np.clip(start_x + dx * fractions, 0, self.width() - 1)).astype(float)
        y = np.rint(np.clip(start_y + dy * fractions, 0, self.height() - 1)).astype(float)
        _, valid = self._model.colors_at_positions(x, y, _widget_size(self))
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            index = int(valid_indices[0])
            return QtCore.QPoint(int(x[index]), int(y[index]))
        return None

    def _is_keyboard_navigation_key(self, key: int) -> bool:
        return key in {
            QtCore.Qt.Key_Left,
            QtCore.Qt.Key_Right,
            QtCore.Qt.Key_Up,
            QtCore.Qt.Key_Down,
            QtCore.Qt.Key_Home,
            QtCore.Qt.Key_End,
            QtCore.Qt.Key_PageUp,
            QtCore.Qt.Key_PageDown,
        }

    @staticmethod
    def _quantized_equal(left: np.ndarray, right: np.ndarray | None) -> bool:
        # INV-4: compare under the model/controller Krita quantization, never
        # raw float ==, or PINNED↔IDLE flickers on near-equal echoes.
        if right is None:
            return False
        return bool(
            np.array_equal(
                normalize_oklab_for_krita(left),
                normalize_oklab_for_krita(right),
            )
        )

    # -- Rendering helpers --------------------------------------------

    def _stroke_circle(
        self, painter: QtGui.QPainter, position: tuple[float, float], *, solid: bool
    ) -> None:
        x, y = position
        center = QtCore.QPointF(x, y)
        if solid:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 3.0))
            painter.drawEllipse(center, 5.0, 5.0)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 1.5))
            painter.drawEllipse(center, 5.0, 5.0)
            return
        # Dashed ring: dark halo first, then a white dashed stroke on top.
        halo = QtGui.QPen(QtCore.Qt.black, 3.0)
        halo.setStyle(QtCore.Qt.DashLine)
        halo.setDashPattern([2.0, 2.0])
        painter.setPen(halo)
        painter.drawEllipse(center, 5.0, 5.0)
        dash = QtGui.QPen(QtCore.Qt.white, 1.5)
        dash.setStyle(QtCore.Qt.DashLine)
        dash.setDashPattern([2.0, 2.0])
        painter.setPen(dash)
        painter.drawEllipse(center, 5.0, 5.0)

    def _selector_image(self) -> QtGui.QImage:
        key = (self._model, self.width(), self.height())
        if self._image_cache_key == key and self._image_cache is not None:
            return self._image_cache

        rgba = renderers.render_rgba(self._model, (self.width(), self.height()))
        bytes_per_line = int(rgba.strides[0])
        image = QtGui.QImage(
            rgba.data,
            self.width(),
            self.height(),
            bytes_per_line,
            QtGui.QImage.Format_RGBA8888,
        )
        self._image_cache_key = key
        self._image_cache_buffer = rgba
        self._image_cache = image
        return image

    def _clear_image_cache(self) -> None:
        self._image_cache_key = None
        self._image_cache_buffer = None
        self._image_cache = None


def _widget_size(widget: QtWidgets.QWidget) -> tuple[int, int]:
    return widget.width(), widget.height()


def _as_oklab(oklab: Sequence[float] | None) -> np.ndarray | None:
    if oklab is None:
        return None
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()


def _keyboard_step(size: QtCore.QSize, modifiers: QtCore.Qt.KeyboardModifiers) -> int:
    if modifiers & QtCore.Qt.ShiftModifier:
        return 1
    base = max(1, min(size.width(), size.height()) // 64)
    if modifiers & QtCore.Qt.ControlModifier:
        return max(1, base * 4)
    return base

"""Qt selector widget backed by pure selector models and NumPy renderers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import renderers


class SelectorModel(Protocol):
    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        ...

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> tuple[float, float] | None:
        ...

    def colors_at_positions(self, x, y, size: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        ...


class SelectorWidget(QtWidgets.QWidget):
    """Paint and interact with a selector model.

    The widget owns only presentation state. Picking and indicator placement
    are delegated to the selector model, and RGBA pixels come from the renderer.
    """

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, model: SelectorModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._selected_colour: np.ndarray | None = None
        self._colour_before_drag: np.ndarray | None = None
        self._last_valid_drag_colour: np.ndarray | None = None
        self._last_interaction_position: tuple[float, float] | None = None
        self._image_cache_key: tuple[SelectorModel, int, int] | None = None
        self._image_cache_buffer: np.ndarray | None = None
        self._image_cache: QtGui.QImage | None = None
        self._pressed = False
        self._keyboard_commit_pending = False
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(32, 32)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

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
        self._last_interaction_position = None
        self._clear_image_cache()
        self.update()

    def set_selected_colour(self, oklab: Sequence[float] | None) -> None:
        new_colour = _as_oklab(oklab)
        # The dock loops set_selected_colour back to the source widget on
        # every previewed/committed signal, so we must keep the recorded
        # interaction position whenever it still resolves to the new colour
        # under the current model. Otherwise on achromatic slices the
        # override is cleared between click and repaint and the indicator
        # falls back to the model's hue=0 round-trip.
        if not self._interaction_position_resolves_to(new_colour):
            self._last_interaction_position = None
        self._selected_colour = new_colour
        self.update()

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        override = self._interaction_indicator_position()
        if override is not None:
            return override
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

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

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self._keyboard_commit_pending = False
        self._pressed = True
        self._colour_before_drag = None if self._selected_colour is None else self._selected_colour.copy()
        self._last_valid_drag_colour = None
        self._preview_at(event.pos())
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if not self._pressed:
            event.ignore()
            return
        self._preview_at(event.pos())
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton or not self._pressed:
            event.ignore()
            return
        colour = self._drag_colour_at(event.pos())
        self._pressed = False
        if colour is not None:
            self._selected_colour = colour.copy()
            self._record_interaction_position(event.pos(), colour)
            self.update()
            self.committed.emit(colour.copy())
        elif self._last_valid_drag_colour is not None:
            self._selected_colour = self._last_valid_drag_colour.copy()
            self.update()
            self.committed.emit(self._last_valid_drag_colour.copy())
        else:
            self._selected_colour = self._colour_before_drag
            self._last_interaction_position = None
            self.update()
            self.previewed.emit(None if self._selected_colour is None else self._selected_colour.copy())
        self._colour_before_drag = None
        self._last_valid_drag_colour = None
        event.accept()

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._flush_keyboard_commit()
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
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
        self._record_interaction_position(point, colour)
        self._keyboard_commit_pending = True
        self.update()
        self.previewed.emit(colour.copy())
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if not self._is_keyboard_navigation_key(event.key()) or not self._keyboard_commit_pending:
            event.ignore()
            return

        self._flush_keyboard_commit()
        event.accept()

    def _preview_at(self, point: QtCore.QPoint) -> None:
        colour = self._drag_colour_at(point) if self._pressed else self._colour_at(point)
        if colour is not None:
            self._selected_colour = colour.copy()
            self._record_interaction_position(point, colour)
            if self._pressed:
                self._last_valid_drag_colour = self._selected_colour.copy()
            self.update()
            self.previewed.emit(colour.copy())
            return

        # If a drag began on an invalid point and has not yet reached any
        # selectable colour, keep cancellation semantics until it does.
        if self._pressed and self._last_valid_drag_colour is not None:
            return

        self._selected_colour = None
        self._last_interaction_position = None
        self.update()
        self.previewed.emit(None)

    def _colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        return self._model.color_at_position((point.x(), point.y()), _widget_size(self))

    def _drag_colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        colour = self._colour_at(point)
        if colour is not None:
            return colour
        if self._last_valid_drag_colour is None:
            return None
        return self._snapped_colour_at(point)

    def _snapped_colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        snapper = getattr(self._model, "snapped_color_at_position", None)
        if not callable(snapper):
            return None
        return snapper((point.x(), point.y()), _widget_size(self))

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

    def _flush_keyboard_commit(self) -> None:
        if not self._keyboard_commit_pending:
            return
        self._keyboard_commit_pending = False
        if self._selected_colour is not None:
            self.committed.emit(self._selected_colour.copy())

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        override = self._interaction_indicator_position()
        if override is not None:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.setBrush(QtCore.Qt.NoBrush)
            self._stroke_circle(painter, override, solid=True)
            return

        desired = self._desired_indicator_position()
        snapped = self._snapped_indicator_position()
        if desired is None and snapped is None:
            return

        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)

        if desired is not None and snapped is not None and not _positions_close(desired, snapped):
            # Out of gamut on this slice: solid ring at the desired position
            # (where the stored colour wants to be), dashed ring at the
            # snapped fallback position the user will actually paint with.
            self._stroke_circle(painter, desired, solid=True)
            self._stroke_circle(painter, snapped, solid=False)
            return

        position = desired if desired is not None else snapped
        self._stroke_circle(painter, position, solid=True)

    def _record_interaction_position(self, point: QtCore.QPoint, colour: np.ndarray) -> None:
        # Only trust the click point as the indicator location when the model
        # picks the same colour directly from that point. Drag snaps past the
        # gamut leaf return colours that don't live at the cursor — there we
        # let the model decide where to draw the indicator.
        direct = self._colour_at(point)
        if direct is not None and np.array_equal(direct, colour):
            self._last_interaction_position = (float(point.x()), float(point.y()))
        else:
            self._last_interaction_position = None

    def _interaction_position_resolves_to(self, colour: np.ndarray | None) -> bool:
        if self._last_interaction_position is None or colour is None:
            return False
        x, y = self._last_interaction_position
        direct = self._model.color_at_position((x, y), _widget_size(self))
        return direct is not None and np.array_equal(direct, colour)

    def _interaction_indicator_position(self) -> tuple[float, float] | None:
        # The recorded position is invalidated whenever the colour changes
        # outside of direct user interaction (see set_selected_colour /
        # set_model), so it is safe to use without re-verifying here.
        return self._last_interaction_position

    def _desired_indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        helper = getattr(self._model, "desired_position_for_color", None)
        if callable(helper):
            return helper(self._selected_colour, _widget_size(self))
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

    def _snapped_indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        helper = getattr(self._model, "snapped_position_for_color", None)
        if callable(helper):
            return helper(self._selected_colour, _widget_size(self))
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

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


def _positions_close(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return abs(a[0] - b[0]) <= 0.5 and abs(a[1] - b[1]) <= 0.5


def _keyboard_step(size: QtCore.QSize, modifiers: QtCore.Qt.KeyboardModifiers) -> int:
    if modifiers & QtCore.Qt.ShiftModifier:
        return 1
    base = max(1, min(size.width(), size.height()) // 64)
    if modifiers & QtCore.Qt.ControlModifier:
        return max(1, base * 4)
    return base

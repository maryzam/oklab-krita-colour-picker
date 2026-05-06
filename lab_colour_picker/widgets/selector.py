"""Qt selector widget backed by pure selector models and NumPy renderers."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from lab_colour_picker import renderers


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
        self._image_cache_key: tuple[SelectorModel, int, int] | None = None
        self._image_cache_buffer: np.ndarray | None = None
        self._image_cache: QtGui.QImage | None = None
        self._pressed = False
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
        self._clear_image_cache()
        self.update()

    def set_selected_colour(self, oklab: Sequence[float] | None) -> None:
        self._selected_colour = _as_oklab(oklab)
        self.update()

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        image_area = self.rect()
        if not event.rect().intersects(image_area):
            return

        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
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
        self._pressed = True
        self._colour_before_drag = None if self._selected_colour is None else self._selected_colour.copy()
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
        self._pressed = False
        colour = self._colour_at(event.pos())
        if colour is not None:
            self._selected_colour = colour
            self.update()
            self.committed.emit(colour)
        else:
            self._selected_colour = self._colour_before_drag
            self.update()
        self._colour_before_drag = None
        event.accept()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._pressed:
            self.previewed.emit(None)
        super().leaveEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        position = self.indicator_position()
        if position is None:
            position = ((self.width() - 1.0) / 2.0, (self.height() - 1.0) / 2.0)

        x, y = position
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            x -= 1.0
        elif key == QtCore.Qt.Key_Right:
            x += 1.0
        elif key == QtCore.Qt.Key_Up:
            y -= 1.0
        elif key == QtCore.Qt.Key_Down:
            y += 1.0
        elif key == QtCore.Qt.Key_Home:
            x = 0.0
        elif key == QtCore.Qt.Key_End:
            x = self.width() - 1.0
        else:
            event.ignore()
            return

        point = QtCore.QPoint(_clamp_round(x, 0, self.width() - 1), _clamp_round(y, 0, self.height() - 1))
        colour = self._colour_at(point)
        if colour is None:
            event.ignore()
            return

        self._selected_colour = colour
        self.update()
        self.previewed.emit(colour)
        self.committed.emit(colour)
        event.accept()

    def _preview_at(self, point: QtCore.QPoint) -> None:
        colour = self._colour_at(point)
        self._selected_colour = colour
        self.update()
        self.previewed.emit(colour)

    def _colour_at(self, point: QtCore.QPoint) -> np.ndarray | None:
        return self._model.color_at_position((point.x(), point.y()), _widget_size(self))

    def _paint_indicator(self, painter: QtGui.QPainter) -> None:
        position = self.indicator_position()
        if position is None:
            return

        x, y = position
        outer = QtCore.QPointF(x, y)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtCore.Qt.black, 3.0))
        painter.drawEllipse(outer, 5.0, 5.0)
        painter.setPen(QtGui.QPen(QtCore.Qt.white, 1.5))
        painter.drawEllipse(outer, 5.0, 5.0)

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


def _clamp_round(value: float, lower: int, upper: int) -> int:
    return max(lower, min(upper, round(value)))

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
        self._pressed = False
        self.setMouseTracking(True)
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
        self.update()

    def set_selected_colour(self, oklab: Sequence[float] | None) -> None:
        with _blocked_signals(self):
            self._selected_colour = _as_oklab(oklab)
            self.update()

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

    def paintEvent(self, event: QtGui.QPaintEvent) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setClipRect(event.rect())

        try:
            image = _selector_image(self._model, self.width(), self.height())
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
        self._pressed = True
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
            self.committed.emit(colour.copy())
        event.accept()

    def leaveEvent(self, event: QtCore.QEvent) -> None:
        if self._pressed:
            self.previewed.emit(None)
        super().leaveEvent(event)

    def _preview_at(self, point: QtCore.QPoint) -> None:
        colour = self._colour_at(point)
        self._selected_colour = colour
        self.update()
        self.previewed.emit(None if colour is None else colour.copy())

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


def _selector_image(model: SelectorModel, width: int, height: int) -> QtGui.QImage:
    rgba = np.ascontiguousarray(renderers.render_rgba(model, (width, height)))
    bytes_per_line = int(rgba.strides[0])
    image = QtGui.QImage(rgba.data, width, height, bytes_per_line, QtGui.QImage.Format_RGBA8888)
    return image.copy()


def _widget_size(widget: QtWidgets.QWidget) -> tuple[int, int]:
    return widget.width(), widget.height()


def _as_oklab(oklab: Sequence[float] | None) -> np.ndarray | None:
    if oklab is None:
        return None
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()


class _blocked_signals:
    def __init__(self, obj: QtCore.QObject) -> None:
        self._obj = obj
        self._blocker: QtCore.QSignalBlocker | None = None

    def __enter__(self):
        self._blocker = QtCore.QSignalBlocker(self._obj)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._blocker = None

"""Qt selector widget backed by the pure selector interaction facade."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import renderers, selector_interaction
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.selector_interaction import Indicator, PickResult, Ring, StateKind
from oklab_colour_picker.selector_models import SelectorModel


class SelectorWidget(QtWidgets.QWidget):
    """Paint a selector model and translate Qt events into interaction commands."""

    previewed = QtCore.pyqtSignal(object)
    committed = QtCore.pyqtSignal(object)

    def __init__(self, model: SelectorModel, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._model = model
        self._selected_colour: np.ndarray | None = None
        self._interaction = selector_interaction.SelectorInteraction()
        self._image_cache_key: tuple[SelectorModel, int, int] | None = None
        self._image_cache_buffer: np.ndarray | None = None
        self._image_cache: QtGui.QImage | None = None
        self.setFocusPolicy(QtCore.Qt.StrongFocus)
        self.setMinimumSize(32, 32)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)

    # -- State-machine surface ----------------------------------------

    @property
    def state(self) -> str:
        return self._interaction.state_name

    @property
    def anchor(self) -> tuple[float, float] | None:
        return self._interaction.anchor

    @property
    def transition_log(self) -> tuple[str, ...]:
        return self._interaction.transition_log

    def _force_state_for_test(
        self, kind: StateKind, *, anchor: tuple[float, float] | None = None
    ) -> None:
        self._interaction.force_for_test(kind, colour=self._selected_colour, anchor=anchor)
        self.update()

    def _apply_interaction(self, result: selector_interaction.InteractionResult) -> None:
        self.update()

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
        self._clear_image_cache()
        self._apply_interaction(
            self._interaction.dispatch(self, selector_interaction.Reframe())
        )

    def show_colour(self, oklab: Sequence[float] | None, kind: object | None = None) -> None:
        """Absorb an inbound colour through the state machine (§3.2/§3.5).

        ``kind`` is informational only and never used to skip a view —
        absorption is decided locally by the state object (INV-3).
        """

        self._apply_interaction(
            self._interaction.dispatch(self, selector_interaction.Broadcast(_as_oklab(oklab)))
        )

    # Backwards-compatible alias used by programmatic/seed callers.
    set_selected_colour = show_colour

    def apply_broadcast(
        self,
        oklab: Sequence[float] | None,
        kind: object,
        model_factory,
    ) -> None:
        """Absorb a controller broadcast and adopt the model only if rendered."""

        result = self._interaction.dispatch(
            self, selector_interaction.Broadcast(_as_oklab(oklab))
        )
        if model_factory is not None and result.rendered_broadcast:
            model = model_factory()
            if self._model != model:
                self._model = model
                self._clear_image_cache()
        self._apply_interaction(result)

    def indicator_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        return self._interaction.indicator_position(self)

    # -- Ctx port (used only by the state machine) ---------------------

    @property
    def colour(self) -> np.ndarray | None:
        return self._selected_colour

    def set_colour(self, colour: np.ndarray | None) -> None:
        self._selected_colour = None if colour is None else np.asarray(colour, dtype=float).copy()

    def preview(self, colour: np.ndarray | None) -> None:
        self.previewed.emit(None if colour is None else np.asarray(colour, dtype=float).copy())

    def commit(self, colour: np.ndarray) -> None:
        self.committed.emit(np.asarray(colour, dtype=float).copy())

    def pick(self, point: tuple[float, float]) -> PickResult:
        colour = self.color_at(point)
        if colour is not None:
            return PickResult.exact(colour)
        snapped = self._model.snapped_color_at_position((point[0], point[1]), _widget_size(self))
        if snapped is not None:
            return PickResult.snapped(snapped)
        return PickResult.invalid()

    def color_at(self, point: tuple[float, float]) -> np.ndarray | None:
        return self._model.color_at_position((point[0], point[1]), _widget_size(self))

    @staticmethod
    def quantized_equal(a: np.ndarray | None, b: np.ndarray | None) -> bool:
        # INV-4: compare under Krita 8-bit quantization, never raw float ==.
        if a is None or b is None:
            return False
        return bool(
            np.array_equal(normalize_oklab_for_krita(a), normalize_oklab_for_krita(b))
        )

    def model_indicator(self) -> Indicator:
        if self._selected_colour is None:
            return Indicator.nothing()
        spec = self._model.indicator_for_color(self._selected_colour, _widget_size(self))
        if spec is None:
            return Indicator.nothing()
        rings = [Ring(spec.desired, True)]
        if spec.snapped is not None and spec.out_of_gamut:
            # Out of gamut on this slice: dashed ring at the snapped fallback.
            rings.append(Ring(spec.snapped, False))
        return Indicator(tuple(rings))

    def model_position(self) -> tuple[float, float] | None:
        if self._selected_colour is None:
            return None
        return self._model.position_for_color(self._selected_colour, _widget_size(self))

    # -- Qt event plumbing (input routing only) -----------------------

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        self.setFocus(QtCore.Qt.MouseFocusReason)
        self._apply_interaction(
            self._interaction.dispatch(self, selector_interaction.PointerPress(_point(event)))
        )
        event.accept()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        result = self._interaction.dispatch(
            self, selector_interaction.PointerMove(_point(event))
        )
        self._apply_interaction(result)
        if not result.handled:
            event.ignore()
            return
        event.accept()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            event.ignore()
            return
        result = self._interaction.dispatch(
            self, selector_interaction.PointerRelease(_point(event))
        )
        self._apply_interaction(result)
        if not result.handled:
            event.ignore()
            return
        event.accept()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        self._apply_interaction(
            self._interaction.dispatch(self, selector_interaction.Reframe())
        )
        super().resizeEvent(event)

    def focusOutEvent(self, event: QtGui.QFocusEvent) -> None:
        self._apply_interaction(
            self._interaction.dispatch(self, selector_interaction.FocusOut())
        )
        super().focusOutEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        fallback = ((self.width() - 1.0) / 2.0, (self.height() - 1.0) / 2.0)
        position = self._interaction.navigation_origin(self, fallback)
        if position is None:
            event.ignore()
            return

        target = self._keyboard_target_position(position, event)
        if target is None:
            event.ignore()
            return
        colour = self.color_at((target.x(), target.y()))
        if colour is None:
            event.ignore()
            return

        result = self._interaction.dispatch(
            self,
            selector_interaction.Navigation(
                (float(target.x()), float(target.y())), colour
            ),
        )
        self._apply_interaction(result)
        event.accept()

    def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.isAutoRepeat():
            event.accept()
            return
        if not self._is_keyboard_navigation_key(event.key()):
            event.ignore()
            return
        result = self._interaction.dispatch(self, selector_interaction.KeyRelease())
        self._apply_interaction(result)
        if not result.handled:
            event.ignore()
            return
        event.accept()

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
        indicator = self._interaction.indicator(self)
        if not indicator.rings:
            return
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtCore.Qt.NoBrush)
        for ring in indicator.rings:
            self._stroke_circle(painter, ring.position, solid=ring.solid)

    def _stroke_circle(
        self, painter: QtGui.QPainter, position: tuple[float, float], *, solid: bool
    ) -> None:
        center = QtCore.QPointF(position[0], position[1])
        if solid:
            painter.setPen(QtGui.QPen(QtCore.Qt.black, 3.0))
            painter.drawEllipse(center, 5.0, 5.0)
            painter.setPen(QtGui.QPen(QtCore.Qt.white, 1.5))
            painter.drawEllipse(center, 5.0, 5.0)
            return
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

    # -- Keyboard navigation maths ------------------------------------

    def _keyboard_target_position(
        self, position: tuple[float, float], event: QtGui.QKeyEvent
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

    def _nearest_valid_point(
        self, position: tuple[float, float], dx: float, dy: float
    ) -> QtCore.QPoint | None:
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


def _widget_size(widget: QtWidgets.QWidget) -> tuple[int, int]:
    return widget.width(), widget.height()


def _point(event: QtGui.QMouseEvent) -> tuple[float, float]:
    return float(event.pos().x()), float(event.pos().y())


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

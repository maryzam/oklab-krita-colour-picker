import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import ChromaLightnessModel
from oklab_colour_picker.widgets import HueRingTabWidget, SelectorWidget
from oklab_colour_picker.widgets.hue_ring_tab import _OklchGradientSlider


def test_lightness_slider_commits_oklab_with_new_l_and_preserved_hue(qtbot):
    initial_hue = math.radians(120.0)
    initial = color_math.oklch_to_oklab([0.4, 0.08, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.4, initial_chroma=0.08, selected=initial)

    l_slider = _slider(widget, axis="L")
    commits, previews = _capture_signals(widget)

    right_edge = QtCore.QPoint(l_slider.width() - 1, l_slider.height() // 2)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonPress, right_edge)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonRelease, right_edge)

    assert len(previews) == 1
    assert len(commits) == 1
    new_l, _, new_h = color_math.oklab_to_oklch(commits[0])
    assert new_l == pytest.approx(1.0, abs=1e-9)
    assert _hue_diff(new_h, initial_hue) < 1e-9


def test_chroma_slider_commits_oklab_with_new_c_and_preserved_hue(qtbot):
    initial_hue = math.radians(220.0)
    initial = color_math.oklch_to_oklab([0.55, 0.05, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.55, initial_chroma=0.05, selected=initial)

    c_slider = _slider(widget, axis="C")
    commits, _ = _capture_signals(widget)

    midpoint = QtCore.QPoint(c_slider.width() // 2, c_slider.height() // 2)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(commits) == 1
    new_l, new_c, new_h = color_math.oklab_to_oklch(commits[0])
    assert new_l == pytest.approx(0.55, abs=1e-9)
    assert new_c > 0.05
    assert _hue_diff(new_h, initial_hue) < 1e-9


def test_ring_click_passes_through_committed_signal(qtbot):
    initial_hue = math.radians(0.0)
    initial = color_math.oklch_to_oklab([0.6, 0.1, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.6, initial_chroma=0.1, selected=initial)

    ring = widget.findChild(SelectorWidget)
    assert ring is not None

    # A few pixels below the top of the ring — solidly inside the donut band,
    # but close enough to the top centre that the hue is ~90°.
    cx = (ring.width() - 1) // 2
    target = QtCore.QPoint(cx, 5)
    commits, _ = _capture_signals(widget)

    _send_mouse(ring, QtCore.QEvent.MouseButtonPress, target)
    _send_mouse(ring, QtCore.QEvent.MouseButtonRelease, target)

    assert len(commits) == 1
    _, _, hue = color_math.oklab_to_oklch(commits[0])
    assert _hue_diff(hue, math.pi / 2.0) < math.radians(2.0)


def test_set_selected_colour_updates_slider_positions(qtbot):
    initial = color_math.oklch_to_oklab([0.3, 0.05, 0.0])
    widget = _build_widget(qtbot, initial_lightness=0.3, initial_chroma=0.05, selected=initial)

    new_l, new_c, hue = 0.85, 0.18, math.radians(40.0)
    widget.set_model(ChromaLightnessModel(lightness=new_l, chroma=new_c))
    widget.set_selected_colour(color_math.oklch_to_oklab([new_l, new_c, hue]))

    l_slider = _slider(widget, axis="L")
    c_slider = _slider(widget, axis="C")
    assert l_slider._value == pytest.approx(new_l)
    assert c_slider._value == pytest.approx(new_c)


def _build_widget(qtbot, *, initial_lightness, initial_chroma, selected):
    model = ChromaLightnessModel(lightness=initial_lightness, chroma=initial_chroma)
    widget = HueRingTabWidget(model)
    widget.resize(240, 240)
    qtbot.addWidget(widget)
    widget.show()
    qtbot.waitExposed(widget)
    widget.set_selected_colour(selected)
    return widget


def _slider(widget: HueRingTabWidget, *, axis: str) -> _OklchGradientSlider:
    for child in widget.findChildren(_OklchGradientSlider):
        if child._axis == axis:
            return child
    raise AssertionError(f"no {axis!r} slider found")


def _capture_signals(widget: HueRingTabWidget):
    commits: list = []
    previews: list = []
    widget.committed.connect(commits.append)
    widget.previewed.connect(previews.append)
    return commits, previews


def _send_mouse(widget, event_type, position):
    button = QtCore.Qt.LeftButton
    buttons = QtCore.Qt.LeftButton if event_type != QtCore.QEvent.MouseButtonRelease else QtCore.Qt.NoButton
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    assert event.isAccepted()


def _hue_diff(a: float, b: float) -> float:
    delta = (a - b) % math.tau
    return min(delta, math.tau - delta)

import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import HueLightnessModel, LightnessSliceModel
from oklab_colour_picker.widgets import SelectorWidget


def test_mouse_drag_emits_previews_and_commit(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    previews = []
    commits = []
    widget.previewed.connect(previews.append)
    widget.committed.connect(commits.append)

    start = QtCore.QPoint(8, 12)
    end = QtCore.QPoint(24, 16)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, start, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseMove, end, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, end, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    assert len(previews) >= 2
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.model.color_at_position((end.x(), end.y()), _size(widget)))


def test_invalid_release_does_not_commit(qtbot):
    widget = SelectorWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(40, 80)
    qtbot.addWidget(widget)
    widget.show()

    previous = widget.model.color_at_position((20, 40), _size(widget))
    assert previous is not None
    widget.set_selected_colour(previous)

    commits = []
    previews = []
    widget.committed.connect(commits.append)
    widget.previewed.connect(previews.append)

    invalid_corner = QtCore.QPoint(0, 0)
    _send_mouse(
        widget,
        QtCore.QEvent.MouseButtonPress,
        invalid_corner,
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
    )
    _send_mouse(
        widget,
        QtCore.QEvent.MouseButtonRelease,
        invalid_corner,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
    )

    assert commits == []
    np.testing.assert_allclose(widget.selected_colour, previous)
    np.testing.assert_allclose(previews[-1], previous)


def test_programmatic_colour_update_blocks_widget_signals(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)

    previews = []
    commits = []
    widget.previewed.connect(previews.append)
    widget.committed.connect(commits.append)

    colour = widget.model.color_at_position((20, 10), (64, 32))
    assert colour is not None
    blocker = QtCore.QSignalBlocker(widget)
    widget.set_selected_colour(colour)
    del blocker

    assert previews == []
    assert commits == []
    np.testing.assert_allclose(widget.selected_colour, colour)


def test_keyboard_nudge_previews_then_commits_on_release(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(start)

    previews = []
    commits = []
    widget.previewed.connect(previews.append)
    widget.committed.connect(commits.append)

    press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, press)

    assert press.isAccepted()
    assert len(previews) == 1
    assert commits == []

    QtWidgets.QApplication.sendEvent(widget, release)
    assert release.isAccepted()
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_signal_payload_mutation_does_not_corrupt_widget_state(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(commits.append)

    point = QtCore.QPoint(24, 16)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    selected = widget.selected_colour
    assert selected is not None
    commits[0][:] = 0.0
    np.testing.assert_allclose(widget.selected_colour, selected)


def test_keyboard_step_at_boundary_keeps_event_handled(qtbot):
    widget = SelectorWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(40, 80)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((38, 40), _size(widget))
    assert start is not None
    widget.set_selected_colour(start)

    event = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)

    assert event.isAccepted()
    assert widget.selected_colour is not None


def test_mouse_interaction_cancels_pending_keyboard_commit(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(start)

    commits = []
    widget.committed.connect(commits.append)

    key_press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()

    point = QtCore.QPoint(24, 16)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, point, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, point, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    key_release = QtGui.QKeyEvent(QtCore.QEvent.KeyRelease, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_release)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_focus_loss_flushes_pending_keyboard_commit(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(64, 32)
    qtbot.addWidget(widget)
    widget.show()

    start = widget.model.color_at_position((20, 10), _size(widget))
    assert start is not None
    widget.set_selected_colour(start)

    commits = []
    widget.committed.connect(commits.append)

    key_press = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, QtCore.Qt.Key_Right, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, key_press)
    assert key_press.isAccepted()

    focus_out = QtGui.QFocusEvent(QtCore.QEvent.FocusOut)
    QtWidgets.QApplication.sendEvent(widget, focus_out)

    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], widget.selected_colour)


def test_indicator_position_comes_from_model(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=math.pi / 3.0))
    widget.resize(100, 50)
    qtbot.addWidget(widget)

    colour = color_math.oklch_to_oklab([0.25, 0.02, math.pi / 3.0])
    widget.set_selected_colour(colour)

    expected = widget.model.position_for_color(colour, (100, 50))
    assert expected is not None
    assert widget.indicator_position() == pytest.approx(expected)


def test_paint_event_renders_selector_image(qtbot):
    widget = SelectorWidget(HueLightnessModel(hue=0.0))
    widget.resize(32, 24)
    qtbot.addWidget(widget)
    widget.show()

    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_RGBA8888)
    image.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(image)
    widget.render(painter)
    painter.end()

    nontransparent = False
    for y in range(image.height()):
        for x in range(image.width()):
            if QtGui.QColor(image.pixel(x, y)).alpha() != 0:
                nontransparent = True
                break
        if nontransparent:
            break
    assert nontransparent


def _send_mouse(widget, event_type, position, button, buttons):
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    assert event.isAccepted()


def _size(widget):
    return widget.width(), widget.height()

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui

from oklab_colour_picker.selector_models import HueLightnessSliceModel
from oklab_colour_picker.widgets import HueLightnessSliceDiskWidget


def test_disk_widget_picks_through_lightness_overlay(qtbot):
    widget = HueLightnessSliceDiskWidget(HueLightnessSliceModel(chroma=0.03))
    widget.resize(120, 120)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(commits.append)

    pos = QtCore.QPoint(75, 60)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    expected = widget.model.color_at_position((pos.x(), pos.y()), (widget.width(), widget.height()))
    assert expected is not None
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)


def test_disk_widget_renders_lightness_guide_rings_on_top_of_base(qtbot):
    from oklab_colour_picker.widgets.selector import SelectorWidget

    model = HueLightnessSliceModel(chroma=0.03)
    overlay = HueLightnessSliceDiskWidget(model)
    overlay.resize(81, 81)
    qtbot.addWidget(overlay)
    overlay.show()

    bare = SelectorWidget(model)
    bare.resize(81, 81)
    qtbot.addWidget(bare)
    bare.show()

    overlay_pixels = _render_to_rgba_array(overlay)
    bare_pixels = _render_to_rgba_array(bare)

    assert overlay_pixels.shape == bare_pixels.shape
    diff = np.any(overlay_pixels != bare_pixels, axis=-1)
    assert int(diff.sum()) > 100

    cx, cy = 40.0, 40.0
    radius_px = 40.0 * 0.50
    cardinals = [
        (int(round(cx + radius_px)), int(cy)),
        (int(round(cx - radius_px)), int(cy)),
        (int(cx), int(round(cy + radius_px))),
        (int(cx), int(round(cy - radius_px))),
    ]
    assert any(diff[y, x] for x, y in cardinals)


def _render_to_rgba_array(widget) -> np.ndarray:
    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()
    return np.dstack((raw[..., 2], raw[..., 1], raw[..., 0], raw[..., 3]))


def _send_mouse(widget, event_type, pos, button, buttons):
    event = QtGui.QMouseEvent(event_type, pos, button, buttons, QtCore.Qt.NoModifier)
    QtCore.QCoreApplication.sendEvent(widget, event)
    assert event.isAccepted()

import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui

from oklab_colour_picker.selector_models import (
    LIGHTNESS_CHART_CHROMA_MAX,
    LightnessSliceModel,
)
from oklab_colour_picker.widgets import LightnessSliceDiskWidget


def test_disk_widget_picks_through_overlay(qtbot):
    widget = LightnessSliceDiskWidget(LightnessSliceModel(lightness=0.6))
    widget.resize(120, 120)
    qtbot.addWidget(widget)
    widget.show()

    commits = []
    widget.committed.connect(commits.append)

    # Stay close to the centre so the position is in-gamut for any hue.
    pos = QtCore.QPoint(65, 55)
    _send_mouse(widget, QtCore.QEvent.MouseButtonPress, pos, QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    _send_mouse(widget, QtCore.QEvent.MouseButtonRelease, pos, QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    expected = widget.model.color_at_position((pos.x(), pos.y()), (widget.width(), widget.height()))
    assert expected is not None
    assert len(commits) == 1
    np.testing.assert_allclose(commits[0], expected)


def test_disk_widget_renders_overlay_pixels_on_top_of_base(qtbot):
    # Render the disk widget twice at the same model and size: once with the
    # overlay subclass, once with the bare SelectorWidget that paints only
    # the disk image. Both share the same disk pixels, so any rendered
    # difference comes from the rings + gamut contour the subclass adds.
    from oklab_colour_picker.widgets.selector import SelectorWidget

    model = LightnessSliceModel(lightness=0.5)
    overlay = LightnessSliceDiskWidget(model)
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
    differing_pixels = int(diff.sum())

    # The contour stroke alone covers ~360 sampled hues × ~3 px (halo + stroke)
    # plus five chroma rings, so the difference should be substantial — in
    # the hundreds of pixels — not a stray rounding artifact.
    assert differing_pixels > 200

    # And the overlay must reach across the disk: the C=0.05 ring sits well
    # inside the leaf at every hue, so at least one of the four cardinal
    # samples on that ring must differ from the bare render.
    cx, cy = 40.0, 40.0
    radius_px = 40.0 * (0.05 / LIGHTNESS_CHART_CHROMA_MAX)
    cardinals = [
        (int(round(cx + radius_px)), int(cy)),
        (int(round(cx - radius_px)), int(cy)),
        (int(cx), int(round(cy + radius_px))),
        (int(cx), int(round(cy - radius_px))),
    ]
    differs_on_inner_ring = any(
        diff[y, x] for x, y in cardinals
    )
    assert differs_on_inner_ring


def _render_to_rgba_array(widget) -> np.ndarray:
    image = QtGui.QImage(widget.size(), QtGui.QImage.Format_ARGB32)
    image.fill(0)
    widget.render(image)
    # Format_ARGB32 packs 0xAARRGGBB as uint32; on little-endian the bytes
    # land as B, G, R, A. Re-pack as RGBA so channel asserts read naturally.
    ptr = image.bits()
    ptr.setsize(image.byteCount())
    raw = np.frombuffer(ptr, dtype=np.uint8).reshape(image.height(), image.width(), 4).copy()
    return np.dstack((raw[..., 2], raw[..., 1], raw[..., 0], raw[..., 3]))


def test_set_model_invalidates_gamut_path_cache(qtbot):
    widget = LightnessSliceDiskWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    # Force the cache to populate.
    first = widget._gamut_path(widget.model.lightness)
    assert first is not None
    assert widget._gamut_path_cache_key is not None

    widget.set_model(LightnessSliceModel(lightness=0.8))
    assert widget._gamut_path_cache_key is None

    second = widget._gamut_path(widget.model.lightness)
    assert second is not None
    # Different lightness → different leaf shape, so a different path object.
    assert second is not first


def test_resize_invalidates_gamut_path_cache(qtbot):
    widget = LightnessSliceDiskWidget(LightnessSliceModel(lightness=0.5))
    widget.resize(80, 80)
    qtbot.addWidget(widget)
    widget.show()

    widget._gamut_path(widget.model.lightness)
    assert widget._gamut_path_cache_key == (0.5, 80, 80)

    widget.resize(100, 100)
    assert widget._gamut_path_cache_key is None


def test_gamut_path_caps_at_disk_radius(qtbot):
    # Pick a lightness whose cusp chroma is comfortably below the disk's
    # radial extent, so every sample on the contour should sit strictly
    # inside the rim.
    widget = LightnessSliceDiskWidget(LightnessSliceModel(lightness=0.95))
    widget.resize(101, 101)
    qtbot.addWidget(widget)
    widget.show()

    path = widget._gamut_path(0.95)
    assert path is not None

    radius = (min(widget.width(), widget.height()) - 1) / 2.0
    cx = (widget.width() - 1) / 2.0
    cy = (widget.height() - 1) / 2.0
    bounds = path.boundingRect()
    # Every point on the contour is within the disk radius (allow a 1 px
    # tolerance for rounding).
    assert bounds.left() >= cx - radius - 1
    assert bounds.right() <= cx + radius + 1
    assert bounds.top() >= cy - radius - 1
    assert bounds.bottom() <= cy + radius + 1


def _send_mouse(widget, event_type, pos, button, buttons):
    event = QtGui.QMouseEvent(event_type, pos, button, buttons, QtCore.Qt.NoModifier)
    QtCore.QCoreApplication.sendEvent(widget, event)

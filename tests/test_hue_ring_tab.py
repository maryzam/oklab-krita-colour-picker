import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5 import QtCore, QtGui, QtWidgets

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    LIGHTNESS_CHART_CHROMA_MAX,
    ChromaLightnessModel,
)
from oklab_colour_picker.widgets import HueRingTabWidget, SelectorWidget
from oklab_colour_picker.widgets.hue_ring_tab import (
    _HueRingSelector,
    _OklchGradientSlider,
    _Swatch,
)


def test_lightness_slider_commits_oklab_with_new_l_and_preserved_hue(qtbot):
    initial_hue = math.radians(120.0)
    initial = color_math.oklch_to_oklab([0.4, 0.08, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.4, initial_chroma=0.08, selected=initial)

    l_slider = _slider(widget, axis="L")
    commits, previews = _capture_signals(widget)

    # Aim ~midway down the slider so the new L stays well inside the per-hue
    # sRGB gamut for the initial chroma — the right edge would force L=1
    # (white), which the gamut clamp would collapse to C=0 and erase the hue.
    midpoint = QtCore.QPoint(l_slider.width() // 2, l_slider.height() // 2)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(previews) == 1
    assert len(commits) == 1
    new_l, _, new_h = color_math.oklab_to_oklch(commits[0])
    assert new_l == pytest.approx(0.5, abs=0.05)
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

    new_l, hue = 0.6, math.radians(40.0)
    new_c = float(color_math.max_chroma_for_lh(new_l, hue)) * 0.5
    widget.set_model(ChromaLightnessModel(lightness=new_l, chroma=new_c))
    widget.set_selected_colour(color_math.oklch_to_oklab([new_l, new_c, hue]))

    l_slider = _slider(widget, axis="L")
    c_slider = _slider(widget, axis="C")
    assert l_slider._value == pytest.approx(new_l)
    assert c_slider._value == pytest.approx(new_c)


def test_chosen_hue_survives_a_chroma_zero_round_trip(qtbot):
    chosen_hue = math.radians(200.0)
    chromatic = color_math.oklch_to_oklab([0.55, 0.1, chosen_hue])
    widget = _build_widget(qtbot, initial_lightness=0.55, initial_chroma=0.1, selected=chromatic)

    # Simulate the dock pushing back an achromatic colour (C ~= 0). OKLab→OKLCh
    # cannot recover the hue from this triple, but the widget must remember it
    # so the user can raise chroma without snapping to red.
    achromatic = color_math.oklch_to_oklab([0.55, 0.0, 0.0])
    widget.set_selected_colour(achromatic)

    c_slider = _slider(widget, axis="C")
    commits, _ = _capture_signals(widget)
    midpoint = QtCore.QPoint(c_slider.width() // 2, c_slider.height() // 2)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(commits) == 1
    _, new_c, new_h = color_math.oklab_to_oklch(commits[0])
    assert new_c > 0.0
    assert _hue_diff(new_h, chosen_hue) < 1e-9


def test_emit_clamps_chroma_against_per_lightness_gamut(qtbot):
    hue = math.radians(30.0)
    initial = color_math.oklch_to_oklab([0.5, 0.15, hue])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.15, selected=initial)

    # Drive the L slider to ~0.97 — at this lightness the per-hue gamut max
    # is far below the initial chroma, so the wrapper must clamp before
    # emitting or the dock would diverge from what Krita actually accepts.
    l_slider = _slider(widget, axis="L")
    target_x = l_slider.width() - 4
    target = QtCore.QPoint(target_x, l_slider.height() // 2)
    commits, _ = _capture_signals(widget)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonPress, target)
    _send_mouse(l_slider, QtCore.QEvent.MouseButtonRelease, target)

    assert len(commits) == 1
    new_l, new_c, _ = color_math.oklab_to_oklch(commits[0])
    gamut_max = float(color_math.max_chroma_for_lh(new_l, hue))
    assert new_c <= gamut_max + 1e-9


def test_ring_click_at_zero_chroma_captures_hue_for_subsequent_chroma_drag(qtbot):
    # Default Hue Ring state: model chroma == 0, selected colour achromatic.
    # The ring click then produces an achromatic OKLab too, so OKLab→OKLCh
    # cannot recover the hue — the wrapper must learn the hue from the click
    # geometry itself, otherwise raising chroma snaps to red regardless of
    # where the user clicked.
    achromatic = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.0, selected=achromatic)

    ring = widget.findChild(SelectorWidget)
    assert ring is not None

    cx = (ring.width() - 1) // 2
    target = QtCore.QPoint(cx, 5)  # top of ring → hue ≈ π/2
    _send_mouse(ring, QtCore.QEvent.MouseButtonPress, target)
    _send_mouse(ring, QtCore.QEvent.MouseButtonRelease, target)

    c_slider = _slider(widget, axis="C")
    commits, _ = _capture_signals(widget)
    midpoint = QtCore.QPoint(c_slider.width() // 2, c_slider.height() // 2)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(commits) == 1
    _, new_c, new_h = color_math.oklab_to_oklch(commits[0])
    assert new_c > 0.0
    assert _hue_diff(new_h, math.pi / 2.0) < math.radians(2.0)


def test_right_button_ring_press_does_not_rotate_chosen_hue(qtbot):
    # SelectorWidget ignores non-left-button presses (event.ignore() in
    # mousePressEvent). The hue capture filter must mirror that — a stray
    # right-click should not silently rewrite the user's chosen hue.
    chosen_hue = math.radians(45.0)
    initial = color_math.oklch_to_oklab([0.5, 0.1, chosen_hue])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.1, selected=initial)
    ring = widget.findChild(SelectorWidget)
    assert ring is not None

    cx = (ring.width() - 1) // 2
    top_of_ring = QtCore.QPoint(cx, 5)  # geometrically implies hue ~ 90°
    _send_mouse(ring, QtCore.QEvent.MouseButtonPress, top_of_ring, button=QtCore.Qt.RightButton)
    _send_mouse(ring, QtCore.QEvent.MouseButtonRelease, top_of_ring, button=QtCore.Qt.RightButton)

    c_slider = _slider(widget, axis="C")
    commits, _ = _capture_signals(widget)
    midpoint = QtCore.QPoint(c_slider.width() // 2, c_slider.height() // 2)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(commits) == 1
    _, _, new_h = color_math.oklab_to_oklch(commits[0])
    assert _hue_diff(new_h, chosen_hue) < 1e-9


def test_cancelled_ring_drag_restores_chosen_hue(qtbot):
    # SelectorWidget treats a release on an invalid position as a cancel —
    # it restores _colour_before_drag and emits previewed(_colour_before_drag).
    # When the pre-drag colour is *achromatic* the wrapper cannot recover the
    # hue from that signal (chroma == 0), so the event filter itself must
    # mirror SelectorWidget's accept/cancel lifecycle — otherwise a partial
    # drag leaks the intermediate hue past the cancellation.
    achromatic = color_math.oklch_to_oklab([0.5, 0.0, 0.0])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.0, selected=achromatic)
    ring = widget.findChild(SelectorWidget)
    assert ring is not None

    cx = (ring.width() - 1) // 2
    cy = (ring.height() - 1) // 2
    radius = (min(ring.width(), ring.height()) - 1) / 2.0

    # Step 1: capture an initial hue of ~45° via a completed ring click at C=0.
    chosen_hue = math.radians(45.0)
    chosen_x = int(round(cx + radius * 0.95 * math.cos(chosen_hue)))
    chosen_y = int(round(cy - radius * 0.95 * math.sin(chosen_hue)))
    chosen_point = QtCore.QPoint(chosen_x, chosen_y)
    _send_mouse(ring, QtCore.QEvent.MouseButtonPress, chosen_point)
    _send_mouse(ring, QtCore.QEvent.MouseButtonRelease, chosen_point)

    # Step 2: start a fresh drag at the top of the ring (~90°), then bail out
    # by releasing on an invalid corner — SelectorWidget cancels.
    top = QtCore.QPoint(cx, 5)
    invalid_corner = QtCore.QPoint(0, 0)
    _send_mouse(ring, QtCore.QEvent.MouseButtonPress, top)
    _send_mouse(ring, QtCore.QEvent.MouseMove, invalid_corner, buttons=QtCore.Qt.LeftButton)
    _send_mouse(ring, QtCore.QEvent.MouseButtonRelease, invalid_corner)

    c_slider = _slider(widget, axis="C")
    commits, _ = _capture_signals(widget)
    midpoint = QtCore.QPoint(c_slider.width() // 2, c_slider.height() // 2)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonPress, midpoint)
    _send_mouse(c_slider, QtCore.QEvent.MouseButtonRelease, midpoint)

    assert len(commits) == 1
    _, _, new_h = color_math.oklab_to_oklch(commits[0])
    assert _hue_diff(new_h, chosen_hue) < math.radians(2.0)


def test_ring_drag_does_not_change_chroma_slider_value(qtbot):
    # Regression: previously _CentralPanel.set_axes called set_max(per-hue
    # gamut max) on the C slider, so the thumb's x-position rescaled every
    # time the ring rotated even though chroma was unchanged. The slider
    # must now keep the same _value and the same track length across hues.
    initial_hue = math.radians(30.0)
    initial_chroma = 0.05
    initial = color_math.oklch_to_oklab([0.55, initial_chroma, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.55, initial_chroma=initial_chroma, selected=initial)

    c_slider = _slider(widget, axis="C")
    assert c_slider._value == pytest.approx(initial_chroma)
    assert c_slider._max == pytest.approx(LIGHTNESS_CHART_CHROMA_MAX)
    initial_thumb_x = c_slider._value_to_x(c_slider._value)

    # Push the same chroma at a different hue back into the widget — this is
    # the path that the dock takes during a ring drag (oklab is recomputed at
    # the new hue but L/C are preserved).
    new_hue = math.radians(210.0)
    rotated = color_math.oklch_to_oklab([0.55, initial_chroma, new_hue])
    widget.set_selected_colour(rotated)

    assert c_slider._value == pytest.approx(initial_chroma)
    assert c_slider._max == pytest.approx(LIGHTNESS_CHART_CHROMA_MAX)
    assert c_slider._value_to_x(c_slider._value) == initial_thumb_x


def test_ring_drag_updates_chroma_gamut_overlay(qtbot):
    # The gamut tail overlay must follow the per-hue gamut so the user sees
    # which portion of the C track is unreachable at the current hue, even
    # though the track length itself stays constant.
    hue_a = math.radians(30.0)
    hue_b = math.radians(210.0)
    gamut_a = float(color_math.max_chroma_for_lh(0.55, hue_a))
    gamut_b = float(color_math.max_chroma_for_lh(0.55, hue_b))
    # Sanity: pick lightness/hues where the two gamut maxima differ enough to
    # make the assertion meaningful — otherwise the test would pass trivially.
    assert abs(gamut_a - gamut_b) > 1e-3

    initial = color_math.oklch_to_oklab([0.55, 0.05, hue_a])
    widget = _build_widget(qtbot, initial_lightness=0.55, initial_chroma=0.05, selected=initial)

    c_slider = _slider(widget, axis="C")
    expected_a = min(LIGHTNESS_CHART_CHROMA_MAX, gamut_a)
    assert c_slider._gamut_max == pytest.approx(expected_a)

    rotated = color_math.oklch_to_oklab([0.55, 0.05, hue_b])
    widget.set_selected_colour(rotated)

    expected_b = min(LIGHTNESS_CHART_CHROMA_MAX, gamut_b)
    assert c_slider._gamut_max == pytest.approx(expected_b)


def test_ring_drag_does_not_change_lightness_slider_value(qtbot):
    # Symmetric regression for the L slider — its _max is constant at 1.0
    # so the thumb position should track only the lightness value, which is
    # untouched by ring rotation.
    hue_a = math.radians(60.0)
    hue_b = math.radians(300.0)
    lightness = 0.42
    initial = color_math.oklch_to_oklab([lightness, 0.04, hue_a])
    widget = _build_widget(qtbot, initial_lightness=lightness, initial_chroma=0.04, selected=initial)

    l_slider = _slider(widget, axis="L")
    initial_thumb_x = l_slider._value_to_x(l_slider._value)

    rotated = color_math.oklch_to_oklab([lightness, 0.04, hue_b])
    widget.set_selected_colour(rotated)

    assert l_slider._value == pytest.approx(lightness)
    assert l_slider._value_to_x(l_slider._value) == initial_thumb_x


def test_sliders_have_visible_axis_labels(qtbot):
    # UX: each slider gets a small "L" / "C" label so the axis is obvious
    # without hovering or guessing from gradient direction.
    initial = color_math.oklch_to_oklab([0.5, 0.05, 0.0])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.05, selected=initial)

    label_texts = {label.text() for label in widget.findChildren(QtWidgets.QLabel)}
    assert "L" in label_texts
    assert "C" in label_texts


def test_hue_text_is_painted_on_swatch_not_in_separate_label(qtbot):
    # UX: the hue readout sits over the swatch (saving vertical space and
    # keeping the central panel compact). No standalone "H ..." QLabel
    # should remain in the widget tree.
    initial_hue = math.radians(120.0)
    initial = color_math.oklch_to_oklab([0.5, 0.05, initial_hue])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.05, selected=initial)

    label_texts = [label.text() for label in widget.findChildren(QtWidgets.QLabel)]
    assert not any(text.startswith("H ") for text in label_texts)

    swatch = widget.findChild(_Swatch)
    assert swatch is not None
    assert "120" in swatch._hue_text


def test_hue_ring_uses_radial_bar_indicator(qtbot):
    # The hue ring's indicator is a radial bar (selects the whole hue slice),
    # not the small circle the base SelectorWidget paints.
    initial = color_math.oklch_to_oklab([0.5, 0.05, math.radians(0.0)])
    widget = _build_widget(qtbot, initial_lightness=0.5, initial_chroma=0.05, selected=initial)

    ring = widget.findChild(SelectorWidget)
    assert isinstance(ring, _HueRingSelector)


def test_widget_exposes_selector_surface_used_by_dock(qtbot):
    initial = color_math.oklch_to_oklab([0.4, 0.08, math.radians(120.0)])
    widget = _build_widget(qtbot, initial_lightness=0.4, initial_chroma=0.08, selected=initial)

    # ColourPickerDockPanel iterates every selector tab and reads these two —
    # without forwarding to the inner ring widget the dock would AttributeError
    # on the Hue Ring tab as soon as a foreground colour arrives.
    np.testing.assert_allclose(widget.selected_colour, initial)
    assert widget.indicator_position() is not None


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


def _send_mouse(widget, event_type, position, *, button=None, buttons=None):
    if button is None:
        button = QtCore.Qt.LeftButton if event_type != QtCore.QEvent.MouseMove else QtCore.Qt.NoButton
    if buttons is None:
        if event_type == QtCore.QEvent.MouseButtonRelease:
            buttons = QtCore.Qt.NoButton
        elif event_type == QtCore.QEvent.MouseMove:
            buttons = QtCore.Qt.LeftButton
        else:
            buttons = button
    event = QtGui.QMouseEvent(
        event_type,
        QtCore.QPointF(position),
        button,
        buttons,
        QtCore.Qt.NoModifier,
    )
    QtWidgets.QApplication.sendEvent(widget, event)
    # Acceptance is not always guaranteed (e.g., right-clicks the widget
    # ignores) — the test asserts on the resulting widget state instead.


def _hue_diff(a: float, b: float) -> float:
    delta = (a - b) % math.tau
    return min(delta, math.tau - delta)

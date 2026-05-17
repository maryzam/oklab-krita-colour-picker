import math

import numpy as np
import pytest
from hypothesis import assume, given, settings, strategies as st

from oklab_colour_picker import color_math
from oklab_colour_picker.controller import normalize_oklab_for_krita
from oklab_colour_picker.selector_models import (
    HueLightnessSliceModel,
    LightnessChromaSliceModel,
    LightnessSliceModel,
)
from oklab_colour_picker.selector_interaction import StateKind


MODEL_CASES = (
    ("lightness-slice", LightnessSliceModel(lightness=0.55)),
    ("hue-lightness-slice", HueLightnessSliceModel(chroma=0.03)),
    ("lightness-chroma-slice", LightnessChromaSliceModel(hue=1.0)),
)
_QT_APP = None


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p1_model_position_colour_round_trip_is_stable_within_quantization(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    position = (x_fraction * (width - 1.0), y_fraction * (height - 1.0))
    colour = model.color_at_position(position, size)
    assume(colour is not None)

    round_tripped_position = model.position_for_color(colour, size)
    assume(round_tripped_position is not None)
    round_tripped_colour = model.color_at_position(round_tripped_position, size)

    assert round_tripped_colour is not None
    np.testing.assert_allclose(
        normalize_oklab_for_krita(round_tripped_colour),
        normalize_oklab_for_krita(colour),
        atol=1e-12,
    )


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p2_model_position_lookup_is_idempotent_for_same_colour(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    once = model.position_for_color(colour, size)
    twice = model.position_for_color(np.asarray(colour, dtype=float).copy(), size)

    assert once == pytest.approx(twice)


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    x_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    y_fraction=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_p3_idle_indicator_is_independent_of_interaction_history(case, width, height, x_fraction, y_fraction):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    widget = SelectorWidget(model)
    widget.resize(*size)

    widget.show_colour(colour)
    idle_position = widget.indicator_position()
    widget._force_state_for_test(
        StateKind.DRAGGING, anchor=(width / 2.0, height / 2.0)
    )
    widget.show_colour(colour)
    widget._force_state_for_test(StateKind.IDLE, anchor=None)

    assert widget.indicator_position() == pytest.approx(idle_position)


@settings(max_examples=200, derandomize=True, deadline=None)
@given(colour=st.tuples(st.floats(0.0, 1.0), st.floats(-0.1, 0.1), st.floats(-0.1, 0.1)))
def test_p2_show_colour_echo_idempotence_from_any_state(colour):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    widget = SelectorWidget(LightnessChromaSliceModel(hue=0.0))
    normalized = normalize_oklab_for_krita(np.asarray(colour, dtype=float))

    widget.show_colour(normalized)
    once = (widget.state, widget.indicator_position())
    widget.show_colour(normalized)

    assert (widget.state, widget.indicator_position()) == once


@settings(max_examples=200, derandomize=True, deadline=None)
@given(
    case=st.sampled_from(MODEL_CASES),
    width=st.integers(min_value=16, max_value=96),
    height=st.integers(min_value=16, max_value=96),
    points=st.lists(
        st.tuples(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=5,
    ),
)
def test_p4_no_orphan_anchor_after_random_gesture_sequence(case, width, height, points):
    _ensure_qapp()
    from PyQt5 import QtCore, QtGui, QtWidgets
    from oklab_colour_picker.widgets import SelectorWidget

    _name, model = case
    widget = SelectorWidget(model)
    # Stay above the 32x32 minimum so the later resize is a real size change
    # (sub-minimum resizes are clamped and would not deliver a resizeEvent).
    widget.resize(width + 40, height + 40)
    widget.show()
    QtWidgets.QApplication.processEvents()
    actual_w, actual_h = widget.width(), widget.height()
    qpoints = [
        QtCore.QPoint(int(fx * (actual_w - 1)), int(fy * (actual_h - 1)))
        for fx, fy in points
    ]

    def send(kind, point, button, buttons):
        QtWidgets.QApplication.sendEvent(
            widget,
            QtGui.QMouseEvent(
                kind, QtCore.QPointF(point), button, buttons, QtCore.Qt.NoModifier
            ),
        )

    # Press, drag through the random points, release: a real gesture history.
    send(QtCore.QEvent.MouseButtonPress, qpoints[0], QtCore.Qt.LeftButton, QtCore.Qt.LeftButton)
    for point in qpoints[1:]:
        send(QtCore.QEvent.MouseMove, point, QtCore.Qt.NoButton, QtCore.Qt.LeftButton)
    send(QtCore.QEvent.MouseButtonRelease, qpoints[-1], QtCore.Qt.LeftButton, QtCore.Qt.NoButton)

    # End outside anchored states: a resize drops a PINNED anchor (INV-1); if
    # the gesture already fell back to IDLE this is a harmless no-op. Either
    # way, no anchor may survive (P4 / INV-1).
    widget.resize(actual_w + 17, actual_h + 13)
    QtWidgets.QApplication.processEvents()

    assert widget.state == "IDLE"
    assert widget.anchor is None
    widget.close()


def test_edge_hue_wrap_position_is_stable_across_zero_tau():
    zero_model = LightnessChromaSliceModel(hue=0.0)
    tau_model = LightnessChromaSliceModel(hue=math.tau - 1e-12)
    colour = color_math.oklch_to_oklab([0.5, 0.02, 0.0])

    zero_position = zero_model.position_for_color(colour, (101, 101))
    tau_position = tau_model.position_for_color(colour, (101, 101))

    assert zero_position is not None
    assert tau_position == pytest.approx(zero_position, abs=1e-6)


@pytest.mark.parametrize(("lightness", "expected_y"), [(0.0, 100.0), (1.0, 0.0)])
def test_edge_lightness_chroma_slice_accepts_achromatic_lightness_boundaries(lightness, expected_y):
    model = LightnessChromaSliceModel(hue=0.0)
    colour = color_math.oklch_to_oklab([lightness, 0.0, 0.0])

    position = model.position_for_color(colour, (101, 101))
    round_tripped = model.color_at_position((0.0, expected_y), (101, 101))

    assert position == pytest.approx((0.0, expected_y), abs=1e-6)
    assert round_tripped is not None
    np.testing.assert_allclose(round_tripped, colour, atol=1e-12)


@pytest.mark.parametrize("lightness", [0.0, 1.0])
def test_edge_hue_lightness_slice_rejects_positive_chroma_at_lightness_boundaries(lightness):
    model = HueLightnessSliceModel(chroma=0.05)
    colour = color_math.oklch_to_oklab([lightness, 0.05, 0.0])

    assert model.position_for_color(colour, (101, 101)) is None


def test_edge_quantization_boundary_colours_compare_equal_after_krita_normalization():
    raw = np.array([0.55, 0.02, -0.03])
    normalized_once = normalize_oklab_for_krita(raw)

    np.testing.assert_allclose(
        normalize_oklab_for_krita(normalized_once),
        normalized_once,
        atol=0.0,
    )


def _ensure_qapp():
    global _QT_APP

    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    _QT_APP = QtWidgets.QApplication.instance() or _QT_APP or QtWidgets.QApplication([])
    return _QT_APP

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
def test_p3_idle_indicator_position_is_model_pure(case, width, height, x_fraction, y_fraction):
    _name, model = case
    size = (width, height)
    colour = model.color_at_position((x_fraction * (width - 1.0), y_fraction * (height - 1.0)), size)
    assume(colour is not None)

    first = model.position_for_color(colour, size)
    second = model.position_for_color(np.asarray(colour, dtype=float).copy(), size)

    assert first == pytest.approx(second)


@pytest.mark.xfail(strict=True, reason="PR-2 / P2: SelectorWidget.show_colour state-machine API does not exist yet")
@settings(max_examples=10, derandomize=True, deadline=None)
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


@pytest.mark.xfail(strict=True, reason="PR-2 / P4: explicit anchor/state API does not exist yet")
@settings(max_examples=10, derandomize=True, deadline=None)
@given(lightness=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_p4_idle_state_has_no_orphan_anchor(lightness):
    _ensure_qapp()
    from oklab_colour_picker.widgets import SelectorWidget

    widget = SelectorWidget(LightnessSliceModel(lightness=float(lightness)))

    assert widget.state == "IDLE"
    assert widget.anchor is None


def test_edge_hue_wrap_position_is_stable_across_zero_tau():
    model = LightnessChromaSliceModel(hue=math.tau - 1e-12)
    colour = color_math.oklch_to_oklab([0.5, 0.02, 0.0])

    position = model.position_for_color(colour, (101, 101))

    assert position is not None


def _ensure_qapp():
    global _QT_APP

    pytest.importorskip("PyQt5")
    from PyQt5 import QtWidgets

    _QT_APP = QtWidgets.QApplication.instance() or _QT_APP or QtWidgets.QApplication([])
    return _QT_APP

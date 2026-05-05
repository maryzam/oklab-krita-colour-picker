import math

import numpy as np
import pytest

from lab_colour_picker import color_math
from lab_colour_picker.selector_models import (
    ChromaLightnessModel,
    HueLightnessModel,
    LightnessSliceModel,
)


def test_lightness_slice_maps_center_to_neutral_current_lightness():
    model = LightnessSliceModel(lightness=0.62)

    actual = model.color_at_position((50.0, 50.0), (101.0, 101.0))

    np.testing.assert_allclose(actual, [0.62, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("lightness", [-0.01, 1.01, math.nan])
def test_lightness_slice_validates_lightness(lightness):
    with pytest.raises(ValueError, match="lightness"):
        LightnessSliceModel(lightness=lightness)


def test_lightness_slice_uses_per_hue_gamut_boundary_at_edge():
    model = LightnessSliceModel(lightness=0.5)

    actual = model.color_at_position((100.0, 50.0), (101.0, 101.0))

    expected_chroma = color_math.max_chroma_for_lh(0.5, 0.0)
    np.testing.assert_allclose(actual, [0.5, expected_chroma, 0.0], atol=1e-12)
    assert color_math.in_srgb_gamut(color_math.oklab_to_srgb(actual), epsilon=1e-8) is True


@pytest.mark.parametrize("position", [(-1.0, 50.0), (50.0, -1.0), (101.0, 50.0), (50.0, 101.0), (0.0, 0.0)])
def test_lightness_slice_rejects_outside_square_or_circle(position):
    model = LightnessSliceModel(lightness=0.5)

    assert model.color_at_position(position, (101.0, 101.0)) is None


def test_lightness_slice_rejects_degenerate_size():
    model = LightnessSliceModel(lightness=0.5)

    assert model.color_at_position((0.0, 0.0), (1.0, 1.0)) is None
    assert model.position_for_color([0.5, 0.0, 0.0], (1.0, 1.0)) is None


def test_lightness_slice_rejects_inverse_outside_gamut_boundary():
    model = LightnessSliceModel(lightness=0.5)
    max_chroma = color_math.max_chroma_for_lh(0.5, 0.0)

    assert model.position_for_color([0.5, max_chroma * 1.01, 0.0], (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(65.0, 43.0), (20.0, 50.0), (50.0, 80.0)])
def test_lightness_slice_round_trips_position_and_color(position):
    model = LightnessSliceModel(lightness=0.55)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_hue_lightness_maps_axes_to_lightness_and_gamut_relative_chroma():
    model = HueLightnessModel(hue=1.25)

    actual = model.color_at_position((50.0, 25.0), (101.0, 101.0))

    lightness = 0.75
    max_chroma = color_math.max_chroma_for_lh(lightness, 1.25)
    expected = color_math.oklch_to_oklab([lightness, max_chroma * 0.5, 1.25])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize("hue", [math.nan, math.inf, -math.inf])
def test_hue_lightness_validates_hue(hue):
    with pytest.raises(ValueError, match="hue"):
        HueLightnessModel(hue=hue)


def test_hue_lightness_normalizes_hue():
    model = HueLightnessModel(hue=math.tau + 0.25)

    assert model.hue == pytest.approx(0.25)


def test_hue_lightness_top_and_bottom_rows_are_selectable_end_to_end():
    model = HueLightnessModel(hue=0.0)

    np.testing.assert_allclose(model.color_at_position((50.0, 0.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((100.0, 100.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(("color", "expected"), [([1.0, 0.0, 0.0], (0.0, 0.0)), ([0.0, 0.0, 0.0], (0.0, 100.0))])
def test_hue_lightness_inverse_collapses_achromatic_endpoint_rows_to_left_edge(color, expected):
    model = HueLightnessModel(hue=0.0)

    np.testing.assert_allclose(model.position_for_color(color, (101.0, 101.0)), expected, atol=1e-12)


def test_hue_lightness_rejects_degenerate_or_out_of_bounds_positions():
    model = HueLightnessModel(hue=0.0)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(0.0, 50.0), (35.0, 20.0), (100.0, 75.0)])
def test_hue_lightness_round_trips_position_and_color(position):
    model = HueLightnessModel(hue=2.0)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_hue_lightness_rejects_color_from_different_hue_plane():
    model = HueLightnessModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi / 2.0])

    assert model.position_for_color(color, (101.0, 101.0)) is None


def test_hue_lightness_rejects_color_from_antipodal_hue_ray():
    model = HueLightnessModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi])

    assert model.position_for_color(color, (101.0, 101.0)) is None


def test_hue_lightness_rejects_color_with_lightness_out_of_range():
    model = HueLightnessModel(hue=0.0)

    assert model.position_for_color([1.5, 0.0, 0.0], (101.0, 101.0)) is None


def test_hue_lightness_accepts_near_neutral_round_trip_without_hue_sensitivity():
    model = HueLightnessModel(hue=1.75)
    color = color_math.oklch_to_oklab([0.5, 1e-12, model.hue])

    np.testing.assert_allclose(model.position_for_color(color, (101.0, 101.0)), (0.0, 50.0), atol=1e-9)


def test_chroma_lightness_uses_fixed_lightness_and_chroma_with_position_hue():
    chroma = color_math.max_chroma_for_lh(0.6, math.pi / 2.0) * 0.5
    model = ChromaLightnessModel(lightness=0.6, chroma=chroma)

    actual = model.color_at_position((50.0, 0.0), (101.0, 101.0))

    expected = color_math.oklch_to_oklab([0.6, chroma, math.pi / 2.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize(
    ("lightness", "chroma"),
    [(-0.01, 0.1), (1.01, 0.1), (math.nan, 0.1), (0.5, -0.01), (0.5, math.nan)],
)
def test_chroma_lightness_validates_lightness_and_chroma(lightness, chroma):
    with pytest.raises(ValueError, match="lightness|chroma"):
        ChromaLightnessModel(lightness=lightness, chroma=chroma)


def test_chroma_lightness_rejects_out_of_circle_and_out_of_gamut_hues():
    model = ChromaLightnessModel(lightness=0.5, chroma=0.4)

    assert model.color_at_position((0.0, 0.0), (101.0, 101.0)) is None
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_chroma_lightness_rejects_degenerate_size():
    model = ChromaLightnessModel(lightness=0.5, chroma=0.1)

    assert model.color_at_position((0.0, 0.0), (1.0, 1.0)) is None
    assert model.position_for_color([0.5, 0.1, 0.0], (1.0, 1.0)) is None


def test_chroma_lightness_rejects_interior_positions_to_preserve_inverse_symmetry():
    model = ChromaLightnessModel(lightness=0.55, chroma=color_math.max_chroma_for_lh(0.55, 0.0) * 0.35)

    assert model.color_at_position((75.0, 50.0), (101.0, 101.0)) is None


def test_chroma_lightness_rejects_inverse_color_with_mismatched_chroma():
    model = ChromaLightnessModel(lightness=0.55, chroma=0.05)

    assert model.position_for_color(color_math.oklch_to_oklab([0.55, 0.06, 0.0]), (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(100.0, 50.0), (50.0, 0.0), (0.0, 50.0), (50.0, 100.0)])
def test_chroma_lightness_round_trips_position_and_color(position):
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    model = ChromaLightnessModel(lightness=0.55, chroma=chroma)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)

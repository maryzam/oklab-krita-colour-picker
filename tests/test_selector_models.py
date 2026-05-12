import math

import numpy as np
import pytest

from oklab_colour_picker import color_math
from oklab_colour_picker.selector_models import (
    CHROMA_LIGHTNESS_BAND_MAX_PX,
    ChromaLightnessModel,
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
    chroma_lightness_band_width,
)


def test_lightness_slice_maps_center_to_neutral_current_lightness():
    model = LightnessSliceModel(lightness=0.62)

    actual = model.color_at_position((50.0, 50.0), (101.0, 101.0))

    np.testing.assert_allclose(actual, [0.62, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("lightness", [-0.01, 1.01, math.nan])
def test_lightness_slice_validates_lightness(lightness):
    with pytest.raises(ValueError, match="lightness"):
        LightnessSliceModel(lightness=lightness)


def test_lightness_slice_radius_maps_to_absolute_chroma():
    model = LightnessSliceModel(lightness=0.5)
    # Halfway from centre to the rim along +x (hue=0). Radius is normalised
    # against color_math.SRGB_MAX_CHROMA, not the per-hue gamut max.
    actual = model.color_at_position((75.0, 50.0), (101.0, 101.0))

    expected_chroma = 0.5 * color_math.SRGB_MAX_CHROMA
    if expected_chroma > color_math.max_chroma_for_lh(0.5, 0.0):
        assert actual is None
    else:
        np.testing.assert_allclose(actual, [0.5, expected_chroma, 0.0], atol=1e-12)


def test_lightness_slice_renders_transparent_outside_per_hue_gamut():
    # At hue=0 (red-ish) the L=0.5 cusp chroma sits well below
    # color_math.SRGB_MAX_CHROMA, so the rim along +x is outside gamut.
    model = LightnessSliceModel(lightness=0.5)
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_slice_keeps_in_gamut_pixel_inside_per_hue_leaf():
    model = LightnessSliceModel(lightness=0.5)
    # Blue (hue=3pi/2). Sit inside the gamut leaf at chroma well below cusp.
    chroma = 0.20
    assert chroma < color_math.max_chroma_for_lh(0.5, 3.0 * math.pi / 2.0)

    fraction = chroma / color_math.SRGB_MAX_CHROMA
    # hue=3pi/2 -> atan2(dy,dx) with dy<0, dx=0; dy = center_y - y, so y > center_y.
    y = 50.0 + fraction * 50.0
    actual = model.color_at_position((50.0, y), (101.0, 101.0))

    np.testing.assert_allclose(
        actual,
        color_math.oklch_to_oklab([0.5, chroma, 3.0 * math.pi / 2.0]),
        atol=1e-12,
    )


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


def test_lightness_slice_snap_clamps_to_per_hue_gamut():
    # At hue=0 (red-ish) and L=0.5 the cusp chroma sits well below
    # color_math.SRGB_MAX_CHROMA, so the rim along +x is past the leaf.
    model = LightnessSliceModel(lightness=0.5)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue % math.tau, 0.0, atol=1e-9)
    np.testing.assert_allclose(chroma, color_math.max_chroma_for_lh(0.5, 0.0), atol=1e-9)


def test_lightness_slice_snap_returns_in_gamut_position_unchanged():
    model = LightnessSliceModel(lightness=0.55)
    position = (60.0, 45.0)
    expected = model.color_at_position(position, (101.0, 101.0))
    assert expected is not None

    snapped = model.snapped_color_at_position(position, (101.0, 101.0))
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


@pytest.mark.parametrize("position", [(-1.0, 50.0), (50.0, 101.0), (0.0, 0.0)])
def test_lightness_slice_snap_projects_outside_positions_to_disk(position):
    model = LightnessSliceModel(lightness=0.5)
    snapped = model.snapped_color_at_position(position, (101.0, 101.0))

    assert snapped is not None
    assert model.position_for_color(snapped, (101.0, 101.0)) is not None


@pytest.mark.parametrize("position", [(60.0, 45.0), (40.0, 50.0), (50.0, 60.0)])
def test_lightness_slice_round_trips_position_and_color(position):
    # Positions chosen to sit well inside the L=0.55 gamut leaf for any hue,
    # so round-tripping doesn't depend on per-hue cusp variation.
    model = LightnessSliceModel(lightness=0.55)

    color = model.color_at_position(position, (101.0, 101.0))
    assert color is not None
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_lightness_chart_chroma_max_envelopes_srgb_cusp():
    # Hardcoded chart extent must stay above the largest sRGB cusp chroma so
    # the entire gamut leaf remains addressable on the picker.
    hues = np.linspace(0.0, math.tau, 4096, endpoint=False)
    _, chroma_cusp = color_math.find_cusp(np.cos(hues), np.sin(hues))
    assert color_math.SRGB_MAX_CHROMA >= float(np.max(chroma_cusp))


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
        ChromaLightnessModel(lightness=0.55, chroma=0.05),
    ],
)
def test_vectorized_selector_render_paths_do_not_use_halley_boundary_solver(monkeypatch, model):
    def fail_on_boundary_solver(*_args, **_kwargs):
        raise AssertionError("colors_at_positions must not call max_chroma_for_lh")

    monkeypatch.setattr(color_math, "max_chroma_for_lh", fail_on_boundary_solver)
    y, x = np.indices((32, 32), dtype=float)

    oklab, valid = model.colors_at_positions(x, y, (32, 32))

    assert oklab.shape == (32, 32, 3)
    assert valid.shape == (32, 32)


def test_lightness_chroma_slice_maps_x_to_absolute_chroma():
    model = LightnessChromaSliceModel(hue=1.25)

    actual = model.color_at_position((25.0, 25.0), (101.0, 101.0))

    lightness = 0.75
    chroma = 0.25 * color_math.SRGB_MAX_CHROMA
    expected = color_math.oklch_to_oklab([lightness, chroma, 1.25])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_lightness_chroma_slice_rejects_position_outside_per_hue_gamut():
    model = LightnessChromaSliceModel(hue=0.0)
    # Hue 0 (red) cusp chroma is well below color_math.SRGB_MAX_CHROMA, so the right
    # edge at mid-lightness lies outside the achievable per-hue gamut.
    assert model.color_at_position((100.0, 50.0), (101.0, 101.0)) is None


def test_lightness_chroma_slice_snap_clamps_to_gamut_leaf():
    model = LightnessChromaSliceModel(hue=0.0)
    snapped = model.snapped_color_at_position((100.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(lightness, 0.5, atol=1e-12)
    np.testing.assert_allclose(hue, 0.0, atol=1e-12)
    np.testing.assert_allclose(chroma, color_math.max_chroma_for_lh(0.5, 0.0), atol=1e-9)


def test_lightness_chroma_slice_snap_projects_outside_rect_to_edge():
    model = LightnessChromaSliceModel(hue=1.0)
    snapped = model.snapped_color_at_position((-20.0, 50.0), (101.0, 101.0))

    expected = model.color_at_position((0.0, 50.0), (101.0, 101.0))
    assert expected is not None
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


@pytest.mark.parametrize("hue", [math.nan, math.inf, -math.inf])
def test_lightness_chroma_slice_validates_hue(hue):
    with pytest.raises(ValueError, match="hue"):
        LightnessChromaSliceModel(hue=hue)


def test_lightness_chroma_slice_normalizes_hue():
    model = LightnessChromaSliceModel(hue=math.tau + 0.25)

    assert model.hue == pytest.approx(0.25)


def test_lightness_chroma_slice_achromatic_endpoints_only_at_left_edge():
    model = LightnessChromaSliceModel(hue=0.0)

    np.testing.assert_allclose(model.color_at_position((0.0, 0.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((0.0, 100.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)
    # Off the L=0/L=1 axis chroma must vanish, so positive x is out of gamut.
    assert model.color_at_position((50.0, 0.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, 100.0), (101.0, 101.0)) is None


@pytest.mark.parametrize(("color", "expected"), [([1.0, 0.0, 0.0], (0.0, 0.0)), ([0.0, 0.0, 0.0], (0.0, 100.0))])
def test_lightness_chroma_slice_inverse_collapses_achromatic_endpoint_rows_to_left_edge(color, expected):
    model = LightnessChromaSliceModel(hue=0.0)

    np.testing.assert_allclose(model.position_for_color(color, (101.0, 101.0)), expected, atol=1e-12)


def test_lightness_chroma_slice_rejects_degenerate_or_out_of_bounds_positions():
    model = LightnessChromaSliceModel(hue=0.0)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(0.0, 50.0), (15.0, 20.0), (10.0, 75.0)])
def test_lightness_chroma_slice_round_trips_position_and_color(position):
    model = LightnessChromaSliceModel(hue=2.0)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_lightness_chroma_slice_rejects_color_from_different_hue_plane():
    model = LightnessChromaSliceModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi / 2.0])

    assert model.position_for_color(color, (101.0, 101.0)) is None


def test_lightness_chroma_slice_rejects_color_from_antipodal_hue_ray():
    model = LightnessChromaSliceModel(hue=0.0)
    color = color_math.oklch_to_oklab([0.5, 0.05, math.pi])

    assert model.position_for_color(color, (101.0, 101.0)) is None


def test_lightness_chroma_slice_rejects_color_with_lightness_out_of_range():
    model = LightnessChromaSliceModel(hue=0.0)

    assert model.position_for_color([1.5, 0.0, 0.0], (101.0, 101.0)) is None


def test_lightness_chroma_slice_accepts_near_neutral_round_trip_without_hue_sensitivity():
    model = LightnessChromaSliceModel(hue=1.75)
    color = color_math.oklch_to_oklab([0.5, 1e-12, model.hue])

    np.testing.assert_allclose(model.position_for_color(color, (101.0, 101.0)), (0.0, 50.0), atol=1e-9)


def test_hue_lightness_slice_maps_angle_to_hue_and_radius_to_lightness():
    chroma = 0.05
    model = HueLightnessSliceModel(chroma=chroma)

    actual = model.color_at_position((50.0, 25.0), (101.0, 101.0))

    expected = color_math.oklch_to_oklab([0.5, chroma, math.pi / 2.0])
    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_hue_lightness_slice_maps_center_to_white_and_rim_to_black():
    model = HueLightnessSliceModel(chroma=0.0)

    np.testing.assert_allclose(model.color_at_position((50.0, 50.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((100.0, 50.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)


@pytest.mark.parametrize("chroma", [-0.01, math.nan])
def test_hue_lightness_slice_validates_chroma(chroma):
    with pytest.raises(ValueError, match="chroma"):
        HueLightnessSliceModel(chroma=chroma)


def test_hue_lightness_slice_rejects_position_outside_per_lightness_hue_gamut():
    model = HueLightnessSliceModel(chroma=0.2)

    assert model.color_at_position((50.0, 50.0), (101.0, 101.0)) is None


def test_hue_lightness_slice_snap_finds_nearest_valid_lightness_on_spoke():
    model = HueLightnessSliceModel(chroma=0.2)
    snapped = model.snapped_color_at_position((50.0, 50.0), (101.0, 101.0))

    assert snapped is not None
    assert model.position_for_color(snapped, (101.0, 101.0)) is not None


def test_hue_lightness_slice_snap_projects_outside_circle_to_rim_angle():
    model = HueLightnessSliceModel(chroma=0.0)
    snapped = model.snapped_color_at_position((150.0, 50.0), (101.0, 101.0))

    expected = model.color_at_position((100.0, 50.0), (101.0, 101.0))
    assert expected is not None
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


def test_hue_lightness_slice_accepts_achromatic_full_lightness_range():
    model = HueLightnessSliceModel(chroma=0.0)

    np.testing.assert_allclose(model.color_at_position((50.0, 50.0), (101.0, 101.0)), [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(model.color_at_position((50.0, 0.0), (101.0, 101.0)), [0.0, 0.0, 0.0], atol=1e-12)


def test_hue_lightness_slice_rejects_degenerate_or_out_of_bounds_positions():
    model = HueLightnessSliceModel(chroma=0.05)

    assert model.color_at_position((0.0, 0.0), (1.0, 100.0)) is None
    assert model.color_at_position((101.0, 50.0), (101.0, 101.0)) is None
    assert model.color_at_position((50.0, -1.0), (101.0, 101.0)) is None
    assert model.color_at_position((0.0, 0.0), (101.0, 101.0)) is None


@pytest.mark.parametrize("position", [(65.0, 50.0), (50.0, 25.0), (35.0, 50.0)])
def test_hue_lightness_slice_round_trips_position_and_color(position):
    chroma = 0.03
    model = HueLightnessSliceModel(chroma=chroma)

    color = model.color_at_position(position, (101.0, 101.0))
    assert color is not None
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)


def test_hue_lightness_slice_rejects_color_with_mismatched_chroma():
    model = HueLightnessSliceModel(chroma=0.05)
    color = color_math.oklch_to_oklab([0.5, 0.06, math.pi / 2.0])

    assert model.position_for_color(color, (101.0, 101.0)) is None


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

    # Pixel sits ~10 px from the centre of a 101x101 widget — well inside the
    # donut hole (outer radius 50, band width 25 → inner edge at 25 px).
    assert model.color_at_position((60.0, 50.0), (101.0, 101.0)) is None


def test_chroma_lightness_snap_projects_interior_to_ring_hue():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    model = ChromaLightnessModel(lightness=0.55, chroma=chroma)
    snapped = model.snapped_color_at_position((60.0, 50.0), (101.0, 101.0))

    expected = model.color_at_position((100.0, 50.0), (101.0, 101.0))
    assert expected is not None
    np.testing.assert_allclose(snapped, expected, atol=1e-12)


def test_chroma_lightness_snap_projects_to_nearest_valid_hue_when_ring_has_gap():
    model = ChromaLightnessModel(lightness=0.5, chroma=0.2)
    assert model.color_at_position((50.0, 0.0), (101.0, 101.0)) is None

    snapped = model.snapped_color_at_position((50.0, 0.0), (101.0, 101.0))

    assert snapped is not None
    assert model.position_for_color(snapped, (101.0, 101.0)) is not None
    lightness, chroma, hue = color_math.oklab_to_oklch(snapped)
    np.testing.assert_allclose(
        color_math.max_chroma_for_lh(lightness, hue),
        chroma,
        atol=1e-5,
    )


def test_chroma_lightness_rejects_inverse_color_with_mismatched_chroma():
    model = ChromaLightnessModel(lightness=0.55, chroma=0.05)

    assert model.position_for_color(color_math.oklch_to_oklab([0.55, 0.06, 0.0]), (101.0, 101.0)) is None


def test_chroma_lightness_band_width_uses_half_radius_below_cap():
    # For small rings the band fills 50% of the outer radius, giving the user
    # the full hue ring even on tiny widgets.
    assert chroma_lightness_band_width(20.0) == pytest.approx(10.0)
    assert chroma_lightness_band_width(60.0) == pytest.approx(30.0)


def test_chroma_lightness_band_width_clamps_to_pixel_cap():
    # On large widgets the band caps at CHROMA_LIGHTNESS_BAND_MAX_PX so the
    # donut doesn't grow unbounded — extra width past the cap is wasted.
    assert chroma_lightness_band_width(200.0) == pytest.approx(CHROMA_LIGHTNESS_BAND_MAX_PX)
    assert chroma_lightness_band_width(1000.0) == pytest.approx(CHROMA_LIGHTNESS_BAND_MAX_PX)


def test_chroma_lightness_band_thickness_caps_on_large_widget():
    # Outer radius 100 → band capped at 40 px → inner edge at 60 px. A pixel
    # 50 px from the centre should now sit inside the donut hole, even though
    # the previous 55%-fraction rule would have placed it on the band.
    model = ChromaLightnessModel(lightness=0.55, chroma=color_math.max_chroma_for_lh(0.55, 0.0) * 0.35)

    assert model.color_at_position((50.0, 100.0), (201.0, 201.0)) is None
    assert model.color_at_position((100.0 + 70.0, 100.0), (201.0, 201.0)) is not None


@pytest.mark.parametrize("position", [(100.0, 50.0), (50.0, 0.0), (0.0, 50.0), (50.0, 100.0)])
def test_chroma_lightness_round_trips_position_and_color(position):
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    model = ChromaLightnessModel(lightness=0.55, chroma=chroma)

    color = model.color_at_position(position, (101.0, 101.0))
    actual = model.position_for_color(color, (101.0, 101.0))

    np.testing.assert_allclose(actual, position, atol=1e-9)

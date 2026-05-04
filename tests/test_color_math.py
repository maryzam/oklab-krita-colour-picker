import math

import numpy as np
import pytest

from lab_colour_picker import color_math


def test_srgb_oklab_known_reference_values():
    # Reference primary values use Bjorn Ottosson's OKLab matrices:
    # https://bottosson.github.io/posts/oklab/
    samples = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    actual = color_math.srgb_to_oklab(samples)

    np.testing.assert_allclose(actual[0], [0.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(actual[1], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(actual[2], [0.62795536, 0.22486306, 0.12584630], atol=1e-7)
    np.testing.assert_allclose(actual[3], [0.86643961, -0.23388757, 0.17949848], atol=1e-7)
    np.testing.assert_allclose(actual[4], [0.45201372, -0.03245698, -0.31152815], atol=1e-7)


def test_srgb_oklab_round_trip_for_scalar_and_vector_inputs():
    samples = np.array(
        [
            [0.12, 0.34, 0.56],
            [0.90, 0.20, 0.10],
            [0.04, 0.95, 0.70],
        ]
    )

    round_tripped = color_math.oklab_to_srgb(color_math.srgb_to_oklab(samples))
    scalar_round_tripped = color_math.oklab_to_srgb(color_math.srgb_to_oklab(samples[0]))

    np.testing.assert_allclose(round_tripped, samples, atol=1e-10)
    np.testing.assert_allclose(scalar_round_tripped, samples[0], atol=1e-10)


def test_oklab_oklch_round_trip_preserves_hue_and_chroma():
    lab = np.array(
        [
            [0.5, 0.1, 0.0],
            [0.7, -0.05, 0.08],
            [0.3, 0.0, -0.2],
        ]
    )

    lch = color_math.oklab_to_oklch(lab)
    actual = color_math.oklch_to_oklab(lch)

    assert np.all((0.0 <= lch[:, 2]) & (lch[:, 2] < math.tau))
    np.testing.assert_allclose(actual, lab, atol=1e-12)


def test_gamut_predicate_and_clip_work_for_scalar_and_vector_inputs():
    srgb = np.array(
        [
            [0.0, 0.5, 1.0],
            [-0.01, 0.5, 1.0],
            [0.0, 0.5, 1.01],
        ]
    )

    np.testing.assert_array_equal(color_math.in_srgb_gamut(srgb), [True, False, False])
    assert color_math.in_srgb_gamut(np.array([0.0, 0.5, 1.0])) is True
    np.testing.assert_allclose(
        color_math.clip_srgb(srgb),
        [[0.0, 0.5, 1.0], [0.0, 0.5, 1.0], [0.0, 0.5, 1.0]],
    )


def test_oklab_to_srgb_returns_unclipped_out_of_gamut_values():
    srgb = color_math.oklab_to_srgb(np.array([0.5, 1.0, 0.0]))

    assert np.any(srgb < 0.0) or np.any(srgb > 1.0)
    assert color_math.in_srgb_gamut(srgb) is False


@pytest.mark.parametrize("hue", [0.0, 1.0, 2.0, 4.0, 5.5])
def test_find_cusp_lands_on_srgb_gamut_boundary(hue):
    a_ = math.cos(hue)
    b_ = math.sin(hue)
    lightness, chroma = color_math.find_cusp(a_, b_)

    srgb = color_math.oklab_to_srgb(np.array([lightness, chroma * a_, chroma * b_]))
    just_outside = color_math.oklab_to_srgb(np.array([lightness, chroma * 1.001 * a_, chroma * 1.001 * b_]))

    assert color_math.in_srgb_gamut(srgb, epsilon=1e-6) is True
    assert color_math.in_srgb_gamut(just_outside, epsilon=1e-6) is False


def test_find_gamut_intersection_lower_half_matches_slow_oracle():
    hue = 0.0
    a_ = math.cos(hue)
    b_ = math.sin(hue)
    l0 = 0.05
    l1 = 0.2
    c1 = 1.0

    actual = color_math.find_gamut_intersection(a_, b_, l1, c1, l0)
    expected = _slow_gamut_intersection(a_, b_, l1, c1, l0)

    assert actual == pytest.approx(expected, abs=1e-6)


def test_find_gamut_intersection_requires_positive_chroma_ray():
    with pytest.raises(ValueError, match="c1 must be greater than zero"):
        color_math.find_gamut_intersection(1.0, 0.0, 0.5, 0.0, 0.5)


@pytest.mark.parametrize(
    ("lightness", "hue"),
    [
        (0.0, 0.0),
        (1.0, 1.25),
        (0.05, 4.0),
        (0.20, 0.0),
        (0.35, 1.0),
        (0.50, 2.0),
        (0.65, 3.0),
        (0.80, 4.0),
        (0.95, 5.0),
        (0.42, math.tau - 0.01),
    ],
)
def test_max_chroma_for_lh_matches_slow_binary_search_oracle(lightness, hue):
    expected = _slow_max_chroma_for_lh(lightness, hue)

    actual = color_math.max_chroma_for_lh(lightness, hue)

    assert actual == pytest.approx(expected, abs=1e-6)


def test_max_chroma_for_lh_vectorizes_over_lightness_and_hue():
    lightness = np.array([0.2, 0.5, 0.8, 0.7858333333333334])
    hue = np.array([0.0, 2.0, 4.0, 5.770272220879211])

    actual = color_math.max_chroma_for_lh(lightness, hue)
    expected = np.array([_slow_max_chroma_for_lh(l, h) for l, h in zip(lightness, hue)])

    np.testing.assert_allclose(actual, expected, atol=1e-6)


def _slow_max_chroma_for_lh(lightness, hue):
    if lightness <= 0.0 or lightness >= 1.0:
        return 0.0

    low = 0.0
    high = 1.0
    direction = np.array([math.cos(hue), math.sin(hue)])

    while _is_oklab_in_gamut(lightness, direction * high):
        high *= 2.0
        if high > 4.0:
            raise AssertionError("binary-search oracle failed to bracket max chroma")

    for _ in range(80):
        mid = (low + high) / 2.0
        if _is_oklab_in_gamut(lightness, direction * mid):
            low = mid
        else:
            high = mid

    return low


def _slow_gamut_intersection(a_, b_, l1, c1, l0):
    low = 0.0
    high = 1.0

    while _is_intersection_point_in_gamut(a_, b_, l1, c1, l0, high):
        high *= 2.0
        if high > 4.0:
            raise AssertionError("binary-search oracle failed to bracket intersection")

    for _ in range(80):
        mid = (low + high) / 2.0
        if _is_intersection_point_in_gamut(a_, b_, l1, c1, l0, mid):
            low = mid
        else:
            high = mid

    return low


def _is_intersection_point_in_gamut(a_, b_, l1, c1, l0, t):
    lightness = l0 * (1.0 - t) + t * l1
    chroma = t * c1
    srgb = color_math.oklab_to_srgb(np.array([lightness, chroma * a_, chroma * b_]))
    return bool(color_math.in_srgb_gamut(srgb, epsilon=1e-12))


def _is_oklab_in_gamut(lightness, ab):
    srgb = color_math.oklab_to_srgb(np.array([lightness, ab[0], ab[1]]))
    return color_math.in_srgb_gamut(srgb, epsilon=1e-12)

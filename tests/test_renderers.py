import math

import numpy as np
import pytest

from lab_colour_picker import color_math
from lab_colour_picker.renderers import render_rgba
from lab_colour_picker.selector_models import (
    ChromaLightnessModel,
    HueLightnessModel,
    LightnessSliceModel,
)


@pytest.mark.parametrize(
    "model",
    [
        LightnessSliceModel(lightness=0.55),
        HueLightnessModel(hue=1.25),
        ChromaLightnessModel(lightness=0.55, chroma=0.05),
    ],
)
def test_renderers_return_uint8_rgba_buffers(model):
    actual = render_rgba(model, (23, 17))

    assert actual.shape == (17, 23, 4)
    assert actual.dtype == np.uint8


@pytest.mark.parametrize(
    ("model", "size", "probes"),
    [
        (
            LightnessSliceModel(lightness=0.55),
            (33, 33),
            [(16, 16), (32, 16), (16, 0), (0, 0)],
        ),
        (
            HueLightnessModel(hue=1.25),
            (33, 21),
            [(0, 0), (16, 10), (32, 20), (33, 10)],
        ),
        (
            ChromaLightnessModel(lightness=0.55, chroma=0.05),
            (33, 33),
            [(32, 16), (16, 0), (16, 16), (0, 0)],
        ),
    ],
)
def test_renderer_pixels_match_model_at_probe_points(model, size, probes):
    rgba = render_rgba(model, size)

    for x, y in probes:
        model_color = model.color_at_position((x, y), size)
        if model_color is None:
            if 0 <= x < size[0] and 0 <= y < size[1]:
                assert rgba[y, x, 3] == 0
            continue

        assert rgba[y, x, 3] == 255
        np.testing.assert_array_equal(rgba[y, x, :3], _quantize8(model_color))


@pytest.mark.parametrize("size", [(64, 64), (200, 120)])
def test_lightness_renderer_preserves_coordinate_semantics_across_sizes(size):
    model = LightnessSliceModel(lightness=0.5)
    rgba = render_rgba(model, size)
    position = model.position_for_color(color_math.oklch_to_oklab([0.5, 0.0, 0.0]), size)
    x, y = round(position[0]), round(position[1])

    assert rgba[y, x, 3] == 255
    np.testing.assert_array_equal(rgba[y, x, :3], _quantize8(model.color_at_position((x, y), size)))


def test_hue_lightness_renderer_alpha_matches_selectable_area():
    model = HueLightnessModel(hue=math.pi / 3.0)
    rgba = render_rgba(model, (17, 9))

    assert np.all(rgba[..., 3] == 255)


def test_chroma_lightness_renderer_masks_out_interior_and_out_of_gamut_hues():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    model = ChromaLightnessModel(lightness=0.55, chroma=chroma)
    rgba = render_rgba(model, (33, 33))

    assert rgba[16, 16, 3] == 0
    assert rgba[16, 32, 3] == 255


def test_chroma_lightness_renderer_has_visible_ring_at_even_sizes():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    model = ChromaLightnessModel(lightness=0.55, chroma=chroma)
    rgba = render_rgba(model, (32, 32))

    assert np.count_nonzero(rgba[..., 3]) > 0


def _quantize8(oklab):
    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
    return np.rint(srgb * 255.0).astype(np.uint8)

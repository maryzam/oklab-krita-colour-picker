import statistics
import time

import pytest

from oklab_colour_picker import color_math
from oklab_colour_picker import renderers
from oklab_colour_picker.renderers import render_rgba
from oklab_colour_picker.selector_models import (
    ChromaLightnessModel,
    LightnessChromaSliceModel,
    HueLightnessSliceModel,
    LightnessSliceModel,
)


PERFORMANCE_BUDGET_SECONDS = 0.005
COLD_RENDER_BUDGET_SECONDS = 0.020
SAMPLE_COUNT = 21
COLD_SAMPLE_COUNT = 7


@pytest.mark.perf
def test_256_renderers_meet_median_budget():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    cases = [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
        ChromaLightnessModel(lightness=0.55, chroma=chroma),
    ]

    for model in cases:
        render_rgba(model, (256, 256))
        timings = []
        for _ in range(SAMPLE_COUNT):
            start = time.perf_counter()
            render_rgba(model, (256, 256))
            timings.append(time.perf_counter() - start)

        assert statistics.median(timings) <= PERFORMANCE_BUDGET_SECONDS


@pytest.mark.perf
def test_256_cold_renderers_meet_startup_budget_without_cache_warmup():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    cases = [
        LightnessSliceModel(lightness=0.55),
        HueLightnessSliceModel(chroma=0.05),
        LightnessChromaSliceModel(hue=1.25),
        ChromaLightnessModel(lightness=0.55, chroma=chroma),
    ]

    for model in cases:
        timings = []
        for _ in range(COLD_SAMPLE_COUNT):
            renderers._render_rgba_cached.cache_clear()
            renderers._pixel_grid.cache_clear()
            start = time.perf_counter()
            render_rgba(model, (256, 256))
            timings.append(time.perf_counter() - start)

        median = statistics.median(timings)
        assert median <= COLD_RENDER_BUDGET_SECONDS, (
            f"{type(model).__name__} cold 256px render took {median:.4f}s; "
            f"budget is {COLD_RENDER_BUDGET_SECONDS:.4f}s"
        )

import statistics
import time

from lab_colour_picker import color_math
from lab_colour_picker.renderers import render_rgba
from lab_colour_picker.selector_models import (
    ChromaLightnessModel,
    HueLightnessModel,
    LightnessSliceModel,
)


PERFORMANCE_BUDGET_SECONDS = 0.005


def test_256_renderers_meet_median_budget():
    chroma = color_math.max_chroma_for_lh(0.55, 0.0) * 0.35
    cases = [
        LightnessSliceModel(lightness=0.55),
        HueLightnessModel(hue=1.25),
        ChromaLightnessModel(lightness=0.55, chroma=chroma),
    ]

    for model in cases:
        render_rgba(model, (256, 256))
        timings = []
        for _ in range(9):
            start = time.perf_counter()
            render_rgba(model, (256, 256))
            timings.append(time.perf_counter() - start)

        assert statistics.median(timings) <= PERFORMANCE_BUDGET_SECONDS

import subprocess
import sys

import pytest


COLD_COLOR_MATH_IMPORT_BUDGET_SECONDS = 0.120


@pytest.mark.perf
def test_color_math_import_meets_cold_start_budget():
    script = """
import time
start = time.perf_counter()
import oklab_colour_picker.color_math
print(time.perf_counter() - start)
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    elapsed = float(completed.stdout.strip())

    assert elapsed <= COLD_COLOR_MATH_IMPORT_BUDGET_SECONDS, (
        f"cold color_math import took {elapsed:.4f}s; "
        f"budget is {COLD_COLOR_MATH_IMPORT_BUDGET_SECONDS:.4f}s"
    )

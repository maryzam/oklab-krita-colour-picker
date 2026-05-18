"""Chroma-fixed hue/lightness selector model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker import color_math
from oklab_colour_picker.models.base import (
    IndicatorSpec,
    Position,
    SelectorModel,
    SnappedSelection,
    indicator_from_positions,
)
from oklab_colour_picker.models.geometry import (
    circle_geometry,
    circle_geometry_arrays,
    circle_geometry_projected,
    empty_color_grid,
    position_from_circle,
    size_bounds,
)


CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9
_LIGHTNESS_SNAP_SAMPLES = np.linspace(0.0, 1.0, 257)
_SNAP_BOUNDARY_ITERATIONS = 20


@dataclass(frozen=True)
class HueLightnessSliceModel(SelectorModel):
    """Hue/lightness selector at a fixed OKLCh chroma.

    Hue is the polar angle and OKLab lightness is inverse radius, so the centre
    is white-lightness (L=1) and the rim is black (L=0). Pixels whose fixed
    chroma exceeds the per-(L, hue) sRGB gamut leaf are not selectable. In
    normal dock use this model is rebuilt from the selected colour's chroma, so
    ``position_for_color`` is expected to receive colours on this fixed-chroma
    slice.
    """

    chroma: float

    def __post_init__(self) -> None:
        _validate_chroma(self.chroma)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        geometry = circle_geometry(position, size)
        if geometry is None:
            return None

        normalized_radius, hue, _, _ = geometry
        lightness = 1.0 - normalized_radius
        max_chroma = color_math.max_chroma_for_lh(lightness, hue)
        if self.chroma > max_chroma + CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([lightness, self.chroma, hue])

    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = circle_geometry_arrays(x, y, size)
        if geometry is None:
            return empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, circle_valid = geometry
        lightness = 1.0 - normalized_radius
        oklch = np.stack(
            (
                lightness,
                np.full_like(lightness, self.chroma, dtype=float),
                hue,
            ),
            axis=-1,
        )
        oklab = color_math.oklch_to_oklab(oklch)
        valid = circle_valid & color_math.in_srgb_gamut(
            color_math.oklab_to_srgb(oklab), epsilon=1e-6
        )
        return oklab, valid

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        bounds = size_bounds(size)
        if bounds is None:
            return None

        width, height = bounds
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if not -LIGHTNESS_EPSILON <= lightness <= 1.0 + LIGHTNESS_EPSILON:
            return None
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None

        lightness = float(np.clip(lightness, 0.0, 1.0))
        hue = float(hue % math.tau)
        if self.chroma > color_math.max_chroma_for_lh(lightness, hue) + CHROMA_EPSILON:
            return None

        return position_from_circle(1.0 - lightness, hue, (width, height))

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """Nearest in-gamut colour along the cursor's hue spoke."""

        snapped = self._snapped_lightness_and_hue(position, size)
        if snapped is None:
            return None
        lightness, hue = snapped
        return color_math.oklch_to_oklab([lightness, self.chroma, hue])

    def snapped_position_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> Position | None:
        snapped = self.snapped_selection_at_position(position, size)
        return None if snapped is None else snapped.position

    def snapped_selection_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> SnappedSelection | None:
        snapped = self._snapped_lightness_and_hue(position, size)
        if snapped is None:
            return None
        lightness, hue = snapped
        return SnappedSelection(
            color_math.oklch_to_oklab([lightness, self.chroma, hue]),
            position_from_circle(1.0 - lightness, hue, size),
        )

    def indicator_for_color(
        self, oklab: Sequence[float], size: Sequence[float]
    ) -> IndicatorSpec | None:
        return indicator_from_positions(
            self._desired_position_for_color(oklab, size),
            self._snapped_position_for_color(oklab, size),
        )

    def _desired_position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        bounds = size_bounds(size)
        if bounds is None:
            return None
        width, height = bounds
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None
        if not -LIGHTNESS_EPSILON <= lightness <= 1.0 + LIGHTNESS_EPSILON:
            return None
        lightness = float(np.clip(lightness, 0.0, 1.0))
        hue = float(hue % math.tau)
        return position_from_circle(1.0 - lightness, hue, (width, height))

    def _snapped_position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        bounds = size_bounds(size)
        if bounds is None:
            return None
        width, height = bounds
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None
        hue = float(hue % math.tau)
        lightness = float(np.clip(lightness, 0.0, 1.0))
        snapped = _snap_lightness_to_gamut(self.chroma, hue, lightness)
        if snapped is None:
            return None
        return position_from_circle(1.0 - snapped, hue, (width, height))

    def _snapped_lightness_and_hue(
        self,
        position: Sequence[float],
        size: Sequence[float],
    ) -> tuple[float, float] | None:
        geometry = circle_geometry_projected(position, size)
        if geometry is None:
            return None

        normalized_radius, hue = geometry
        lightness = _snap_lightness_to_gamut(
            self.chroma,
            hue,
            1.0 - normalized_radius,
        )
        if lightness is None:
            return None
        return lightness, hue


def _snap_lightness_to_gamut(chroma: float, hue: float, desired_lightness: float) -> float | None:
    # The scalar fast path avoids the 257-sample sweep on normal in-gamut
    # drags, which is the common case.
    if _lightness_in_gamut(chroma, hue, desired_lightness):
        return desired_lightness

    valid = chroma <= color_math.max_chroma_for_lh(_LIGHTNESS_SNAP_SAMPLES, hue) + CHROMA_EPSILON
    valid_indices = np.flatnonzero(valid)
    if not valid_indices.size:
        return None

    first = int(valid_indices[0])
    last = int(valid_indices[-1])
    lower = float(_LIGHTNESS_SNAP_SAMPLES[first])
    upper = float(_LIGHTNESS_SNAP_SAMPLES[last])
    if desired_lightness < lower and first > 0:
        return _bisect_lightness_boundary(
            chroma,
            hue,
            invalid_lightness=float(_LIGHTNESS_SNAP_SAMPLES[first - 1]),
            valid_lightness=lower,
        )
    if desired_lightness > upper and last + 1 < _LIGHTNESS_SNAP_SAMPLES.size:
        return _bisect_lightness_boundary(
            chroma,
            hue,
            invalid_lightness=float(_LIGHTNESS_SNAP_SAMPLES[last + 1]),
            valid_lightness=upper,
        )

    raise AssertionError("expected contiguous lightness gamut interval")


def _bisect_lightness_boundary(
    chroma: float,
    hue: float,
    *,
    invalid_lightness: float,
    valid_lightness: float,
) -> float:
    invalid = invalid_lightness
    valid = valid_lightness
    for _ in range(_SNAP_BOUNDARY_ITERATIONS):
        midpoint = (invalid + valid) / 2.0
        if _lightness_in_gamut(chroma, hue, midpoint):
            valid = midpoint
        else:
            invalid = midpoint
    return float(valid)


def _lightness_in_gamut(chroma: float, hue: float, lightness: float) -> bool:
    return bool(chroma <= color_math.max_chroma_for_lh(lightness, hue) + CHROMA_EPSILON)


def _validate_chroma(chroma: float) -> None:
    if not math.isfinite(chroma) or chroma < 0.0:
        raise ValueError("chroma must be finite and non-negative")

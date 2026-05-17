"""Hue-fixed lightness/chroma selector model."""

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
    indicator_from_positions,
)
from oklab_colour_picker.models.geometry import (
    empty_color_grid,
    position_in_bounds,
    position_in_bounds_arrays,
    rect_geometry_projected,
    size_bounds,
)


AB_EPSILON = 1e-9
CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9


@dataclass(frozen=True)
class LightnessChromaSliceModel(SelectorModel):
    """Lightness/chroma selector at a fixed OKLab hue.

    The x axis spans absolute OKLCh chroma in ``[0, color_math.SRGB_MAX_CHROMA]``,
    so the selectable region traces the per-hue sRGB gamut leaf rather than
    filling the whole rectangle.
    """

    hue: float

    def __post_init__(self) -> None:
        _validate_hue(self.hue)
        object.__setattr__(self, "hue", self.hue % math.tau)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        bounds = position_in_bounds(position, size)
        if bounds is None:
            return None

        x, y, width, height = bounds
        lightness = 1.0 - y / (height - 1.0)
        chroma = (x / (width - 1.0)) * color_math.SRGB_MAX_CHROMA
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if chroma > max_chroma + CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([lightness, chroma, self.hue])

    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        bounds = position_in_bounds_arrays(x, y, size)
        if bounds is None:
            return empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        x, y, width, height, in_bounds = bounds
        lightness = 1.0 - y / (height - 1.0)
        chroma = (x / (width - 1.0)) * color_math.SRGB_MAX_CHROMA
        oklch = np.stack(
            (
                lightness,
                chroma,
                np.full_like(lightness, self.hue, dtype=float),
            ),
            axis=-1,
        )
        oklab = color_math.oklch_to_oklab(oklch)
        valid = in_bounds & color_math.in_srgb_gamut(
            color_math.oklab_to_srgb(oklab), epsilon=1e-6
        )
        return oklab, valid

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        bounds = size_bounds(size)
        if bounds is None:
            return None

        width, height = bounds
        lightness, chroma, _ = color_math.oklab_to_oklch(oklab)
        if not -LIGHTNESS_EPSILON <= lightness <= 1.0 + LIGHTNESS_EPSILON:
            return None
        if not _on_hue_plane(oklab, self.hue):
            return None

        lightness = float(np.clip(lightness, 0.0, 1.0))
        chroma = max(0.0, float(chroma))
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if chroma > max_chroma + CHROMA_EPSILON:
            return None
        if chroma > color_math.SRGB_MAX_CHROMA + CHROMA_EPSILON:
            return None

        chroma_fraction = min(chroma / color_math.SRGB_MAX_CHROMA, 1.0)
        return (
            float(chroma_fraction * (width - 1.0)),
            float((1.0 - lightness) * (height - 1.0)),
        )

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """In-gamut colour nearest the drag cursor on this hue plane."""

        geometry = rect_geometry_projected(position, size)
        if geometry is None:
            return None

        x, y, width, height = geometry
        lightness = 1.0 - y / (height - 1.0)
        desired_chroma = (x / (width - 1.0)) * color_math.SRGB_MAX_CHROMA
        max_chroma = float(color_math.max_chroma_for_lh(lightness, self.hue))
        chroma = max(0.0, min(desired_chroma, max_chroma))
        return color_math.oklch_to_oklab([lightness, chroma, self.hue])

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
        lightness, chroma, _ = color_math.oklab_to_oklch(oklab)
        if not -LIGHTNESS_EPSILON <= lightness <= 1.0 + LIGHTNESS_EPSILON:
            return None
        if not _on_hue_plane(oklab, self.hue):
            return None
        if chroma > color_math.SRGB_MAX_CHROMA + CHROMA_EPSILON:
            return None
        lightness = float(np.clip(lightness, 0.0, 1.0))
        chroma_fraction = float(np.clip(float(chroma) / color_math.SRGB_MAX_CHROMA, 0.0, 1.0))
        return (
            float(chroma_fraction * (width - 1.0)),
            float((1.0 - lightness) * (height - 1.0)),
        )

    def _snapped_position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        bounds = size_bounds(size)
        if bounds is None:
            return None
        width, height = bounds
        lightness, chroma, _ = color_math.oklab_to_oklch(oklab)
        if not _on_hue_plane(oklab, self.hue):
            return None
        lightness = float(np.clip(lightness, 0.0, 1.0))
        max_chroma = float(color_math.max_chroma_for_lh(lightness, self.hue))
        clamped = max(0.0, min(float(chroma), max_chroma))
        chroma_fraction = float(np.clip(clamped / color_math.SRGB_MAX_CHROMA, 0.0, 1.0))
        return (
            float(chroma_fraction * (width - 1.0)),
            float((1.0 - lightness) * (height - 1.0)),
        )


def _on_hue_plane(oklab: Sequence[float], hue: float) -> bool:
    a = float(oklab[1])
    b = float(oklab[2])
    perpendicular = a * math.sin(hue) - b * math.cos(hue)
    along = a * math.cos(hue) + b * math.sin(hue)
    return abs(perpendicular) <= AB_EPSILON and along >= -AB_EPSILON


def _validate_hue(hue: float) -> None:
    if not math.isfinite(hue):
        raise ValueError("hue must be finite")

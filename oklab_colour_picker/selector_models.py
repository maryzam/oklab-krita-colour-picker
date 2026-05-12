"""Pure coordinate models for OKLab selector surfaces.

These scalar models are for pointer interaction and indicator placement.
Renderers should call the vectorized colour-math helpers on coordinate grids
instead of looping over these per-position methods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker import color_math


POSITION_EPSILON = 1e-12
AB_EPSILON = 1e-9
CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9
CHROMA_LIGHTNESS_RING_HALF_WIDTH = 0.5
# Donut band thickness: band_width = min(outer_radius * 0.5, 40 px). Capping
# at 40 px keeps the donut from eating the dock on big monitors — past a
# point a thicker band gives no extra hue precision, only wasted space.
CHROMA_LIGHTNESS_BAND_FRACTION = 0.5
CHROMA_LIGHTNESS_BAND_MAX_PX = 40.0


def chroma_lightness_band_width(outer_radius: float) -> float:
    """Pixel thickness of the hue donut at a given outer radius."""
    return min(float(outer_radius) * CHROMA_LIGHTNESS_BAND_FRACTION, CHROMA_LIGHTNESS_BAND_MAX_PX)

_LIGHTNESS_SNAP_SAMPLES = np.linspace(0.0, 1.0, 257)
_HUE_SNAP_SAMPLES = np.linspace(0.0, math.tau, 361, endpoint=False)
_SNAP_BOUNDARY_ITERATIONS = 20

Position = tuple[float, float]
Size = tuple[float, float]


@dataclass(frozen=True)
class LightnessSliceModel:
    """Circular hue/chroma selector at a fixed OKLab lightness.

    Radius maps linearly to absolute OKLCh chroma in
    ``[0, color_math.SRGB_MAX_CHROMA]``, matching the Lightness tab's x-axis
    extent. Pixels whose chroma exceeds the per-hue sRGB cusp at this
    lightness fall outside the gamut leaf and render transparent, so the
    irregular gamut outline is visible directly on the disk.
    """

    lightness: float

    def __post_init__(self) -> None:
        _validate_lightness(self.lightness)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        geometry = _circle_geometry(position, size)
        if geometry is None:
            return None

        normalized_radius, hue, _, _ = geometry
        chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        if chroma > max_chroma + CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([self.lightness, chroma, hue])

    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = _circle_geometry_arrays(x, y, size)
        if geometry is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, circle_valid = geometry
        chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        oklch = np.stack(
            (
                np.full_like(normalized_radius, self.lightness, dtype=float),
                chroma,
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
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None

        max_chroma = color_math.max_chroma_for_lh(lightness, hue)
        if chroma > max_chroma + CHROMA_EPSILON:
            return None
        if chroma > color_math.SRGB_MAX_CHROMA + CHROMA_EPSILON:
            return None

        normalized_radius = float(np.clip(chroma / color_math.SRGB_MAX_CHROMA, 0.0, 1.0))
        return _position_from_circle(normalized_radius, hue, size)

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """In-gamut colour at the cursor's hue, with chroma clamped to the leaf.

        Used during drag to keep the preview continuous when the cursor leaves
        the gamut leaf. ``color_at_position`` returns ``None`` past the leaf;
        this variant instead clamps to the per-(L, hue) sRGB cusp at the same
        hue so the preview slides along the boundary. Cursor positions outside
        the disk are projected back to the disk rim at the cursor's angle.
        """
        geometry = _circle_geometry_projected(position, size)
        if geometry is None:
            return None

        normalized_radius, hue = geometry
        desired_chroma = normalized_radius * color_math.SRGB_MAX_CHROMA
        max_chroma = float(color_math.max_chroma_for_lh(self.lightness, hue))
        chroma = max(0.0, min(desired_chroma, max_chroma))
        return color_math.oklch_to_oklab([self.lightness, chroma, hue])


@dataclass(frozen=True)
class LightnessChromaSliceModel:
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
        bounds = _position_in_bounds(position, size)
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
        bounds = _position_in_bounds_arrays(x, y, size)
        if bounds is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

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
        bounds = _size_bounds(size)
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
        geometry = _rect_geometry_projected(position, size)
        if geometry is None:
            return None

        x, y, width, height = geometry
        lightness = 1.0 - y / (height - 1.0)
        desired_chroma = (x / (width - 1.0)) * color_math.SRGB_MAX_CHROMA
        max_chroma = float(color_math.max_chroma_for_lh(lightness, self.hue))
        chroma = max(0.0, min(desired_chroma, max_chroma))
        return color_math.oklch_to_oklab([lightness, chroma, self.hue])


@dataclass(frozen=True)
class HueLightnessSliceModel:
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
        geometry = _circle_geometry(position, size)
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
        geometry = _circle_geometry_arrays(x, y, size)
        if geometry is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

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
        bounds = _size_bounds(size)
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

        return _position_from_circle(1.0 - lightness, hue, (width, height))

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """Nearest in-gamut colour along the cursor's hue spoke."""
        geometry = _circle_geometry_projected(position, size)
        if geometry is None:
            return None

        normalized_radius, hue = geometry
        desired_lightness = 1.0 - normalized_radius
        lightness = _snap_lightness_to_gamut(self.chroma, hue, desired_lightness)
        if lightness is None:
            return None
        return color_math.oklch_to_oklab([lightness, self.chroma, hue])


@dataclass(frozen=True)
class ChromaLightnessModel:
    """Circular hue selector at fixed OKLab lightness and chroma."""

    lightness: float
    chroma: float

    def __post_init__(self) -> None:
        _validate_lightness(self.lightness)
        _validate_chroma(self.chroma)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        geometry = _circle_geometry(position, size)
        if geometry is None:
            return None

        normalized_radius, hue, _, radius = geometry
        if not _on_chroma_lightness_ring(normalized_radius, radius):
            return None
        if self.chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([self.lightness, self.chroma, hue])

    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        geometry = _circle_geometry_arrays(x, y, size)
        if geometry is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, circle_valid = geometry
        oklch = np.stack(
            (
                np.full_like(normalized_radius, self.lightness, dtype=float),
                np.full_like(normalized_radius, self.chroma, dtype=float),
                hue,
            ),
            axis=-1,
        )
        oklab = color_math.oklch_to_oklab(oklch)
        valid = (
            circle_valid
            & _on_chroma_lightness_ring(normalized_radius, _radius_for_size(size))
            & color_math.in_srgb_gamut(color_math.oklab_to_srgb(oklab), epsilon=1e-6)
        )
        return oklab, valid

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None
        if self.chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return _position_from_circle(1.0, hue, size)

    def snapped_color_at_position(
        self, position: Sequence[float], size: Sequence[float]
    ) -> np.ndarray | None:
        """Hue at the cursor angle, projected to the selectable ring."""
        geometry = _circle_geometry_projected(position, size)
        if geometry is None:
            return None

        _, hue = geometry
        hue = _snap_hue_to_gamut(self.lightness, self.chroma, hue)
        if hue is None:
            return None
        return color_math.oklch_to_oklab([self.lightness, self.chroma, hue])


def _circle_geometry(position: Sequence[float], size: Sequence[float]):
    return _circle_geometry_core(position, size, project=False)


def _circle_geometry_projected(position: Sequence[float], size: Sequence[float]):
    geometry = _circle_geometry_core(position, size, project=True)
    if geometry is None:
        return None
    normalized_radius, hue, _, _ = geometry
    return normalized_radius, hue


def _circle_geometry_core(position: Sequence[float], size: Sequence[float], *, project: bool):
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = (float(position[0]), float(position[1]))
    if not project and not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return None
    radius = (min(width, height) - 1.0) / 2.0
    if radius <= 0.0:
        return None

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    dx = x - center_x
    dy = center_y - y
    distance = math.hypot(dx, dy)
    if not project and distance > radius + POSITION_EPSILON:
        return None

    hue = 0.0 if distance <= POSITION_EPSILON else math.atan2(dy, dx) % math.tau
    return min(distance / radius, 1.0), hue, center_x, radius


def _rect_geometry_projected(position: Sequence[float], size: Sequence[float]):
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x = float(np.clip(float(position[0]), 0.0, width - 1.0))
    y = float(np.clip(float(position[1]), 0.0, height - 1.0))
    return x, y, width, height


def _circle_geometry_arrays(x, y, size: Sequence[float]):
    bounds = _position_in_bounds_arrays(x, y, size)
    if bounds is None:
        return None

    x, y, width, height, bounds_valid = bounds
    radius = (min(width, height) - 1.0) / 2.0
    if radius <= 0.0:
        return None

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    dx = x - center_x
    dy = center_y - y
    distance = np.hypot(dx, dy)
    circle_valid = bounds_valid & (distance <= radius + POSITION_EPSILON)
    normalized_radius = np.minimum(distance / radius, 1.0)
    hue = np.where(distance <= POSITION_EPSILON, 0.0, np.mod(np.arctan2(dy, dx), math.tau))
    return normalized_radius, hue, circle_valid


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


def _snap_hue_to_gamut(lightness: float, chroma: float, desired_hue: float) -> float | None:
    desired_hue = float(desired_hue % math.tau)
    if _hue_in_gamut(lightness, chroma, desired_hue):
        return desired_hue

    valid = chroma <= color_math.max_chroma_for_lh(lightness, _HUE_SNAP_SAMPLES) + CHROMA_EPSILON
    valid_hues = _HUE_SNAP_SAMPLES[np.flatnonzero(valid)]
    if not valid_hues.size:
        return None

    clockwise = (valid_hues - desired_hue) % math.tau
    counterclockwise = (desired_hue - valid_hues) % math.tau
    cw_hue = float(valid_hues[int(np.argmin(clockwise))])
    ccw_hue = float(valid_hues[int(np.argmin(counterclockwise))])
    cw_boundary = _bisect_hue_boundary(
        lightness,
        chroma,
        invalid_hue=desired_hue,
        valid_hue=cw_hue,
        clockwise=True,
    )
    ccw_boundary = _bisect_hue_boundary(
        lightness,
        chroma,
        invalid_hue=desired_hue,
        valid_hue=ccw_hue,
        clockwise=False,
    )
    cw_distance = (cw_boundary - desired_hue) % math.tau
    ccw_distance = (desired_hue - ccw_boundary) % math.tau
    return cw_boundary if cw_distance <= ccw_distance else ccw_boundary


def _bisect_hue_boundary(
    lightness: float,
    chroma: float,
    *,
    invalid_hue: float,
    valid_hue: float,
    clockwise: bool,
) -> float:
    invalid_offset = 0.0
    if clockwise:
        valid_offset = (valid_hue - invalid_hue) % math.tau
    else:
        valid_offset = (invalid_hue - valid_hue) % math.tau

    for _ in range(_SNAP_BOUNDARY_ITERATIONS):
        midpoint_offset = (invalid_offset + valid_offset) / 2.0
        if clockwise:
            midpoint = (invalid_hue + midpoint_offset) % math.tau
        else:
            midpoint = (invalid_hue - midpoint_offset) % math.tau
        if _hue_in_gamut(lightness, chroma, midpoint):
            valid_offset = midpoint_offset
        else:
            invalid_offset = midpoint_offset

    if clockwise:
        return float((invalid_hue + valid_offset) % math.tau)
    return float((invalid_hue - valid_offset) % math.tau)


def _hue_in_gamut(lightness: float, chroma: float, hue: float) -> bool:
    return bool(chroma <= color_math.max_chroma_for_lh(lightness, hue) + CHROMA_EPSILON)


def _position_from_circle(normalized_radius: float, hue: float, size: Sequence[float]) -> Position | None:
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    radius = (min(width, height) - 1.0) / 2.0
    if radius <= 0.0:
        return None

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    distance = normalized_radius * radius
    return (
        center_x + distance * math.cos(hue),
        center_y - distance * math.sin(hue),
    )


def _position_in_bounds(position: Sequence[float], size: Sequence[float]):
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = (float(position[0]), float(position[1]))
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return None
    return x, y, width, height


def _position_in_bounds_arrays(x, y, size: Sequence[float]):
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = np.broadcast_arrays(np.asarray(x, dtype=float), np.asarray(y, dtype=float))
    valid = (0.0 <= x) & (x <= width - 1.0) & (0.0 <= y) & (y <= height - 1.0)
    return x, y, width, height, valid


def _size_bounds(size: Sequence[float]):
    width, height = (float(size[0]), float(size[1]))
    if width <= 1.0 or height <= 1.0:
        return None
    return width, height


def _radius_for_size(size: Sequence[float]) -> float:
    width, height = (float(size[0]), float(size[1]))
    return (min(width, height) - 1.0) / 2.0


def _on_chroma_lightness_ring(normalized_radius, radius):
    assert radius > 0.0
    half_pixel_inner = 1.0 - CHROMA_LIGHTNESS_RING_HALF_WIDTH / radius
    band_inner = 1.0 - chroma_lightness_band_width(radius) / radius
    inner = min(half_pixel_inner, band_inner)
    return normalized_radius >= max(0.0, inner)


def _empty_color_grid(x):
    return np.zeros(np.asarray(x).shape + (3,), dtype=float)


def _on_hue_plane(oklab: Sequence[float], hue: float) -> bool:
    a = float(oklab[1])
    b = float(oklab[2])
    perpendicular = a * math.sin(hue) - b * math.cos(hue)
    along = a * math.cos(hue) + b * math.sin(hue)
    return abs(perpendicular) <= AB_EPSILON and along >= -AB_EPSILON


def _validate_lightness(lightness: float) -> None:
    if not math.isfinite(lightness) or not 0.0 <= lightness <= 1.0:
        raise ValueError("lightness must be finite and in [0, 1]")


def _validate_chroma(chroma: float) -> None:
    if not math.isfinite(chroma) or chroma < 0.0:
        raise ValueError("chroma must be finite and non-negative")


def _validate_hue(hue: float) -> None:
    if not math.isfinite(hue):
        raise ValueError("hue must be finite")

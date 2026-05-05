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

from lab_colour_picker import color_math


POSITION_EPSILON = 1e-12
AB_EPSILON = 1e-9
CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9
CHROMA_LIGHTNESS_RING_HALF_WIDTH = 0.5

Position = tuple[float, float]
Size = tuple[float, float]


@dataclass(frozen=True)
class LightnessSliceModel:
    """Circular hue/chroma selector at a fixed OKLab lightness."""

    lightness: float

    def __post_init__(self) -> None:
        _validate_lightness(self.lightness)

    def color_at_position(self, position: Sequence[float], size: Sequence[float]) -> np.ndarray | None:
        geometry = _circle_geometry(position, size)
        if geometry is None:
            return None

        normalized_radius, hue, _, _ = geometry
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        chroma = normalized_radius * max_chroma
        return color_math.oklch_to_oklab([self.lightness, chroma, hue])

    def colors_at_positions(self, x, y, size: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        geometry = _circle_geometry_arrays(x, y, size)
        if geometry is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, valid = geometry
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        oklch = np.stack(
            (
                np.full_like(normalized_radius, self.lightness, dtype=float),
                normalized_radius * max_chroma,
                hue,
            ),
            axis=-1,
        )
        return color_math.oklch_to_oklab(oklch), valid

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None

        max_chroma = color_math.max_chroma_for_lh(lightness, hue)
        if max_chroma <= CHROMA_EPSILON:
            normalized_radius = 0.0 if chroma <= CHROMA_EPSILON else math.inf
        else:
            normalized_radius = chroma / max_chroma
        if normalized_radius > 1.0 + POSITION_EPSILON:
            return None

        return _position_from_circle(float(np.clip(normalized_radius, 0.0, 1.0)), hue, size)


@dataclass(frozen=True)
class HueLightnessModel:
    """Rectangular chroma/lightness selector at a fixed OKLab hue."""

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
        chroma_fraction = x / (width - 1.0)
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if max_chroma <= CHROMA_EPSILON:
            return color_math.oklch_to_oklab([lightness, 0.0, self.hue])
        return color_math.oklch_to_oklab([lightness, chroma_fraction * max_chroma, self.hue])

    def colors_at_positions(self, x, y, size: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        bounds = _position_in_bounds_arrays(x, y, size)
        if bounds is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        x, y, width, height, valid = bounds
        lightness = 1.0 - y / (height - 1.0)
        chroma_fraction = x / (width - 1.0)
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        oklch = np.stack(
            (
                lightness,
                chroma_fraction * max_chroma,
                np.full_like(lightness, self.hue, dtype=float),
            ),
            axis=-1,
        )
        return color_math.oklch_to_oklab(oklch), valid

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
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if max_chroma <= CHROMA_EPSILON:
            chroma_fraction = 0.0 if chroma <= CHROMA_EPSILON else math.inf
        else:
            chroma_fraction = chroma / max_chroma
        if chroma_fraction > 1.0 + POSITION_EPSILON:
            return None

        # L=0 and L=1 are achromatic bands: every x maps to the same colour,
        # so inverse placement uses the left edge as the canonical position.
        return (
            float(np.clip(chroma_fraction, 0.0, 1.0) * (width - 1.0)),
            float((1.0 - lightness) * (height - 1.0)),
        )


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

    def colors_at_positions(self, x, y, size: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        geometry = _circle_geometry_arrays(x, y, size)
        if geometry is None:
            return _empty_color_grid(x), np.zeros_like(np.asarray(x), dtype=bool)

        normalized_radius, hue, circle_valid = geometry
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        valid = (
            circle_valid
            & _on_chroma_lightness_ring(normalized_radius, _radius_for_size(size))
            & (self.chroma <= max_chroma + CHROMA_EPSILON)
        )
        oklch = np.stack(
            (
                np.full_like(normalized_radius, self.lightness, dtype=float),
                np.full_like(normalized_radius, self.chroma, dtype=float),
                hue,
            ),
            axis=-1,
        )
        return color_math.oklch_to_oklab(oklch), valid

    def position_for_color(self, oklab: Sequence[float], size: Sequence[float]) -> Position | None:
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None
        if self.chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return _position_from_circle(1.0, hue, size)


def _circle_geometry(position: Sequence[float], size: Sequence[float]):
    bounds = _position_in_bounds(position, size)
    if bounds is None:
        return None

    x, y, width, height = bounds
    radius = (min(width, height) - 1.0) / 2.0
    if radius <= 0.0:
        return None

    center_x = (width - 1.0) / 2.0
    center_y = (height - 1.0) / 2.0
    dx = x - center_x
    dy = center_y - y
    distance = math.hypot(dx, dy)
    if distance > radius + POSITION_EPSILON:
        return None

    hue = 0.0 if distance <= POSITION_EPSILON else math.atan2(dy, dx) % math.tau
    return min(distance / radius, 1.0), hue, center_x, radius


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
    if radius <= 0.0:
        return False
    return normalized_radius >= max(0.0, 1.0 - CHROMA_LIGHTNESS_RING_HALF_WIDTH / radius)


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

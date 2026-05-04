"""Pure coordinate models for OKLab selector surfaces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lab_colour_picker import color_math


POSITION_EPSILON = 1e-12
HUE_EPSILON = 1e-9
CHROMA_EPSILON = 1e-9
LIGHTNESS_EPSILON = 1e-9


@dataclass(frozen=True)
class LightnessSliceModel:
    """Circular hue/chroma selector at a fixed OKLab lightness."""

    lightness: float

    def color_at_position(self, position, size):
        geometry = _circle_geometry(position, size)
        if geometry is None:
            return None

        normalized_radius, hue, _, _ = geometry
        max_chroma = color_math.max_chroma_for_lh(self.lightness, hue)
        chroma = normalized_radius * max_chroma
        return color_math.oklch_to_oklab([self.lightness, chroma, hue])

    def position_for_color(self, oklab, size):
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

    def color_at_position(self, position, size):
        bounds = _position_in_bounds(position, size)
        if bounds is None:
            return None

        x, y, width, height = bounds
        lightness = 1.0 - y / (height - 1.0)
        chroma_fraction = x / (width - 1.0)
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if max_chroma <= CHROMA_EPSILON and chroma_fraction > CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([lightness, chroma_fraction * max_chroma, self.hue])

    def position_for_color(self, oklab, size):
        bounds = _size_bounds(size)
        if bounds is None:
            return None

        width, height = bounds
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if not -LIGHTNESS_EPSILON <= lightness <= 1.0 + LIGHTNESS_EPSILON:
            return None
        if chroma > CHROMA_EPSILON and not _same_hue(hue, self.hue):
            return None

        lightness = float(np.clip(lightness, 0.0, 1.0))
        max_chroma = color_math.max_chroma_for_lh(lightness, self.hue)
        if max_chroma <= CHROMA_EPSILON:
            chroma_fraction = 0.0 if chroma <= CHROMA_EPSILON else math.inf
        else:
            chroma_fraction = chroma / max_chroma
        if chroma_fraction > 1.0 + POSITION_EPSILON:
            return None

        return (
            float(np.clip(chroma_fraction, 0.0, 1.0) * (width - 1.0)),
            float((1.0 - lightness) * (height - 1.0)),
        )


@dataclass(frozen=True)
class ChromaLightnessModel:
    """Circular hue selector at fixed OKLab lightness and chroma."""

    lightness: float
    chroma: float

    def color_at_position(self, position, size):
        geometry = _circle_geometry(position, size)
        if geometry is None:
            return None

        _, hue, _, _ = geometry
        if self.chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return color_math.oklch_to_oklab([self.lightness, self.chroma, hue])

    def position_for_color(self, oklab, size):
        lightness, chroma, hue = color_math.oklab_to_oklch(oklab)
        if abs(lightness - self.lightness) > LIGHTNESS_EPSILON:
            return None
        if abs(chroma - self.chroma) > CHROMA_EPSILON:
            return None
        if self.chroma > color_math.max_chroma_for_lh(self.lightness, hue) + CHROMA_EPSILON:
            return None
        return _position_from_circle(1.0, hue, size)


def _circle_geometry(position, size):
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
    return min(distance / radius, 1.0), hue, center_x, center_y


def _position_from_circle(normalized_radius, hue, size):
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


def _position_in_bounds(position, size):
    bounds = _size_bounds(size)
    if bounds is None:
        return None

    width, height = bounds
    x, y = (float(position[0]), float(position[1]))
    if not (0.0 <= x <= width - 1.0 and 0.0 <= y <= height - 1.0):
        return None
    return x, y, width, height


def _size_bounds(size):
    width, height = (float(size[0]), float(size[1]))
    if width <= 1.0 or height <= 1.0:
        return None
    return width, height


def _same_hue(left, right):
    delta = abs((left - right + math.pi) % math.tau - math.pi)
    return delta <= HUE_EPSILON

"""NumPy-backed RGBA renderers for selector models."""

from __future__ import annotations

from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
import numpy.typing as npt

from oklab_colour_picker import color_math


class VectorizedSelectorModel(Protocol):
    def colors_at_positions(
        self,
        x: npt.ArrayLike,
        y: npt.ArrayLike,
        size: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        ...


def render_rgba(model: VectorizedSelectorModel, size: Sequence[int]) -> np.ndarray:
    """Render ``model`` to a ``(height, width, 4)`` uint8 RGBA buffer."""

    width, height = _validate_size(size)
    return _render_rgba_cached(model, width, height).copy()


@lru_cache(maxsize=16)
def _render_rgba_cached(model: VectorizedSelectorModel, width: int, height: int) -> np.ndarray:
    x, y = _pixel_grid(width, height)
    oklab, selectable = model.colors_at_positions(x, y, (width, height))
    rgb = _quantize_srgb8(oklab)
    rgba = np.zeros((height, width, 4), dtype=np.uint8)
    rgba[..., :3] = rgb
    rgba[..., 3] = np.where(selectable, 255, 0).astype(np.uint8)
    rgba.setflags(write=False)
    return rgba


def _quantize_srgb8(oklab) -> np.ndarray:
    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
    return np.rint(srgb * 255.0).astype(np.uint8)


@lru_cache(maxsize=16)
def _pixel_grid(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    y, x = np.indices((height, width), dtype=float)
    return x, y


def _validate_size(size: Sequence[int]) -> tuple[int, int]:
    width, height = int(size[0]), int(size[1])
    if width <= 1 or height <= 1:
        raise ValueError("renderer size must be at least 2x2")
    return width, height

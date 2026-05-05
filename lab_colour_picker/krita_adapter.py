"""Krita and Qt boundary adapters for the colour picker controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from lab_colour_picker import color_math
from lab_colour_picker.controller import normalize_oklab_for_krita


class QtSingleShotScheduler:
    """Coalesce work onto the next Qt event-loop turn."""

    def call_soon(self, callback: Callable[[], None]) -> None:
        from PyQt5.QtCore import QTimer

        QTimer.singleShot(0, callback)


class QtForegroundTimer:
    """Small QTimer wrapper with the controller's testable timer interface."""

    def __init__(self) -> None:
        from PyQt5.QtCore import QTimer

        self._timer = QTimer()
        self._callback: Callable[[], None] | None = None
        self._timer.timeout.connect(self._on_timeout)

    def start(self, interval_ms: int, callback: Callable[[], None]) -> None:
        self._callback = callback
        self._timer.start(interval_ms)

    def stop(self) -> None:
        self._timer.stop()

    def _on_timeout(self) -> None:
        if self._callback is not None:
            self._callback()


class KritaForegroundAdapter:
    """Read and write Krita's active foreground colour with null guards."""

    def __init__(self, krita_instance=None) -> None:
        self._krita = krita_instance if krita_instance is not None else _krita_instance()

    def set_foreground(self, oklab: Sequence[float]) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
        managed = _managed_color_from_srgb(srgb)
        view.setForeGroundColor(managed)
        return normalize_oklab_for_krita(oklab)

    def get_foreground(self) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        foreground_color = view.foregroundColor()
        components = _components_from_managed_color(foreground_color)
        if components is None:
            return None
        return color_math.srgb_to_oklab(np.asarray(components[:3], dtype=float))

    def _active_view(self):
        if self._krita is None:
            return None

        active_window = getattr(self._krita, "activeWindow", lambda: None)()
        if active_window is None:
            return None
        return getattr(active_window, "activeView", lambda: None)()


def _krita_instance():
    from krita import Krita

    return Krita.instance()


def _managed_color_from_srgb(srgb: Sequence[float]):
    from krita import ManagedColor

    managed = ManagedColor("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc")
    managed.setComponents([float(srgb[0]), float(srgb[1]), float(srgb[2]), 1.0])
    return managed


def _components_from_managed_color(managed_color) -> list[float] | None:
    if managed_color is None:
        return None

    components = list(managed_color.components())
    if len(components) < 3:
        return None

    rgb = [float(component) for component in components[:3]]
    if max(rgb) > 1.0:
        rgb = [component / 255.0 for component in rgb]
    return [float(np.clip(component, 0.0, 1.0)) for component in rgb]

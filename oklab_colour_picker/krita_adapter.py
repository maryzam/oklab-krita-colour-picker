"""Krita and Qt boundary adapters for the colour picker controller."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from oklab_colour_picker import color_math
from oklab_colour_picker.controller import normalize_oklab_for_krita


SUPPORTED_SRGB_PROFILES = {
    "srgb-elle-v2-srgbtrc.icc",
    "srgb built-in",
    "srgb iec61966-2.1",
}


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

    def __init__(self, krita_instance=None, *, managed_color_factory=None) -> None:
        self._krita = krita_instance if krita_instance is not None else _krita_instance()
        self._managed_color_factory = managed_color_factory

    def set_foreground(self, oklab: Sequence[float]) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        srgb = color_math.clip_srgb(color_math.oklab_to_srgb(oklab))
        managed = _managed_color_from_srgb(srgb, self._managed_color_factory)
        view.setForeGroundColor(managed)
        readback = self.get_foreground()
        return readback if readback is not None else normalize_oklab_for_krita(oklab)

    def get_foreground(self) -> np.ndarray | None:
        view = self._active_view()
        if view is None:
            return None

        foreground_color = view.foregroundColor()
        components = _srgb_components_from_managed_color(foreground_color)
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


def _managed_color_from_srgb(srgb: Sequence[float], managed_color_factory=None):
    if managed_color_factory is None:
        from krita import ManagedColor

        managed_color_factory = ManagedColor

    managed = managed_color_factory("RGBA", "U8", "sRGB-elle-V2-srgbtrc.icc")
    managed.setComponents([float(srgb[0]), float(srgb[1]), float(srgb[2]), 1.0])
    return managed


def _srgb_components_from_managed_color(managed_color) -> list[float] | None:
    if managed_color is None:
        return None

    qcolor_components = _qcolor_srgb_components(managed_color)
    if qcolor_components is not None:
        return qcolor_components

    if not _is_srgb_rgba_u8(managed_color):
        return None

    components = list(managed_color.components())
    if len(components) < 3:
        return None

    rgb = [float(component) for component in components[:3]]
    return [float(np.clip(component, 0.0, 1.0)) for component in rgb]


def _qcolor_srgb_components(managed_color) -> list[float] | None:
    try:
        qcolor = managed_color.toQColor()
    except (AttributeError, TypeError, RuntimeError):
        return None
    if qcolor is None:
        return None
    try:
        rgb = [float(qcolor.redF()), float(qcolor.greenF()), float(qcolor.blueF())]
    except (AttributeError, TypeError, RuntimeError, ValueError):
        return None
    return [float(np.clip(component, 0.0, 1.0)) for component in rgb]


def _is_srgb_rgba_u8(managed_color) -> bool:
    try:
        color_model = managed_color.colorModel()
        color_depth = managed_color.colorDepth()
        color_profile = managed_color.colorProfile()
    except (AttributeError, TypeError, RuntimeError):
        return False

    return (
        str(color_model).upper() == "RGBA"
        and str(color_depth).upper() == "U8"
        and str(color_profile).strip().lower() in SUPPORTED_SRGB_PROFILES
    )

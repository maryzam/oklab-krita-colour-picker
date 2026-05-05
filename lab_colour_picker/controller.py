"""Controller state and Krita foreground synchronization."""

from __future__ import annotations

from typing import Callable, Protocol, Sequence

import numpy as np

from lab_colour_picker import color_math


ForegroundListener = Callable[[np.ndarray], None]


class ForegroundAdapter(Protocol):
    def set_foreground(self, oklab: Sequence[float]) -> np.ndarray | None:
        ...

    def get_foreground(self) -> np.ndarray | None:
        ...


class CommitScheduler(Protocol):
    def call_soon(self, callback: Callable[[], None]) -> None:
        ...


class ForegroundTimer(Protocol):
    def start(self, interval_ms: int, callback: Callable[[], None]) -> None:
        ...

    def stop(self) -> None:
        ...


class ImmediateScheduler:
    """Synchronous fallback for tests or non-Qt hosts."""

    def call_soon(self, callback: Callable[[], None]) -> None:
        callback()


class ColourPickerController:
    """Own colour state and all foreground reads/writes through an adapter."""

    def __init__(
        self,
        adapter: ForegroundAdapter,
        *,
        scheduler: CommitScheduler | None = None,
        foreground_timer: ForegroundTimer | None = None,
        foreground_poll_interval_ms: int = 250,
    ) -> None:
        self._adapter = adapter
        self._scheduler = scheduler if scheduler is not None else ImmediateScheduler()
        self._foreground_timer = foreground_timer
        self._foreground_poll_interval_ms = foreground_poll_interval_ms
        self._foreground_listeners: list[ForegroundListener] = []
        self._selected_colour: np.ndarray | None = None
        self._pending_commit: np.ndarray | None = None
        self._commit_scheduled = False
        self._commit_token = 0
        self._last_committed_token: int | None = None
        self._last_committed_colour: np.ndarray | None = None
        self._dock_visible = True

        if self._foreground_timer is not None:
            self._foreground_timer.start(self._foreground_poll_interval_ms, self.sync_external_foreground)

    @property
    def selected_colour(self) -> np.ndarray | None:
        return None if self._selected_colour is None else self._selected_colour.copy()

    @property
    def last_committed_token(self) -> int | None:
        return self._last_committed_token

    @property
    def last_committed_colour(self) -> np.ndarray | None:
        return None if self._last_committed_colour is None else self._last_committed_colour.copy()

    def add_foreground_listener(self, listener: ForegroundListener) -> None:
        self._foreground_listeners.append(listener)

    def set_preview_colour(self, oklab: Sequence[float] | None) -> None:
        self._selected_colour = None if oklab is None else _as_oklab(oklab)

    def request_foreground_commit(self, oklab: Sequence[float] | None) -> None:
        if oklab is None:
            return

        colour = _as_oklab(oklab)
        self._selected_colour = colour
        self._pending_commit = colour
        if self._commit_scheduled:
            return

        self._commit_scheduled = True
        self._scheduler.call_soon(self._flush_pending_commit)

    def sync_external_foreground(self) -> bool:
        if not self._dock_visible:
            return False

        foreground = self._adapter.get_foreground()
        if foreground is None:
            return False

        colour = _as_oklab(foreground)
        normalized = normalize_oklab_for_krita(colour)
        if self._is_self_feedback(normalized):
            return False
        if self._selected_colour is not None and _normalized_equal(normalize_oklab_for_krita(self._selected_colour), normalized):
            return False

        self._selected_colour = colour
        self._last_committed_token = None
        self._last_committed_colour = None
        for listener in self._foreground_listeners:
            listener(colour.copy())
        return True

    def set_dock_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._dock_visible == visible:
            return

        self._dock_visible = visible
        if self._foreground_timer is None:
            return
        if visible:
            self._foreground_timer.start(self._foreground_poll_interval_ms, self.sync_external_foreground)
        else:
            self._foreground_timer.stop()

    def _flush_pending_commit(self) -> None:
        self._commit_scheduled = False
        colour = self._pending_commit
        self._pending_commit = None
        if colour is None:
            return

        normalized = normalize_oklab_for_krita(colour)
        if self._last_committed_colour is not None and _normalized_equal(normalized, self._last_committed_colour):
            return

        committed = self._adapter.set_foreground(colour)
        if committed is None:
            return

        self._commit_token += 1
        self._last_committed_token = self._commit_token
        self._last_committed_colour = normalize_oklab_for_krita(committed)

    def _is_self_feedback(self, normalized_colour: np.ndarray) -> bool:
        return (
            self._last_committed_token == self._commit_token
            and self._last_committed_colour is not None
            and _normalized_equal(normalized_colour, self._last_committed_colour)
        )


def normalize_oklab_for_krita(oklab: Sequence[float]) -> np.ndarray:
    """Normalize OKLab through Krita's 8-bit sRGB foreground precision."""

    srgb = color_math.clip_srgb(color_math.oklab_to_srgb(_as_oklab(oklab)))
    srgb8 = np.rint(srgb * 255.0).astype(np.uint8)
    return color_math.srgb_to_oklab(srgb8.astype(float) / 255.0)


def _as_oklab(oklab: Sequence[float]) -> np.ndarray:
    colour = np.asarray(oklab, dtype=float)
    if colour.shape != (3,):
        raise ValueError("OKLab colour must contain exactly three components")
    return colour.copy()


def _normalized_equal(left: np.ndarray, right: np.ndarray) -> bool:
    return bool(np.array_equal(normalize_oklab_for_krita(left), normalize_oklab_for_krita(right)))

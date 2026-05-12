"""Controller state and Krita foreground synchronization."""

from __future__ import annotations

import logging
from typing import Callable, Protocol, Sequence

import numpy as np

from oklab_colour_picker import color_math


ForegroundListener = Callable[[np.ndarray], None]
LOGGER = logging.getLogger(__name__)


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
        initially_visible: bool = True,
    ) -> None:
        self._adapter = adapter
        self._scheduler = scheduler if scheduler is not None else ImmediateScheduler()
        self._foreground_timer = foreground_timer
        self._foreground_poll_interval_ms = foreground_poll_interval_ms
        self._foreground_listeners: list[ForegroundListener] = []
        self._selected_colour: np.ndarray | None = None
        self._pending_commit: np.ndarray | None = None
        self._selection_before_pending_commit: np.ndarray | None = None
        self._commit_scheduled = False
        self._commit_token = 0
        self._last_committed_token: int | None = None
        self._last_committed_colour: np.ndarray | None = None
        self._dock_visible = bool(initially_visible)

        if self._dock_visible:
            # Initial foreground is pulled before UI listeners exist; dock
            # consumers seed themselves by reading selected_colour afterward.
            self.sync_external_foreground()
        if self._foreground_timer is not None and self._dock_visible:
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

    def remove_foreground_listener(self, listener: ForegroundListener) -> None:
        try:
            self._foreground_listeners.remove(listener)
        except ValueError:
            pass

    def set_preview_colour(self, oklab: Sequence[float] | None) -> None:
        """Set transient UI preview state without replacing any pending commit."""

        self._selected_colour = None if oklab is None else _as_oklab(oklab)

    def request_foreground_commit(self, oklab: Sequence[float] | None) -> None:
        if oklab is None:
            return

        colour = _as_oklab(oklab)
        if self._pending_commit is None:
            self._selection_before_pending_commit = None if self._selected_colour is None else self._selected_colour.copy()
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
        selected_normalized = None if self._selected_colour is None else normalize_oklab_for_krita(self._selected_colour)
        if selected_normalized is not None and _quantized_equal(selected_normalized, normalized):
            return False

        self._selected_colour = colour
        if self._pending_commit is not None:
            self._selection_before_pending_commit = colour.copy()
        self._last_committed_token = None
        self._last_committed_colour = None
        for listener in self._foreground_listeners:
            try:
                listener(colour.copy())
            except Exception:
                LOGGER.exception("foreground listener failed")
        return True

    def set_dock_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._dock_visible == visible:
            return

        self._dock_visible = visible
        if visible:
            self.sync_external_foreground()
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
        selection_before_commit = self._selection_before_pending_commit
        self._selection_before_pending_commit = None
        if colour is None:
            return

        normalized = normalize_oklab_for_krita(colour)
        if self._last_committed_colour is not None and _quantized_equal(normalized, self._last_committed_colour):
            self._selected_colour = colour
            return

        committed = self._adapter.set_foreground(colour)
        if committed is None:
            self._selected_colour = None if selection_before_commit is None else selection_before_commit.copy()
            return

        self._commit_token += 1
        self._last_committed_token = self._commit_token
        self._last_committed_colour = normalize_oklab_for_krita(committed)
        self._selected_colour = colour

    def _is_self_feedback(self, normalized_colour: np.ndarray) -> bool:
        return (
            self._last_committed_token == self._commit_token
            and self._last_committed_colour is not None
            and _quantized_equal(normalized_colour, self._last_committed_colour)
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


def _quantized_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Compare colours already returned by ``normalize_oklab_for_krita``."""

    return bool(np.array_equal(left, right))

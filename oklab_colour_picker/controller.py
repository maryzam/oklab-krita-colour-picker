"""Controller state and Krita foreground synchronization."""

from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable, Protocol, Sequence

import numpy as np

from oklab_colour_picker import color_math


class ChangeKind(Enum):
    """Why the controller's colour state changed (north-star §2.4).

    ``kind`` is informational for views that need it; it is **not** a source
    tag and must never be used to skip a view. Echo absorption stays local in
    each view's state machine (INV-3).
    """

    PREVIEW = "preview"
    COMMIT = "commit"
    ROLLBACK = "rollback"
    EXTERNAL = "external"
    INITIAL = "initial"


# Listeners receive the broadcast colour and the reason it changed.
ColourListener = Callable[[np.ndarray, ChangeKind], None]
LOGGER = logging.getLogger(__name__)
LOCAL_INTERACTION_SYNC_GRACE_SECONDS = 0.75


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._scheduler = scheduler if scheduler is not None else ImmediateScheduler()
        self._foreground_timer = foreground_timer
        self._foreground_poll_interval_ms = foreground_poll_interval_ms
        self._colour_listeners: list[ColourListener] = []
        self._selected_colour: np.ndarray | None = None
        self._pending_commit: np.ndarray | None = None
        self._selection_before_pending_commit: np.ndarray | None = None
        self._commit_scheduled = False
        self._commit_token = 0
        self._last_committed_token: int | None = None
        self._last_committed_colour: np.ndarray | None = None
        self._clock = clock
        self._local_interaction_deadline: float | None = None
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

    def add_colour_listener(self, listener: ColourListener) -> None:
        self._colour_listeners.append(listener)

    def remove_colour_listener(self, listener: ColourListener) -> None:
        try:
            self._colour_listeners.remove(listener)
        except ValueError:
            pass

    def _broadcast(self, colour: np.ndarray, kind: ChangeKind) -> None:
        """Notify every listener uniformly (no skip-the-originator logic).

        Each view's state machine decides whether to honour or absorb the
        inbound colour; this is what keeps the data flow genuinely one-way
        (north-star §2.1 / §2.4, INV-3).
        """

        for listener in list(self._colour_listeners):
            try:
                listener(colour.copy(), kind)
            except Exception:
                LOGGER.exception("colour listener failed")

    def set_preview_colour(self, oklab: Sequence[float] | None) -> None:
        """Set transient UI preview state without replacing any pending commit.

        Broadcasts ``PREVIEW`` so *other* views can track a mid-drag preview;
        the emitting view self-absorbs the echo via its own state machine
        (north-star §2.4).
        """

        self._selected_colour = None if oklab is None else _as_oklab(oklab)
        if oklab is None:
            return
        self._extend_local_interaction_guard()
        self._broadcast(self._selected_colour, ChangeKind.PREVIEW)

    def request_foreground_commit(self, oklab: Sequence[float] | None) -> None:
        if oklab is None:
            return

        colour = _as_oklab(oklab)
        if self._pending_commit is None:
            self._selection_before_pending_commit = None if self._selected_colour is None else self._selected_colour.copy()
        self._selected_colour = colour
        self._pending_commit = colour
        self._extend_local_interaction_guard()
        if self._commit_scheduled:
            return

        self._commit_scheduled = True
        self._scheduler.call_soon(self._flush_pending_commit)

    def sync_external_foreground(self) -> bool:
        if not self._dock_visible:
            return False
        if self._local_interaction_blocks_external_sync():
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
        self._broadcast(colour, ChangeKind.EXTERNAL)
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
        self._local_interaction_deadline = None
        if colour is None:
            return

        normalized = normalize_oklab_for_krita(colour)
        if self._last_committed_colour is not None and _quantized_equal(normalized, self._last_committed_colour):
            # No adapter write needed (quantizes to the live foreground), but
            # still broadcast COMMIT so views converge on the committed value.
            self._selected_colour = colour
            self._broadcast(self._last_committed_colour, ChangeKind.COMMIT)
            return

        committed = self._adapter.set_foreground(colour)
        if committed is None:
            restored = None if selection_before_commit is None else selection_before_commit.copy()
            self._selected_colour = restored
            if restored is not None:
                self._broadcast(restored, ChangeKind.ROLLBACK)
            return

        self._commit_token += 1
        self._last_committed_token = self._commit_token
        self._last_committed_colour = normalize_oklab_for_krita(committed)
        self._selected_colour = colour
        self._broadcast(self._last_committed_colour, ChangeKind.COMMIT)

    def _is_self_feedback(self, normalized_colour: np.ndarray) -> bool:
        return (
            self._last_committed_token == self._commit_token
            and self._last_committed_colour is not None
            and _quantized_equal(normalized_colour, self._last_committed_colour)
        )

    def _extend_local_interaction_guard(self) -> None:
        self._local_interaction_deadline = self._clock() + LOCAL_INTERACTION_SYNC_GRACE_SECONDS

    def _local_interaction_blocks_external_sync(self) -> bool:
        if self._pending_commit is not None or self._commit_scheduled:
            return True
        if self._local_interaction_deadline is None:
            return False
        if self._clock() < self._local_interaction_deadline:
            return True
        self._local_interaction_deadline = None
        return False


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

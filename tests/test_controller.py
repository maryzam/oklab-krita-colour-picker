import numpy as np

from lab_colour_picker.controller import ColourPickerController, normalize_oklab_for_krita
from lab_colour_picker.krita_adapter import KritaForegroundAdapter


def test_foreground_commits_are_coalesced_to_latest_colour():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)

    first = np.array([0.45, 0.01, 0.02])
    latest = np.array([0.62, -0.03, 0.04])
    controller.request_foreground_commit(first)
    controller.request_foreground_commit(latest)

    assert adapter.set_foreground_calls == []
    assert scheduler.pending_count == 1

    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1
    np.testing.assert_allclose(adapter.set_foreground_calls[0], latest)
    np.testing.assert_allclose(controller.selected_colour, latest)


def test_duplicate_commits_are_suppressed_after_normalization():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    colour = np.array([0.55, 0.02, -0.03])
    same_quantized_colour = normalize_oklab_for_krita(colour)

    controller.request_foreground_commit(colour)
    scheduler.run_pending()
    controller.request_foreground_commit(same_quantized_colour)
    scheduler.run_pending()

    assert len(adapter.set_foreground_calls) == 1


def test_missing_active_krita_view_does_not_record_a_commit():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    colour = np.array([0.5, 0.01, 0.02])

    controller.request_foreground_commit(colour)
    scheduler.run_pending()

    assert adapter.set_foreground_calls == [colour.tolist()]
    assert controller.last_committed_token is None
    assert controller.last_committed_colour is None


def test_external_foreground_sync_updates_selected_colour_once():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    external = np.array([0.4, -0.03, 0.07])
    observed = []
    controller.add_foreground_listener(observed.append)

    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    assert controller.sync_external_foreground() is False

    np.testing.assert_allclose(controller.selected_colour, external)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0], external)


def test_commit_echo_is_suppressed_by_token_and_normalized_colour_match():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    observed = []
    controller.add_foreground_listener(observed.append)

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is False
    assert observed == []
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_external_change_clears_stale_self_feedback_suppression():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    external = np.array([0.38, -0.02, 0.06])
    observed = []
    controller.add_foreground_listener(observed.append)

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is True
    assert len(observed) == 2
    np.testing.assert_allclose(observed[-1], normalize_oklab_for_krita(committed))


def test_hidden_dock_stops_polling_and_visible_dock_restarts_it():
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(FakeKritaAdapter(), scheduler=FakeScheduler(), foreground_timer=timer)

    assert timer.started_intervals == [250]
    controller.set_dock_visible(False)
    controller.set_dock_visible(False)
    controller.set_dock_visible(True)

    assert timer.stop_count == 1
    assert timer.started_intervals == [250, 250]


def test_timer_tick_syncs_foreground_only_while_visible():
    adapter = FakeKritaAdapter()
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler(), foreground_timer=timer)

    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])
    timer.tick()
    np.testing.assert_allclose(controller.selected_colour, adapter.foreground_colour)

    controller.set_dock_visible(False)
    adapter.foreground_colour = np.array([0.72, -0.02, 0.06])
    timer.tick()
    assert not np.allclose(controller.selected_colour, adapter.foreground_colour)


def test_krita_adapter_returns_none_without_active_window():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=None))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


def test_krita_adapter_returns_none_without_active_view():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=None)))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


class FakeScheduler:
    def __init__(self):
        self._callbacks = []

    @property
    def pending_count(self):
        return len(self._callbacks)

    def call_soon(self, callback):
        self._callbacks.append(callback)

    def run_pending(self):
        callbacks = self._callbacks
        self._callbacks = []
        for callback in callbacks:
            callback()


class FakeRepeatingTimer:
    def __init__(self):
        self.started_intervals = []
        self.stop_count = 0
        self._callback = None
        self._running = False

    def start(self, interval_ms, callback):
        self.started_intervals.append(interval_ms)
        self._callback = callback
        self._running = True

    def stop(self):
        self.stop_count += 1
        self._running = False

    def tick(self):
        if self._running and self._callback is not None:
            self._callback()


class FakeKritaAdapter:
    def __init__(self, *, available=True):
        self.available = available
        self.foreground_colour = None
        self.set_foreground_calls = []

    def set_foreground(self, oklab):
        colour = np.asarray(oklab, dtype=float)
        self.set_foreground_calls.append(colour.tolist())
        if not self.available:
            return None
        self.foreground_colour = normalize_oklab_for_krita(colour)
        return self.foreground_colour

    def get_foreground(self):
        return self.foreground_colour


class FakeKrita:
    def __init__(self, *, active_window):
        self._active_window = active_window

    def activeWindow(self):
        return self._active_window


class FakeWindow:
    def __init__(self, *, active_view):
        self._active_view = active_view

    def activeView(self):
        return self._active_view

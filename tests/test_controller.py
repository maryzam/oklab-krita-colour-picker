import numpy as np

from oklab_colour_picker import color_math
from oklab_colour_picker.controller import ChangeKind, ColourPickerController, normalize_oklab_for_krita
from oklab_colour_picker.krita_adapter import KritaForegroundAdapter


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


def test_duplicate_suppression_normalizes_adapter_readback_before_storage():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(readback_offset=np.array([1e-10, -1e-10, 1e-10]))
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

    assert len(adapter.set_foreground_calls) == 1
    np.testing.assert_allclose(adapter.set_foreground_calls[0], colour)
    assert controller.last_committed_token is None
    assert controller.last_committed_colour is None
    assert controller.selected_colour is None


def test_failed_commit_rolls_back_to_previous_selected_colour():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    previous = np.array([0.45, -0.01, 0.03])
    requested = np.array([0.5, 0.01, 0.02])

    controller.set_preview_colour(previous)
    controller.request_foreground_commit(requested)
    scheduler.run_pending()

    np.testing.assert_allclose(controller.selected_colour, previous)


def test_external_foreground_sync_updates_selected_colour_once():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    external = np.array([0.4, -0.03, 0.07])
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    assert controller.sync_external_foreground() is False

    np.testing.assert_allclose(controller.selected_colour, external)
    assert len(observed) == 1
    np.testing.assert_allclose(observed[0][0], external)
    assert observed[0][1] is ChangeKind.EXTERNAL


def test_subscribe_replays_initial_seed_for_listeners_added_after_startup():
    adapter = FakeKritaAdapter()
    external = np.array([0.4, -0.03, 0.07])
    adapter.foreground_colour = external
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []

    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    assert len(observed) == 1
    np.testing.assert_allclose(observed[0][0], external)
    assert observed[0][1] is ChangeKind.INITIAL


def test_subscribe_does_not_replay_when_no_colour_is_available():
    controller = ColourPickerController(FakeKritaAdapter(), scheduler=FakeScheduler())
    observed = []

    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    assert observed == []


def test_visible_controller_reads_foreground_during_initial_startup():
    adapter = FakeKritaAdapter()
    external = np.array([0.4, -0.03, 0.07])
    adapter.foreground_colour = external

    controller = ColourPickerController(adapter, scheduler=FakeScheduler())

    np.testing.assert_allclose(controller.selected_colour, external)


def test_showing_hidden_controller_syncs_foreground_immediately():
    adapter = FakeKritaAdapter()
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(
        adapter,
        scheduler=FakeScheduler(),
        foreground_timer=timer,
        initially_visible=False,
    )
    external = np.array([0.48, 0.02, 0.01])
    adapter.foreground_colour = external

    controller.set_dock_visible(True)

    np.testing.assert_allclose(controller.selected_colour, external)
    assert timer.started_intervals == [250]


def test_showing_hidden_controller_without_timer_still_syncs_foreground():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(
        adapter,
        scheduler=FakeScheduler(),
        initially_visible=False,
    )
    external = np.array([0.38, -0.02, 0.06])
    adapter.foreground_colour = external

    controller.set_dock_visible(True)

    np.testing.assert_allclose(controller.selected_colour, external)


def test_commit_echo_is_suppressed_by_token_and_normalized_colour_match():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is False
    # The commit broadcasts COMMIT; the self-feedback sync adds no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.COMMIT]
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_external_change_clears_stale_self_feedback_suppression():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.65, 0.04, -0.02])
    external = np.array([0.38, -0.02, 0.06])
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.request_foreground_commit(committed)
    scheduler.run_pending()
    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is True
    adapter.foreground_colour = normalize_oklab_for_krita(committed)

    assert controller.sync_external_foreground() is True
    assert [kind for _colour, kind in observed] == [
        ChangeKind.COMMIT,
        ChangeKind.EXTERNAL,
        ChangeKind.EXTERNAL,
    ]
    np.testing.assert_allclose(observed[-1][0], normalize_oklab_for_krita(committed))


def test_external_sync_does_not_refresh_failed_commit_snapshot_during_pending_interaction():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter(available=False)
    controller = ColourPickerController(adapter, scheduler=scheduler)
    first_pending = np.array([0.65, 0.04, -0.02])
    external = np.array([0.38, -0.02, 0.06])
    latest_pending = np.array([0.5, 0.01, 0.02])

    controller.request_foreground_commit(first_pending)
    adapter.foreground_colour = external
    assert controller.sync_external_foreground() is False
    controller.request_foreground_commit(latest_pending)
    scheduler.run_pending()

    assert controller.selected_colour is None


def test_hidden_dock_stops_polling_and_visible_dock_restarts_it():
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(FakeKritaAdapter(), scheduler=FakeScheduler(), foreground_timer=timer)

    assert timer.started_intervals == [250]
    controller.set_dock_visible(False)
    controller.set_dock_visible(False)
    controller.set_dock_visible(True)

    assert timer.stop_count == 1
    assert timer.started_intervals == [250, 250]


def test_initially_hidden_dock_does_not_start_polling_until_visible():
    timer = FakeRepeatingTimer()
    controller = ColourPickerController(
        FakeKritaAdapter(),
        scheduler=FakeScheduler(),
        foreground_timer=timer,
        initially_visible=False,
    )

    assert timer.started_intervals == []
    controller.set_dock_visible(True)

    assert timer.started_intervals == [250]


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


def test_removed_foreground_listener_does_not_receive_updates():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []
    listener = lambda colour, kind: observed.append((colour, kind))
    controller.add_colour_listener(listener)
    controller.remove_colour_listener(listener)

    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])
    assert controller.sync_external_foreground() is True

    assert observed == []


def test_raising_foreground_listener_does_not_block_later_listeners():
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=FakeScheduler())
    observed = []

    def raising_listener(_colour, _kind):
        raise RuntimeError("deleted widget")

    controller.add_colour_listener(raising_listener)
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))
    adapter.foreground_colour = np.array([0.48, 0.02, 0.01])

    assert controller.sync_external_foreground() is True
    assert len(observed) == 1


def test_preview_does_not_replace_pending_commit_before_flush():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    controller = ColourPickerController(adapter, scheduler=scheduler)
    committed = np.array([0.45, 0.01, 0.02])
    preview = np.array([0.62, -0.03, 0.04])

    controller.request_foreground_commit(committed)
    controller.set_preview_colour(preview)
    scheduler.run_pending()

    np.testing.assert_allclose(adapter.set_foreground_calls[0], committed)
    np.testing.assert_allclose(controller.selected_colour, committed)


def test_external_sync_does_not_override_local_preview_before_commit():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.set_preview_colour(preview)

    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, preview)
    # Subscribe replays INITIAL (foreground existed at startup); then
    # set_preview_colour broadcasts PREVIEW (§2.4). The blocked external sync
    # adds no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]
    np.testing.assert_allclose(observed[-1][0], preview)


def test_external_sync_does_not_override_preview_across_repeated_polls():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.set_preview_colour(preview)

    assert controller.sync_external_foreground() is False
    clock.advance(0.25)
    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, preview)
    # INITIAL replay on subscribe, then one PREVIEW; the repeated blocked
    # syncs add no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]


def test_preview_cancellation_does_not_drop_external_sync_guard():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.set_preview_colour(preview)
    controller.set_preview_colour(None)

    assert controller.sync_external_foreground() is False
    # INITIAL replay on subscribe, then PREVIEW for the non-None preview; the
    # None cancel does not broadcast, and the blocked sync adds no EXTERNAL.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.PREVIEW]


def test_external_sync_resumes_after_preview_guard_expires():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    clock = FakeClock()
    previous = np.array([0.45, -0.01, 0.02])
    preview = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler, clock=clock)

    controller.set_preview_colour(preview)
    clock.advance(0.76)

    assert controller.sync_external_foreground() is True
    np.testing.assert_allclose(controller.selected_colour, previous)


def test_external_sync_does_not_override_pending_local_commit_before_flush():
    scheduler = FakeScheduler()
    adapter = FakeKritaAdapter()
    previous = np.array([0.45, -0.01, 0.02])
    committed = np.array([0.62, 0.03, -0.04])
    adapter.foreground_colour = previous
    controller = ColourPickerController(adapter, scheduler=scheduler)
    observed = []
    controller.add_colour_listener(lambda colour, kind: observed.append((colour, kind)))

    controller.request_foreground_commit(committed)

    assert controller.sync_external_foreground() is False
    np.testing.assert_allclose(controller.selected_colour, committed)
    # Only the INITIAL replay so far; the commit broadcast happens at flush.
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL]

    scheduler.run_pending()
    np.testing.assert_allclose(controller.selected_colour, committed)
    assert len(adapter.set_foreground_calls) == 1
    assert [kind for _colour, kind in observed] == [ChangeKind.INITIAL, ChangeKind.COMMIT]


def test_krita_adapter_returns_none_without_active_window():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=None))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


def test_krita_adapter_returns_none_without_active_view():
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=None)))

    assert adapter.get_foreground() is None
    assert adapter.set_foreground([0.5, 0.0, 0.0]) is None


def test_krita_adapter_reads_foreground_through_qcolor_srgb_components():
    view = FakeView(foreground_color=FakeManagedColor(qcolor=FakeQColor(0.25, 0.5, 0.75)))
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    np.testing.assert_allclose(
        adapter.get_foreground(),
        color_math.srgb_to_oklab([0.25, 0.5, 0.75]),
    )


def test_krita_adapter_rejects_non_srgb_foreground_without_qcolor_conversion():
    view = FakeView(foreground_color=FakeManagedColor(model="CMYK", profile="Chemical proof"))
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    assert adapter.get_foreground() is None


def test_krita_adapter_rejects_linear_srgb_fallback_without_qcolor_conversion():
    view = FakeView(foreground_color=FakeManagedColor(profile="linear-sRGB-elle-V2.icc"))
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    assert adapter.get_foreground() is None


def test_krita_adapter_does_not_rescale_normalized_components_above_one():
    # ManagedColor.components() is BGRA; values below pick up red=1.2, green=0.5, blue=0.25.
    managed = FakeManagedColor(components=[0.25, 0.5, 1.2, 1.0])
    view = FakeView(foreground_color=managed)
    adapter = KritaForegroundAdapter(FakeKrita(active_window=FakeWindow(active_view=view)))

    expected = color_math.srgb_to_oklab([1.0, 0.5, 0.25])
    actual = adapter.get_foreground()

    np.testing.assert_allclose(actual, expected)


def test_krita_adapter_writes_components_in_bgra_order():
    managed = FakeManagedColor()
    view = FakeView(foreground_color=managed)

    def factory(*args, **kwargs):
        return managed

    adapter = KritaForegroundAdapter(
        FakeKrita(active_window=FakeWindow(active_view=view)),
        managed_color_factory=factory,
    )

    red_oklab = color_math.srgb_to_oklab([1.0, 0.0, 0.0])
    adapter.set_foreground(red_oklab)

    assert len(view.set_foreground_calls) == 1
    np.testing.assert_allclose(managed.components(), [0.0, 0.0, 1.0, 1.0], atol=1e-6)


def test_krita_adapter_returns_readback_colour_after_setting_foreground():
    readback = FakeManagedColor(qcolor=FakeQColor(0.25, 0.5, 0.75))
    view = FakeView(foreground_color=readback)
    adapter = KritaForegroundAdapter(
        FakeKrita(active_window=FakeWindow(active_view=view)),
        managed_color_factory=FakeManagedColor,
    )

    actual = adapter.set_foreground([0.5, 0.0, 0.0])

    assert len(view.set_foreground_calls) == 1
    np.testing.assert_allclose(actual, adapter.get_foreground())


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


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += float(seconds)


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
    def __init__(self, *, available=True, readback_offset=None):
        self.available = available
        self.readback_offset = None if readback_offset is None else np.asarray(readback_offset, dtype=float)
        self.foreground_colour = None
        self.set_foreground_calls = []

    def set_foreground(self, oklab):
        colour = np.asarray(oklab, dtype=float)
        self.set_foreground_calls.append(colour.tolist())
        if not self.available:
            return None
        self.foreground_colour = normalize_oklab_for_krita(colour)
        if self.readback_offset is not None:
            return self.foreground_colour + self.readback_offset
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


class FakeView:
    def __init__(self, *, foreground_color):
        self._foreground_color = foreground_color
        self.set_foreground_calls = []

    def foregroundColor(self):
        return self._foreground_color

    def setForeGroundColor(self, managed):
        self.set_foreground_calls.append(managed)


class FakeManagedColor:
    def __init__(
        self,
        *managed_color_args,
        components=None,
        model="RGBA",
        depth="U8",
        profile="sRGB-elle-V2-srgbtrc.icc",
        qcolor=None,
    ):
        self._components = components if components is not None else [0.25, 0.5, 0.75, 1.0]
        self._model = model
        self._depth = depth
        self._profile = profile
        self._qcolor = qcolor

    def setComponents(self, components):
        self._components = components

    def components(self):
        return self._components

    def colorModel(self):
        return self._model

    def colorDepth(self):
        return self._depth

    def colorProfile(self):
        return self._profile

    def toQColor(self):
        if self._qcolor is None:
            raise AttributeError("toQColor unavailable")
        return self._qcolor


class FakeQColor:
    def __init__(self, red, green, blue):
        self._red = red
        self._green = green
        self._blue = blue

    def redF(self):
        return self._red

    def greenF(self):
        return self._green

    def blueF(self):
        return self._blue

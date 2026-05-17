"""Pure, Qt-free tests for the selector interaction state machine.

These exercise the State pattern directly through a fake ``Ctx`` — no
QApplication, no widgets — which is the whole point of the humble-object
split: the interaction logic is unit-testable in isolation.
"""

from oklab_colour_picker.selector_interaction import (
    Dragging,
    Idle,
    Keyboard,
    Pinned,
    state_from_name,
)


class FakeCtx:
    """A 1-D selector: x in [0, 10) is in gamut; colour == float(x)."""

    WIDTH = 10

    def __init__(self):
        self.colour = None
        self.previews = []
        self.commits = []

    # Ctx port -------------------------------------------------------
    def set_colour(self, colour):
        self.colour = colour

    def preview(self, colour):
        self.previews.append(colour)

    def commit(self, colour):
        self.commits.append(colour)

    def color_at(self, point):
        x = point[0]
        return float(x) if 0 <= x < self.WIDTH else None

    def drag_colour_at(self, point, last_valid):
        colour = self.color_at(point)
        if colour is not None:
            return colour
        if last_valid is None:
            return None
        return float(min(self.WIDTH - 1, max(0, point[0])))

    def quantized_equal(self, a, b):
        return a is not None and b is not None and round(a) == round(b)

    def model_indicator(self):
        return None

    def model_position(self):
        if self.colour is None or not (0 <= self.colour < self.WIDTH):
            return None
        return (self.colour, 0.0)


def test_idle_absorb_sets_colour_and_stays_idle():
    ctx = FakeCtx()
    state = Idle().absorb(ctx, 3.0)
    assert isinstance(state, Idle)
    assert ctx.colour == 3.0
    assert ctx.previews == [] and ctx.commits == []


def test_click_drag_release_is_idle_dragging_pinned():
    ctx = FakeCtx()
    state = Idle().press(ctx, (2.0, 0.0))
    assert isinstance(state, Dragging)
    assert state.anchor == (2.0, 0.0)
    assert ctx.previews == [2.0]

    state = state.move(ctx, (5.0, 0.0))
    assert isinstance(state, Dragging) and state.anchor == (5.0, 0.0)
    assert ctx.previews == [2.0, 5.0]

    state = state.release(ctx, (7.0, 0.0))
    assert isinstance(state, Pinned)
    assert state.anchor == (7.0, 0.0)
    assert ctx.commits == [7.0]


def test_dragging_ignores_inbound_colour_in_flight():
    ctx = FakeCtx()
    state = Idle().press(ctx, (2.0, 0.0))
    same = state
    state = state.absorb(ctx, 9.0)
    assert state is same
    assert ctx.colour == 2.0  # unchanged by the ignored broadcast


def test_drag_that_never_hits_valid_restores_before_and_does_not_commit():
    ctx = FakeCtx()
    ctx.set_colour(4.0)
    state = Idle().press(ctx, (50.0, 0.0))  # invalid press
    assert isinstance(state, Dragging)
    assert ctx.previews == [None]

    state = state.release(ctx, (60.0, 0.0))
    assert isinstance(state, Idle)
    assert ctx.colour == 4.0
    assert ctx.previews[-1] == 4.0
    assert ctx.commits == []


def test_drag_leaving_gamut_snaps_continuously_and_commits_snapped():
    ctx = FakeCtx()
    state = Idle().press(ctx, (3.0, 0.0))            # valid: last_valid set
    state = state.move(ctx, (99.0, 0.0))             # off-gamut -> snapped
    assert None not in ctx.previews
    state = state.release(ctx, (99.0, 0.0))
    assert isinstance(state, Pinned)
    assert ctx.commits == [9.0]                      # snapped to the rim


def test_keyboard_nav_then_release_commits():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0)
    assert isinstance(state, Keyboard)
    assert ctx.previews == [4.0] and ctx.commits == []

    state = state.nav(ctx, (5.0, 0.0), 5.0)
    assert isinstance(state, Keyboard)
    state = state.key_release(ctx)
    assert isinstance(state, Pinned)
    assert ctx.commits == [5.0]


def test_keyboard_focus_out_flushes_commit():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0).focus_out(ctx)
    assert isinstance(state, Pinned)
    assert ctx.commits == [4.0]


def test_mouse_press_during_keyboard_cancels_without_commit():
    ctx = FakeCtx()
    state = Idle().nav(ctx, (4.0, 0.0), 4.0)
    state = state.press(ctx, (2.0, 0.0))
    assert isinstance(state, Dragging)
    assert ctx.commits == []


def test_pinned_swallows_quantized_equal_echo_but_yields_to_difference():
    ctx = FakeCtx()
    pinned = Pinned(5.0, (5.0, 0.0))

    same = pinned.absorb(ctx, 5.4)  # round-equal -> echo
    assert same is pinned
    assert ctx.colour is None  # not re-applied

    other = pinned.absorb(ctx, 8.0)
    assert isinstance(other, Idle)
    assert ctx.colour == 8.0


def test_reframe_only_resets_pinned():
    ctx = FakeCtx()
    assert isinstance(Pinned(1.0, (1.0, 0.0)).reframe(ctx), Idle)
    idle = Idle()
    assert idle.reframe(ctx) is idle
    drag = Idle().press(ctx, (2.0, 0.0))
    assert drag.reframe(ctx) is drag


def test_state_from_name_round_trips_and_rejects_unknown():
    assert state_from_name("IDLE").name == "IDLE"
    assert state_from_name("DRAGGING", anchor=(1.0, 2.0)).anchor == (1.0, 2.0)
    assert state_from_name("KEYBOARD", anchor=(3.0, 4.0)).anchor == (3.0, 4.0)
    assert state_from_name("PINNED", colour=1.0, anchor=(5.0, 6.0)).anchor == (5.0, 6.0)
    try:
        state_from_name("BOGUS")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

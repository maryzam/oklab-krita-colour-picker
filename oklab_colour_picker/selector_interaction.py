"""Pure, Qt-free interaction state machine for the selector view (north-star §3).

States are *objects* (GoF State pattern): each one owns its data, answers its
own questions (``anchor``, ``broadcast``, ``indicator``) and returns typed
interaction results. There is no conditional dispatch on a state tag and no
ad-hoc "anchored" / "in-flight" state groups — those questions are answered
polymorphically by the state itself.

The module imports nothing from Qt, Krita, or the widget/dock layer, so the
whole machine is unit-testable against a plain fake ``Ctx`` (see
``tests/test_selector_interaction.py``). The Qt ``SelectorWidget`` is a humble
adapter: it translates Qt events into state calls and renders what the state
reports.

``Ctx`` is the narrow port the widget implements. Colours are opaque to this
module; copying/quantization are the widget's responsibility through ``Ctx``.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


Position = tuple[float, float]
Point = tuple[float, float]


class StateKind(str, Enum):
    IDLE = "IDLE"
    DRAGGING = "DRAGGING"
    KEYBOARD = "KEYBOARD"
    PINNED = "PINNED"


class PickKind(str, Enum):
    EXACT = "exact"
    SNAPPED = "snapped"
    INVALID = "invalid"


@dataclass(frozen=True)
class PickResult:
    kind: PickKind
    colour: object | None = None

    @classmethod
    def exact(cls, colour: object) -> "PickResult":
        return cls(PickKind.EXACT, colour)

    @classmethod
    def snapped(cls, colour: object) -> "PickResult":
        return cls(PickKind.SNAPPED, colour)

    @classmethod
    def invalid(cls) -> "PickResult":
        return cls(PickKind.INVALID)


@dataclass(frozen=True)
class PointerPress:
    point: Point


@dataclass(frozen=True)
class PointerMove:
    point: Point


@dataclass(frozen=True)
class PointerRelease:
    point: Point


@dataclass(frozen=True)
class Navigation:
    point: Point
    colour: object


@dataclass(frozen=True)
class KeyRelease:
    pass


@dataclass(frozen=True)
class FocusOut:
    pass


@dataclass(frozen=True)
class Reframe:
    pass


@dataclass(frozen=True)
class Broadcast:
    colour: object | None


SelectorCommand = (
    PointerPress
    | PointerMove
    | PointerRelease
    | Navigation
    | KeyRelease
    | FocusOut
    | Reframe
    | Broadcast
)


@dataclass(frozen=True)
class Ring:
    """One indicator ring to stroke; ``solid`` picks solid vs. dashed."""

    position: Position
    solid: bool


@dataclass(frozen=True)
class Indicator:
    """What the view should draw for the current state (empty == nothing)."""

    rings: tuple[Ring, ...] = ()

    @classmethod
    def nothing(cls) -> "Indicator":
        return cls(())

    @classmethod
    def at(cls, position: Position) -> "Indicator":
        return cls((Ring(position, True),))


class Ctx(Protocol):
    """Narrow port the widget implements for the state machine to use."""

    @property
    def colour(self) -> object | None: ...
    def set_colour(self, colour: object | None) -> None: ...
    def preview(self, colour: object | None) -> None: ...
    def commit(self, colour: object) -> None: ...
    def pick(self, point: Point) -> PickResult: ...
    def quantized_equal(self, a: object | None, b: object | None) -> bool: ...
    def model_indicator(self) -> Indicator: ...
    def model_position(self) -> Position | None: ...


@dataclass(frozen=True)
class InteractionResult:
    state: "State"
    handled: bool = True
    rendered_broadcast: bool = False


class State(ABC):
    """Base state. Every event defaults to a no-op self-transition, so each
    concrete state only overrides the events it actually handles."""

    kind: StateKind

    @property
    def name(self) -> str:
        return self.kind.value

    @property
    def anchor(self) -> Position | None:
        return None

    def indicator(self, ctx: Ctx) -> Indicator:
        return ctx.model_indicator()

    def indicator_position(self, ctx: Ctx) -> Position | None:
        anchor = self.anchor
        return anchor if anchor is not None else ctx.model_position()

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return self.indicator_position(ctx) or fallback

    # Events ----------------------------------------------------------
    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self, handled=True, rendered_broadcast=False)

    def press(self, ctx: Ctx, point: Point) -> InteractionResult:
        return _begin_drag(ctx, point)

    def move(self, ctx: Ctx, point: Point) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def release(self, ctx: Ctx, point: Point) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def nav(self, ctx: Ctx, point: Point, colour: object) -> InteractionResult:
        return _begin_keyboard(ctx, point, colour)

    def key_release(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def focus_out(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(self, handled=False)

    def reframe(self, ctx: Ctx) -> InteractionResult:
        """Model or widget-size change (north-star §3.2)."""

        return InteractionResult(self)


class _Anchored(State):
    """States that pin the indicator to a concrete pixel (DRAGGING/KEYBOARD/
    PINNED). The anchor *is* the indicator — no model round trip (INV-2)."""

    _anchor: Position

    @property
    def anchor(self) -> Position | None:
        return self._anchor

    def indicator(self, ctx: Ctx) -> Indicator:
        return Indicator.at(self._anchor) if ctx.colour is not None else Indicator.nothing()


class Idle(State):
    """Rendering an externally pushed colour; no anchor (INV-1)."""

    kind = StateKind.IDLE

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        ctx.set_colour(colour)
        return InteractionResult(self, rendered_broadcast=True)


class Dragging(_Anchored):
    """Pointer held; emitting previews. Owns the cancel target and the last
    in-gamut colour for the out-of-gamut release fallback (INV-6)."""

    kind = StateKind.DRAGGING

    def __init__(
        self, anchor: Position, before: object | None, last_valid: object | None
    ) -> None:
        self._anchor = anchor
        self._before = before
        self._last_valid = last_valid

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self)

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return None

    def move(self, ctx: Ctx, point: Point) -> InteractionResult:
        return self._pick(ctx, point)

    def release(self, ctx: Ctx, point: Point) -> InteractionResult:
        picked = ctx.pick(point)
        colour = _drag_colour(picked, has_last_valid=self._last_valid is not None)
        if colour is not None:
            ctx.set_colour(colour)
            ctx.commit(colour)
            return InteractionResult(Pinned(colour, (float(point[0]), float(point[1]))))
        if self._last_valid is not None:
            ctx.set_colour(self._last_valid)
            ctx.commit(self._last_valid)
            return InteractionResult(Pinned(self._last_valid, self._anchor))
        ctx.set_colour(self._before)
        ctx.preview(self._before)
        return InteractionResult(Idle())

    def _pick(self, ctx: Ctx, point: Point) -> InteractionResult:
        picked = ctx.pick(point)
        colour = _drag_colour(picked, has_last_valid=self._last_valid is not None)
        if colour is not None:
            ctx.set_colour(colour)
            ctx.preview(colour)
            return InteractionResult(
                Dragging((float(point[0]), float(point[1])), self._before, colour)
            )
        if self._last_valid is not None:
            return InteractionResult(self)
        ctx.set_colour(None)
        ctx.preview(None)
        return InteractionResult(Dragging(self._anchor, self._before, None))


class Keyboard(_Anchored):
    """Arrow/page navigation in flight; commit pending until key-up/blur."""

    kind = StateKind.KEYBOARD

    def __init__(self, anchor: Position) -> None:
        self._anchor = anchor

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        return InteractionResult(self)

    def nav(self, ctx: Ctx, point: Point, colour: object) -> InteractionResult:
        ctx.set_colour(colour)
        ctx.preview(colour)
        return InteractionResult(Keyboard((float(point[0]), float(point[1]))))

    def key_release(self, ctx: Ctx) -> InteractionResult:
        return self._flush(ctx)

    def focus_out(self, ctx: Ctx) -> InteractionResult:
        return self._flush(ctx)

    def _flush(self, ctx: Ctx) -> InteractionResult:
        colour = ctx.colour
        if colour is None:
            return InteractionResult(Idle())
        ctx.commit(colour)
        return InteractionResult(Pinned(colour, self._anchor))


class Pinned(_Anchored):
    """Post-commit: holds the committed colour at its pixel until something
    external supersedes it. The deliberate UX terminal state (§3.4)."""

    kind = StateKind.PINNED

    def __init__(self, colour: object, anchor: Position) -> None:
        self._colour = colour
        self._anchor = anchor

    def broadcast(self, ctx: Ctx, colour: object | None) -> InteractionResult:
        if colour is not None and ctx.quantized_equal(colour, self._colour):
            return InteractionResult(self)
        return Idle().broadcast(ctx, colour)

    def reframe(self, ctx: Ctx) -> InteractionResult:
        return InteractionResult(Idle())


def _begin_drag(ctx: Ctx, point: Point) -> InteractionResult:
    return Dragging((float(point[0]), float(point[1])), ctx.colour, None)._pick(ctx, point)


def _begin_keyboard(ctx: Ctx, point: Point, colour: object) -> InteractionResult:
    ctx.set_colour(colour)
    ctx.preview(colour)
    return InteractionResult(Keyboard((float(point[0]), float(point[1]))))


class SelectorInteraction:
    def __init__(self, initial: State | None = None) -> None:
        self._state = initial or Idle()
        self._transition_log: list[StateKind] = [self._state.kind]

    @property
    def state_kind(self) -> StateKind:
        return self._state.kind

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def anchor(self) -> Position | None:
        return self._state.anchor

    @property
    def transition_log(self) -> tuple[str, ...]:
        return tuple(kind.value for kind in self._transition_log)

    def dispatch(self, ctx: Ctx, command: SelectorCommand) -> InteractionResult:
        result = self._handle(ctx, command)
        self._adopt(result.state)
        return result

    def indicator(self, ctx: Ctx) -> Indicator:
        return self._state.indicator(ctx)

    def indicator_position(self, ctx: Ctx) -> Position | None:
        return self._state.indicator_position(ctx)

    def navigation_origin(self, ctx: Ctx, fallback: Position) -> Position | None:
        return self._state.navigation_origin(ctx, fallback)

    def force_for_test(
        self,
        kind: StateKind,
        *,
        colour: object | None = None,
        anchor: Position | None = None,
    ) -> None:
        try:
            factory = _STATE_FACTORIES[kind]
        except KeyError as err:
            raise ValueError(f"unknown selector state: {kind!r}")
        state = factory(colour, anchor or (0.0, 0.0))
        self._adopt(state)

    def _handle(self, ctx: Ctx, command: SelectorCommand) -> InteractionResult:
        try:
            handler = _COMMAND_HANDLERS[type(command)]
        except KeyError as err:
            raise TypeError(f"unknown selector command: {command!r}") from err
        return handler(self._state, ctx, command)

    def _adopt(self, state: State) -> None:
        if state.kind is not self._state.kind:
            self._transition_log.append(state.kind)
        self._state = state


def _drag_colour(picked: PickResult, *, has_last_valid: bool) -> object | None:
    if picked.kind is PickKind.EXACT:
        if picked.colour is None:
            raise AssertionError("exact pick without a colour")
        return picked.colour
    if picked.kind is PickKind.SNAPPED and has_last_valid:
        if picked.colour is None:
            raise AssertionError("snapped pick without a colour")
        return picked.colour
    return None


StateFactory = Callable[[object | None, Position], State]
CommandHandler = Callable[[State, Ctx, Any], InteractionResult]


def _new_idle(_colour: object | None, _anchor: Position) -> State:
    return Idle()


def _new_dragging(colour: object | None, anchor: Position) -> State:
    return Dragging(anchor, colour, None)


def _new_keyboard(_colour: object | None, anchor: Position) -> State:
    return Keyboard(anchor)


def _new_pinned(colour: object | None, anchor: Position) -> State:
    return Pinned(colour, anchor)


_STATE_FACTORIES: dict[StateKind, StateFactory] = {
    StateKind.IDLE: _new_idle,
    StateKind.DRAGGING: _new_dragging,
    StateKind.KEYBOARD: _new_keyboard,
    StateKind.PINNED: _new_pinned,
}


def _handle_press(state: State, ctx: Ctx, command: PointerPress) -> InteractionResult:
    return state.press(ctx, command.point)


def _handle_move(state: State, ctx: Ctx, command: PointerMove) -> InteractionResult:
    return state.move(ctx, command.point)


def _handle_release(state: State, ctx: Ctx, command: PointerRelease) -> InteractionResult:
    return state.release(ctx, command.point)


def _handle_navigation(state: State, ctx: Ctx, command: Navigation) -> InteractionResult:
    return state.nav(ctx, command.point, command.colour)


def _handle_key_release(state: State, ctx: Ctx, _command: object) -> InteractionResult:
    return state.key_release(ctx)


def _handle_focus_out(state: State, ctx: Ctx, _command: object) -> InteractionResult:
    return state.focus_out(ctx)


def _handle_reframe(state: State, ctx: Ctx, _command: object) -> InteractionResult:
    return state.reframe(ctx)


def _handle_broadcast(state: State, ctx: Ctx, command: Broadcast) -> InteractionResult:
    return state.broadcast(ctx, command.colour)


_COMMAND_HANDLERS: dict[type[object], CommandHandler] = {
    PointerPress: _handle_press,
    PointerMove: _handle_move,
    PointerRelease: _handle_release,
    Navigation: _handle_navigation,
    KeyRelease: _handle_key_release,
    FocusOut: _handle_focus_out,
    Reframe: _handle_reframe,
    Broadcast: _handle_broadcast,
}

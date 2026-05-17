"""Pure, Qt-free interaction state machine for the selector view (north-star §3).

States are *objects* (GoF State pattern): each one owns its data, answers its
own questions (``anchor``, ``absorb``, ``indicator``) and returns the next
state from every event. There is no conditional dispatch on a state tag and no
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
from typing import Protocol


Position = tuple[float, float]
Point = tuple[float, float]


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
    def color_at(self, point: Point) -> object | None: ...
    def drag_colour_at(self, point: Point, last_valid: object | None) -> object | None: ...
    def quantized_equal(self, a: object | None, b: object | None) -> bool: ...
    def model_indicator(self) -> Indicator: ...
    def model_position(self) -> Position | None: ...


class State(ABC):
    """Base state. Every event defaults to a no-op self-transition, so each
    concrete state only overrides the events it actually handles."""

    name: str = "?"

    @property
    def anchor(self) -> Position | None:
        return None

    def indicator(self, ctx: Ctx) -> Indicator:
        return ctx.model_indicator()

    def indicator_position(self, ctx: Ctx) -> Position | None:
        anchor = self.anchor
        return anchor if anchor is not None else ctx.model_position()

    # Events ----------------------------------------------------------
    def absorb(self, ctx: Ctx, colour: object | None) -> "State":
        return self

    def press(self, ctx: Ctx, point: Point) -> "State":
        return _begin_drag(ctx, point)

    def move(self, ctx: Ctx, point: Point) -> "State":
        return self

    def release(self, ctx: Ctx, point: Point) -> "State":
        return self

    def nav(self, ctx: Ctx, point: Point, colour: object) -> "State":
        return _begin_keyboard(ctx, point, colour)

    def key_release(self, ctx: Ctx) -> "State":
        return self

    def focus_out(self, ctx: Ctx) -> "State":
        return self

    def reframe(self, ctx: Ctx) -> "State":
        """Model or widget-size change (north-star §3.2)."""

        return self


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

    name = "IDLE"

    def absorb(self, ctx: Ctx, colour: object | None) -> State:
        ctx.set_colour(colour)
        return self


class Dragging(_Anchored):
    """Pointer held; emitting previews. Owns the cancel target and the last
    in-gamut colour for the out-of-gamut release fallback (INV-6)."""

    name = "DRAGGING"

    def __init__(self, anchor: Position, before: object | None, last_valid: object | None) -> None:
        self._anchor = anchor
        self._before = before
        self._last_valid = last_valid

    # An in-flight local gesture always wins; the inbound colour is dropped.
    def absorb(self, ctx: Ctx, colour: object | None) -> State:
        return self

    def move(self, ctx: Ctx, point: Point) -> State:
        return self._pick(ctx, point)

    def release(self, ctx: Ctx, point: Point) -> State:
        colour = ctx.drag_colour_at(point, self._last_valid)
        if colour is not None:
            ctx.set_colour(colour)
            ctx.commit(colour)
            return Pinned(colour, (float(point[0]), float(point[1])))
        if self._last_valid is not None:
            ctx.set_colour(self._last_valid)
            ctx.commit(self._last_valid)
            return Pinned(self._last_valid, self._anchor)
        ctx.set_colour(self._before)
        ctx.preview(self._before)
        return Idle()

    def _pick(self, ctx: Ctx, point: Point) -> State:
        colour = ctx.drag_colour_at(point, self._last_valid)
        if colour is not None:
            ctx.set_colour(colour)
            ctx.preview(colour)
            return Dragging((float(point[0]), float(point[1])), self._before, colour)
        if self._last_valid is not None:
            # Keep cancellation semantics until a valid colour is reached.
            return self
        ctx.set_colour(None)
        ctx.preview(None)
        return Dragging(self._anchor, self._before, None)


class Keyboard(_Anchored):
    """Arrow/page navigation in flight; commit pending until key-up/blur."""

    name = "KEYBOARD"

    def __init__(self, anchor: Position) -> None:
        self._anchor = anchor

    def absorb(self, ctx: Ctx, colour: object | None) -> State:
        return self

    def nav(self, ctx: Ctx, point: Point, colour: object) -> State:
        ctx.set_colour(colour)
        ctx.preview(colour)
        return Keyboard((float(point[0]), float(point[1])))

    def key_release(self, ctx: Ctx) -> State:
        return self._flush(ctx)

    def focus_out(self, ctx: Ctx) -> State:
        return self._flush(ctx)

    def _flush(self, ctx: Ctx) -> State:
        colour = ctx.colour
        if colour is None:
            return Idle()
        ctx.commit(colour)
        return Pinned(colour, self._anchor)


class Pinned(_Anchored):
    """Post-commit: holds the committed colour at its pixel until something
    external supersedes it. The deliberate UX terminal state (§3.4)."""

    name = "PINNED"

    def __init__(self, colour: object, anchor: Position) -> None:
        self._colour = colour
        self._anchor = anchor

    def absorb(self, ctx: Ctx, colour: object | None) -> State:
        if colour is not None and ctx.quantized_equal(colour, self._colour):
            return self  # the echo — swallow it, stay PINNED (INV-3 / INV-4)
        return Idle().absorb(ctx, colour)  # a different colour supersedes the pin

    def reframe(self, ctx: Ctx) -> State:
        return Idle()


def _begin_drag(ctx: Ctx, point: Point) -> State:
    return Dragging((float(point[0]), float(point[1])), ctx.colour, None)._pick(ctx, point)


def _begin_keyboard(ctx: Ctx, point: Point, colour: object) -> State:
    ctx.set_colour(colour)
    ctx.preview(colour)
    return Keyboard((float(point[0]), float(point[1])))


def state_from_name(
    name: str, *, colour: object | None = None, anchor: Position | None = None
) -> State:
    """Build a state by name. Orchestration/test hook for §3.2."""

    if name == "IDLE":
        return Idle()
    if name == "DRAGGING":
        return Dragging(anchor or (0.0, 0.0), colour, None)
    if name == "KEYBOARD":
        return Keyboard(anchor or (0.0, 0.0))
    if name == "PINNED":
        return Pinned(colour, anchor or (0.0, 0.0))
    raise ValueError(f"unknown selector state: {name!r}")

# OKLab Colour Picker — Architecture North Star

Status: aligned. Single source of truth for the in-place widget/dock refactor.

---

## 1. Why this document exists

The lower layers — `color_math.py`, `renderers.py`, `selector_models.py`,
`controller.py` — are in good shape, and that claim is enforced, not asserted:

- **Isolation invariants** (AGENTS.md hard rules, enforced by
  `tests/test_import_discipline.py`): only `plugin.py` / `controller.py` /
  `krita_adapter.py` import Krita; `color_math.py` / `selector_models.py` import
  no Qt or Krita.
- **Test coverage**: `tests/test_color_math.py`, `test_renderers.py`,
  `test_selector_models.py`, `test_controller.py` cover these layers directly.
- **Dependency constraints**: data flows up the layer stack only
  (math → models → renderers → controller); none import the widget/dock layer.

The problems are concentrated in `dock.py`, `widgets/selector.py`, and
`widgets/readout_panel.py`, and are concrete:

1. **Behavioural divergence between selectors.** The tab selectors and the
   shared LCH slider/swatch selector handle the same user gestures
   inconsistently — indicator placement, commit timing, and achromatic
   behaviour differ depending on which surface drove the change.
2. **Edge cases handled poorly across user flows.** Achromatic (chroma=0),
   out-of-gamut drag, keyboard nav, resize, and Krita round-trip each have
   their own ad-hoc handling that breaks when flows combine.
3. **Coupling and bloat.** The widget/dock code grows by near-duplicated
   if/else branches added one edge case at a time, making it long, hard to
   read, and hard to modify without regressing another flow.

### Root cause: the echo loop

`dock._preview_colour` / `_commit_colour` take every `previewed` / `committed`
signal and push `set_selected_colour` back into **all** selectors *including the
widget that emitted it*. The originating widget must then reconcile a colour it
just produced against its own pixel position. That single round-trip is the
parent of every recent bug and hack:

- `SelectorWidget._last_interaction_position`, `_interaction_position_resolves_to`,
  `_record_interaction_position`, the `set_selected_colour` "keep the recorded
  position" block, the `resizeEvent` override drop, the disk-widget overrides.
- The achromatic-indicator series (`9efd159`..`aa7cad7`) — five commits patching
  one missing state transition, not five independent bugs.
- Behaviour differs per selector because each model round-trips colour→position
  differently (achromatic hue collapses to 0, off-slice rejected, chroma
  clamps). The *same* echo yields a *different* indicator depending on the
  active model. This is structurally unfixable by adding more `if`s.

There are four copies of the selected colour (controller, dock panel, each
selector, readout panel) kept coherent only by signal round-trips, each with its
own bolted-on re-entrancy guard.

### Goal

A single source of truth, strictly one-way state flow into dumb views, and an
**explicit** interaction state machine in the views so every transition is named
and tested instead of emergent from a tangle of booleans.

The design patterns, named tech-stack-agnostically:

- **Single source of truth** — one owner (the controller) holds colour state;
  no other component stores an authoritative copy.
- **Unidirectional data flow** — state flows owner → views in one direction;
  views never write shared state directly (Redux-style, for those who want a
  one-word reference).
- **Command/event (intent dispatch)** — views emit `previewed` / `committed`
  intents; the controller decides the next state and broadcasts it.
- **Finite state machine** — per-view interaction state is explicit, local UI
  state only; it never holds colour truth.

Today's defect is the inverse of all four: four mutable copies of the colour
kept in sync by bidirectional signal round-trips.

---

## 2. North-star architecture

### 2.1 One-way data flow

```
 user gesture
      │
      ▼
 SelectorWidget / ReadoutPanel ──(previewed | committed)──► ColourPickerDockPanel
                                                                   │
                                                                   ▼
                                                          ColourPickerController
                                                          (SINGLE SOURCE OF TRUTH:
                                                           colour state, Krita I/O,
                                                           commit debounce, self-feedback)
                                                                   │
                                            controller change broadcast
                                                                   │
                                                                   ▼
                                  dock.show_colour(c) ──► every view.show_colour(c)
```

- Views emit **intent** upward only (`previewed`, `committed`). They never call
  the controller or Krita directly (existing hard rule preserved).
- The controller is the only colour-state owner. The dock is a thin broadcaster
  with no colour memory of its own.
- The controller's outbound broadcast goes to **every** view uniformly. There is
  no "skip the originator" logic and no source tag on signals. Each view's own
  state machine decides whether to honour or absorb an inbound colour
  (see §3). This is what makes the flow genuinely one-way.

### 2.4 Controller change contract (resolves review #1)

Today `controller.py` only notifies listeners during external foreground sync
(`controller.py:150`); `set_preview_colour` / `request_foreground_commit` mutate
state silently. The refactor introduces **one** explicit notification:

```
controller.colour_changed(colour: np.ndarray, kind: ChangeKind)
```

`ChangeKind ∈ {PREVIEW, COMMIT, ROLLBACK, EXTERNAL, INITIAL}`. The
`add/remove_foreground_listener` API is renamed to
`add/remove_colour_listener`; the dock subscribes one listener that calls
`view.show_colour(colour, kind)` on every view.

| Trigger (controller method) | Emits | `kind` | Notes |
|-----------------------------|-------|--------|-------|
| `set_preview_colour(c)` | yes | `PREVIEW` | broadcast so *other* views track mid-drag; emitter self-absorbs via its state machine |
| `request_foreground_commit` → flush success | yes | `COMMIT` | after successful adapter write; carries the normalized committed colour |
| flush failure / rollback | yes | `ROLLBACK` | carries the restored pre-commit colour |
| `sync_external_foreground` (Krita-originated) | yes | `EXTERNAL` | the only kind that may force a `PINNED` view to `IDLE` |
| initial startup foreground pull | yes | `INITIAL` | replay seed for views created before/after (see §2.5) |
| self-feedback / no-op (token+quantized match) | **no** | — | suppressed exactly as today (`_is_self_feedback`) |

`kind` is informational for views that need it (`COMMIT` updates ReadoutPanel's
revert target; `PREVIEW` does not). It is **not** a source tag and must never be
used to skip a view — echo absorption stays local (INV-3).

### 2.5 Dock state ownership & lazy tabs (resolves review #2)

The dock keeps **no authoritative** colour state. The controller is replayable
via `controller.selected_colour`. A lazily-created selector tab seeds itself by
calling `show_colour(controller.selected_colour, kind=INITIAL)` at construction,
then receives all subsequent broadcasts. The dock *may* hold a single
non-authoritative `_last_shown` cache **only** as a perf shortcut for seeding
new tabs without re-reading the controller; it is write-only from the broadcast
path, never read by any commit/preview logic, and is not a fifth source of
truth. Slice-model construction from the seed colour is covered by §5 slice 2,
not slice 4 (see review #5).

**Slice-model rebuild policy (decided in slice 2).** The dock builds the
per-mode slice model from each broadcast colour for every `kind`, but a view
that is mid-gesture (`DRAGGING`/`KEYBOARD`) is skipped entirely — neither its
model nor its colour is touched until the gesture releases (§3.5). This is a
uniform rule for all views, not a source tag. Slice 4 only adds caching so the
rebuild is skipped when the fixed slice coordinate is unchanged; it does not
change this policy.

### 2.2 Layer boundaries (unchanged hard rules, restated)

- Only `plugin.py`, `controller.py`, `krita_adapter.py` may import Krita.
- `color_math.py`, `selector_models.py` stay free of Qt and Krita.
- Widgets never call `setForeGroundColor`; Krita writes go through the
  controller/adapter boundary.
- Selection comes from selector models, never from `QImage.pixelColor`.

### 2.3 Honest model contract

`SelectorModel` becomes an explicit ABC. Instead of the widget probing
`desired_`/`snapped_position_for_color` via `getattr`, the contract exposes a
single `indicator_for_color(colour, size) -> IndicatorSpec | None` returning:

```
IndicatorSpec(desired: Position, snapped: Position | None, out_of_gamut: bool)
```

When `snapped` is set and differs from `desired`, the view draws the existing
dual ring (solid at `desired`, dashed at `snapped`) — preserving the
out-of-gamut cue (resolves review #3). In-gamut colours return
`snapped is None`. Every `getattr(model, ...)` probe in `selector.py` and the
`LightnessSliceDiskWidget._snapped_colour_at` override are deleted. Shared disk
geometry collapses to one helper reused by widgets and
`selector_models._position_from_circle`.

---

## 3. The SelectorWidget state machine

The widget is *already* a state machine smeared across `_pressed`,
`_keyboard_commit_pending`, `_colour_before_drag`, `_last_valid_drag_colour`,
`_last_interaction_position`. We make it explicit.

### 3.1 States

| State      | Meaning                                                            | Anchor        | Indicator source |
|------------|--------------------------------------------------------------------|---------------|------------------|
| `IDLE`     | Rendering an externally pushed colour                              | none          | `model.position_for_color` |
| `DRAGGING` | Pointer held; emitting `previewed`                                 | cursor pixel  | anchor |
| `KEYBOARD` | Arrow/Page navigation in flight; commit pending                    | target pixel  | anchor |
| `PINNED`   | Post-commit; holds `(committed_colour, anchor_pixel)`              | committed pixel | anchor |

`DRAGGING` also retains `colour_before` (cancel target) and `last_valid`
(off-gamut fallback). These belong to the `DRAGGING` state object only and are
discarded on exit.

### 3.2 Transition table

| From | Event | Guard | To | Side effect |
|------|-------|-------|----|-------------|
| IDLE | mouse press (LMB) | point yields colour or starts drag | DRAGGING | snapshot `colour_before`; `previewed` |
| IDLE | nav key | target point valid | KEYBOARD | `previewed` |
| DRAGGING | mouse move | — | DRAGGING | `previewed` (or hold on invalid w/ last_valid) |
| DRAGGING | release | valid colour at point | PINNED | `committed` |
| DRAGGING | release | invalid but `last_valid` set | PINNED | `committed(last_valid)` |
| DRAGGING | release | no valid colour ever | IDLE | restore `colour_before`; `previewed(prev)` |
| KEYBOARD | nav key | target valid | KEYBOARD | `previewed` |
| KEYBOARD | key release / focus out | commit pending | PINNED | `committed` |
| PINNED | `show_colour(c)` | `c` ≈ pinned (model quantization) | PINNED | swallow (this is the echo) |
| PINNED | `show_colour(c)` | `c` ≉ pinned | IDLE | render from model |
| PINNED | model/size change | — | IDLE | render from model |
| any non-IDLE | mouse press | — | DRAGGING | cancels pending keyboard commit |

### 3.3 Core invariants

- **INV-1 — anchor lifetime.** An anchor pixel exists *only* in
  `DRAGGING`/`KEYBOARD`/`PINNED`. `IDLE` has no anchor; its indicator is purely
  `model.position_for_color`. (Kills `_last_interaction_position` persistence and
  the resize/model reconciliation hacks.)
- **INV-2 — indicator is a pure function of state.** States with an anchor draw
  a solid ring at the anchor. `IDLE` draws from
  `model.indicator_for_color(colour, size)`: solid at `desired`, plus a dashed
  ring at `snapped` when present (out-of-gamut). No `if`-ladder and no `getattr`
  probing in the widget — the dual-ring OOG cue lives in the model contract, not
  widget branching.
- **INV-3 — echo absorption is local.** Only `PINNED` swallows an inbound
  colour, and only when it quantizes-equal to the pinned colour. The dock/
  controller never special-cases the source.
- **INV-4 — quantization parity.** The PINNED equality test uses the *model/
  controller* quantization (`controller.normalize_oklab_for_krita` /
  `_quantized_equal`), never raw float `==`, or PINNED↔IDLE will flicker.
- **INV-5 — one writer.** A view emits intent; only the controller mutates
  colour state. No view reads another view's state.
- **INV-6 — out-of-gamut continuity.** During `DRAGGING`, leaving the gamut leaf
  snaps via `model.snapped_color_at_position`; the preview stays continuous and
  `last_valid` tracks the last in-gamut colour for release fallback.

### 3.4 PINNED — the deliberate UX decision

`PINNED` encodes "the indicator stays where the user clicked, even for
achromatic / off-slice colours, until something external changes the colour."
This is the behaviour the achromatic commit series fought for, now expressed as
**one terminal state with one exit rule** instead of five scattered patches.
Dropping `PINNED` (IDLE straight after commit) is simpler but reintroduces the
visible snap-to-canonical-hue=0 on achromatic clicks. Adopted: keep `PINNED`
(see Open Decisions §6 — reversible).

### 3.5 External colour changes (another widget, canvas eyedropper)

"External" means any colour change the controller did not originate from *this*
view: another selector tab, the LCH slider/swatch, Krita's canvas colour picker
/ eyedropper, or a script setting the foreground. They all arrive identically —
as a controller broadcast — so the state machine needs no per-source logic; it
reacts to `kind` (§2.4):

- **`COMMIT` / `EXTERNAL` / `INITIAL` / `ROLLBACK`** while this view is `IDLE`
  → render the new colour from the model (this is the normal cross-widget and
  canvas-pick path: pick on canvas → controller `EXTERNAL` → every idle
  selector and the readout repaint).
- The same kinds while this view is `PINNED` → exit to `IDLE` and render. The
  user's pin is local to the gesture that set it; an external pick supersedes
  it (this is the `PINNED → IDLE` row of §3.2 / INV-3 false branch).
- The same kinds while this view is `DRAGGING` or `KEYBOARD` → **ignored**.
  An in-flight local gesture wins; the controller's local-interaction guard
  (`LOCAL_INTERACTION_SYNC_GRACE_SECONDS`) already suppresses competing
  `EXTERNAL` syncs during this window, so a canvas pick mid-drag is dropped
  rather than fighting the drag.
- `PREVIEW`-kind from another view while this view is `IDLE` → render
  (cross-widget live preview); while `PINNED`/`DRAGGING`/`KEYBOARD` → same rules
  as above.

`ReadoutPanel`'s edit-latch (§3.6) is the readout-specific instance of the
`DRAGGING`/`KEYBOARD`-ignore rule above. No view ever reads the originating
widget; absorption stays local (INV-3).

### 3.6 ReadoutPanel — reduced machine

`ReadoutPanel` adopts the same contract with a two-state machine: `IDLE` and
`EDITING` (slider down / spinbox or hex focused). `EDITING` emits `previewed`;
exit emits `committed`. It has no anchor concept (no spatial indicator). The
private `_syncing` guard is replaced by `state == EDITING` ⇒ defer inbound
`show_colour`.

**External-change-during-edit policy (resolves review #4).** While `EDITING`,
an inbound broadcast is *latched* (last value kept, not applied), regardless of
`kind`. On `EDITING` exit:

- exit via **commit** → the user's commit wins; the latched value is discarded
  (the controller's self-feedback/rollback path then reconciles state).
- exit via **cancel** (Esc / focus-out with no change) → the latched value, if
  any, is applied immediately so the panel re-converges on Krita's colour.

Rationale: never destroy an in-flight edit, but never strand the panel on a
stale colour after the user backs out. `EXTERNAL`-kind changes do not interrupt
editing; they only take effect once the user is no longer editing. Same one-way
contract as selectors; no special dock handling.

---

## 4. Behavioural test plan

Tests are the contract for this refactor and **must be written and green before
the production change in each slice** (red-green per PR). Existing
`tests/test_widgets.py`, `tests/test_dock_integration.py`,
`tests/test_controller.py` already cover much of the legacy behaviour; the new
suite re-expresses intended behaviour against the state machine.

### 4.1 Quality indicators (acceptance gates)

| Indicator | Target |
|-----------|--------|
| State coverage | Every state in §3.1 entered by ≥1 test |
| Transition coverage | Every row of the §3.2 table exercised, including guard-false branches |
| Invariant coverage | One named test per INV-1..INV-6 asserting the invariant directly |
| Core-flow regression | Click, drag, keyboard, tab-switch, Krita round-trip pass on **all three** selector models |
| Randomization | Property-based fuzz (see §4.4) over colour space + widget sizes, ≥200 examples/property, deterministic seed in CI |
| No-flicker | Echo round-trip asserts zero spurious state changes (transition log length == expected) |
| Determinism | Offscreen Qt; no sleeps; no wall-clock; fake clock/timer/adapter (existing pattern) |

A slice is not mergeable until its row of the §5 table is green and the
indicators above hold for the touched surface.

### 4.2 Core flows (per model: LightnessSlice, HueLightnessSlice, LightnessChromaSlice)

1. Click in-gamut → `committed` once → state `PINNED` → indicator at click pixel.
2. Press-drag across gamut → ordered `previewed`s → release → single `committed`.
3. Keyboard nudge → `previewed` per step → key release → single `committed`.
4. Tab switch preserves colour; new selector enters `IDLE` and draws indicator
   from its model.
5. External Krita foreground change (via controller) → active view in `IDLE`
   re-renders; a `PINNED` view exits to `IDLE` (INV-3 false branch).

### 4.3 Secondary flows

- Drag leaving the gamut leaf → snapped continuous preview (INV-6); release
  commits `last_valid`.
- Drag that never hits a valid colour → release restores `colour_before`, no
  `committed`.
- Mouse press during pending keyboard commit cancels it (last table row).
- Focus loss during `KEYBOARD` flushes the pending commit.
- Resize during `PINNED`: indicator follows the model after size change
  (INV-1) — no stale absolute pixel.
- ReadoutPanel slider drag emits `previewed`s then one `committed`; hex/spin
  commit-only paths.
- IDLE indicator for an out-of-gamut colour draws the dual ring (solid
  `desired` + dashed `snapped`); in-gamut draws a single ring (INV-2 /
  `IndicatorSpec`).
- `EXTERNAL` broadcast arriving mid-edit in ReadoutPanel is latched; commit-exit
  discards it, cancel-exit applies it (§3.6).

### 4.4 Edge cases (named tests, including randomized)

- **Chroma == 0 (achromatic).** Click a neutral pixel: `PINNED` keeps the click
  pixel; echo (`show_colour` with the same neutral) is swallowed, **no** snap to
  hue=0 (this is the regression that drove `9efd159`..`aa7cad7`). Explicit
  test asserting the transition log has no `PINNED→IDLE→PINNED`.
- **Lightness 0 and 1 boundaries** on HueLightness/LightnessChroma models.
- **Hue wrap** at 0/τ — `position_for_color` stability across the seam.
- **Off-slice colour** pushed to a model that rejects it → view falls to a
  defined state (no indicator) without raising.
- **Tiny/zero widget size** (width or height ≤ 1) → no exceptions, no indicator.
- **Quantization boundary** — colours that differ in float but collapse under
  Krita 8-bit normalization must be swallowed by `PINNED` (INV-4).
- **Signal payload mutation** — caller mutating an emitted array does not
  corrupt widget state (existing test, retained).

### 4.5 Randomized / property-based testing

Add `hypothesis` as a dev dependency (`requirements-dev.txt`). Properties:

- **P1 round-trip:** for random in-gamut OKLab and random widget size,
  `model.position_for_color` then `model.color_at_position` is stable within
  quantization for all three models.
- **P2 echo idempotence:** from any state, `show_colour(controller-normalized c)`
  applied twice equals applied once (no oscillation).
- **P3 indicator purity:** in `IDLE`, indicator depends only on `(colour, model,
  size)` — independent of prior interaction history (shuffled gesture prefix).
- **P4 no orphan anchor:** after any random gesture sequence ending outside
  `DRAGGING`/`KEYBOARD`/`PINNED`, no anchor is retained (INV-1).

CI runs Hypothesis with a fixed seed/profile for determinism; a separate
nightly/extended profile may widen example counts.

---

## 5. PR sequence (warm context kept inline)

Small, ordered slices. Each PR: tests first (red), implementation (green), no
behaviour change beyond its row. Branch per slice (e.g. `rewrite/02-echo-kill`).

| # | Status | Slice | Scope | Acceptance gate |
|---|--------|-------|-------|-----------------|
| 0a | **Done — handover doc on merge** | Guardrail characterization | Enforce lower-layer structure claims: hard-rule import/write/pixel checks, lower-layer no UI-layer imports, direct coverage-module checks, and strict xfails for the PR-1 §2.3 model contract. | Suite green; xfails enumerated and linked to PR-1 §2.3 |
| 0b | **Done — handover doc on merge** | Behaviour characterization | Add §4.2–4.4 acceptance tests against *current* behaviour where it matches intent; mark known-bad with xfail referencing the invariant they will satisfy post-refactor. Add `hypothesis` dep + §4.5 properties (xfail allowed). | Suite green; xfails enumerated and linked to INV-/edge IDs |
| 1 | In review | Honest model contract | `SelectorModel` ABC + defaults; split selector-model implementations under `oklab_colour_picker/models/` behind the stable `selector_models.py` facade; delete all `getattr` probes and disk `_snapped_colour_at` override; unify disk geometry helper. | No behaviour change; model tests + import-discipline green; model package covered by pure/lower-layer guardrails |
| 2 | In review | Echo kill + state machine | Introduce explicit state machine in `SelectorWidget` (§3); controller change contract §2.4 (rename listener API, emit `colour_changed`/`kind`); dock dumb broadcaster + lazy-tab seeding §2.5; `IndicatorSpec` model contract §2.3; delete `_last_interaction_position` family. **Decide here:** who builds the per-mode slice model from a broadcast colour and on which `kind` (the *policy*); slice 4 only optimizes its caching. Flip Phase-0 xfails for INV-1..INV-4 and chroma=0. | All §4 core/secondary/edge + INV-1..INV-6; achromatic + OOG dual-ring regression green; transition+state coverage met |
| 3 | Pending | ReadoutPanel unification | Two-state machine; drop `_syncing`; latch/exit policy §3.6; same one-way contract. | ReadoutPanel flows §4.3 incl. external-change-during-edit; P2 holds for panel |
| 4 | Pending | Slice-model rebuild optimization | Keep slice-2 rebuild *policy*; add caching so a mode's slice model is reconstructed only when its fixed slice coordinate actually changes, not per preview tick. | Perf tests stable; no model rebuild during a drag (counter assertion) |

Phases 0–2 deliver the bug fix and the bulk of the simplification. 3–4 are
cleanup/perf and may land later without blocking the fix.

---

## 6. Open decisions

1. **PINNED retained** (§3.4). Adopted with rationale; reversible if product
   prefers the simpler always-model-derived indicator. Flag for sign-off.
2. **Hypothesis dependency** added to `requirements-dev.txt`. If the project
   wants zero new dev deps, P1–P4 degrade to table-driven parametrized tests
   with a fixed pseudo-random seed (lower coverage signal).
3. **Doc vs. PR-plan split.** Kept inline (§5) because the invariants drive the
   slicing; warm context outweighs separation here. Split out only if §5 grows
   its own lifecycle.

---

## 7. Anti-goals

- No restoring or diffing against deleted legacy plugin / old rewrite docs.
- No new colour-state owners; the controller stays the only one.
- No source tags on signals or "skip originator" logic in the dock.
- No absolute-pixel indicator memory surviving outside an interaction state.

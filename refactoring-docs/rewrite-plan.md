# Rewrite Plan: OKLab Colour Selector Krita Plugin

## Position

Rewrite the plugin from scratch while preserving the product intent and the
known-good OKLab conversion formulas. The current implementation is small, but
the core boundaries are wrong: low-level selector widgets render pixels, own
interaction state, sample colours from display images, and call Krita directly.
That coupling makes the observable lag hard to fix cleanly.

The rewrite should use mature boundaries: pure colour math, pure selector
models, vectorized rendering, Qt widgets for interaction only, and a controller
that is solely responsible for communicating with Krita.

### Rewrite over strangler-fig refactor

A strangler-fig refactor would normally be the lower-risk path, but it pays off mainly when the system is in active production use, where every step must keep serving traffic. This plugin is a single-author tool with no live users to protect, and its existing boundaries are wrong in ways that resist incremental correction (selection-from-image, widgets calling Krita, tree-as-LUT). A ground-up rewrite is cheaper here.

Concrete approach:

The entire original plugin implementation has been moved under `legacy-plugin/lab_colour_picker/` folder. The `legacy-plugin` folder also contains original author README and some other artifacts. Treat this folder as readonly. Avoid modifying it or reference any of its content outside the `legacy-plugin` or `refactoring-docs`.

Keep behaviour unchanged, but allow the minimal packaging edits required to
make it load as a separate Krita plugin: distinct desktop file,`X-KDE-Library`, Python package path, dock factory id, and dock title (e.g.
"OKLab Colour Selector (legacy)").

Build the rewrite version under `lab_colour_picker/` from empty.

Both plugins remain installable simultaneously throughout the rewrite, which enables side-by-side exploratory testing in real Krita: same foreground colour, two pickers, immediate visual comparison of latency, gamut handling,indicator placement, and feel.

Once the rewrite reaches the Definition of Done and an acceptance pass, the
tracked refactoring docs and any rewrite-only tracked scaffolding will be
deleted by human. `legacy-plugin/` remains an ignored local reference folder
and is not part of repository cleanup.

## Rationale

The independent review in `refactoring-docs/review.md` correctly identifies most of the
technical failure modes. The strongest verified causes are:

- `SelectorSurface.mouseMoveEvent` calls `updateForegroundColour` on every raw
  mouse movement, and that calls `view.setForeGroundColor` synchronously. This
  can flood Krita with foreground changes and block the UI thread.
- All selector redraws are nested Python loops over a pixel grid followed by
  `QImage`/`QPixmap` allocation. This is too slow for interactive slider
  movement.
- Picked colours are read from the rendered `QImage` via `pixelColor` instead
  of being computed from the selector model. That loses precision, makes
  selection depend on display scaling, and allows transparent/out-of-gamut
  pixels to become selected colours.
- The foreground sync timer polls Krita every 100 ms without null guards and
  compares `ManagedColor` objects directly.
- The 2-3 tree used for hue-to-max-chroma lookup has correctness bugs and is
  unnecessary for fixed hue samples.
- Qt signal handling uses broad `disconnect()` calls instead of
  `QSignalBlocker`.
- Several UI bugs and hard-coded values are symptoms of weak ownership:
  duplicate radio-button wiring, `6.28` instead of `math.tau`, hard-coded
  selector sizes, and dead legacy code.

The rewrite should address these as architectural constraints, not isolated
patches.

## Target Architecture

### `plugin.py`

Owns Krita plugin registration and dock construction only.

Responsibilities:

- Register the `DockWidgetFactory`.
- Create the dock widget.
- Wire the dock to the controller.
- Avoid colour math, rendering, and selector-specific logic.

### `controller.py`

Owns application state and all Krita communication.

Responsibilities:

- Track the current selected colour in model form.
- Receive preview/commit signals from widgets.
- Coalesce foreground updates with an event-loop-tick scheduler first, then
  fall back to a frame-rate throttle only if profiling proves Krita cannot
  process foreground commits at event-loop speed.
- Call `view.setForeGroundColor` only from this layer.
- Guard `activeWindow()` and `activeView()` access.
- Suppress self-feedback using a commit token and normalized colour match (see
  below).
- Sync external Krita foreground changes into the active selector.

#### Coalescing via `QTimer.singleShot(0, …)`, with measured fallback

Rationale: a fixed timed throttle (e.g. 16-33 ms) inserts visible latency into
every commit even when Krita can keep up. The cheaper primitive is a dirty
flag plus a zero-delay single-shot timer, which collapses any number of
preview events fired before the next event-loop tick into a single Krita
commit, with no artificial delay.

Important limitation: mouse move events often arrive as separate event-loop
turns. `singleShot(0)` may therefore still commit once per mouse event. That is
acceptable only if measured Krita foreground commits remain cheap. If profiling
shows `setForeGroundColor` itself is the bottleneck, replace the zero-delay
flush with a bounded frame-rate commit timer (for example 16 ms for about
60 Hz, or 33 ms for about 30 Hz) while still always committing the latest
pending colour.

```python
def request_commit(self, colour):
    self._pending = colour
    if not self._scheduled:
        self._scheduled = True
        QTimer.singleShot(0, self._flush)

def _flush(self):
    self._scheduled = False
    colour, self._pending = self._pending, None
    if colour is None or colour == self._last_committed:
        return
    self._commit_token += 1
    self._last_committed = colour
    self._last_committed_token = self._commit_token
    self._krita.set_foreground(colour)
```

#### Self-feedback suppression via commit token and normalized colour match

Rationale: comparing `ManagedColor` objects or floating-point components to
decide whether an external foreground change is "really external" is fragile
(identity equality, quantisation noise, race with Krita's own rounding). A
controller-owned commit token plus normalized colour comparison is
deterministic and testable.

Mechanism:

- The controller holds an integer `_commit_token`, incremented on every
  plugin-initiated commit.
- Each commit records `_last_committed_token = _commit_token` and a normalized
  copy of the colour actually sent to Krita.
- The external-foreground watcher reads Krita's current foreground and
  compares it to the normalized `_last_committed` colour using the same
  quantisation policy used to create `ManagedColor`. If it matches, the change
  is treated as the echo of the plugin's own write and ignored.
- An external change that does not match (different colour) advances state
  exactly once and updates the active selector model.

This keeps the loop-suppression rule expressible as a unit test against a fake
Krita adapter without any timing dependencies.

### `color_math.py`

Pure OKLab/OKLCh/sRGB logic.

Responsibilities:

- Scalar and vectorized conversions exposed under one API. NumPy ufuncs work
  for both scalars and arrays; a single implementation must serve both render
  and pick paths to prevent drift.
- Gamut checking and clipping policy.
- Analytic gamut-boundary helpers for hue/chroma limits (see below). No binary
  search and no tree required at runtime.
- Use standard math functions and `math.tau`. No hand-rolled `cbrt`; use
  `numpy.cbrt` / `math.cbrt`.

#### Analytic max chroma from Ottosson gamut helpers

Source: Björn Ottosson, "How software gets color wrong" / "Finding the
maximum saturation possible for a given hue", supplementary post to the
original OKLab article (<https://bottosson.github.io/posts/gamutclipping/>).
The post gives a closed-form expression for the maximum sRGB-in-gamut
saturation at a given OKLab hue direction `(a_, b_)` where
`a_ = cos(h), b_ = sin(h)`.

Rationale: replaces both the buggy 2-3 tree and the binary search in the
existing `calculateMaxChromas`. It has no arbitrary precision parameter,
vectorises across an entire image, and removes the 2-3 tree as a load-bearing
structure.

Implementation note: `max_saturation(a_, b_)` alone is not the same thing as
"maximum chroma at this lightness". It gives the lower-half saturation ray.
For the lightness selector, implement the full Ottosson helper chain:

- `compute_max_saturation(a_, b_)`
- `find_cusp(a_, b_)`
- `find_gamut_intersection(a_, b_, L1, C1, L0)`
- `max_chroma_for_lh(L, hue)`

Then test `max_chroma_for_lh` against a slow binary-search oracle for a fixed
set of lightness/hue samples before deleting the legacy tree/search approach.
A small flat cache of sampled hue limits is acceptable as an optimization, but
it must be derived from the analytic helper and treated as disposable cache
data, not core logic.

Core snippet (per-channel form; sRGB matrix coefficients elided for brevity):

```python
def compute_max_saturation(a_, b_):
    # Pick the channel whose linear combination first hits 0 or 1.
    # k0..k4 below are precomputed per-channel polynomial coefficients
    # of the form derived in Ottosson's post.
    if   -1.88170328 * a_ - 0.80936493 * b_ > 1: k = R_COEFFS
    elif  1.81444104 * a_ - 1.19445276 * b_ > 1: k = G_COEFFS
    else:                                         k = B_COEFFS
    S = k[0] + k[1]*a_ + k[2]*b_ + k[3]*a_*a_ + k[4]*a_*b_
    # One Halley iteration on f(L) = channel(L; a_, b_) refines S to
    # machine precision; see the source post for the derivation.
    return S
```

Vectorised: `a_`, `b_`, and `L` become NumPy arrays of shape `(H, W)`; channel
selection becomes a `np.where` / boolean-mask blend; the final
`max_chroma_for_lh` result is consumed directly by the model/renderer.

### `selector_models.py`

Pure selector coordinate models.

Responsibilities:

- Convert widget coordinates to OKLab/OKLCh colours.
- Convert OKLab colours back to selector positions.
- Return `None` for non-selectable/out-of-gamut positions.
- Hold selector parameters such as lightness, hue, and chroma.
- Avoid Qt and Krita dependencies.

### `renderers.py`

Fast image generation.

Responsibilities:

- Precompute geometry arrays per selector size.
- Render selector state into RGBA `numpy.uint8` buffers.
- Wrap buffers in `QImage` without per-pixel Python loops.
- Cache reusable data such as hue, radius, masks, and max-chroma tables.

#### Invariant: renderer depends on model, never duplicates it

Both the renderer and the selector model perform the same coordinate-to-colour
math, just at different granularities (scalar at the cursor, vectorised across
the image). To prevent drift between what the user sees and what gets picked,
this is a hard architectural invariant:

> The renderer obtains every colour by calling vectorised methods on
> `selector_models` (or `color_math`). It must not contain its own copy of any
> coordinate or gamut formula.

Deterministic validation (run as part of the renderer test suite, not as
visual review):

- For each selector, render at a small size (e.g. 64×64).
- Pick a fixed grid of probe points covering interior, edge, and out-of-gamut
  regions.
- For each probe `p`: assert
  `renderer_pixel(p) == quantize8(model.color_at_position(p))` for unscaled,
  non-antialiased render buffers, and that alpha agrees on
  in-gamut/out-of-gamut classification. If a later presentation layer adds
  interpolation or antialiasing, keep this invariant at the raw renderer
  buffer layer rather than at the displayed pixmap layer.
- Repeat at a second size (e.g. 200×200) to catch geometry assumptions that
  silently bake in a fixed resolution.

Any divergence is a test failure, not a tolerance question — the renderer is
not allowed to compute colour independently.

### `widgets/`

Qt presentation and interaction only. One file per widget — keep modules
small and focused (`widgets/indicator.py`, `widgets/selector_surface.py`,
`widgets/hue_strip_slider.py`, `widgets/lightness_view.py`,
`widgets/hue_plane_view.py`, `widgets/chroma_view.py`, etc.). The widget set
is expected to grow after the rewrite, so the directory is the right shape
from day one.

Responsibilities:

- Paint selector images and indicators.
- Handle mouse and slider input.
- Emit typed signals such as `previewChanged`, `commitRequested`, and
  `parameterChanged`.
- Use `QSignalBlocker` for programmatic updates.
- Never call Krita APIs directly.

### Krita import discipline

`from krita import *` re-exports Qt symbols, which has caused every selector
in the legacy code to depend on Krita transitively. To keep layers 1-3
testable without Krita:

- Only `plugin.py` and the Krita adapter inside `controller.py` may import
  from `krita`.
- All other modules import Qt directly: `from PyQt5.QtCore import …`,
  `from PyQt5.QtGui import …`, `from PyQt5.QtWidgets import …`.
- `color_math.py` and `selector_models.py` must have zero Qt imports.
- A CI/test-time check (a small `pytest` that scans imports) enforces this.

## Core Interaction Contract

Mouse movement should follow this path:

```text
mouse position
-> selector widget
-> selector model color_at_position(x, y)
-> previewChanged(Lab/RGB)
-> controller
-> coalesced setForeGroundColor
```

The rendered image is not part of colour selection. It is only a visual output
of the model.

## Testing Strategy

Use test-driven development for the rewrite. Each iteration starts by adding or
updating tests that describe the intended behaviour, then implementing only the
code needed to pass those tests.

Recommended test layers:

- Pure unit tests for colour math and selector models. These should run outside
  Krita.
- Renderer tests that verify image shape, dtype, alpha masks, and per-pixel
  agreement with the model (see "renderer depends on model" invariant).
- Controller tests using fake Krita window/view objects to verify coalescing,
  null handling, deduplication, and commit-token feedback suppression.
- Qt widget tests using `pytest-qt` for signal emission and programmatic
  slider updates.
- Manual Krita smoke tests for the final integration path, run side-by-side
  against the legacy plugin.

Prefer deterministic tests over screenshot comparisons. Screenshot/manual tests
are useful for final visual validation, but correctness should mostly live in
pure Python tests.

### Running tests

- `pytest` from the repository root. Layers 1-4 (color_math, selector_models,
  renderers, controller-with-fake-adapter) require no Krita and no display.
- Layer 5 (widgets) requires `pytest-qt` and a display (or `QT_QPA_PLATFORM=
  offscreen`).
- Krita is only required for the layer 7 manual acceptance pass.

### Performance budget

Each iteration's "fast enough" claim must be backed by a numeric measurement,
not a feel test. Targets, on a mid-range desktop CPU (single core, NumPy on
top of the system BLAS shipped with Krita's bundled Python):

- 256×256 lightness-selector full redraw: ≤ 5 ms median, ≤ 10 ms p99.
- 256×256 hue-plane redraw: ≤ 5 ms median.
- 256×256 chroma/lightness redraw: ≤ 5 ms median.
- Per-mouse-move pick (model-only, no render): ≤ 50 µs.
- One coalesced commit per event-loop tick under sustained mouse drag.

A small `tests/perf/` benchmark script exercises each renderer at the target
size and fails CI if the median exceeds budget. The budget is the exit
criterion for iteration 3, not iteration 7.

## Iteration Plan

### 0. Quarantine the legacy plugin

Keep the existing legacy implementation under ignored `legacy-plugin/` as a
local readonly reference. If side-by-side Krita testing requires local legacy
packaging edits, make them outside Git. From this point on the legacy code is
frozen and untracked.

Exit criteria:

- Both the legacy plugin and an empty rewrite skeleton install side by side
  in Krita without colliding on dock id or factory name.

### 1. Test Harness and Pure Colour Math

Tests first:

- `srgb -> OKLab -> sRGB` round trips representative colours within tolerance.
- Out-of-gamut RGB conversion reports a mask or invalid state predictably.
- `math.tau` wrap-around hue cases produce equivalent colours.
- `compute_max_saturation(a_, b_)`, `find_cusp`, and
  `max_chroma_for_lh(L, hue)` match reference values or a slow binary-search
  oracle for a fixed table of hue/lightness samples, and are consistent between
  scalar and vectorised paths.

Implementation:

- Create the new module skeleton.
- Move or reimplement OKLab formulas in `color_math.py`.
- Add scalar conversion functions and the analytic gamut-boundary helpers.
- Add vectorized variants once scalar tests are stable. Use the same
  implementation where possible (NumPy ufuncs handle scalars).
- Use `dataclass(frozen=True)` (or typed `NamedTuple`) for `Lab`, `LCh`,
  `RGB` value types, with type hints throughout the module.

Exit criteria:

- Pure tests pass without Krita.
- No Qt imports in `color_math.py`.
- Import-discipline test passes for `color_math.py`.

### 2. Selector Models

Tests first:

- Lightness selector maps centre to neutral colour at the current lightness.
- Hue-plane selector maps horizontal movement to chroma and vertical movement
  to lightness.
- Chroma/lightness selector maps angle to hue while slider state controls
  chroma and lightness.
- Out-of-bounds and out-of-gamut positions return `None`.
- `position_for_color(color_at_position(p))` is approximately reversible for
  valid points.

Implementation:

- Build pure model classes.
- Replace tree-based hue lookup with calls to `max_chroma_for_lh`. Add a flat
  max-chroma cache only if model tests or profiling show it is useful.
- Keep all geometry independent from Qt widget size assumptions.

Exit criteria:

- Model tests pass without Krita or Qt.
- No selected colour is derived from a rendered image.

### 3. Vectorized Renderers

Tests first:

- Renderers return `(height, width, 4)` `uint8` buffers.
- Alpha is zero for non-selectable/out-of-gamut areas.
- Per-pixel renderer output equals `quantize8(model.color_at_position(p))` on
  a fixed grid of probe points (the model-renderer invariant from the
  `renderers.py` section).
- Rendering at multiple sizes preserves coordinate semantics.
- Performance benchmark passes the budget (5 ms median at 256×256).

Implementation:

- Precompute geometry arrays per size.
- Render Lightness, Hue Plane, and Chroma/Lightness views using NumPy by
  calling vectorised methods on `selector_models`/`color_math` — never a
  duplicate inline implementation.
- Avoid nested Python loops in render paths.

Exit criteria:

- Render budget met (see Performance budget section).
- Render output is disposable; the model remains the source of truth.
- Renderer modules have zero formulas not delegated to `selector_models` or
  `color_math`.

### 4. Controller with Coalesced Krita Boundary

Tests first:

- Multiple `request_commit` calls within the same event-loop tick produce one
  Krita foreground update with the latest colour (`singleShot(0)` coalescing).
- Duplicate colours are not committed repeatedly.
- Missing active window/view does not raise.
- A plugin-initiated commit followed by Krita echoing the same colour does
  not produce a second model update (commit-token and normalized-colour
  suppression).
- A genuinely external foreground change updates the active model exactly
  once.
- Dock visibility off → external-sync watcher and any timers stop; visibility
  on → they resume from current foreground.

Implementation:

- Introduce fake Krita adapters for tests.
- Implement a real Krita adapter for production, behind a thin protocol so
  tests do not need Krita.
- Implement `request_commit` / `_flush` with `QTimer.singleShot(0, …)` and
  the dirty-flag pattern from the controller section.
- Implement commit-token and normalized-colour feedback suppression as
  specified.
- Gate the external-foreground watcher on
  `QDockWidget.visibilityChanged`.
- Keep direct `setForeGroundColor` calls out of widgets.

Exit criteria:

- Krita sync behaviour is covered with deterministic tests.
- Dragging cannot enqueue unbounded foreground updates.
- Plugin performs no work while the dock is hidden.

### 5. Qt Widgets

Tests first:

- Mouse movement over a valid point emits `previewChanged`.
- Mouse press/release emits commit semantics as designed.
- Programmatic slider updates use `QSignalBlocker` and do not recursively emit
  parameter changes.
- View switching updates controls from model state without committing a new
  foreground colour.

Implementation:

- Build a reusable selector image widget.
- Paint the indicator with `QPainter`.
- Build sliders and mode controls.
- Replace `HueEffect` with a proper painted hue strip or dedicated slider
  subclass.

Exit criteria:

- Widgets contain no Krita imports.
- UI state can be driven entirely by controller/model state.

### 6. Dock Integration and Resizable Layout

Tests first:

- Dock construction creates all three selector views.
- Chroma mode selection switches to the chroma view (regression test for the
  legacy radio-wiring bug).
- View switching preserves selected colour and updates selector indicators.
- Resizing the dock changes the rendered selector size without coordinate
  drift; an indicator placed at colour `c` before resize ends up at the same
  colour `c` after resize within 1 px.

Implementation:

- Wire `plugin.py`, dock widget, controller, and widgets.
- Hook `resizeEvent` on selector surfaces; rebuild geometry caches lazily on
  size change.
- The legacy plugin remains available from ignored `legacy-plugin/` for local
  side-by-side testing through iteration 7 acceptance.

Exit criteria:

- The plugin loads in Krita alongside the legacy plugin.
- All three selector modes work.
- Foreground updates propagate promptly without visible backlog.
- Selectors visibly resize with the dock.

### 7. Manual Krita Acceptance Pass

Run the rewrite and the legacy plugin side by side in a single Krita session
and compare directly. Manual checks:

- Dragging in each selector feels immediate, and visibly faster than legacy.
- Krita's other colour widgets track the latest colour without delayed replay.
- Rapid slider movement remains responsive.
- Switching modes positions the indicator correctly.
- Closing all documents or changing active views does not spam errors.
- Hidden dock does not perform unnecessary work (verify with a CPU sampler).
- Out-of-gamut/transparent areas do not set accidental black or transparent
  foreground colours.
- Resizing the dock keeps the picker readable and indicator-accurate at a
  range of sizes.
- After acceptance, delete tracked rewrite-only planning artifacts in a single
  commit if the human approves. Do not attempt to delete ignored
  `legacy-plugin/` from Git.

## Migration Policy

Do not port existing classes one-for-one. The legacy code lives behaviourally
unchanged in ignored `legacy-plugin/` until iteration 7 acceptance, aside from
local packaging edits needed for side-by-side loading. From the legacy code,
reuse only:

- The product concept.
- The three selector modes.
- The OKLab conversion constants, validated by tests.
- Any documentation or screenshots that remain accurate.

Do not carry forward:

- `b_tree.py` (replaced by Ottosson's analytic gamut-boundary helpers and,
  only if useful, a disposable flat cache).
- The single-file `lab_colour_picker.py` legacy module (already dead in
  `__init__.py`).
- Broad `disconnect()` signal patterns (replaced by `QSignalBlocker`).
- Image-sampling selection logic (replaced by model-driven picking).
- Direct Krita calls from selector surfaces (replaced by controller).
- Hand-rolled `cbrt` (replaced by `math.cbrt` / `numpy.cbrt`).
- 100 ms polling timer (replaced by Krita signals + visibility-gated
  watcher).
- `from krita import *` outside `plugin.py` and the Krita adapter.

## Definition of Done

- Tests cover colour math, selector models, renderer-vs-model invariants, and
  controller coalescing + commit-token feedback suppression.
- Performance budget met (5 ms median 256×256 redraw per selector).
- No nested Python per-pixel loops exist in interactive render paths.
- No widget calls `view.setForeGroundColor` directly.
- No selection path reads colour from `QImage.pixelColor`.
- Renderer modules contain no coordinate or gamut formula not delegated to
  `selector_models` / `color_math`.
- Krita access is null-guarded and isolated behind the controller/adapter.
- `from krita import *` appears only in `plugin.py` and the Krita adapter,
  enforced by an import-discipline test.
- Selectors resize with the dock without coordinate drift.
- Plugin performs no work while the dock is hidden.
- The plugin remains responsive during drag and slider operations inside
  Krita, and observably faster than the legacy plugin in side-by-side use.
- Tracked rewrite-only planning artifacts are removed in the same commit that
  closes acceptance, if the human approves that cleanup.

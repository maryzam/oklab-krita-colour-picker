# PR Execution Plan

This document translates `rewrite-plan.md` into the branch and PR sequence for
the rewrite. Keep PRs ordered and merge each one before starting the next,
unless a later PR is explicitly split into independent parallel work.

## PR Rules

- Branch from updated `main`.
- Use one branch per rewrite slice.
- Add or update tests before implementation.
- Keep `legacy-plugin/` untracked and readonly.
- Run `python3 scripts/checks/dev_checks.py --scope staged` before commit.
- Run `python3 scripts/checks/dev_checks.py --scope tracked --pytest` before
  push.

## PR 0: Dev Loop Guardrails

Branch: `rewrite/00-dev-loop-guardrails`

Status: merged.

Scope: track rewrite docs, keep legacy code ignored, add local hooks, add CI,
and document agent rules. This PR creates the deterministic development loop
used by all later rewrite work.

Verification: dev checks pass locally and in GitHub Actions.

## PR 1: Colour Math

Branch: `rewrite/01-color-math`

Scope: create the test harness and implement pure OKLab, OKLCh, sRGB, gamut,
and analytic max-chroma helpers in `lab_colour_picker/color_math.py`.

Expected files: `lab_colour_picker/color_math.py`, package skeleton files,
`tests/` color-math tests, and minimal project test configuration if needed.

Out of scope: selector geometry, Qt widgets, renderers, Krita imports, and
plugin registration.

Verification: pure tests pass outside Krita; import-discipline checks confirm
`color_math.py` has no Qt or Krita dependency.

## PR 2: Selector Models

Branch: `rewrite/02-selector-models`

Scope: implement pure selector coordinate models that convert positions to
colors and colors back to positions. Invalid or out-of-gamut positions return
`None`.

Expected files: `lab_colour_picker/selector_models.py` and focused model
tests.

Out of scope: rendered images, Qt event handling, controller logic, and Krita
foreground updates.

Verification: model tests cover valid picks, invalid areas, gamut boundaries,
and approximate position/color reversibility.

## PR 3: Vectorized Renderers

Branch: `rewrite/03-renderers`

Scope: implement NumPy-backed renderers that produce RGBA buffers from selector
models without per-pixel Python loops.

Expected files: `lab_colour_picker/renderers.py`, renderer tests, and initial
performance benchmark scripts.

Out of scope: Qt painting widgets, Krita integration, and selector interaction
state.

Verification: renderer buffers match selector-model output at probe points,
alpha masks match selectable areas, multiple sizes preserve coordinate
semantics, and 256x256 redraw benchmarks meet the documented budget.

## PR 4: Controller And Krita Boundary

Branch: `rewrite/04-controller`

Scope: implement controller state, fake Krita adapters for tests, real Krita
adapter boundaries, coalesced foreground commits, null guards, and
self-feedback suppression.

Expected files: `lab_colour_picker/controller.py`,
`lab_colour_picker/krita_adapter.py` if separated, and controller tests.

Out of scope: selector widget UI, renderer implementation, and plugin dock
registration beyond adapter interfaces needed for tests.

Verification: fake-adapter tests cover coalescing, duplicate suppression,
missing active window/view handling, external foreground sync, and hidden-dock
timer behavior.

## PR 5: Qt Widgets

Branch: `rewrite/05-widgets`

Scope: implement Qt presentation and interaction widgets that paint selector
images, draw indicators, handle input, and emit typed preview/commit signals.

Expected files: `lab_colour_picker/widgets/` modules and widget tests using
`pytest-qt` where practical.

Out of scope: direct Krita calls, color math formulas duplicated from models,
and plugin registration.

Verification: widget tests cover mouse signals, commit semantics,
`QSignalBlocker` behavior for programmatic updates, and no Krita imports in
widgets.

## PR 6: Dock Integration

Branch: `rewrite/06-dock-integration`

Scope: wire plugin registration, dock construction, controller, widgets,
resizable layout, and mode switching into a loadable Krita plugin.

Expected files: `lab_colour_picker/plugin.py`, dock/layout modules, desktop
plugin metadata, and integration-level tests that can run outside Krita where
possible.

Out of scope: deleting legacy reference files and broad UI redesign beyond the
rewrite plan.

Verification: dock construction creates all selector views, mode switching
works, resizing preserves indicator/color semantics, and the plugin can load
alongside the local legacy reference plugin during manual testing.

## PR 7: Manual Acceptance And Cleanup

Branch: `rewrite/07-acceptance`

Scope: run side-by-side manual Krita acceptance, record the result, and remove
temporary rewrite-only artifacts only when acceptance is complete.

Expected files: acceptance notes and cleanup edits approved by the human.

Out of scope: new feature work or architecture changes unrelated to acceptance
findings.

Verification: manual checks from `rewrite-plan.md` pass, all automated checks
pass, and the rewrite is observably more responsive than the legacy reference
plugin.

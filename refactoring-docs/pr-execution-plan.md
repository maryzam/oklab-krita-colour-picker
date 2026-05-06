# PR Execution Plan

This document translates `rewrite-plan.md` into the branch and PR sequence for
the rewrite. Keep PRs ordered and merge each one before starting the next.

## PR Rules

- Branch from updated `main`.
- Use one branch per rewrite slice.
- Add or update tests before implementation.
- Keep `legacy-plugin/` untracked and readonly.
- Install dev/test dependencies from `requirements-dev.txt` before running
  pytest-backed checks.
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
and analytic max-chroma helpers in `oklab_colour_picker/color_math.py`. Add the
initial import-discipline pytest scan in this PR.

Expected files: `oklab_colour_picker/color_math.py`, package skeleton files,
`tests/` color-math tests, import-discipline tests, and minimal project test
configuration. Establish `requirements-dev.txt` as the dev/test dependency
file and add NumPy and pytest there.

Out of scope: selector geometry, Qt widgets, renderers, Krita imports, and
plugin registration.

Verification: pure tests pass outside Krita; import-discipline tests use an
allow-list shape that permits Krita imports only in plugin/adapter boundary
files and confirm `color_math.py` has no Qt or Krita dependency;
`max_chroma_for_lh(L, hue)` is validated against a slow binary-search oracle
for fixed hue/lightness samples before any tree/search approach is dropped.

## PR 2: Selector Models

Branch: `rewrite/02-selector-models`

Scope: implement pure selector coordinate models that convert positions to
colors and colors back to positions. Invalid or out-of-gamut positions return
`None`.

Expected files: `oklab_colour_picker/selector_models.py` and focused model
tests. Extend the import-discipline tests from PR 1 so `selector_models.py` is
also enforced as Qt-free and Krita-free.

Out of scope: rendered images, Qt event handling, controller logic, and Krita
foreground updates. Do not add a flat max-chroma cache unless profiling proves
it is needed.

Verification: model tests cover valid picks, invalid areas, gamut boundaries,
and approximate position/color reversibility. Import-discipline tests cover
`selector_models.py`.

## PR 3: Vectorized Renderers

Branch: `rewrite/03-renderers`

Scope: implement NumPy-backed renderers that produce RGBA buffers from selector
models without per-pixel Python loops.

Expected files: `oklab_colour_picker/renderers.py`, renderer tests, and
`tests/perf/` performance benchmark tests or scripts wired into the normal
pytest/dev-check path.

Out of scope: Qt painting widgets, Krita integration, and selector interaction
state.

Verification: renderer buffers match selector-model output at probe points,
alpha masks match selectable areas, multiple sizes preserve coordinate
semantics, and 256x256 redraw benchmarks meet the documented budget. The
performance budget is the PR 3 exit gate, and CI must fail if the benchmark
regresses beyond budget.

## PR 4: Controller And Krita Boundary

Branch: `rewrite/04-controller`

Scope: implement controller state, fake Krita adapters for tests, real Krita
adapter boundaries, coalesced foreground commits, null guards, and
self-feedback suppression using the commit-token plus normalized-colour match
policy.

Expected files: `oklab_colour_picker/controller.py`,
`oklab_colour_picker/krita_adapter.py` if separated, and controller tests.

Out of scope: selector widget UI, renderer implementation, and plugin dock
registration beyond adapter interfaces needed for tests.

Verification: fake-adapter tests cover coalescing, duplicate suppression,
missing active window/view handling, external foreground sync,
commit-token/normalized-colour feedback suppression, and hidden-dock timer
behavior.

## PR 5: Qt Widgets

Branch: `rewrite/05-widgets`

Scope: implement Qt presentation and interaction widgets that paint selector
images, draw indicators, handle input, and emit typed preview/commit signals.

Expected files: `oklab_colour_picker/widgets/` modules and widget tests using
`pytest-qt` where practical. Add `pytest-qt` to declared dev dependencies in
the same requirements/config location established by PR 1.

Out of scope: direct Krita calls, color math formulas duplicated from models,
and plugin registration.

Verification: widget tests cover mouse signals, commit semantics,
`QSignalBlocker` behavior for programmatic updates, and the PR 1
import-discipline test enforces no Krita imports in widgets.

## PR 6: Dock Integration

Branch: `rewrite/06-dock-integration`

Scope: wire plugin registration, dock construction, controller, widgets,
resizable layout, mode switching, and `QDockWidget.visibilityChanged` into a
loadable Krita plugin. The visibility signal wiring connects the real dock to
the controller behavior already tested with fakes in PR 4.

Expected files: `oklab_colour_picker/plugin.py`, dock/layout modules, desktop
plugin metadata, and integration-level tests that can run outside Krita where
possible.

Out of scope: deleting legacy reference files and broad UI redesign beyond the
rewrite plan.

Verification: dock construction creates all selector views, mode switching
works, an indicator placed at colour `c` before resize lands at colour `c`
within 1 px after resize, the visibility signal reaches the controller, and
the plugin can load alongside the local legacy reference plugin during manual
testing.

## PR 7: Manual Acceptance And Cleanup

Branch: `rewrite/07-acceptance`

Scope: run side-by-side manual Krita acceptance, record the result, and remove
tracked temporary rewrite artifacts only when acceptance is complete.

Expected files: acceptance notes and cleanup edits approved by the human,
likely including `refactoring-docs/` and any rewrite-only scripts. Do not
delete `legacy-plugin/` from Git because it is intentionally untracked.

Out of scope: new feature work or architecture changes unrelated to acceptance
findings.

Verification: manual checks from `rewrite-plan.md` pass, all automated checks
pass, and the rewrite is observably more responsive than the legacy reference
plugin.

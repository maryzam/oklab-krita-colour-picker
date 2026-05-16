# Agent Development Rules

Follow `refactoring-docs/architecture-north-star.md` — the single source of
truth for the widget/dock refactor (architecture, invariants, state machine,
test plan, and PR sequence). Keep PRs small and ordered by the slice table
there.

The legacy plugin and all earlier rewrite/PR docs are deleted. Do not restore
them, import from them, or validate changes against old/deleted code.

## Hard Rules

- `refactoring-docs/` is tracked and should stay current when the dev loop
  changes.
- Only `oklab_colour_picker/plugin.py`, `oklab_colour_picker/controller.py`, or
  `oklab_colour_picker/krita_adapter.py` may import Krita.
- `oklab_colour_picker/color_math.py` and `oklab_colour_picker/selector_models.py`
  must stay free of Qt and Krita imports.
- Widgets must not call `setForeGroundColor`; Krita writes belong behind the
  controller/Krita adapter boundary.
- Selection must come from selector models, never from `QImage.pixelColor`.

## First-Time Setup

Install tracked Git hooks before making changes in a fresh clone:

```sh
python3 scripts/checks/dev_checks.py --install-hooks
```

The hooks run staged-index checks before commit and tracked-tree checks before
push. GitHub Actions also runs the tracked-tree check on PRs.

## Local Loop

1. Create a branch per slice from the north-star table, e.g. `rewrite/02-echo-kill`.
2. Add or update deterministic tests before implementation (red-green per slice).
3. Run focused tests while developing, then run the pre-push check before PR.

Run checks manually:

```sh
python3 scripts/checks/dev_checks.py --scope staged
python3 scripts/checks/dev_checks.py --scope tracked --pytest
```

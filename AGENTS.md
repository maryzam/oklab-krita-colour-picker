# Agent Development Rules

Follow the rewrite plan in `refactoring-docs/rewrite-plan.md`. Keep PRs small
and ordered by rewrite slice.

## Hard Rules

- `legacy-plugin/` is local reference material only. Read it if needed, but do
  not track it, import it, or copy modules from it.
- `refactoring-docs/` is tracked and should stay current when the dev loop
  changes.
- Only `lab_colour_picker/plugin.py`, `lab_colour_picker/controller.py`, or
  `lab_colour_picker/krita_adapter.py` may import Krita.
- `lab_colour_picker/color_math.py` and `lab_colour_picker/selector_models.py`
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

1. Create a branch per rewrite slice, for example `rewrite/01-color-math`.
2. Add or update deterministic tests before implementation.
3. Run focused tests while developing, then run the pre-push check before PR.
4. Inspect staged files before commit and reject anything under
   `legacy-plugin/`.

Run checks manually:

```sh
python3 scripts/checks/dev_checks.py --scope staged
python3 scripts/checks/dev_checks.py --scope tracked --pytest
```

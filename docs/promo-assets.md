# Promo Assets

Keep the public README visual, but small. The target set is one hero screenshot,
two supporting screenshots, and one optional short motion asset.

## Required

`docs/assets/readme/hero.png`

Show Krita with the OKLab Colour Selector docker open in a realistic workspace.
The docker should be large enough to read, with a saturated but in-gamut colour
selected.

`docs/assets/readme/hue-ring.png`

Show the Hue Ring tab with the central swatch and L/C controls visible.

`docs/assets/readme/selector-tabs.png`

Show the alternate selector views. A single composite image is enough; avoid
adding a separate screenshot for every tab.

## Optional

`docs/assets/readme/demo.webp`

Use a 5-10 second loop that shows dragging a selector, previewing a colour, and
committing it to Krita's foreground colour.

## Capture Notes

- Use a clean Krita workspace with no unrelated panels stealing attention.
- Prefer a medium-light UI theme so selector colours remain readable.
- Capture at 2x scale if possible, then export final PNGs at README size.
- Keep filenames stable so the README can link them directly later.
- Avoid heavy annotations; the screenshot should show the plugin, not a tour.

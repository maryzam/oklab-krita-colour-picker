# OKLab Colour Selector for Krita

A perceptual colour picker docker for Krita, built around OKLab and OKLCh.
Pick colours by lightness, chroma, and hue, preview them interactively, and
commit the result to Krita's foreground colour.

OKLab is designed to make colour edits feel more even to human vision than
RGB or HSL controls. This plugin brings that workflow into Krita as a compact
docker with multiple selector views, numeric controls, swatches, and hex input.

## Highlights

- Pick in OKLab/OKLCh space with perceptual lightness, chroma, and hue controls.
- Switch between Hue/Chroma, Hue/Lightness, Lightness/Chroma, and Hue Ring views.
- Use gradient L/C/H sliders, numeric inputs, current/previous swatches, and hex entry.
- Preview colours while dragging, then commit the chosen colour to Krita.
- Keep selections inside the visible sRGB gamut.

## Install

Requires Krita 5.2 or newer and NumPy available to Krita's Python.

1. Copy `oklab_colour_picker/` and `oklab_colour_picker.desktop` into Krita's
   `pykrita/` folder.
2. Restart Krita.
3. Enable **OKLab Colour Selector** in **Settings > Configure Krita... >
   Python Plugin Manager**.
4. Restart Krita again.
5. Open **Settings > Dockers > OKLab Colour Selector**.

See [docs/install.md](docs/install.md) for platform-specific install commands
and NumPy setup.

## Use

Open the docker, choose a selector tab, then drag inside the selector or adjust
the L/C/H controls. The docker previews the colour as you work and commits
valid selections to Krita's foreground colour.

See [docs/usage.md](docs/usage.md) for a short guide to the selector modes and
controls.

## Promo Assets

The README is designed for a compact screenshot set:

- `docs/assets/readme/hero.png` - the docker inside Krita.
- `docs/assets/readme/hue-ring.png` - Hue Ring view with swatch and sliders.
- `docs/assets/readme/selector-tabs.png` - the alternate selector modes.
- `docs/assets/readme/demo.webp` - optional short drag-and-preview loop.

Capture guidance lives in [docs/promo-assets.md](docs/promo-assets.md).

## Troubleshooting

If the plugin does not appear, confirm the `.desktop` file is directly inside
Krita's `pykrita/` folder. If the docker reports a missing dependency, install
NumPy into Krita's Python and restart Krita.

More help: [docs/troubleshooting.md](docs/troubleshooting.md)

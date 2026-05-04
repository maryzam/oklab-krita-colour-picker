# Deep Review: OKLab Colour Picker Krita Plugin

Scope: full read of the active codebase — `colour_selector_main.py`,
`selector_common.py`, `oklab.py`, `util.py`, `b_tree.py`, the three
selector views (`lightness_selector.py`, `hue_plane_selector.py`,
`chroma_selector.py`), plus the legacy `lab_colour_picker.py`.

Each item includes file:line evidence and rationale. Recommended order
of work and a verdict on rewrite-vs-iterate are at the end.

---

## 1. Per-pixel Python in the GUI thread (dominant performance issue)

Every selector renders a 150×150 grid by looping in Python and calling
`oklab_to_srgb` per pixel, then upscales with `QImage.scaled` to 256×256.

**Evidence**
- `lightness_selector.py:34-54`, `chroma_selector.py:39-60`,
  `hue_plane_selector.py:62-78`: nested `for y / for x` loops over
  22 500 pixels.
- `oklab.py:62-78`: each call does ~15 multiplies, 3 cubes, then
  `rgb_channel_to_srbg_channel` does up to 3 `x**(1/2.4)` calls, all in
  Python.
- `selector_common.py:114`: every redraw allocates a fresh
  `bytes(self.rgb_data)` from a list of ~90 000 ints (`extend` ×
  22 500).
- The slider has 1001 steps (`selector_common.py:55`); each tick
  triggers a synchronous full recalculation on the GUI thread. This is
  the cause of the "not as responsive as they should be" item the
  README already calls out.

**Why it matters**
A typical OKLab→sRGB CPython call is ~3–5 µs. 22 500 × 5 µs ≈ 110 ms
per redraw, plus list allocation, `bytes()`, and Qt scaling. That is
well outside a 16 ms / 60 fps budget — exactly the laggy/jagged
experience documented.

**Fix (highest ROI)**
Vectorise with NumPy (Krita ships NumPy in its bundled Python). One
redraw becomes:

```python
# precomputed once: lab_a, lab_b float32 arrays of shape (size, size)
l_ = lightness + 0.3963377774*lab_a + 0.2158037573*lab_b
m_ = lightness - 0.1055613458*lab_a - 0.0638541728*lab_b
s_ = lightness - 0.0894841775*lab_a - 1.2914855480*lab_b
l, m, s = l_**3, m_**3, s_**3
r = 4.0767416621*l - 3.3077115913*m + 0.2309699292*s
# ...
mask = (r < 0) | (r > 1) | ...
rgb = np.where(rgb > 0.0031308, 1.055*rgb**(1/2.4) - 0.055, 12.92*rgb)
out = (rgb*255).clip(0, 255).astype(np.uint8)
out[mask] = 0  # transparent
```

Realistic speed-up: **50–200×**. Lightness slider becomes interactive.
This single change invalidates most of the README's "Issues" list.

---

## 2. Concrete bugs in the 2-3 tree (`b_tree.py`)

The tree stores max-chroma per hue
(`lightness_selector.py:62-96`). The implementation has hard bugs.

- **Wrong arity.** `b_tree.py:63` returns
  `TNode(Lk, Lv, k, v, L, right)` — 6 args. `TNode` is defined with 7
  fields (`b_tree.py:29`:
  `low_key, low_value, high_key, high_value, left, middle, right`).
  Line 71 has the same 6-arg `TNode`. These branches will raise
  `TypeError` whenever they execute (TNode→ONode promotion during a
  cascade). They appear not to fire often only because the typical
  insertion order from `range(0, 360, 2)` doesn't hit deeply
  unbalanced cases — but it *will* with adversarial orders.
- **Typo on line 85**: `ONode(k1, v2, ...)` — should be `v1`. Silently
  corrupts the tree (one node's value duplicated) on insertion into the
  middle of a TNode whose left is Empty.
- **`search` may return `(low, None)` or `(None, high)`** at boundaries
  (`b_tree.py:127, 142, 164`). Callers like
  `lightness_selector.py:39, 114` unpack
  `((h_low, c_low), (h_high, c_high))` unconditionally → crash on hues
  at exactly `±π`, which `math.atan2` does emit.

**Recommendation: delete the entire tree.** The keys are 180 fixed,
equally-spaced hue values. Use a flat `numpy.ndarray` of length 180
indexed by `int((hue + π) / step)` for O(1) lookup with
linear-interpolation between neighbours. Simpler, faster, no
allocation per insert, removes ~165 LOC.

---

## 3. The third radio button never wires up its handler

`colour_selector_main.py:30-33`:

```python
self.chroma = QRadioButton('Chroma', self.main)
self.plane.toggled.connect(self.set_display)   # should be self.chroma
```

`chroma.toggled` is never connected. Selecting "Chroma" therefore does
nothing on its own — the view changes only because clicking it
un-toggles "Lightness" or "Hue Plane", whose handlers do fire.
`current_view` becomes stale. Same typo exists in legacy
`lab_colour_picker.py:323-324`.

---

## 4. Polling timer for foreground colour is wrong on multiple axes

`colour_selector_main.py:54-64`:

```python
self.startTimer(100)
def timerEvent(self, ev):
    fg = Krita.instance().activeWindow().activeView().foregroundColor()
```

- 10 Hz polling forever, even when the dock is hidden.
- No null guard — if `activeWindow()` or `activeView()` is `None`
  (startup, all views closed) this raises `AttributeError` 10×/s into
  Krita's log.
- `fg != self.foreground` compares two `ManagedColor` objects with
  default identity equality; this almost always evaluates True after
  `setForeGroundColor`, causing an update loop with the user's own
  clicks.

Krita exposes `Window.activeViewChanged` and `Notifier` signals; a
500 ms timer with proper null-checks and equality on `.components()`
would be enough.

---

## 5. `Indicator` widget construction is wasteful and brittle

`selector_common.py:35-46` (and duplicated at
`lab_colour_picker.py:79-91`): generates two pixmaps from the
`krita_tool_ellipse` toolbar icon, masks one, and stacks them as
`QLabel`s. The author's own comment ("I'm sure there's a better way to
do this … this abomination") flags it. Replacement is one method:

```python
def paintEvent(self, ev):
    p = QPainter(self)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(QPen(Qt.black, 2)); p.drawEllipse(1, 1, 10, 10)
    p.setPen(QPen(Qt.white, 1)); p.drawEllipse(2, 2, 8, 8)
```

Also gives anti-aliased edges (the README's "jagged pixelated edges"
item).

---

## 6. Reading colour back from the rendered image, not from the model

`selector_common.py:127`:

```python
argb = self.image.pixelColor(*self.position)
selected_colour = ManagedColor.fromQColor(argb)
view.setForeGroundColor(selected_colour)
```

The selected colour is sampled from the **already gamma-encoded,
gamut-clipped, bilinearly-upscaled QImage** rather than computed from
the cursor's OKLab coordinates directly. Consequences:

- Click into a transparent (out-of-gamut) hole → black/transparent set
  as foreground.
- Sub-pixel accuracy is destroyed by the 150→256 scale.
- The whole point of using OKLab is lost: you're round-tripping
  through 8-bit sRGB.

Compute Lab from `(x, y)` → `oklab_to_srgb` → `setForeGroundColor`
directly.

---

## 7. `Surface.pixel_metrics` data layout

`selector_common.py:87-97`: `dict[int, dict[int, tuple]]` of ~17 000
entries, looked up as `metrics.get(x, {}).get(y)` per pixel per redraw
(`lightness_selector.py:36`, `chroma_selector.py:44`). Two dict
lookups per pixel × 22 500 pixels per redraw is significant overhead,
and each miss allocates a fresh empty dict.

Replace with three `numpy.ndarray((size, size), float32)` for
`length`, `cos`, `sin` plus a boolean `inside_disc` mask. Eliminates
the outer-pixel branch *and* makes the calculation vectorisable
(point 1).

---

## 8. `cbrt` reinvention in `util.py`

`util.py:1-10` is a hand-written Newton-Raphson cube root. CPython
exposes `math.cbrt` (3.11+, which Krita 5.2+ ships) or `x**(1/3)`.
Both are C-level and handle negative `l` correctly without a
divergence check. Used in `srgb_to_oklab` (only called on view switch,
so impact is minor — but the function is a footgun: with `x = 0` it
divides by zero on line 9).

---

## 9. Slider signal management

Every `modify*` method does `self.slider.disconnect(); … ;
self.slider.valueChanged.connect(self.modify…)`. This:

- Disconnects **all** signals (other code subscribing to
  `valueChanged` would silently break),
- Reallocates a Python lambda each time in some paths
  (`lab_colour_picker.py:238, 247`),
- Is unnecessary — `QSignalBlocker(self.slider)` (or
  `self.slider.blockSignals(True)`) is the idiomatic Qt approach.

---

## 10. Hue slider rainbow strip (`HueEffect`)

`hue_plane_selector.py:18-48`:

- Generates a "rainbow" by computing OKLab→sRGB at L=0.5, chroma=0.085,
  then converts each result to HSL and re-emits at HSL L=0.5 — i.e. it
  does *not* show OKLab hues, it shows HSL hues with OKLab hue angles
  as input. Defeats the plugin's purpose; the rainbow does not match
  the colours in the gamut display below.
- `draw()` paints the pixmap and the arrow but never calls
  `self.drawSource(painter)` (line 48 is commented out), so the
  slider's actual handle/groove disappears. That's why "the hue slider
  is just a normal slider and doesn't display the hues" remained a
  README gripe — the fix tried `QGraphicsEffect` and silently dropped
  the source rendering.

Replacement: subclass `QSlider` with a `paintEvent` that paints an
OKLab strip at fixed L and per-hue *max in-gamut* chroma, then calls
`super().paintEvent(ev)` so the handle still draws.

---

## 11. `2π ≠ 6.28`

`hue_plane_selector.py:108`, `lab_colour_picker.py:229`:
`lerp(0, 6.28, value/1000)`. `6.28` is short of `2π` by 0.0032 rad
(≈0.18°). Small discontinuity at wrap-around. Use `math.tau`.

---

## 12. Hard-coded sizes & no resize support

Confirmed in README issues, plus `selector_common.py:162`
(`makeSurface(150)`) and the `256` display sizes scattered across
views. The display is upscaled 1.7× with bilinear filtering, which is
the source of the "jagged" look. Once point 1 is done you can simply
render at the actual widget size and react to `resizeEvent`.

---

## 13. Dead legacy code shipped in the plugin

`lab_colour_picker.py` (370 lines) is *not* imported by `__init__.py`
and contains an older, separately-buggy implementation. Krita scans
the plugin folder; this file isn't loaded as a dock factory through
the main path, but `Krita.instance().addDockWidgetFactory(...)` at
line 370 *will* execute if anything imports the module. Remove or move
under a clearly archived path.

---

## 14. Smaller correctness items

- `oklab.py:80` function name typo `srbg`. Cosmetic.
- `lightness_selector.py:74` `del(self.chromas)` immediately followed
  by reassignment — pointless.
- `lightness_selector.py:116` computes `chroma_delta` and never uses
  it.
- `chroma_selector.py:30` indicator is positioned at a fixed
  `0.5 * half_size` radius regardless of the current chroma slider, so
  the indicator does not move when the user drags chroma — it only
  moves on view switch.
- `hue_plane_selector.py:89` divides by `0.35` (the display's chroma
  extent) but `image_size` is also multiplied — for `display_size=256`,
  chroma 0.35 gives `int(256*1)=256`, one pixel past the surface edge.

---

## Recommended order of work

1. **Vectorise the three `calculateColours` paths with NumPy.** Single
   biggest user-visible win; removes the README's top three issues.
2. **Replace `b_tree.py` with a 180-element NumPy LUT** indexed by
   hue. Fixes real bugs (#2), removes ~165 LOC.
3. **Fix the chroma radio-button wire-up** (#3) — one-character bug,
   blocks a feature.
4. **Replace polling timer with proper null-checked Krita signals**
   (#4).
5. **Sample colour from the model, not from the QImage** (#6).
6. **Rewrite `Indicator` with `paintEvent`** (#5) — also fixes
   anti-aliasing.
7. **Replace `HueEffect` with a true OKLab strip slider** (#10).
8. **Make widgets resize** (#12), enabled by step 1.
9. **Delete `lab_colour_picker.py` legacy file and the `cbrt`
   hand-roll**.

---

## Verdict: rewrite or iterate?

**Iterate.** The architecture (one `SelectorSurface` base + three
views + a shared dock with a stacked layout) is sound, and the OKLab
math in `oklab.py` is a faithful copy of Ottosson's reference. The
pain is concentrated in two layers: (a) per-pixel Python loops that
should be NumPy, and (b) a hand-rolled tree that should be a flat
array. Replace those two and fix the half-dozen Qt-wiring bugs and
you have a fast, correct plugin with the same module boundaries it
has today — roughly a 2–3 day effort, not a rewrite.

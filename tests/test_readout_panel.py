"""Tests for the expanded LCH readout panel."""

import math

import numpy as np
import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from oklab_colour_picker import color_math, renderers
from oklab_colour_picker.widgets.readout_panel import (
    ReadoutPanel,
    hex_to_oklab,
    is_in_srgb_gamut,
    oklab_to_hex,
)


# -- pure helpers -----------------------------------------------------------


@pytest.mark.parametrize(
    "hex_value",
    ["#000000", "#ffffff", "#4a8fb2", "#7f3322"],
)
def test_hex_round_trip(hex_value):
    oklab = hex_to_oklab(hex_value)
    assert oklab is not None
    assert oklab_to_hex(oklab) == hex_value


def test_hex_accepts_uppercase_and_missing_hash():
    assert oklab_to_hex(hex_to_oklab("4A8FB2")) == "#4a8fb2"
    assert oklab_to_hex(hex_to_oklab("#FF00AA")) == "#ff00aa"


@pytest.mark.parametrize("bad", ["", "not-hex", "#12345", "#1234567", "#zzzzzz"])
def test_hex_rejects_malformed(bad):
    assert hex_to_oklab(bad) is None


def test_in_gamut_detects_displayable_colour():
    assert is_in_srgb_gamut(color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])))


def test_in_gamut_flags_super_saturated_oklch():
    # A high-chroma OKLCh point that lives outside the sRGB cusp.
    oklab = color_math.oklch_to_oklab([0.6, color_math.SRGB_MAX_CHROMA, 0.0])
    assert not is_in_srgb_gamut(oklab)


# -- gamut-gap rendering ----------------------------------------------------


def test_axis_track_hue_marks_out_of_gamut_with_checker():
    # At high chroma and L=0.95 most hues are unreachable in sRGB; we expect
    # a substantial fraction of the hue track to be flagged out-of-gamut.
    rgba = renderers.render_axis_track(
        renderers.AXIS_H,
        (0.95, color_math.SRGB_MAX_CHROMA * 0.9),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # In-gamut pixels never use the (200, 200, 200) / (120, 120, 120) tones
    # exclusively for entire pattern columns, so look for the dark-tile colour.
    has_dark_tile = np.any(np.all(rgba[..., :3] == 120, axis=-1))
    assert has_dark_tile


def test_axis_track_chroma_low_l_is_fully_in_gamut_at_zero_chroma_start():
    # At chroma=0 the swept C axis starts in gamut and crosses the cusp once.
    rgba = renderers.render_axis_track(
        renderers.AXIS_C,
        (0.5, 0.0),
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # The leftmost column (C=0) must be in gamut (no checker tile colour).
    left = rgba[:, 0, :3]
    assert not np.any(np.all(left == 120, axis=-1))
    assert not np.any(np.all(left == 200, axis=-1))


def test_axis_track_l_at_extremes_is_out_of_gamut_for_nonzero_chroma():
    # At L=0 or L=1 any positive chroma is out of gamut.
    rgba = renderers.render_axis_track(
        renderers.AXIS_L,
        (0.15, 0.0),  # chroma=0.15, hue=0
        color_math.SRGB_MAX_CHROMA,
        (256, 12),
    )
    # First column corresponds to L=0 and last to L=1; both should be flagged.
    for col in (0, -1):
        pixel = rgba[0, col, :3]
        assert tuple(pixel) in {(120, 120, 120), (200, 200, 200)}


def test_axis_track_unknown_axis_raises():
    with pytest.raises(ValueError):
        renderers.render_axis_track("Q", (0.5, 0.0), color_math.SRGB_MAX_CHROMA, (32, 10))


# -- panel round-trips ------------------------------------------------------


def test_readout_panel_round_trips_through_sliders(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)
    panel.resize(320, 200)

    target = color_math.oklch_to_oklab([0.62, 0.11, math.radians(245.0)])
    panel.set_current_colour(target)

    assert panel._row_l.value() == pytest.approx(0.62, abs=1e-3)
    assert panel._row_c.value() == pytest.approx(0.11, abs=1e-3)
    assert panel._row_h.value() == pytest.approx(245.0, abs=0.5)


def test_readout_panel_hex_field_reflects_current_colour(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    oklab = color_math.srgb_to_oklab(np.array([0x4A, 0x8F, 0xB2]) / 255.0)
    panel.set_current_colour(oklab)

    assert panel._hex_field.text() == "#4a8fb2"


def test_readout_panel_hex_edit_emits_committed_colour(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)
    panel.set_current_colour(color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])))

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(np.asarray(colour, dtype=float)))

    panel._hex_field.setText("#4a8fb2")
    panel._hex_field.editingFinished.emit()

    assert received
    expected = color_math.srgb_to_oklab(np.array([0x4A, 0x8F, 0xB2]) / 255.0)
    np.testing.assert_allclose(received[-1], expected, atol=1e-4)


def test_readout_panel_previous_swatch_click_restores_previous(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    first = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    second = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    panel.set_current_colour(first)
    panel.set_current_colour(second)

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(np.asarray(colour, dtype=float)))

    panel._previous_swatch.clicked.emit()

    assert received
    np.testing.assert_allclose(received[-1], first, atol=1e-4)


def test_readout_panel_set_previous_seeds_revert_target(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    seed = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    panel.set_current_colour(seed)
    panel.set_previous_colour(seed)

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(np.asarray(colour, dtype=float)))

    panel._previous_swatch.clicked.emit()

    assert received
    np.testing.assert_allclose(received[-1], seed, atol=1e-4)


def test_readout_panel_preview_does_not_advance_previous(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    first = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    panel.set_current_colour(first, committed=True)
    snapshot = panel._previous_oklab.copy() if panel._previous_oklab is not None else None

    preview = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    panel.set_current_colour(preview, committed=False)

    if snapshot is None:
        assert panel._previous_oklab is None
    else:
        np.testing.assert_allclose(panel._previous_oklab, snapshot, atol=1e-12)
    np.testing.assert_allclose(panel._current_oklab, preview, atol=1e-12)


def test_readout_panel_committed_updates_advance_previous(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    a = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    b = color_math.srgb_to_oklab(np.array([0.7, 0.3, 0.1]))
    panel.set_current_colour(a, committed=True)
    panel.set_current_colour(b, committed=True)

    np.testing.assert_allclose(panel._previous_oklab, a, atol=1e-12)
    np.testing.assert_allclose(panel._current_oklab, b, atol=1e-12)


def test_readout_panel_hex_enter_sets_previous_only_once(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    a = color_math.srgb_to_oklab(np.array([0.2, 0.4, 0.6]))
    panel.set_current_colour(a, committed=True)

    received: list[np.ndarray] = []
    panel.committed.connect(lambda colour: received.append(np.asarray(colour, dtype=float)))

    panel._hex_field.setText("#4a8fb2")
    # Simulate Enter: both signals fire in the real Qt event flow, but only
    # editingFinished is connected, so the handler runs exactly once.
    panel._hex_field.returnPressed.emit()
    panel._hex_field.editingFinished.emit()

    assert len(received) == 1
    np.testing.assert_allclose(panel._previous_oklab, a, atol=1e-12)


def test_readout_panel_out_of_gamut_warning_visibility(qtbot):
    panel = ReadoutPanel()
    qtbot.addWidget(panel)

    panel.set_current_colour(color_math.srgb_to_oklab(np.array([0.5, 0.5, 0.5])))
    assert not panel._gamut_warning.isVisible() or panel._gamut_warning.isHidden()

    panel.set_current_colour(
        color_math.oklch_to_oklab([0.6, color_math.SRGB_MAX_CHROMA, 0.0])
    )
    assert panel._gamut_warning.isVisibleTo(panel)

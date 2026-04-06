"""Tests for ass.py — alignment mapping, style, and ASS file assembly."""

from pathlib import Path
from unittest.mock import patch

import pytest
import pysubs2

from v2a.ass import (
    MAX_SUBTITLE_MS,
    MIN_SUBTITLE_MS,
    alignment_from_area,
    build_ass,
    make_style,
)


# ---------------------------------------------------------------------------
# alignment_from_area
# ---------------------------------------------------------------------------

class TestAlignmentFromArea:
    """
    DVD frame is 720×576 divided into a 3×3 grid.
    Column thresholds: left < 240, center < 480, right >= 480.
    Row thresholds:    top  < 192, mid   < 384, bot  >= 384.

    Numpad mapping:
      7 8 9
      4 5 6
      1 2 3
    """

    def test_bottom_center_returns_2(self):
        # cx=360 (center), cy=500 (bot)
        assert alignment_from_area(200, 470, 520, 530) == 2

    def test_top_left_returns_7(self):
        # cx=50 (left), cy=50 (top)
        assert alignment_from_area(0, 0, 100, 100) == 7

    def test_top_center_returns_8(self):
        # cx=360 (center), cy=50 (top)
        assert alignment_from_area(200, 0, 520, 100) == 8

    def test_top_right_returns_9(self):
        # cx=670 (right), cy=50 (top)
        assert alignment_from_area(620, 0, 720, 100) == 9

    def test_mid_left_returns_4(self):
        # cx=50 (left), cy=288 (mid)
        assert alignment_from_area(0, 200, 100, 370) == 4

    def test_mid_center_returns_5(self):
        # cx=360 (center), cy=288 (mid)
        assert alignment_from_area(200, 200, 520, 370) == 5

    def test_mid_right_returns_6(self):
        # cx=670 (right), cy=288 (mid)
        assert alignment_from_area(620, 200, 720, 370) == 6

    def test_bot_left_returns_1(self):
        # cx=50 (left), cy=500 (bot)
        assert alignment_from_area(0, 470, 100, 530) == 1

    def test_bot_right_returns_3(self):
        # cx=670 (right), cy=500 (bot)
        assert alignment_from_area(620, 470, 720, 530) == 3

    def test_none_coords_return_2(self):
        assert alignment_from_area(None, None, None, None) == 2

    def test_partial_none_coords_return_2(self):
        assert alignment_from_area(100, None, 500, 500) == 2

    def test_boundary_exactly_at_column_threshold(self):
        # cx = 240 exactly → should be 'center' (cx < 480 but not < 240)
        assert alignment_from_area(230, 470, 250, 530) == 2   # bot-center

    def test_boundary_exactly_at_row_threshold(self):
        # cy = 192 exactly → should be 'mid' (cy < 384 but not < 192)
        assert alignment_from_area(200, 184, 520, 200) == 5   # mid-center


# ---------------------------------------------------------------------------
# make_style
# ---------------------------------------------------------------------------

class TestMakeStyle:
    def test_returns_ssastyle(self):
        style = make_style()
        assert isinstance(style, pysubs2.SSAStyle)

    def test_font_and_size(self):
        style = make_style()
        assert style.fontname == "Arial"
        assert style.fontsize == 52

    def test_default_alignment_is_bottom_center(self):
        assert make_style().alignment == 2

    def test_margins(self):
        style = make_style()
        assert style.marginl == 40
        assert style.marginr == 40
        assert style.marginv == 30


# ---------------------------------------------------------------------------
# build_ass
# ---------------------------------------------------------------------------

def _spu(end_ms=None, x1=100, x2=620, y1=500, y2=540):
    return {"end_ms": end_ms, "x1": x1, "y1": y1, "x2": x2, "y2": y2}


class TestBuildAss:
    """
    build_ass calls read_spu_at internally; we patch it at v2a.ass.read_spu_at
    to avoid needing real .sub files.
    """

    def _run(self, entries, texts, spu_results, tmp_path):
        out = tmp_path / "output.ass"
        sub = tmp_path / "subs.sub"
        sub.write_bytes(b'')
        spu_iter = iter(spu_results)
        with patch("v2a.ass.read_spu_at", side_effect=lambda *_: next(spu_iter)):
            count = build_ass(entries, texts, sub, out)
        return count, out

    def test_writes_ass_file(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Hello world"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        assert out.exists()

    def test_returns_event_count(self, tmp_path):
        entries = [(1000, 0x1000), (5000, 0x2000)]
        texts   = ["Line one", "Line two"]
        count, _ = self._run(entries, texts, [_spu(2000), _spu(1000)], tmp_path)
        assert count == 2

    def test_empty_text_entries_skipped(self, tmp_path):
        entries = [(1000, 0x1000), (3000, 0x2000)]
        texts   = ["", "Visible line"]
        count, _ = self._run(entries, texts, [_spu(2000), _spu(1000)], tmp_path)
        assert count == 1

    def test_end_time_from_spu(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Hello"]
        _, out = self._run(entries, texts, [_spu(end_ms=2500)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].start == 1000
        assert subs[0].end   == 3500   # 1000 + 2500

    def test_end_time_fallback_to_next_start(self, tmp_path):
        # No SPU end_ms; should use next start - 100
        entries = [(1000, 0x1000), (5000, 0x2000)]
        texts   = ["First", "Second"]
        _, out = self._run(entries, texts, [_spu(end_ms=None), _spu(end_ms=1000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 4900   # 5000 - 100

    def test_end_time_last_entry_fallback(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Only line"]
        _, out = self._run(entries, texts, [_spu(end_ms=None)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 4000   # 1000 + LAST_FALLBACK_MS (3000)

    def test_max_cap_applied(self, tmp_path):
        entries = [(0, 0x1000)]
        texts   = ["Too long"]
        # SPU claims 60 seconds — should be capped at MAX_SUBTITLE_MS
        _, out = self._run(entries, texts, [_spu(end_ms=60_000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == MAX_SUBTITLE_MS

    def test_min_floor_applied(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Flash"]
        # SPU claims 10 ms — should be raised to MIN_SUBTITLE_MS
        _, out = self._run(entries, texts, [_spu(end_ms=10)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 1000 + MIN_SUBTITLE_MS

    def test_bottom_center_has_no_an_tag(self, tmp_path):
        # Bottom-center (an=2) should NOT inject an override tag
        entries = [(1000, 0x1000)]
        texts   = ["Normal sub"]
        # y centroid ≈ 515 (bot), x centroid ≈ 360 (center) → an=2
        _, out = self._run(entries, texts, [_spu(end_ms=2000, x1=200, x2=520, y1=500, y2=530)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "{\\an" not in subs[0].text

    def test_sign_has_an_tag(self, tmp_path):
        # Top-center (an=8) should inject {\an8}
        entries = [(1000, 0x1000)]
        texts   = ["Sign text"]
        # y centroid ≈ 50 (top), x centroid ≈ 360 (center) → an=8
        _, out = self._run(entries, texts, [_spu(end_ms=2000, x1=200, x2=520, y1=0, y2=100)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].text.startswith("{\\an8}")

    def test_newlines_converted_to_ass_format(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Line one\nLine two"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "\\N" in subs[0].text

    def test_default_style_is_set(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts   = ["Hello"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "Default" in subs.styles
        assert subs.styles["Default"].fontname == "Arial"

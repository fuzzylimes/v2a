"""Tests for ass.py — alignment mapping, style, and ASS file assembly."""

from pathlib import Path
from unittest.mock import patch

import pytest
import pysubs2

from PIL import Image

from v2a.ass import (
    LAST_FALLBACK_MS,
    MAX_SUBTITLE_MS,
    MIN_SUBTITLE_MS,
    _bbox_is_fullframe,
    alignment_from_area,
    alignment_from_image,
    build_ass,
    build_ass_pgs,
    make_style,
)


# ---------------------------------------------------------------------------
# alignment_from_area
# ---------------------------------------------------------------------------

class TestAlignmentFromArea:
    """
    DVD frame is 720x576 divided into a 3x3 grid.
    Column thresholds: left < 240, center < 480, right >= 480.
    Row thresholds:    top < 144,  mid < 346,    bot >= 346.

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
        # cy = 192 → not < 144 (top), is < 346 (mid) → mid-center
        assert alignment_from_area(200, 184, 520, 200) == 5   # mid-center


# ---------------------------------------------------------------------------
# _bbox_is_fullframe
# ---------------------------------------------------------------------------

class TestBboxIsFullframe:
    def test_full_ntsc_frame_detected(self):
        assert _bbox_is_fullframe(0, 2, 719, 479) is True

    def test_small_dialogue_box_not_fullframe(self):
        assert _bbox_is_fullframe(100, 500, 620, 540) is False

    def test_none_coords_are_fullframe(self):
        assert _bbox_is_fullframe(None, None, None, None) is True

    def test_partial_none_is_fullframe(self):
        assert _bbox_is_fullframe(0, None, 719, 479) is True

    def test_wide_but_short_not_fullframe(self):
        # Full width but only a dialogue strip — not full-frame
        assert _bbox_is_fullframe(0, 450, 719, 530) is False


# ---------------------------------------------------------------------------
# alignment_from_image
# ---------------------------------------------------------------------------

def _make_frame_png(tmp_path, w, h, content_y1, content_y2, cx_frac=0.5) -> Path:
    """Create a synthetic RGBA PNG with visible pixels in a horizontal band."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    cx = int(w * cx_frac)
    half = w // 6
    for y in range(content_y1, content_y2 + 1):
        for x in range(cx - half, cx + half):
            if 0 <= x < w:
                img.putpixel((x, y), (255, 255, 255, 255))
    path = tmp_path / "frame.png"
    img.save(path)
    return path


class TestAlignmentFromImage:
    def test_bottom_center_content(self, tmp_path):
        # Content in the bottom 20% of a 480px frame → bot-center → an=2
        path = _make_frame_png(tmp_path, 720, 480, 420, 460)
        assert alignment_from_image(path) == 2

    def test_top_center_content(self, tmp_path):
        # Content in the top 10% → top-center → an=8
        path = _make_frame_png(tmp_path, 720, 480, 10, 40)
        assert alignment_from_image(path) == 8

    def test_mid_center_content(self, tmp_path):
        # Content in the middle → mid-center → an=5
        path = _make_frame_png(tmp_path, 720, 480, 200, 250)
        assert alignment_from_image(path) == 5

    def test_transparent_frame_returns_2(self, tmp_path):
        img = Image.new("RGBA", (720, 480), (0, 0, 0, 0))
        path = tmp_path / "empty.png"
        img.save(path)
        assert alignment_from_image(path) == 2

    def test_missing_file_returns_2(self, tmp_path):
        assert alignment_from_image(tmp_path / "nonexistent.png") == 2


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
        assert style.fontsize == 36

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

    def _run(self, entries, texts, spu_results, tmp_path, frames=None):
        out = tmp_path / "output.ass"
        sub = tmp_path / "subs.sub"
        sub.write_bytes(b'')
        spu_iter = iter(spu_results)
        with patch("v2a.ass.read_spu_at", side_effect=lambda *_: next(spu_iter)):
            count = build_ass(entries, texts, sub, out, frames=frames)
        return count, out

    def test_writes_ass_file(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Hello world"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        assert out.exists()

    def test_returns_event_count(self, tmp_path):
        entries = [(1000, 0x1000), (5000, 0x2000)]
        texts = ["Line one", "Line two"]
        count, _ = self._run(
            entries, texts, [_spu(2000), _spu(1000)], tmp_path)
        assert count == 2

    def test_empty_text_entries_skipped(self, tmp_path):
        entries = [(1000, 0x1000), (3000, 0x2000)]
        texts = ["", "Visible line"]
        count, _ = self._run(
            entries, texts, [_spu(2000), _spu(1000)], tmp_path)
        assert count == 1

    def test_end_time_from_spu(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Hello"]
        _, out = self._run(entries, texts, [_spu(end_ms=2500)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].start == 1000
        assert subs[0].end == 3500   # 1000 + 2500

    def test_end_time_fallback_to_next_start(self, tmp_path):
        # No SPU end_ms; should use next start - 100
        entries = [(1000, 0x1000), (5000, 0x2000)]
        texts = ["First", "Second"]
        _, out = self._run(entries, texts, [_spu(
            end_ms=None), _spu(end_ms=1000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 4900   # 5000 - 100

    def test_end_time_last_entry_fallback(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Only line"]
        _, out = self._run(entries, texts, [_spu(end_ms=None)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 4000   # 1000 + LAST_FALLBACK_MS (3000)

    def test_max_cap_applied(self, tmp_path):
        entries = [(0, 0x1000)]
        texts = ["Too long"]
        # SPU claims 60 seconds — should be capped at MAX_SUBTITLE_MS
        _, out = self._run(entries, texts, [_spu(end_ms=60_000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == MAX_SUBTITLE_MS

    def test_min_floor_applied(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Flash"]
        # SPU claims 10 ms — should be raised to MIN_SUBTITLE_MS
        _, out = self._run(entries, texts, [_spu(end_ms=10)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 1000 + MIN_SUBTITLE_MS

    def test_bottom_center_has_no_an_tag(self, tmp_path):
        # Bottom-center (an=2) should NOT inject an override tag
        entries = [(1000, 0x1000)]
        texts = ["Normal sub"]
        # y centroid ≈ 515 (bot), x centroid ≈ 360 (center) → an=2
        _, out = self._run(entries, texts, [_spu(
            end_ms=2000, x1=200, x2=520, y1=500, y2=530)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "{\\an" not in subs[0].text

    def test_sign_has_an_tag(self, tmp_path):
        # Top-center (an=8) should inject {\an8}
        entries = [(1000, 0x1000)]
        texts = ["Sign text"]
        # y centroid ≈ 50 (top), x centroid ≈ 360 (center) → an=8
        _, out = self._run(entries, texts, [_spu(
            end_ms=2000, x1=200, x2=520, y1=0, y2=100)], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].text.startswith("{\\an8}")

    def test_newlines_converted_to_ass_format(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Line one\nLine two"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "\\N" in subs[0].text

    def test_image_based_alignment_used_for_fullframe_bbox(self, tmp_path):
        # Full-frame bbox (0,2)-(719,479) — should use PNG pixel analysis.
        # Content at the bottom of the PNG → bottom-center → no {\an} tag.
        frame = _make_frame_png(tmp_path, 720, 478, 430, 470)
        entries = [(1000, 0x1000)]
        texts = ["Dialogue"]
        spu = {"end_ms": 2000, "x1": 0, "y1": 2, "x2": 719, "y2": 479}
        out = tmp_path / "output.ass"
        sub = tmp_path / "subs.sub"
        sub.write_bytes(b'')
        with patch("v2a.ass.read_spu_at", return_value=spu):
            build_ass(entries, texts, sub, out, frames=[frame])
        loaded = pysubs2.load(str(out))
        assert "{\\an" not in loaded[0].text

    def test_image_based_alignment_top_content(self, tmp_path):
        # Full-frame bbox with content at the top → top-center → {\an8}.
        frame = _make_frame_png(tmp_path, 720, 478, 10, 40)
        entries = [(1000, 0x1000)]
        texts = ["Title card"]
        spu = {"end_ms": 2000, "x1": 0, "y1": 2, "x2": 719, "y2": 479}
        out = tmp_path / "output.ass"
        sub = tmp_path / "subs.sub"
        sub.write_bytes(b'')
        with patch("v2a.ass.read_spu_at", return_value=spu):
            build_ass(entries, texts, sub, out, frames=[frame])
        loaded = pysubs2.load(str(out))
        assert loaded[0].text.startswith("{\\an8}")

    def test_default_style_is_set(self, tmp_path):
        entries = [(1000, 0x1000)]
        texts = ["Hello"]
        _, out = self._run(entries, texts, [_spu(end_ms=2000)], tmp_path)
        subs = pysubs2.load(str(out))
        assert "Default" in subs.styles
        assert subs.styles["Default"].fontname == "Arial"


# ---------------------------------------------------------------------------
# build_ass_pgs
# ---------------------------------------------------------------------------

def _rec(start_ms=1000, end_ms=3000, x1=900, y1=1000, x2=1020, y2=1050,
         canvas_w=1920, canvas_h=1080):
    return {"start_ms": start_ms, "end_ms": end_ms,
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "canvas_w": canvas_w, "canvas_h": canvas_h}


class TestBuildAssPgs:
    def _run(self, records, texts, tmp_path):
        out = tmp_path / "output.ass"
        count = build_ass_pgs(records, texts, out)
        return count, out

    def test_writes_file_and_counts_events(self, tmp_path):
        count, out = self._run([_rec(), _rec()], ["One", "Two"], tmp_path)
        assert out.exists()
        assert count == 2

    def test_empty_text_skipped(self, tmp_path):
        count, _ = self._run([_rec(), _rec()], ["", "Visible"], tmp_path)
        assert count == 1

    def test_uses_pgs_timing_directly(self, tmp_path):
        # PGS end time is trusted as-is — no 8s cap, unlike VobSub.
        _, out = self._run([_rec(start_ms=0, end_ms=20_000)], ["Long"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].start == 0
        assert subs[0].end == 20_000

    def test_min_floor_applied(self, tmp_path):
        _, out = self._run([_rec(start_ms=1000, end_ms=1010)], ["Flash"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 1000 + MIN_SUBTITLE_MS

    def test_none_end_uses_last_fallback(self, tmp_path):
        _, out = self._run([_rec(start_ms=1000, end_ms=None)], ["Trailing"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].end == 1000 + LAST_FALLBACK_MS

    def test_exact_position_for_dialogue(self, tmp_path):
        # Box (900,1000)-(1020,1050) → center (960, 1025) → \pos at those coords.
        _, out = self._run([_rec(x1=900, y1=1000, x2=1020, y2=1050)], ["Dialogue"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].text == "{\\an5\\pos(960,1025)}Dialogue"

    def test_exact_position_for_raised_dialogue(self, tmp_path):
        # Dialogue raised to the top (e.g. two speakers) is positioned exactly,
        # not snapped to a zone — center (960, 60).
        _, out = self._run([_rec(x1=900, y1=40, x2=1020, y2=80)], ["Up here"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].text == "{\\an5\\pos(960,60)}Up here"

    def test_no_zone_an_tags_used(self, tmp_path):
        # The 9-zone \an2/\an8/etc. tags must not appear — only the \an5 anchor.
        _, out = self._run([_rec(x1=0, y1=0, x2=40, y2=40)], ["Corner"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs[0].text.startswith("{\\an5\\pos(")

    def test_playres_is_native_canvas(self, tmp_path):
        _, out = self._run([_rec(canvas_w=1920, canvas_h=1080)], ["Hello"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs.info["PlayResX"] == "1920"
        assert subs.info["PlayResY"] == "1080"

    def test_font_family_and_color_preserved(self, tmp_path):
        # "Same styling" = font/color unchanged from the DVD style.
        _, out = self._run([_rec()], ["Hello"], tmp_path)
        subs = pysubs2.load(str(out))
        base = make_style()
        assert subs.styles["Default"].fontname == base.fontname
        assert subs.styles["Default"].primarycolor == base.primarycolor
        assert subs.styles["Default"].outlinecolor == base.outlinecolor

    def test_font_size_scaled_to_canvas(self, tmp_path):
        # 36px @ 576h scales to 36*1080/576 = 67.5 → 68 @ 1080h.
        _, out = self._run([_rec(canvas_h=1080)], ["Hello"], tmp_path)
        subs = pysubs2.load(str(out))
        assert subs.styles["Default"].fontsize == round(36 * 1080 / 576)

    def test_newlines_converted(self, tmp_path):
        _, out = self._run([_rec()], ["Line one\nLine two"], tmp_path)
        subs = pysubs2.load(str(out))
        assert "\\N" in subs[0].text

    def test_none_bbox_falls_back_to_bottom_center(self, tmp_path):
        # No location data → no \pos, plain bottom-center default (no \an5).
        _, out = self._run(
            [_rec(x1=None, y1=None, x2=None, y2=None)], ["No box"], tmp_path)
        subs = pysubs2.load(str(out))
        assert "\\pos" not in subs[0].text
        assert "{\\an" not in subs[0].text

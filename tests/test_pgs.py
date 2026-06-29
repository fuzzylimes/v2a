"""Tests for pgs.py — Blu-ray PGS (.sup) parsing and rendering.

PGS segments are built by hand here (mirroring tests/test_spu.py's approach for
SPU packets) so the parser, RLE decoder, and display-set state machine can be
exercised without a real .sup file.
"""

import struct

import pytest
from PIL import Image

from v2a.pgs import (
    SEG_END,
    SEG_ODS,
    SEG_PCS,
    SEG_PDS,
    _decode_rle,
    _ycrcb_to_rgb,
    count_sup,
    parse_sup,
)


# ---------------------------------------------------------------------------
# Synthetic .sup builders
# ---------------------------------------------------------------------------

def _segment(seg_type: int, payload: bytes, pts_ms: float = 0.0) -> bytes:
    """Wrap a payload in a 13-byte PG segment header (PTS in 90 kHz units)."""
    pts = int(round(pts_ms * 90))
    return (b"PG" + struct.pack(">I", pts) + struct.pack(">I", 0)
            + bytes([seg_type]) + struct.pack(">H", len(payload)) + payload)


def _pcs(objects: list[tuple[int, int, int]], width=1920, height=1080,
         state=0x80, palette_id=0) -> bytes:
    """Build a PCS payload. objects = [(object_id, x, y), ...]."""
    p = struct.pack(">H", width) + struct.pack(">H", height)
    p += bytes([0x10])                         # frame rate
    p += struct.pack(">H", 0)                   # composition number
    p += bytes([state, 0x00, palette_id, len(objects)])
    for object_id, x, y in objects:
        p += struct.pack(">H", object_id) + bytes([0x00, 0x00])  # window id, not cropped
        p += struct.pack(">H", x) + struct.pack(">H", y)
    return p


def _pds(palette_id: int, entries: list[tuple[int, int, int, int, int]]) -> bytes:
    """Build a PDS payload. entries = [(index, Y, Cr, Cb, A), ...]."""
    p = bytes([palette_id, 0x00])
    for idx, y, cr, cb, a in entries:
        p += bytes([idx, y, cr, cb, a])
    return p


def _ods(object_id: int, width: int, height: int, rle: bytes) -> bytes:
    """Build a single (first+last) ODS payload."""
    data_len = 2 + 2 + len(rle)                 # width + height + rle
    p = struct.pack(">H", object_id) + bytes([0x00, 0xC0])
    p += bytes([(data_len >> 16) & 0xFF, (data_len >> 8) & 0xFF, data_len & 0xFF])
    p += struct.pack(">H", width) + struct.pack(">H", height) + rle
    return p


def _solid_rle(width: int, height: int, color: int) -> bytes:
    """RLE for a solid `width`x`height` block of one palette colour (one pixel per byte)."""
    line = bytes([color]) * width + b"\x00\x00"   # pixels then end-of-line
    return line * height


# White, opaque palette entry at index 1.
WHITE = [(1, 235, 128, 128, 255)]


def _simple_sup(x=100, y=900, w=4, h=4, start_ms=1000.0, end_ms=3000.0,
                canvas=(1920, 1080)) -> bytes:
    """One subtitle: presented at start_ms, cleared at end_ms."""
    cw, ch = canvas
    present = (
        _segment(SEG_PCS, _pcs([(0, x, y)], width=cw, height=ch), start_ms)
        + _segment(SEG_PDS, _pds(0, WHITE), start_ms)
        + _segment(SEG_ODS, _ods(0, w, h, _solid_rle(w, h, 1)), start_ms)
        + _segment(SEG_END, b"", start_ms)
    )
    clear = (
        _segment(SEG_PCS, _pcs([], width=cw, height=ch, state=0x00), end_ms)
        + _segment(SEG_END, b"", end_ms)
    )
    return present + clear


# ---------------------------------------------------------------------------
# _decode_rle
# ---------------------------------------------------------------------------

class TestDecodeRle:
    def test_solid_block(self):
        idx = _decode_rle(_solid_rle(3, 2, 1), 3, 2)
        assert idx == [1, 1, 1, 1, 1, 1]

    def test_short_run_of_background(self):
        # 00 05 -> 5 pixels of colour 0, then fill the rest of a width-8 line
        idx = _decode_rle(b"\x00\x05\x00\x00", 8, 1)
        assert idx == [0] * 8

    def test_run_with_colour(self):
        # 00 84 07 -> 4 pixels of colour 7
        idx = _decode_rle(b"\x00\x84\x07\x00\x00", 6, 1)
        assert idx[:4] == [7, 7, 7, 7]
        assert idx[4:] == [0, 0]

    def test_long_run(self):
        # 00 C1 00 09 -> ((1<<8)|0)=256 pixels of colour 9
        idx = _decode_rle(b"\x00\xC1\x00\x09", 256, 1)
        assert len(idx) == 256
        assert all(c == 9 for c in idx)

    def test_pads_to_full_size(self):
        # Truncated data must still produce width*height entries.
        idx = _decode_rle(b"\x01", 4, 4)
        assert len(idx) == 16


# ---------------------------------------------------------------------------
# _ycrcb_to_rgb
# ---------------------------------------------------------------------------

class TestYcrcbToRgb:
    def test_white(self):
        assert _ycrcb_to_rgb(235, 128, 128) == (235, 235, 235)

    def test_black(self):
        assert _ycrcb_to_rgb(16, 128, 128) == (16, 16, 16)

    def test_clamps_to_byte_range(self):
        r, g, b = _ycrcb_to_rgb(255, 255, 255)
        assert all(0 <= c <= 255 for c in (r, g, b))


# ---------------------------------------------------------------------------
# parse_sup
# ---------------------------------------------------------------------------

class TestParseSup:
    def _write(self, tmp_path, data: bytes):
        sup = tmp_path / "subs.sup"
        sup.write_bytes(data)
        return sup

    def test_single_subtitle_timing(self, tmp_path):
        sup = self._write(tmp_path, _simple_sup(start_ms=1000, end_ms=3000))
        records = parse_sup(sup, tmp_path / "frames")
        assert len(records) == 1
        assert records[0]["start_ms"] == 1000
        assert records[0]["end_ms"] == 3000

    def test_bounding_box_and_canvas(self, tmp_path):
        sup = self._write(tmp_path, _simple_sup(x=100, y=900, w=4, h=4))
        rec = parse_sup(sup, tmp_path / "frames")[0]
        assert (rec["x1"], rec["y1"], rec["x2"], rec["y2"]) == (100, 900, 103, 903)
        assert (rec["canvas_w"], rec["canvas_h"]) == (1920, 1080)

    def test_renders_image_file(self, tmp_path):
        sup = self._write(tmp_path, _simple_sup(w=4, h=4))
        rec = parse_sup(sup, tmp_path / "frames")[0]
        assert rec["image_path"].exists()
        img = Image.open(rec["image_path"]).convert("RGBA")
        assert img.size == (4, 4)
        # The white opaque object should produce visible (opaque) pixels.
        assert img.split()[3].getextrema()[1] == 255

    def test_two_subtitles(self, tmp_path):
        sup = self._write(
            tmp_path,
            _simple_sup(start_ms=1000, end_ms=2000)
            + _simple_sup(x=200, y=100, start_ms=4000, end_ms=6000),
        )
        records = parse_sup(sup, tmp_path / "frames")
        assert len(records) == 2
        assert records[1]["start_ms"] == 4000
        assert records[1]["end_ms"] == 6000

    def test_new_presentation_without_explicit_clear_closes_previous(self, tmp_path):
        # Two presentations back-to-back; the second's PTS ends the first.
        first = (
            _segment(SEG_PCS, _pcs([(0, 100, 900)]), 1000)
            + _segment(SEG_PDS, _pds(0, WHITE), 1000)
            + _segment(SEG_ODS, _ods(0, 4, 4, _solid_rle(4, 4, 1)), 1000)
            + _segment(SEG_END, b"", 1000)
        )
        second = (
            _segment(SEG_PCS, _pcs([(1, 200, 200)], state=0x00), 2500)
            + _segment(SEG_ODS, _ods(1, 4, 4, _solid_rle(4, 4, 1)), 2500)
            + _segment(SEG_END, b"", 2500)
        )
        sup = self._write(tmp_path, first + second)
        records = parse_sup(sup, tmp_path / "frames")
        assert records[0]["end_ms"] == 2500

    def test_trailing_subtitle_has_none_end(self, tmp_path):
        # Presentation with no following clear segment.
        data = (
            _segment(SEG_PCS, _pcs([(0, 100, 900)]), 1000)
            + _segment(SEG_PDS, _pds(0, WHITE), 1000)
            + _segment(SEG_ODS, _ods(0, 4, 4, _solid_rle(4, 4, 1)), 1000)
            + _segment(SEG_END, b"", 1000)
        )
        sup = self._write(tmp_path, data)
        rec = parse_sup(sup, tmp_path / "frames")[0]
        assert rec["end_ms"] is None

    def test_resyncs_past_leading_garbage(self, tmp_path):
        sup = self._write(tmp_path, b"\xDE\xAD\xBE\xEF" + _simple_sup())
        records = parse_sup(sup, tmp_path / "frames")
        assert len(records) == 1

    def test_empty_stream_returns_no_records(self, tmp_path):
        sup = self._write(tmp_path, b"")
        assert parse_sup(sup, tmp_path / "frames") == []


# ---------------------------------------------------------------------------
# count_sup
# ---------------------------------------------------------------------------

class TestCountSup:
    def _write(self, tmp_path, data: bytes):
        sup = tmp_path / "subs.sup"
        sup.write_bytes(data)
        return sup

    def test_counts_single_subtitle(self, tmp_path):
        sup = self._write(tmp_path, _simple_sup())
        assert count_sup(sup) == 1

    def test_counts_multiple_subtitles(self, tmp_path):
        sup = self._write(
            tmp_path,
            _simple_sup(start_ms=1000, end_ms=2000)
            + _simple_sup(x=200, y=100, start_ms=4000, end_ms=6000),
        )
        assert count_sup(sup) == 2

    def test_clear_only_display_sets_not_counted(self, tmp_path):
        # A lone clear (empty PCS) must not count as a subtitle appearance.
        clear = (_segment(SEG_PCS, _pcs([], state=0x00), 5000)
                 + _segment(SEG_END, b"", 5000))
        sup = self._write(tmp_path, _simple_sup() + clear)
        assert count_sup(sup) == 1

    def test_empty_stream_counts_zero(self, tmp_path):
        assert count_sup(self._write(tmp_path, b"")) == 0

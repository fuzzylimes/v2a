"""Tests for spu.py — SPU binary parser."""

import struct
from pathlib import Path

import pytest

from v2a.spu import _skip_ps_header, parse_spu, read_spu_at

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_darea_bytes(x1: int, x2: int, y1: int, y2: int) -> bytes:
    """Pack x1, x2, y1, y2 into the 6-byte SET_DAREA payload."""
    b = bytearray(6)
    b[0] = x1 >> 4
    b[1] = ((x1 & 0x0F) << 4) | (x2 >> 8)
    b[2] = x2 & 0xFF
    b[3] = y1 >> 4
    b[4] = ((y1 & 0x0F) << 4) | (y2 >> 8)
    b[5] = y2 & 0xFF
    return bytes(b)


def _make_spu(delay_units: int, x1=100, x2=620, y1=500, y2=540,
              include_stp=True, include_darea=True) -> bytes:
    """
    Build a minimal SPU payload with a single DCSQ block.

    SPU layout:
      [0:2]  SP_SIZE       (total length, big-endian)
      [2:4]  DCSQ_START    (offset of first DCSQ = 4)
      [4:]   DCSQ block:
               [0:2] delay_units (big-endian)
               [2:4] next_offset = self (marks last block)
               commands...
    """
    CMD_STP_DSP  = 0x02
    CMD_SET_DAREA = 0x05
    CMD_END      = 0xFF

    cmds = bytearray()
    if include_stp:
        cmds += bytes([CMD_STP_DSP])
    if include_darea:
        cmds += bytes([CMD_SET_DAREA]) + _make_darea_bytes(x1, x2, y1, y2)
    cmds += bytes([CMD_END])

    dcsq_start = 4
    # next_offset == dcsq_start → signals last block
    dcsq = struct.pack('>HH', delay_units, dcsq_start) + bytes(cmds)
    total = dcsq_start + len(dcsq)
    header = struct.pack('>HH', total, dcsq_start)
    return header + dcsq


def _make_ps_packet(spu_payload: bytes, pes_hdr_extra: bytes = b'') -> bytes:
    """
    Wrap an SPU payload in a minimal MPEG-PS private_stream_1 packet.

    Structure:
      00 00 01 BD  — start code + stream id
      2 bytes      — PES packet length (unused by our parser)
      2 bytes      — PES flags
      1 byte       — PES header data length N
      N bytes      — optional PES header fields
      1 byte       — sub-stream id (0x20 for first VobSub track)
      payload...
    """
    N = len(pes_hdr_extra)
    header = (
        b'\x00\x00\x01\xbd'
        + b'\x00\x00'          # PES packet length (ignored)
        + b'\x80\x00'          # PES flags
        + bytes([N])           # PES header data length
        + pes_hdr_extra        # optional fields
        + b'\x20'              # sub-stream id
    )
    return header + spu_payload


# ---------------------------------------------------------------------------
# _skip_ps_header
# ---------------------------------------------------------------------------

class TestSkipPsHeader:
    def test_standard_packet_no_optional_fields(self):
        # N=0 → spu starts at byte 10
        packet = _make_ps_packet(b'\xAB\xCD', pes_hdr_extra=b'')
        assert _skip_ps_header(packet) == 10

    def test_packet_with_optional_fields(self):
        # N=5 → spu starts at byte 15
        packet = _make_ps_packet(b'\xAB\xCD', pes_hdr_extra=b'\x00' * 5)
        assert _skip_ps_header(packet) == 15

    def test_wrong_start_code_returns_zero(self):
        bad = b'\x00\x00\x02\xbd' + b'\x00' * 10
        assert _skip_ps_header(bad) == 0

    def test_wrong_stream_id_returns_zero(self):
        bad = b'\x00\x00\x01\xe0' + b'\x00' * 10   # video stream, not 0xBD
        assert _skip_ps_header(bad) == 0

    def test_too_short_returns_zero(self):
        assert _skip_ps_header(b'\x00\x00\x01') == 0

    def test_empty_returns_zero(self):
        assert _skip_ps_header(b'') == 0


# ---------------------------------------------------------------------------
# parse_spu
# ---------------------------------------------------------------------------

class TestParseSpu:
    def test_extracts_end_ms_and_bounding_box(self):
        # delay_units=176 → 176 * 1024/90 ≈ 2000 ms
        spu = _make_spu(delay_units=176, x1=100, x2=620, y1=500, y2=540)
        result = parse_spu(spu)
        assert result['end_ms'] == int(176 * 1024 / 90)
        assert result['x1'] == 100
        assert result['x2'] == 620
        assert result['y1'] == 500
        assert result['y2'] == 540

    def test_zero_delay_gives_zero_end_ms(self):
        spu = _make_spu(delay_units=0)
        result = parse_spu(spu)
        assert result['end_ms'] == 0

    def test_missing_stp_dsp_gives_none_end_ms(self):
        spu = _make_spu(delay_units=100, include_stp=False)
        result = parse_spu(spu)
        assert result['end_ms'] is None

    def test_missing_set_darea_gives_none_coords(self):
        spu = _make_spu(delay_units=100, include_darea=False)
        result = parse_spu(spu)
        assert result['x1'] is None
        assert result['y1'] is None
        assert result['x2'] is None
        assert result['y2'] is None

    def test_too_short_returns_all_none(self):
        result = parse_spu(b'\x00\x01')
        assert result == dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)

    def test_empty_returns_all_none(self):
        result = parse_spu(b'')
        assert result == dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)

    def test_bounding_box_extreme_values(self):
        # x1=0, x2=719, y1=0, y2=575 — full DVD frame
        spu = _make_spu(delay_units=0, x1=0, x2=719, y1=0, y2=575)
        result = parse_spu(spu)
        assert result['x1'] == 0
        assert result['x2'] == 719
        assert result['y1'] == 0
        assert result['y2'] == 575

    def test_skips_set_color_and_set_alpha(self):
        """SET_COLOR (0x03) and SET_ALPHA (0x04) each consume 2 bytes; parser must skip them."""
        CMD_STA_DSP   = 0x01
        CMD_STP_DSP   = 0x02
        CMD_SET_COLOR = 0x03
        CMD_SET_ALPHA = 0x04
        CMD_SET_DAREA = 0x05
        CMD_END       = 0xFF

        cmds = bytes([
            CMD_STA_DSP,
            CMD_SET_COLOR, 0x00, 0x00,
            CMD_SET_ALPHA, 0x00, 0x00,
            CMD_STP_DSP,
            CMD_SET_DAREA,
        ]) + _make_darea_bytes(50, 200, 100, 150) + bytes([CMD_END])

        dcsq_start = 4
        delay_units = 90   # arbitrary
        dcsq = struct.pack('>HH', delay_units, dcsq_start) + cmds
        total = dcsq_start + len(dcsq)
        header = struct.pack('>HH', total, dcsq_start)
        spu = header + dcsq

        result = parse_spu(spu)
        assert result['end_ms'] == int(90 * 1024 / 90)
        assert result['x1'] == 50


# ---------------------------------------------------------------------------
# read_spu_at
# ---------------------------------------------------------------------------

class TestReadSpuAt:
    def test_reads_correct_offset(self, tmp_path):
        spu_payload = _make_spu(delay_units=90, x1=10, x2=100, y1=400, y2=450)
        packet = _make_ps_packet(spu_payload)
        # Write 512 bytes of garbage, then the packet at offset 512
        sub_file = tmp_path / "subs.sub"
        sub_file.write_bytes(b'\x00' * 512 + packet)

        result = read_spu_at(sub_file, filepos=512)
        assert result['end_ms'] == int(90 * 1024 / 90)
        assert result['x1'] == 10
        assert result['x2'] == 100

    def test_bad_offset_returns_all_none(self, tmp_path):
        sub_file = tmp_path / "subs.sub"
        sub_file.write_bytes(b'\x00' * 16)
        result = read_spu_at(sub_file, filepos=0)
        assert result == dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)

    def test_missing_file_returns_all_none(self, tmp_path):
        result = read_spu_at(tmp_path / "missing.sub", filepos=0)
        assert result == dict(end_ms=None, x1=None, y1=None, x2=None, y2=None)
